//! [`FffRetriever`]: fff-search FilePicker behind [`memory_core::Retriever`].

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use fff_search::file_picker::FilePicker;
use fff_search::frecency::FrecencyTracker;
use fff_search::grep::{parse_grep_query, GrepMode as FffGrepMode, GrepSearchOptions};
use fff_search::query_tracker::QueryTracker;
use fff_search::{
    Constraint, FFFMode, FilePickerOptions, FuzzySearchOptions, PaginationArgs, QueryParser,
    SharedFilePicker, SharedFrecency, SharedQueryTracker,
};
use memory_core::{
    ContentHit, Error, FileHit, GrepMode, IndexSnapshot, IndexState, Result, Retriever,
};

/// Default scan wait when opening or reindexing.
const SCAN_TIMEOUT: Duration = Duration::from_secs(30);

/// Max hits returned per find/grep call (MCP ranks/budgets further).
const PAGE_LIMIT: usize = 50;

/// fff-backed retriever. Index state lives under `{root}/.index/`.
///
/// Construction is the only fallible setup: after [`FffRetriever::open`] succeeds,
/// search methods map fff errors into [`Error::Retriever`] so the write path can
/// treat them as best-effort skips.
pub struct FffRetriever {
    root: PathBuf,
    shared_picker: SharedFilePicker,
    shared_frecency: SharedFrecency,
    shared_query_tracker: SharedQueryTracker,
    last_scan_ms: AtomicU64,
}

impl FffRetriever {
    /// Open a memory root, create `.index/` DBs, scan, and wait until ready.
    ///
    /// Init order (required by fff): frecency → query tracker → FilePicker → wait_for_scan.
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let scan_started = Instant::now();
        let root = root
            .as_ref()
            .canonicalize()
            .map_err(|e| Error::io(root.as_ref().to_path_buf(), e))?;

        let index_dir = root.join(".index");
        std::fs::create_dir_all(index_dir.join("frecency"))
            .map_err(|e| Error::io(index_dir.join("frecency"), e))?;
        std::fs::create_dir_all(index_dir.join("queries"))
            .map_err(|e| Error::io(index_dir.join("queries"), e))?;

        let shared_picker = SharedFilePicker::default();
        let shared_frecency = SharedFrecency::default();
        let shared_query_tracker = SharedQueryTracker::default();

        let frecency = FrecencyTracker::open(index_dir.join("frecency"))
            .map_err(|e| Error::Retriever(format!("open frecency db: {e}")))?;
        shared_frecency
            .init(frecency)
            .map_err(|e| Error::Retriever(format!("init frecency: {e}")))?;

        let query_tracker = QueryTracker::open(index_dir.join("queries"))
            .map_err(|e| Error::Retriever(format!("open query tracker: {e}")))?;
        shared_query_tracker
            .init(query_tracker)
            .map_err(|e| Error::Retriever(format!("init query tracker: {e}")))?;

        FilePicker::new_with_shared_state(
            shared_picker.clone(),
            shared_frecency.clone(),
            FilePickerOptions {
                base_path: root.to_string_lossy().into_owned(),
                mode: FFFMode::Ai,
                watch: true,
                ..Default::default()
            },
        )
        .map_err(|e| Error::Retriever(format!("start file picker: {e}")))?;

        if !shared_picker.wait_for_scan(SCAN_TIMEOUT) {
            return Err(Error::Retriever(
                "initial scan timed out; call reindex() or retry open".into(),
            ));
        }
        if !shared_picker.wait_for_watcher(SCAN_TIMEOUT) {
            return Err(Error::Retriever(
                "filesystem watcher timed out; retry open".into(),
            ));
        }

        Ok(Self {
            root,
            shared_picker,
            shared_frecency,
            shared_query_tracker,
            last_scan_ms: AtomicU64::new(elapsed_ms(scan_started)),
        })
    }

    /// Absolute path of the memory root being indexed.
    pub fn root(&self) -> &Path {
        &self.root
    }
}

impl Retriever for FffRetriever {
    fn find_files(
        &self,
        query: &str,
        scope: Option<&str>,
        include_archived: bool,
    ) -> Result<Vec<FileHit>> {
        let guard = self
            .shared_picker
            .read()
            .map_err(|e| Error::Retriever(format!("picker lock: {e}")))?;
        let picker = guard
            .as_ref()
            .ok_or_else(|| Error::Retriever("file picker not initialized".into()))?;

        let qt_guard = self
            .shared_query_tracker
            .read()
            .map_err(|e| Error::Retriever(format!("query tracker lock: {e}")))?;

        let mut hits = find_in_picker(picker, qt_guard.as_ref(), query, scope, None);
        drop(qt_guard);
        drop(guard);

        if include_archived {
            if let Some(archive) = open_archive_picker(&self.root)? {
                hits.extend(find_in_picker(
                    &archive,
                    None,
                    query,
                    scope,
                    Some(Path::new(".archive")),
                ));
            }
        }
        Ok(hits)
    }

    fn grep(
        &self,
        query: &str,
        mode: GrepMode,
        scope: Option<&str>,
        include_archived: bool,
    ) -> Result<Vec<ContentHit>> {
        let guard = self
            .shared_picker
            .read()
            .map_err(|e| Error::Retriever(format!("picker lock: {e}")))?;
        let picker = guard
            .as_ref()
            .ok_or_else(|| Error::Retriever("file picker not initialized".into()))?;

        let fff_mode = match mode {
            GrepMode::Plain => FffGrepMode::PlainText,
            GrepMode::Regex => FffGrepMode::Regex,
            GrepMode::Fuzzy => FffGrepMode::Fuzzy,
        };

        let mut hits = grep_in_picker(picker, query, fff_mode, scope, None);
        drop(guard);

        if include_archived {
            if let Some(archive) = open_archive_picker(&self.root)? {
                hits.extend(grep_in_picker(
                    &archive,
                    query,
                    fff_mode,
                    scope,
                    Some(Path::new(".archive")),
                ));
            }
        }
        Ok(hits)
    }

    fn index_state(&self) -> IndexState {
        let Ok(guard) = self.shared_picker.read() else {
            return IndexState::Unavailable;
        };
        let Some(picker) = guard.as_ref() else {
            return IndexState::Unavailable;
        };
        if picker.is_scan_active() {
            IndexState::Cold
        } else {
            IndexState::Ready
        }
    }

    fn index_snapshot(&self) -> IndexSnapshot {
        let Ok(guard) = self.shared_picker.read() else {
            return IndexSnapshot::default();
        };
        let Some(picker) = guard.as_ref() else {
            return IndexSnapshot::default();
        };
        IndexSnapshot {
            state: if picker.is_scan_active() {
                IndexState::Cold
            } else {
                IndexState::Ready
            },
            files_indexed: picker.live_file_count() as u64,
            last_scan_ms: self.last_scan_ms.load(Ordering::Relaxed),
        }
    }

    fn track_access(&self, path: &Path) -> Result<()> {
        let absolute = self.root.join(path);
        {
            let guard = self
                .shared_frecency
                .read()
                .map_err(|e| Error::Retriever(format!("track access frecency lock: {e}")))?;
            let tracker = guard
                .as_ref()
                .ok_or_else(|| Error::Retriever("frecency tracker not initialized".into()))?;
            tracker
                .track_access(&absolute)
                .map_err(|e| Error::Retriever(format!("track access {}: {e}", path.display())))?;
        }

        let mut picker_guard = self
            .shared_picker
            .write()
            .map_err(|e| Error::Retriever(format!("track access picker lock: {e}")))?;
        let picker = picker_guard
            .as_mut()
            .ok_or_else(|| Error::Retriever("file picker not initialized".into()))?;
        let frecency_guard = self
            .shared_frecency
            .read()
            .map_err(|e| Error::Retriever(format!("track access frecency lock: {e}")))?;
        let tracker = frecency_guard
            .as_ref()
            .ok_or_else(|| Error::Retriever("frecency tracker not initialized".into()))?;
        picker
            .update_single_file_frecency(&absolute, tracker)
            .map_err(|e| Error::Retriever(format!("refresh access score: {e}")))
    }

    fn reindex(&self) -> Result<()> {
        let scan_started = Instant::now();
        let frecency_path = self.root.join(".index/frecency");
        let queries_path = self.root.join(".index/queries");

        self.shared_frecency
            .destroy()
            .map_err(|e| Error::Retriever(format!("reindex destroy frecency: {e}")))?;
        self.shared_query_tracker
            .destroy()
            .map_err(|e| Error::Retriever(format!("reindex destroy queries: {e}")))?;

        let frecency = FrecencyTracker::open(&frecency_path)
            .map_err(|e| Error::Retriever(format!("reindex open frecency: {e}")))?;
        self.shared_frecency
            .init(frecency)
            .map_err(|e| Error::Retriever(format!("reindex init frecency: {e}")))?;
        let queries = QueryTracker::open(&queries_path)
            .map_err(|e| Error::Retriever(format!("reindex open queries: {e}")))?;
        self.shared_query_tracker
            .init(queries)
            .map_err(|e| Error::Retriever(format!("reindex init queries: {e}")))?;

        self.shared_picker
            .trigger_full_rescan_async(&self.shared_frecency)
            .map_err(|e| Error::Retriever(format!("reindex: {e}")))?;
        if !self.shared_picker.wait_for_scan(SCAN_TIMEOUT) {
            return Err(Error::Retriever("reindex scan timed out".into()));
        }
        self.last_scan_ms
            .store(elapsed_ms(scan_started), Ordering::Relaxed);
        Ok(())
    }
}

fn elapsed_ms(started: Instant) -> u64 {
    started.elapsed().as_millis().clamp(1, u64::MAX as u128) as u64
}

fn open_archive_picker(root: &Path) -> Result<Option<FilePicker>> {
    let archive = root.join(".archive");
    if !archive.is_dir() {
        return Ok(None);
    }
    let mut picker = FilePicker::new(FilePickerOptions {
        base_path: archive.to_string_lossy().into_owned(),
        mode: FFFMode::Ai,
        watch: false,
        ..Default::default()
    })
    .map_err(|e| Error::Retriever(format!("open archive picker: {e}")))?;
    picker
        .collect_files()
        .map_err(|e| Error::Retriever(format!("scan archive: {e}")))?;
    Ok(Some(picker))
}

fn find_in_picker(
    picker: &FilePicker,
    query_tracker: Option<&QueryTracker>,
    query: &str,
    scope: Option<&str>,
    prefix: Option<&Path>,
) -> Vec<FileHit> {
    let mut parsed = QueryParser::default().parse(query);
    let scope_glob = scope
        .filter(|s| !s.is_empty())
        .map(|scope| format!("{scope}/**"));
    if let Some(scope_glob) = scope_glob.as_deref() {
        parsed.constraints.push(Constraint::Glob(scope_glob));
    }
    let results = picker.fuzzy_search(
        &parsed,
        query_tracker,
        FuzzySearchOptions {
            max_threads: 0,
            current_file: None,
            pagination: PaginationArgs {
                offset: 0,
                limit: PAGE_LIMIT,
            },
            ..Default::default()
        },
    );
    let mut hits: Vec<FileHit> = results
        .items
        .iter()
        .zip(results.scores.iter())
        .filter_map(|(item, score)| {
            let logical = PathBuf::from(item.relative_path(picker));
            keep_path(&logical, scope).then(|| FileHit {
                path: prefix.map_or(logical.clone(), |p| p.join(&logical)),
                score: score.total as f32,
            })
        })
        .collect();
    normalize_file_hits(&mut hits);
    hits
}

fn grep_in_picker(
    picker: &FilePicker,
    query: &str,
    mode: FffGrepMode,
    scope: Option<&str>,
    prefix: Option<&Path>,
) -> Vec<ContentHit> {
    let mut parsed = parse_grep_query(query);
    let scope_glob = scope
        .filter(|s| !s.is_empty())
        .map(|scope| format!("{scope}/**"));
    if let Some(scope_glob) = scope_glob.as_deref() {
        parsed.constraints.push(Constraint::Glob(scope_glob));
    }
    let mut hits = Vec::new();
    let mut file_offset = 0;
    loop {
        let results = picker.grep(
            &parsed,
            &GrepSearchOptions {
                mode,
                file_offset,
                page_limit: PAGE_LIMIT,
                trim_whitespace: true,
                max_matches_per_file: 5,
                ..Default::default()
            },
        );
        let mut page = Vec::new();
        for m in &results.matches {
            let Some(file) = results.files.get(m.file_index) else {
                continue;
            };
            let logical = PathBuf::from(file.relative_path(picker));
            if keep_path(&logical, scope) {
                page.push(ContentHit {
                    path: prefix.map_or(logical.clone(), |p| p.join(&logical)),
                    snippet: m.line_content.trim().to_string(),
                    line: m.line_number as u32,
                    score: m.fuzzy_score.map(|s| s as f32).unwrap_or(1.0),
                });
            }
        }
        normalize_content_hits(&mut page);
        hits.extend(page);
        if hits.len() >= PAGE_LIMIT
            || results.next_file_offset == 0
            || results.next_file_offset == file_offset
        {
            hits.truncate(PAGE_LIMIT);
            return hits;
        }
        file_offset = results.next_file_offset;
    }
}

fn normalize_file_hits(hits: &mut [FileHit]) {
    let min = hits.iter().map(|h| h.score).fold(f32::INFINITY, f32::min);
    let max = hits
        .iter()
        .map(|h| h.score)
        .fold(f32::NEG_INFINITY, f32::max);
    for hit in hits {
        hit.score = normalize_score(hit.score, min, max);
    }
}

fn normalize_content_hits(hits: &mut [ContentHit]) {
    let min = hits.iter().map(|h| h.score).fold(f32::INFINITY, f32::min);
    let max = hits
        .iter()
        .map(|h| h.score)
        .fold(f32::NEG_INFINITY, f32::max);
    for hit in hits {
        hit.score = normalize_score(hit.score, min, max);
    }
}

fn normalize_score(score: f32, min: f32, max: f32) -> f32 {
    if max <= min {
        1.0
    } else {
        (score - min) / (max - min)
    }
}

/// Drop reserved paths and enforce namespace scope at a segment boundary.
fn keep_path(path: &Path, scope: Option<&str>) -> bool {
    let s = path.to_string_lossy();
    if s == ".archive" || s.starts_with(".archive/") || s.contains("/.archive/") {
        return false;
    }
    // Skip the index itself if it ever appears.
    if s == ".index" || s.starts_with(".index/") {
        return false;
    }
    if let Some(scope) = scope {
        if scope.is_empty() {
            return true;
        }
        return s == scope || s.starts_with(&format!("{scope}/"));
    }
    true
}

#[cfg(test)]
mod path_filter_tests {
    use super::*;

    #[test]
    fn drops_archive_and_index() {
        assert!(!keep_path(Path::new(".archive/old.md"), None));
        assert!(!keep_path(Path::new("ns/.archive/x.md"), None));
        assert!(!keep_path(Path::new(".index/frecency"), None));
        assert!(keep_path(Path::new("ns/note.md"), None));
    }

    #[test]
    fn scope_is_segment_boundary() {
        assert!(keep_path(Path::new("proj/a.md"), Some("proj")));
        assert!(!keep_path(Path::new("project/a.md"), Some("proj")));
        assert!(!keep_path(Path::new("other/a.md"), Some("proj")));
    }
}
