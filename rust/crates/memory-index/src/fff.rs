//! [`FffRetriever`]: fff-search FilePicker behind [`memory_core::Retriever`].

use std::path::{Path, PathBuf};
use std::time::Duration;

use fff_search::file_picker::FilePicker;
use fff_search::frecency::FrecencyTracker;
use fff_search::grep::{parse_grep_query, GrepMode as FffGrepMode, GrepSearchOptions};
use fff_search::query_tracker::QueryTracker;
use fff_search::{
    FFFMode, FilePickerOptions, FuzzySearchOptions, PaginationArgs, QueryParser, SharedFilePicker,
    SharedFrecency, SharedQueryTracker,
};
use memory_core::{ContentHit, Error, FileHit, GrepMode, IndexState, Result, Retriever};

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
}

impl FffRetriever {
    /// Open a memory root, create `.index/` DBs, scan, and wait until ready.
    ///
    /// Init order (required by fff): frecency → query tracker → FilePicker → wait_for_scan.
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
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
                // Tests and short-lived processes: no watcher. Binary can re-open with watch later.
                watch: false,
                ..Default::default()
            },
        )
        .map_err(|e| Error::Retriever(format!("start file picker: {e}")))?;

        if !shared_picker.wait_for_scan(SCAN_TIMEOUT) {
            return Err(Error::Retriever(
                "initial scan timed out; call reindex() or retry open".into(),
            ));
        }

        Ok(Self {
            root,
            shared_picker,
            shared_frecency,
            shared_query_tracker,
        })
    }

    /// Absolute path of the memory root being indexed.
    pub fn root(&self) -> &Path {
        &self.root
    }
}

impl Retriever for FffRetriever {
    fn find_files(&self, query: &str, scope: Option<&str>) -> Result<Vec<FileHit>> {
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

        // Scope as path-ish prefix in the fuzzy query; post-filter enforces it.
        let effective = match scope {
            Some(s) if !s.is_empty() => format!("{s} {query}"),
            _ => query.to_string(),
        };

        let parser = QueryParser::default();
        let parsed = parser.parse(&effective);

        let results = picker.fuzzy_search(
            &parsed,
            qt_guard.as_ref(),
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

        let mut hits = Vec::with_capacity(results.items.len());
        for (item, score) in results.items.iter().zip(results.scores.iter()) {
            let path = PathBuf::from(item.relative_path(picker));
            if !keep_path(&path, scope) {
                continue;
            }
            hits.push(FileHit {
                path,
                score: score.total as f32,
            });
        }
        Ok(hits)
    }

    fn grep(&self, query: &str, mode: GrepMode, scope: Option<&str>) -> Result<Vec<ContentHit>> {
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

        let parsed = parse_grep_query(query);
        let options = GrepSearchOptions {
            mode: fff_mode,
            page_limit: PAGE_LIMIT,
            trim_whitespace: true,
            max_matches_per_file: 5,
            ..Default::default()
        };

        let results = picker.grep(&parsed, &options);

        let mut hits = Vec::with_capacity(results.matches.len());
        for m in &results.matches {
            let Some(file) = results.files.get(m.file_index) else {
                continue;
            };
            let path = PathBuf::from(file.relative_path(picker));
            if !keep_path(&path, scope) {
                continue;
            }
            let score = m.fuzzy_score.map(|s| s as f32).unwrap_or(1.0);
            hits.push(ContentHit {
                path,
                snippet: m.line_content.trim().to_string(),
                line: m.line_number as u32,
                score,
            });
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

    fn reindex(&self) -> Result<()> {
        self.shared_picker
            .trigger_full_rescan_async(&self.shared_frecency)
            .map_err(|e| Error::Retriever(format!("reindex: {e}")))?;
        if !self.shared_picker.wait_for_scan(SCAN_TIMEOUT) {
            return Err(Error::Retriever("reindex scan timed out".into()));
        }
        Ok(())
    }
}

/// Drop `.archive/` and enforce namespace scope (path prefix at segment boundary).
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
