//! Tool business logic: thin orchestration over store + retriever + pipeline.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::search::{
    apply_budget, merge_hits, path_to_handle, rank_hits, BudgetedSearch, MatchStage, NoteMeta,
    BUDGET_BYTES_DEFAULT, MIN_RESULTS_FOR_FUZZY, SEARCH_LIMIT_DEFAULT,
};
use memory_core::{
    extract_wikilinks, AccessLog, AccessSnapshot, AccessVia, GrepMode, IndexState, MemoryStore,
    MergeMode, Note, NoteFrontmatter, NoteType, Retriever, StoreAction, StoreInput, StoreOutcome,
};
use memory_core::{Error, Result};
use serde::{Deserialize, Serialize};
use time::Date;

const AUTOMATION_QUERY_PREFIX: &str = "__fff_memory_user_prompt__:";

/// Options for [`MemoryService::search`].
#[derive(Debug, Clone)]
pub struct SearchOptions {
    /// Query string (required by the tool; empty → empty report).
    pub query: String,
    /// Namespace subtree; `None` or empty = whole root.
    pub namespace: Option<String>,
    /// Max full results (default 8).
    pub limit: usize,
    /// Byte budget (default 4096, max 16384).
    pub budget_bytes: usize,
    /// Include `.archive/` hits when the retriever returns them.
    pub include_archived: bool,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            query: String::new(),
            namespace: None,
            limit: SEARCH_LIMIT_DEFAULT,
            budget_bytes: BUDGET_BYTES_DEFAULT,
            include_archived: false,
        }
    }
}

/// Search tool response (JSON shape from spec 04).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchResponse {
    /// Budgeted hits.
    pub results: Vec<crate::search::SearchHit>,
    /// Overflow handle+title.
    pub more: Vec<crate::search::MoreHit>,
    /// Stages executed.
    pub stages_run: Vec<String>,
    /// Scope string.
    pub scope: String,
    /// Empty-result guidance.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub empty_hint: Option<String>,
}

impl From<BudgetedSearch> for SearchResponse {
    fn from(b: BudgetedSearch) -> Self {
        Self {
            results: b.results,
            more: b.more,
            stages_run: b.stages_run,
            scope: b.scope,
            empty_hint: b.empty_hint,
        }
    }
}

/// Input for store tool (maps to [`StoreInput`]).
#[derive(Debug, Clone)]
pub struct StoreRequest {
    /// Title.
    pub title: String,
    /// Body markdown.
    pub body: String,
    /// Aliases (min 2).
    pub aliases: Vec<String>,
    /// Namespace (default `default` at the tool layer; service accepts any validated ns).
    pub namespace: String,
    /// Note type.
    pub note_type: NoteType,
    /// Tags.
    pub tags: Vec<String>,
    /// Optional expiry.
    pub expires: Option<Date>,
    /// append | replace.
    pub mode: MergeMode,
}

/// Full note read + linked breadcrumbs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadOutcome {
    /// Parsed frontmatter as JSON-friendly value.
    pub frontmatter: NoteFrontmatter,
    /// Body markdown.
    pub body: String,
    /// One-hop wikilink targets with titles when resolvable.
    pub linked: Vec<LinkedNote>,
}

/// Linked note breadcrumb.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinkedNote {
    /// Handle.
    pub handle: String,
    /// Title when the target exists.
    pub title: Option<String>,
}

/// Forget result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForgetOutcome {
    /// Handle.
    pub handle: String,
    /// archived | deleted.
    pub action: ForgetAction,
}

/// Archive vs hard delete.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ForgetAction {
    /// Moved to `.archive/`.
    Archived,
    /// File removed.
    Deleted,
}

/// Stats snapshot for `memory_stats`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatsSnapshot {
    /// Total note files (excluding `.index`).
    pub total_notes: u64,
    /// Counts by namespace (first path segment or `""`).
    pub by_namespace: HashMap<String, u64>,
    /// Counts by type string.
    pub by_type: HashMap<String, u64>,
    /// Approximate disk usage of note files.
    pub disk_bytes: u64,
    /// Index readiness from retriever when present.
    pub index: IndexStats,
    /// Lightweight decay signals.
    pub decay: DecayStats,
}

/// Index portion of stats.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexStats {
    /// `ready` | `cold` | `unavailable`.
    pub state: String,
    /// Live, non-tombstoned files in the active index.
    pub files_indexed: u64,
    /// Wall-clock duration of the most recently completed full scan.
    pub last_scan_ms: u64,
    /// Access log line count.
    pub access_log_events: u64,
}

/// Decay portion of stats (advisory).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecayStats {
    /// Notes with no access-log entries and created > 30d ago (best-effort).
    pub never_accessed_30d: u64,
    /// Notes with `expires` in the past or within 7 days.
    pub expiring_soon: u64,
}

/// Orchestrates store + search + lifecycle tools for MCP adapters.
#[derive(Clone)]
pub struct MemoryService {
    store: MemoryStore,
    access: AccessLog,
    retriever: Option<Arc<dyn Retriever>>,
}

impl MemoryService {
    /// Build a service over an existing store root.
    pub fn new(root: impl Into<PathBuf>, retriever: Option<Arc<dyn Retriever>>) -> Self {
        let root = root.into();
        let store = MemoryStore::new(root.clone());
        let access = AccessLog::open(&root);
        Self {
            store,
            access,
            retriever,
        }
    }

    /// Memory root path.
    pub fn root(&self) -> &Path {
        self.store.root()
    }

    /// Store or update a note. Never fails for search/index reasons.
    pub fn store(&self, req: StoreRequest) -> Result<StoreOutcome> {
        let input = StoreInput {
            title: req.title,
            body: req.body,
            aliases: req.aliases,
            namespace: req.namespace,
            note_type: req.note_type,
            tags: req.tags,
            expires: req.expires,
            source: None,
            mode: req.mode,
        };
        let r = self.retriever.as_deref();
        self.store.store(input, r)
    }

    /// Layered search: stages 1–3 via retriever, 4–6 via pipeline.
    pub fn search(&self, opts: SearchOptions) -> Result<SearchResponse> {
        let raw_query = opts.query.trim();
        let (query, automatic) = raw_query
            .strip_prefix(AUTOMATION_QUERY_PREFIX)
            .map_or((raw_query, false), |query| (query.trim(), true));
        let scope_owned = opts
            .namespace
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string());
        let scope = scope_owned.as_deref();
        let scope_label = scope.unwrap_or("").to_string();

        if automatic && should_skip_automatic_query(query) {
            return Ok(SearchResponse {
                results: Vec::new(),
                more: Vec::new(),
                stages_run: Vec::new(),
                scope: scope_label,
                empty_hint: Some(
                    "search intentionally skipped for a short, command, or greeting prompt".into(),
                ),
            });
        }
        if query.is_empty() {
            return Ok(SearchResponse {
                results: Vec::new(),
                more: Vec::new(),
                stages_run: Vec::new(),
                scope: scope_label,
                empty_hint: Some(
                    "no search stages ran; supply a non-empty query, then try broader terms".into(),
                ),
            });
        }
        let Some(retriever) = self.retriever.as_ref() else {
            return Ok(index_unavailable_response(
                &scope_label,
                IndexState::Unavailable,
            ));
        };
        let index_state = retriever.index_state();
        if index_state != IndexState::Ready {
            return Ok(index_unavailable_response(&scope_label, index_state));
        }

        let mut stages = Vec::new();
        let mut file_raw: Vec<(PathBuf, f32)> = Vec::new();
        let mut plain_raw = Vec::new();
        let mut fuzzy_raw = Vec::new();

        stages.push(MatchStage::FindFiles);
        stages.push(MatchStage::GrepPlain);
        let (find_result, plain_result) = std::thread::scope(|threads| {
            let find = threads.spawn(|| retriever.find_files(query, scope, opts.include_archived));
            let plain = threads
                .spawn(|| retriever.grep(query, GrepMode::Plain, scope, opts.include_archived));
            (find.join(), plain.join())
        });
        let find_hits = find_result
            .map_err(|_| stage_error("find_files", query, scope, "worker panicked"))?
            .map_err(|e| stage_error("find_files", query, scope, &e.to_string()))?;
        let plain_hits = plain_result
            .map_err(|_| stage_error("grep_plain", query, scope, "worker panicked"))?
            .map_err(|e| stage_error("grep_plain", query, scope, &e.to_string()))?;

        for h in find_hits {
            if keep_path(self.store.root(), &h.path, opts.include_archived) {
                file_raw.push((h.path, h.score));
            }
        }
        for h in plain_hits {
            if keep_path(self.store.root(), &h.path, opts.include_archived) {
                plain_raw.push(h);
            }
        }

        let mut merged = merge_hits(&file_raw, &plain_raw, &[]);

        // Stage 3: fuzzy escalate
        if merged.len() < MIN_RESULTS_FOR_FUZZY {
            stages.push(MatchStage::GrepFuzzy);
            let hits = retriever
                .grep(query, GrepMode::Fuzzy, scope, opts.include_archived)
                .map_err(|e| stage_error("grep_fuzzy", query, scope, &e.to_string()))?;
            for h in hits {
                if keep_path(self.store.root(), &h.path, opts.include_archived) {
                    fuzzy_raw.push(h);
                }
            }
            merged = merge_hits(&file_raw, &plain_raw, &fuzzy_raw);
        }

        // Access history is advisory for ranking; corrupt state gives neutral frecency.
        let access_snapshot = self.access.snapshot().ok();
        let mut meta = HashMap::new();
        for m in &merged {
            if let Some(nm) = self.load_meta(&m.path, access_snapshot.as_ref()) {
                meta.insert(m.path.clone(), nm);
            }
        }

        let ranked = rank_hits(merged, &meta);

        let backend_paths: HashMap<String, PathBuf> = ranked
            .iter()
            .map(|hit| (hit.handle.clone(), hit.path.clone()))
            .collect();
        let budgeted = apply_budget(ranked, opts.limit, opts.budget_bytes, &stages, &scope_label);
        for hit in &budgeted.results {
            let _ = self.access.append(&hit.handle, AccessVia::SearchHit);
            if let Some(path) = backend_paths.get(&hit.handle) {
                let _ = retriever.track_access(path);
            }
        }
        Ok(budgeted.into())
    }

    /// Read full note; records a read access event.
    pub fn read(&self, handle: &str) -> Result<ReadOutcome> {
        let (ns, slug) = parse_handle(handle)?;
        let note = self.read_active_or_archived(&ns, &slug)?;
        let handle_norm = MemoryStore::handle(&ns, &slug);
        let _ = self.access.append(&handle_norm, AccessVia::Read);

        let mut linked = Vec::new();
        for link in extract_wikilinks(&note.body) {
            let link_ns = link.namespace.clone().unwrap_or_else(|| ns.clone());
            let link_handle = MemoryStore::handle(&link_ns, &link.slug);
            let title = self
                .store
                .read(&link_ns, &link.slug)
                .ok()
                .map(|n| n.frontmatter.title);
            linked.push(LinkedNote {
                handle: link_handle,
                title,
            });
        }

        Ok(ReadOutcome {
            frontmatter: note.frontmatter,
            body: note.body,
            linked,
        })
    }

    fn read_active_or_archived(&self, namespace: &str, slug: &str) -> Result<Note> {
        match self.store.read(namespace, slug) {
            Ok(note) => Ok(note),
            Err(Error::NotFound { .. }) => {
                let path = self
                    .store
                    .root()
                    .join(".archive")
                    .join(MemoryStore::relative_path(namespace, slug));
                let text = fs::read_to_string(&path).map_err(|e| {
                    if e.kind() == std::io::ErrorKind::NotFound {
                        Error::NotFound {
                            handle: MemoryStore::handle(namespace, slug),
                        }
                    } else {
                        Error::io(&path, e)
                    }
                })?;
                Note::parse(&text)
            }
            Err(error) => Err(error),
        }
    }

    /// Archive (default) or hard-delete a note.
    pub fn forget(&self, handle: &str, hard: bool) -> Result<ForgetOutcome> {
        let (ns, slug) = parse_handle(handle)?;
        let mut abs = self.store.absolute_path(&ns, &slug);
        if hard && !abs.is_file() {
            abs = self
                .store
                .root()
                .join(".archive")
                .join(MemoryStore::relative_path(&ns, &slug));
        }
        if !abs.is_file() {
            return Err(Error::NotFound {
                handle: MemoryStore::handle(&ns, &slug),
            });
        }

        if hard {
            fs::remove_file(&abs).map_err(|e| Error::io(&abs, e))?;
            return Ok(ForgetOutcome {
                handle: MemoryStore::handle(&ns, &slug),
                action: ForgetAction::Deleted,
            });
        }

        self.store.archive_note(&ns, &slug)?;
        Ok(ForgetOutcome {
            handle: MemoryStore::handle(&ns, &slug),
            action: ForgetAction::Archived,
        })
    }

    /// Store health snapshot.
    pub fn stats(&self, namespace: Option<&str>) -> Result<StatsSnapshot> {
        let access_snapshot = self.access.snapshot()?;
        let mut total = 0u64;
        let mut by_namespace: HashMap<String, u64> = HashMap::new();
        let mut by_type: HashMap<String, u64> = HashMap::new();
        let mut disk_bytes = 0u64;
        let mut never_accessed_30d = 0u64;
        let mut expiring_soon = 0u64;

        let today = time::OffsetDateTime::now_utc().date();
        let ns_filter = namespace.map(str::trim).filter(|s| !s.is_empty());

        for (rel, note, size) in walk_notes(self.store.root())? {
            if is_under_reserved(&rel) {
                continue;
            }
            if let Some(ns) = ns_filter {
                if !path_in_namespace(&rel, ns) {
                    continue;
                }
            }

            total += 1;
            disk_bytes += size;

            let (ns, _slug) = split_rel(&rel);
            *by_namespace.entry(ns.clone()).or_insert(0) += 1;

            let type_key = note_type_key(note.frontmatter.note_type);
            *by_type.entry(type_key.into()).or_insert(0) += 1;

            let handle = path_to_handle(&rel);
            if !access_snapshot.ever_accessed(&handle) {
                let age_days = (today - note.frontmatter.created).whole_days();
                if age_days >= 30 {
                    never_accessed_30d += 1;
                }
            }
            if let Some(exp) = note.frontmatter.expires {
                let days = (exp - today).whole_days();
                if days <= 7 {
                    expiring_soon += 1;
                }
            }
        }

        let snapshot = self
            .retriever
            .as_ref()
            .map(|r| r.index_snapshot())
            .unwrap_or_default();
        let state = match snapshot.state {
            IndexState::Ready => "ready",
            IndexState::Cold => "cold",
            IndexState::Unavailable => "unavailable",
        };

        Ok(StatsSnapshot {
            total_notes: total,
            by_namespace,
            by_type,
            disk_bytes,
            index: IndexStats {
                state: state.into(),
                files_indexed: snapshot.files_indexed,
                last_scan_ms: snapshot.last_scan_ms,
                access_log_events: access_snapshot.event_count() as u64,
            },
            decay: DecayStats {
                never_accessed_30d,
                expiring_soon,
            },
        })
    }

    fn load_meta(&self, rel: &Path, access: Option<&AccessSnapshot>) -> Option<NoteMeta> {
        let abs = self.store.root().join(rel);
        let text = fs::read_to_string(abs).ok()?;
        let note = Note::parse(&text).ok()?;
        let handle = path_to_handle(rel);
        let access_counts = access
            .map(|snapshot| snapshot.recent_counts(&handle, 30))
            .unwrap_or_default();
        Some(NoteMeta {
            title: note.frontmatter.title,
            note_type: note.frontmatter.note_type,
            read_count_30d: access_counts.read as u32,
            search_hit_count_30d: access_counts.search_hit as u32,
            handle,
        })
    }
}

fn should_skip_automatic_query(query: &str) -> bool {
    if query.chars().count() < 20 || query.starts_with('/') {
        return true;
    }
    matches!(
        query.to_ascii_lowercase().as_str(),
        "hi" | "hello"
            | "hey"
            | "hello there"
            | "hey there"
            | "good morning"
            | "good afternoon"
            | "good evening"
            | "how are you?"
            | "hello, how are you doing?"
    )
}

fn index_unavailable_response(scope: &str, state: IndexState) -> SearchResponse {
    let state = match state {
        IndexState::Cold => "cold",
        IndexState::Unavailable => "unavailable",
        IndexState::Ready => "ready",
    };
    SearchResponse {
        results: Vec::new(),
        more: Vec::new(),
        stages_run: Vec::new(),
        scope: scope.to_string(),
        empty_hint: Some(format!(
            "search index {state}; no search stages ran; retry after startup or run `fff-memory reindex` and check index path permissions"
        )),
    }
}

fn stage_error(stage: &str, query: &str, scope: Option<&str>, detail: &str) -> Error {
    Error::Retriever(format!(
        "stage={stage} query={query:?} scope={:?}: {detail}",
        scope.unwrap_or("")
    ))
}

fn keep_path(root: &Path, path: &Path, include_archived: bool) -> bool {
    let archived = path.components().any(|c| c.as_os_str() == ".archive");
    let index = path.components().any(|c| c.as_os_str() == ".index");
    if index {
        return false;
    }
    if archived && !include_archived {
        return false;
    }
    if archived {
        let Ok(logical) = path.strip_prefix(".archive") else {
            return false;
        };
        if root.join(logical).is_file() {
            return false;
        }
    }
    path.extension().and_then(|e| e.to_str()) == Some("md")
}

fn is_under_reserved(rel: &Path) -> bool {
    rel.components().any(|c| {
        let s = c.as_os_str();
        s == ".index" || s == ".archive"
    })
}

fn path_in_namespace(rel: &Path, namespace: &str) -> bool {
    let p = rel.to_string_lossy().replace('\\', "/");
    p == namespace || p.starts_with(&format!("{namespace}/"))
}

fn split_rel(rel: &Path) -> (String, String) {
    let stem = rel
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_string();
    let parent = rel.parent().unwrap_or_else(|| Path::new(""));
    let ns = if parent.as_os_str().is_empty() {
        String::new()
    } else {
        parent.to_string_lossy().replace('\\', "/")
    };
    (ns, stem)
}

fn note_type_key(t: NoteType) -> &'static str {
    match t {
        NoteType::Fact => "fact",
        NoteType::Decision => "decision",
        NoteType::Preference => "preference",
        NoteType::Lesson => "lesson",
        NoteType::Reference => "reference",
        NoteType::SessionSummary => "session-summary",
    }
}

/// Parse `namespace/slug` or `slug` (delegates to [`MemoryStore::parse_handle`]).
pub fn parse_handle(handle: &str) -> Result<(String, String)> {
    MemoryStore::parse_handle(handle)
}

fn walk_notes(root: &Path) -> Result<Vec<(PathBuf, Note, u64)>> {
    let mut out = Vec::new();
    if !root.exists() {
        return Ok(out);
    }
    walk_dir(root, root, &mut out)?;
    Ok(out)
}

fn walk_dir(root: &Path, dir: &Path, out: &mut Vec<(PathBuf, Note, u64)>) -> Result<()> {
    let entries = fs::read_dir(dir).map_err(|e| Error::io(dir, e))?;
    for entry in entries {
        let entry = entry.map_err(|e| Error::io(dir, e))?;
        let path = entry.path();
        let name = entry.file_name();
        let name_s = name.to_string_lossy();
        if name_s == ".index" {
            continue;
        }
        // Skip .archive for default stats (counts active notes only)
        if name_s == ".archive" {
            continue;
        }
        if path.is_dir() {
            walk_dir(root, &path, out)?;
        } else if path.extension().and_then(|e| e.to_str()) == Some("md") {
            let meta = fs::metadata(&path).map_err(|e| Error::io(&path, e))?;
            let text = fs::read_to_string(&path).map_err(|e| Error::io(&path, e))?;
            if let Ok(note) = Note::parse(&text) {
                let rel = path.strip_prefix(root).unwrap_or(&path).to_path_buf();
                out.push((rel, note, meta.len()));
            }
        }
    }
    Ok(())
}

/// Map store action to tool JSON string.
pub fn store_action_str(a: StoreAction) -> &'static str {
    match a {
        StoreAction::Created => "created",
        StoreAction::Updated => "updated",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use memory_core::testing::FakeRetriever;
    use memory_core::{ContentHit, ContentMatch, FileHit, IndexSnapshot};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Barrier, Mutex};
    use std::time::{Duration, Instant};
    use tempfile::tempdir;

    fn seed_note(root: &Path, rel: &str, title: &str, aliases: &[&str], body: &str) {
        let path = root.join(rel);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        let aliases_yaml = aliases
            .iter()
            .map(|a| format!("  - {a}"))
            .collect::<Vec<_>>()
            .join("\n");
        let md = format!(
            "---\ntitle: {title}\naliases:\n{aliases_yaml}\ntype: fact\ncreated: 2026-01-01\nupdated: 2026-01-01\n---\n{body}\n"
        );
        fs::write(path, md).unwrap();
    }

    #[test]
    fn parse_handle_variants() {
        assert_eq!(
            parse_handle("proj/sub/note").unwrap(),
            ("proj/sub".into(), "note".into())
        );
        assert_eq!(parse_handle("note").unwrap(), ("".into(), "note".into()));
        assert_eq!(
            parse_handle("proj/note.md").unwrap(),
            ("proj".into(), "note".into())
        );
        assert!(parse_handle("").is_err());
        assert!(parse_handle("../x").is_err());
    }

    #[test]
    fn search_default_budget_is_4096_bytes() {
        assert_eq!(SearchOptions::default().budget_bytes, BUDGET_BYTES_DEFAULT);
        assert_eq!(BUDGET_BYTES_DEFAULT, 4096);
    }

    #[test]
    fn store_and_read_roundtrip() {
        let dir = tempdir().unwrap();
        let svc = MemoryService::new(dir.path(), None);
        let out = svc
            .store(StoreRequest {
                title: "Hello World".into(),
                body: "Body with [[other]].".into(),
                aliases: vec!["hi".into(), "hello".into()],
                namespace: "proj".into(),
                note_type: NoteType::Fact,
                tags: vec![],
                expires: None,
                mode: MergeMode::Append,
            })
            .unwrap();
        assert_eq!(out.action, StoreAction::Created);
        assert_eq!(out.slug, "hello-world");

        // Create linked target
        seed_note(dir.path(), "proj/other.md", "Other", &["o1", "o2"], "x");

        let read = svc.read("proj/hello-world").unwrap();
        assert_eq!(read.frontmatter.title, "Hello World");
        assert!(read.body.contains("Body"));
        assert_eq!(read.linked.len(), 1);
        assert_eq!(read.linked[0].handle, "proj/other");
        assert_eq!(read.linked[0].title.as_deref(), Some("Other"));

        // Access log recorded
        assert!(svc.access.count_recent("proj/hello-world", 30) >= 1);
    }

    #[test]
    fn search_merges_find_and_grep_with_fake() {
        let dir = tempdir().unwrap();
        seed_note(
            dir.path(),
            "proj/shipping.md",
            "Shipping Cadence",
            &["release frequency", "deploys"],
            "We ship weekly.",
        );

        let fake = FakeRetriever {
            state: IndexState::Ready,
            files: Mutex::new(vec![FileHit {
                path: PathBuf::from("proj/shipping.md"),
                score: 1.0,
            }]),
            contents: Mutex::new(vec![ContentHit {
                path: PathBuf::from("proj/shipping.md"),
                snippet: "aliases:\n  - release frequency".into(),
                line: 3,
                score: 0.9,
                matched: ContentMatch::Alias("release frequency".into()),
            }]),
            find_error: Mutex::new(None),
            grep_error: Mutex::new(None),
            reindex_calls: Mutex::new(0),
        };

        let svc = MemoryService::new(dir.path(), Some(Arc::new(fake)));
        let resp = svc
            .search(SearchOptions {
                query: "shipping".into(),
                namespace: Some("proj".into()),
                limit: 8,
                budget_bytes: 4096,
                include_archived: false,
            })
            .unwrap();

        assert!(!resp.results.is_empty());
        assert_eq!(resp.results[0].handle, "proj/shipping");
        assert!(resp.stages_run.iter().any(|s| s == "find_files"));
        assert!(resp.stages_run.iter().any(|s| s == "grep_plain"));
        assert!(resp.empty_hint.is_none());
    }

    #[test]
    fn search_why_names_the_alias_that_matched() {
        let dir = tempdir().unwrap();
        seed_note(
            dir.path(),
            "proj/shipping.md",
            "Shipping Cadence",
            &["first alias", "release frequency"],
            "We ship weekly.",
        );
        let fake = FakeRetriever {
            state: IndexState::Ready,
            contents: Mutex::new(vec![ContentHit {
                path: PathBuf::from("proj/shipping.md"),
                snippet: "aliases: [first alias, release frequency]".into(),
                line: 3,
                score: 1.0,
                matched: ContentMatch::Alias("release frequency".into()),
            }]),
            ..FakeRetriever::ready()
        };
        let svc = MemoryService::new(dir.path(), Some(Arc::new(fake)));

        let response = svc
            .search(SearchOptions {
                query: "release frequency".into(),
                namespace: Some("proj".into()),
                ..Default::default()
            })
            .unwrap();

        assert_eq!(response.results[0].why, "alias:release frequency");
    }

    #[test]
    fn search_why_reports_title_tags_and_body() {
        let dir = tempdir().unwrap();
        for (name, title, body) in [
            ("title", "Needle Title", "body"),
            ("tags", "Tag Note", "body"),
            ("body", "Body Note", "needle body"),
        ] {
            seed_note(
                dir.path(),
                &format!("proj/{name}.md"),
                title,
                &["first alias", "second alias"],
                body,
            );
        }
        let fake = FakeRetriever {
            state: IndexState::Ready,
            contents: Mutex::new(vec![
                ContentHit {
                    path: PathBuf::from("proj/title.md"),
                    snippet: "title: Needle Title".into(),
                    line: 2,
                    score: 1.0,
                    matched: ContentMatch::Title,
                },
                ContentHit {
                    path: PathBuf::from("proj/tags.md"),
                    snippet: "- needle".into(),
                    line: 6,
                    score: 0.9,
                    matched: ContentMatch::Tags,
                },
                ContentHit {
                    path: PathBuf::from("proj/body.md"),
                    snippet: "needle body".into(),
                    line: 9,
                    score: 0.8,
                    matched: ContentMatch::Body,
                },
            ]),
            ..FakeRetriever::ready()
        };
        let svc = MemoryService::new(dir.path(), Some(Arc::new(fake)));

        let response = svc
            .search(SearchOptions {
                query: "needle".into(),
                namespace: Some("proj".into()),
                ..Default::default()
            })
            .unwrap();
        let why: HashMap<_, _> = response
            .results
            .into_iter()
            .map(|hit| (hit.handle, hit.why))
            .collect();

        assert_eq!(why["proj/title"], "title");
        assert_eq!(why["proj/tags"], "tags");
        assert_eq!(why["proj/body"], "content");
    }

    #[test]
    fn search_uses_neutral_frecency_when_access_state_is_corrupt() {
        let dir = tempdir().unwrap();
        seed_note(
            dir.path(),
            "proj/shipping.md",
            "Shipping Cadence",
            &["release frequency", "deploys"],
            "We ship weekly.",
        );
        let access = AccessLog::open(dir.path());
        fs::create_dir_all(access.path().parent().unwrap()).unwrap();
        fs::write(access.path(), "{corrupt}\n").unwrap();
        let fake = FakeRetriever {
            state: IndexState::Ready,
            files: Mutex::new(vec![FileHit {
                path: PathBuf::from("proj/shipping.md"),
                score: 1.0,
            }]),
            ..Default::default()
        };
        let svc = MemoryService::new(dir.path(), Some(Arc::new(fake)));

        let response = svc
            .search(SearchOptions {
                query: "shipping".into(),
                namespace: Some("proj".into()),
                ..Default::default()
            })
            .unwrap();

        assert_eq!(response.results[0].handle, "proj/shipping");
        assert!(!response.results[0].why.contains("frecency"));
    }

    #[test]
    fn search_empty_explains_stages() {
        let dir = tempdir().unwrap();
        let fake = FakeRetriever::ready();
        let svc = MemoryService::new(dir.path(), Some(Arc::new(fake)));
        let resp = svc
            .search(SearchOptions {
                query: "zzzz-no-match".into(),
                ..Default::default()
            })
            .unwrap();
        assert!(resp.results.is_empty());
        let hint = resp.empty_hint.expect("hint");
        assert!(hint.contains("find_files") || hint.contains("grep"));
    }

    #[test]
    fn unavailable_or_cold_index_reports_no_unexecuted_stages() {
        let dir = tempdir().unwrap();
        for (retriever, state) in [
            (None, "unavailable"),
            (
                Some(Arc::new(FakeRetriever {
                    state: IndexState::Cold,
                    ..Default::default()
                }) as Arc<dyn Retriever>),
                "cold",
            ),
        ] {
            let response = MemoryService::new(dir.path(), retriever)
                .search(SearchOptions {
                    query: "anything".into(),
                    ..Default::default()
                })
                .unwrap();
            assert!(response.results.is_empty());
            assert!(response.stages_run.is_empty());
            let hint = response.empty_hint.unwrap();
            assert!(hint.contains(state), "hint={hint}");
            assert!(
                hint.contains("reindex") || hint.contains("retry"),
                "hint={hint}"
            );
        }
    }

    #[test]
    fn empty_query_reports_no_unexecuted_stages() {
        let dir = tempdir().unwrap();
        let response = MemoryService::new(dir.path(), Some(Arc::new(FakeRetriever::ready())))
            .search(SearchOptions::default())
            .unwrap();
        assert!(response.results.is_empty());
        assert!(response.stages_run.is_empty());
        assert!(response.empty_hint.unwrap().contains("non-empty query"));
    }

    const TEST_AUTOMATION_QUERY_PREFIX: &str = "__fff_memory_user_prompt__:";

    struct CountingRetriever {
        calls: Arc<AtomicUsize>,
        queries: Arc<Mutex<Vec<String>>>,
    }

    impl Retriever for CountingRetriever {
        fn find_files(
            &self,
            query: &str,
            _scope: Option<&str>,
            _include_archived: bool,
        ) -> Result<Vec<FileHit>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.queries.lock().unwrap().push(query.to_string());
            Ok(Vec::new())
        }

        fn grep(
            &self,
            query: &str,
            _mode: GrepMode,
            _scope: Option<&str>,
            _include_archived: bool,
        ) -> Result<Vec<ContentHit>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.queries.lock().unwrap().push(query.to_string());
            Ok(Vec::new())
        }

        fn index_state(&self) -> IndexState {
            self.calls.fetch_add(1, Ordering::SeqCst);
            IndexState::Ready
        }

        fn indexed_paths(&self) -> Result<Vec<PathBuf>> {
            Ok(Vec::new())
        }

        fn reindex(&self) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn hook_noise_queries_skip_without_touching_the_retriever() {
        for query in [
            "Unicode 🦀 query",
            "/memory-status",
            "hello, how are you doing?",
        ] {
            let dir = tempdir().unwrap();
            let calls = Arc::new(AtomicUsize::new(0));
            let queries = Arc::new(Mutex::new(Vec::new()));
            let service = MemoryService::new(
                dir.path(),
                Some(Arc::new(CountingRetriever {
                    calls: calls.clone(),
                    queries: queries.clone(),
                })),
            );

            let response = service
                .search(SearchOptions {
                    query: format!("{TEST_AUTOMATION_QUERY_PREFIX}{query}"),
                    ..Default::default()
                })
                .unwrap();

            assert!(response.results.is_empty());
            assert!(response.stages_run.is_empty());
            assert_eq!(calls.load(Ordering::SeqCst), 0, "query={query:?}");
            assert!(queries.lock().unwrap().is_empty());
            assert!(response
                .empty_hint
                .unwrap()
                .contains("intentionally skipped"));
        }
    }

    #[test]
    fn manual_short_and_slash_queries_reach_the_retriever() {
        for query in ["deploy", "/memory-status"] {
            let dir = tempdir().unwrap();
            let calls = Arc::new(AtomicUsize::new(0));
            let queries = Arc::new(Mutex::new(Vec::new()));
            let service = MemoryService::new(
                dir.path(),
                Some(Arc::new(CountingRetriever {
                    calls: calls.clone(),
                    queries: queries.clone(),
                })),
            );

            service
                .search(SearchOptions {
                    query: query.into(),
                    ..Default::default()
                })
                .unwrap();

            assert!(calls.load(Ordering::SeqCst) > 0, "query={query:?}");
            assert!(queries.lock().unwrap().iter().all(|seen| seen == query));
        }
    }

    #[test]
    fn marked_substantive_query_is_stripped_before_retrieval() {
        let dir = tempdir().unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let queries = Arc::new(Mutex::new(Vec::new()));
        let service = MemoryService::new(
            dir.path(),
            Some(Arc::new(CountingRetriever {
                calls: calls.clone(),
                queries: queries.clone(),
            })),
        );
        let query = "explain the deployment policy decision";

        service
            .search(SearchOptions {
                query: format!("{TEST_AUTOMATION_QUERY_PREFIX}{query}"),
                ..Default::default()
            })
            .unwrap();

        assert!(calls.load(Ordering::SeqCst) > 0);
        assert!(queries.lock().unwrap().iter().all(|seen| seen == query));
    }

    #[test]
    fn archived_search_uses_logical_handle() {
        let dir = tempdir().unwrap();
        seed_note(
            dir.path(),
            ".archive/proj/old.md",
            "Old Note",
            &["retired note", "archived note"],
            "retired body",
        );
        let fake = FakeRetriever {
            state: IndexState::Ready,
            contents: Mutex::new(vec![ContentHit {
                path: PathBuf::from(".archive/proj/old.md"),
                snippet: "retired body".into(),
                line: 8,
                score: 1.0,
                matched: ContentMatch::Body,
            }]),
            ..FakeRetriever::ready()
        };
        let svc = MemoryService::new(dir.path(), Some(Arc::new(fake)));

        let response = svc
            .search(SearchOptions {
                query: "retired".into(),
                namespace: Some("proj".into()),
                include_archived: true,
                ..Default::default()
            })
            .expect("archived search");

        assert_eq!(response.results[0].handle, "proj/old");
        let read = svc
            .read(&response.results[0].handle)
            .expect("read archived search result");
        assert_eq!(read.frontmatter.title, "Old Note");
        assert_eq!(read.body.trim(), "retired body");
    }

    #[test]
    fn active_and_archived_collision_returns_one_active_result() {
        let dir = tempdir().unwrap();
        seed_note(
            dir.path(),
            "proj/same.md",
            "Active Note",
            &["current note", "live note"],
            "active body",
        );
        seed_note(
            dir.path(),
            ".archive/proj/same.md",
            "Archived Note",
            &["retired note", "old note"],
            "archived body",
        );
        let fake = FakeRetriever {
            state: IndexState::Ready,
            files: Mutex::new(vec![
                FileHit {
                    path: PathBuf::from(".archive/proj/same.md"),
                    score: 1.0,
                },
                FileHit {
                    path: PathBuf::from("proj/same.md"),
                    score: 0.5,
                },
            ]),
            ..FakeRetriever::ready()
        };
        let svc = MemoryService::new(dir.path(), Some(Arc::new(fake)));

        let response = svc
            .search(SearchOptions {
                query: "same".into(),
                namespace: Some("proj".into()),
                include_archived: true,
                ..Default::default()
            })
            .expect("collision search");

        assert_eq!(response.results.len(), 1, "response={response:?}");
        assert_eq!(response.results[0].handle, "proj/same");
        assert_eq!(response.results[0].title, "Active Note");
        assert_eq!(svc.read("proj/same").unwrap().body.trim(), "active body");
    }

    #[test]
    fn archived_hit_is_shadowed_when_same_handle_is_active() {
        let dir = tempdir().unwrap();
        seed_note(
            dir.path(),
            "proj/same.md",
            "Active Note",
            &["current note", "live note"],
            "active body",
        );
        seed_note(
            dir.path(),
            ".archive/proj/same.md",
            "Archived Note",
            &["retired note", "old note"],
            "archive-only-needle",
        );
        let fake = FakeRetriever {
            state: IndexState::Ready,
            contents: Mutex::new(vec![ContentHit {
                path: PathBuf::from(".archive/proj/same.md"),
                snippet: "archive-only-needle".into(),
                line: 8,
                score: 1.0,
                matched: ContentMatch::Body,
            }]),
            ..FakeRetriever::ready()
        };
        let svc = MemoryService::new(dir.path(), Some(Arc::new(fake)));

        let response = svc
            .search(SearchOptions {
                query: "archive-only-needle".into(),
                namespace: Some("proj".into()),
                include_archived: true,
                ..Default::default()
            })
            .expect("shadowed archive search");

        assert!(response.results.is_empty(), "response={response:?}");
    }

    struct BarrierRetriever {
        barrier: Barrier,
        active: AtomicUsize,
    }

    impl BarrierRetriever {
        fn overlap(&self) {
            let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
            if active == 1 {
                std::thread::sleep(Duration::from_millis(250));
                if self.active.load(Ordering::SeqCst) == 2 {
                    self.barrier.wait();
                }
            } else {
                self.barrier.wait();
            }
            self.active.fetch_sub(1, Ordering::SeqCst);
        }
    }

    impl Retriever for BarrierRetriever {
        fn find_files(
            &self,
            _query: &str,
            _scope: Option<&str>,
            _include_archived: bool,
        ) -> Result<Vec<FileHit>> {
            self.overlap();
            Ok((0..3)
                .map(|i| FileHit {
                    path: PathBuf::from(format!("proj/{i}.md")),
                    score: 1.0,
                })
                .collect())
        }

        fn grep(
            &self,
            _query: &str,
            _mode: GrepMode,
            _scope: Option<&str>,
            _include_archived: bool,
        ) -> Result<Vec<ContentHit>> {
            self.overlap();
            Ok(vec![])
        }

        fn index_state(&self) -> IndexState {
            IndexState::Ready
        }

        fn indexed_paths(&self) -> Result<Vec<PathBuf>> {
            Ok(Vec::new())
        }

        fn reindex(&self) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn search_overlaps_find_and_plain_grep() {
        let dir = tempdir().unwrap();
        let fake = BarrierRetriever {
            barrier: Barrier::new(2),
            active: AtomicUsize::new(0),
        };
        let svc = MemoryService::new(dir.path(), Some(Arc::new(fake)));

        let started = Instant::now();
        let response = svc
            .search(SearchOptions {
                query: "needle".into(),
                namespace: Some("proj".into()),
                ..Default::default()
            })
            .expect("search");

        assert_eq!(response.results.len(), 3);
        assert!(
            started.elapsed() < Duration::from_millis(400),
            "find and grep ran sequentially in {:?}",
            started.elapsed()
        );
    }

    struct StageErrorRetriever {
        find: bool,
        grep: Option<GrepMode>,
    }

    impl Retriever for StageErrorRetriever {
        fn find_files(
            &self,
            _query: &str,
            _scope: Option<&str>,
            _include_archived: bool,
        ) -> Result<Vec<FileHit>> {
            if self.find {
                Err(Error::Retriever("boom".into()))
            } else {
                Ok(vec![])
            }
        }

        fn grep(
            &self,
            _query: &str,
            mode: GrepMode,
            _scope: Option<&str>,
            _include_archived: bool,
        ) -> Result<Vec<ContentHit>> {
            if self.grep == Some(mode) {
                Err(Error::Retriever("boom".into()))
            } else {
                Ok(vec![])
            }
        }

        fn index_state(&self) -> IndexState {
            IndexState::Ready
        }

        fn indexed_paths(&self) -> Result<Vec<PathBuf>> {
            Ok(Vec::new())
        }

        fn reindex(&self) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn search_errors_name_the_failed_stage_query_and_scope() {
        for (fake, stage) in [
            (
                StageErrorRetriever {
                    find: true,
                    grep: None,
                },
                "find_files",
            ),
            (
                StageErrorRetriever {
                    find: false,
                    grep: Some(GrepMode::Plain),
                },
                "grep_plain",
            ),
            (
                StageErrorRetriever {
                    find: false,
                    grep: Some(GrepMode::Fuzzy),
                },
                "grep_fuzzy",
            ),
        ] {
            let dir = tempdir().unwrap();
            let svc = MemoryService::new(dir.path(), Some(Arc::new(fake)));
            let error = svc
                .search(SearchOptions {
                    query: "needle".into(),
                    namespace: Some("proj".into()),
                    ..Default::default()
                })
                .unwrap_err()
                .to_string();
            assert!(error.contains(stage), "error={error}");
            assert!(error.contains("needle"), "error={error}");
            assert!(error.contains("proj"), "error={error}");
        }
    }

    #[derive(Default)]
    struct TrackingRetriever {
        tracked: Mutex<Vec<PathBuf>>,
    }

    impl Retriever for TrackingRetriever {
        fn find_files(
            &self,
            _query: &str,
            _scope: Option<&str>,
            _include_archived: bool,
        ) -> Result<Vec<FileHit>> {
            Ok((0..3)
                .map(|i| FileHit {
                    path: PathBuf::from(format!("proj/{i}.md")),
                    score: 1.0 - i as f32 * 0.1,
                })
                .collect())
        }

        fn grep(
            &self,
            _query: &str,
            _mode: GrepMode,
            _scope: Option<&str>,
            _include_archived: bool,
        ) -> Result<Vec<ContentHit>> {
            Ok(vec![])
        }

        fn index_state(&self) -> IndexState {
            IndexState::Ready
        }

        fn indexed_paths(&self) -> Result<Vec<PathBuf>> {
            Ok(Vec::new())
        }

        fn track_access(&self, path: &Path) -> Result<()> {
            self.tracked.lock().unwrap().push(path.to_path_buf());
            Ok(())
        }

        fn reindex(&self) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn search_reinforces_only_results_not_limit_overflow() {
        let dir = tempdir().unwrap();
        let fake = Arc::new(TrackingRetriever::default());
        let svc = MemoryService::new(dir.path(), Some(fake.clone()));

        let response = svc
            .search(SearchOptions {
                query: "needle".into(),
                limit: 1,
                ..Default::default()
            })
            .expect("search");

        assert_eq!(response.results.len(), 1);
        assert_eq!(response.results[0].handle, "proj/0");
        assert_eq!(response.more.len(), 2);
        assert_eq!(svc.access.len(), 1);
        assert_eq!(svc.access.count_recent("proj/0", 30), 1);
        assert_eq!(svc.access.count_recent("proj/1", 30), 0);
        assert_eq!(svc.access.count_recent("proj/2", 30), 0);
        assert_eq!(
            fake.tracked.lock().unwrap().as_slice(),
            &[PathBuf::from("proj/0.md")]
        );
    }

    #[test]
    fn search_does_not_reinforce_budget_overflow() {
        let dir = tempdir().unwrap();
        let fake = Arc::new(TrackingRetriever::default());
        let svc = MemoryService::new(dir.path(), Some(fake.clone()));

        let response = svc
            .search(SearchOptions {
                query: "needle".into(),
                budget_bytes: 0,
                ..Default::default()
            })
            .expect("search");

        assert!(response.results.is_empty());
        assert_eq!(response.more.len(), 3);
        assert_eq!(svc.access.len(), 0);
        assert!(fake.tracked.lock().unwrap().is_empty());
    }

    #[test]
    fn stats_does_not_call_an_old_access_never_accessed() {
        let dir = tempdir().unwrap();
        let created = (time::OffsetDateTime::now_utc() - time::Duration::days(120)).date();
        seed_note(
            dir.path(),
            "proj/old.md",
            "Old",
            &["old one", "old two"],
            "body",
        );
        let path = dir.path().join("proj/old.md");
        let text = fs::read_to_string(&path)
            .unwrap()
            .replace("created: 2026-01-01", &format!("created: {created}"));
        fs::write(path, text).unwrap();
        let access = AccessLog::open(dir.path());
        access
            .append_event(memory_core::AccessEvent {
                ts: (time::OffsetDateTime::now_utc() - time::Duration::days(100))
                    .format(&time::format_description::well_known::Rfc3339)
                    .unwrap(),
                handle: "proj/old".into(),
                via: AccessVia::Read,
            })
            .unwrap();
        access.compact(90).unwrap();

        let stats = MemoryService::new(dir.path(), None).stats(None).unwrap();

        assert_eq!(stats.decay.never_accessed_30d, 0);
    }

    #[test]
    fn forget_archives_by_default() {
        let dir = tempdir().unwrap();
        let svc = MemoryService::new(dir.path(), None);
        svc.store(StoreRequest {
            title: "Temp Note".into(),
            body: "x".into(),
            aliases: vec!["a".into(), "b".into()],
            namespace: "proj".into(),
            note_type: NoteType::Fact,
            tags: vec![],
            expires: None,
            mode: MergeMode::Append,
        })
        .unwrap();
        let out = svc.forget("proj/temp-note", false).unwrap();
        assert_eq!(out.action, ForgetAction::Archived);
        assert!(!dir.path().join("proj/temp-note.md").exists());
        assert!(dir.path().join(".archive/proj/temp-note.md").is_file());
    }

    #[test]
    fn forget_hard_deletes() {
        let dir = tempdir().unwrap();
        let svc = MemoryService::new(dir.path(), None);
        svc.store(StoreRequest {
            title: "Gone".into(),
            body: "x".into(),
            aliases: vec!["a".into(), "b".into()],
            namespace: "".into(),
            note_type: NoteType::Fact,
            tags: vec![],
            expires: None,
            mode: MergeMode::Append,
        })
        .unwrap();
        let out = svc.forget("gone", true).unwrap();
        assert_eq!(out.action, ForgetAction::Deleted);
        assert!(!dir.path().join("gone.md").exists());
    }

    #[test]
    fn forget_hard_deletes_archived_note_when_active_is_absent() {
        let dir = tempdir().unwrap();
        seed_note(
            dir.path(),
            ".archive/proj/gone.md",
            "Archived Gone",
            &["retired note", "old note"],
            "archived body",
        );
        let svc = MemoryService::new(dir.path(), None);

        let out = svc.forget("proj/gone", true).unwrap();

        assert_eq!(out.action, ForgetAction::Deleted);
        assert!(!dir.path().join(".archive/proj/gone.md").exists());
    }

    #[test]
    fn forget_hard_prefers_active_note_over_same_handle_archive() {
        let dir = tempdir().unwrap();
        seed_note(
            dir.path(),
            "proj/gone.md",
            "Active Gone",
            &["current note", "live note"],
            "active body",
        );
        seed_note(
            dir.path(),
            ".archive/proj/gone.md",
            "Archived Gone",
            &["retired note", "old note"],
            "archived body",
        );
        let svc = MemoryService::new(dir.path(), None);

        let out = svc.forget("proj/gone", true).unwrap();

        assert_eq!(out.action, ForgetAction::Deleted);
        assert!(!dir.path().join("proj/gone.md").exists());
        assert!(dir.path().join(".archive/proj/gone.md").is_file());
    }

    #[test]
    fn stats_counts_notes() {
        let dir = tempdir().unwrap();
        let svc = MemoryService::new(dir.path(), None);
        svc.store(StoreRequest {
            title: "One".into(),
            body: "x".into(),
            aliases: vec!["a".into(), "b".into()],
            namespace: "proj".into(),
            note_type: NoteType::Decision,
            tags: vec![],
            expires: None,
            mode: MergeMode::Append,
        })
        .unwrap();
        let stats = svc.stats(None).unwrap();
        assert_eq!(stats.total_notes, 1);
        assert_eq!(stats.by_namespace.get("proj").copied(), Some(1));
        assert_eq!(stats.by_type.get("decision").copied(), Some(1));
        assert_eq!(stats.index.state, "unavailable");
        assert_eq!(stats.index.files_indexed, 0);
        assert_eq!(stats.index.last_scan_ms, 0);
    }

    struct SnapshotRetriever;

    impl Retriever for SnapshotRetriever {
        fn find_files(
            &self,
            _query: &str,
            _scope: Option<&str>,
            _include_archived: bool,
        ) -> Result<Vec<FileHit>> {
            Ok(vec![])
        }

        fn grep(
            &self,
            _query: &str,
            _mode: GrepMode,
            _scope: Option<&str>,
            _include_archived: bool,
        ) -> Result<Vec<ContentHit>> {
            Ok(vec![])
        }

        fn index_state(&self) -> IndexState {
            IndexState::Ready
        }

        fn index_snapshot(&self) -> IndexSnapshot {
            IndexSnapshot {
                state: IndexState::Ready,
                files_indexed: 7,
                last_scan_ms: 11,
            }
        }

        fn indexed_paths(&self) -> Result<Vec<PathBuf>> {
            Ok(Vec::new())
        }

        fn reindex(&self) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn stats_reports_retriever_snapshot() {
        let dir = tempdir().unwrap();
        let svc = MemoryService::new(dir.path(), Some(Arc::new(SnapshotRetriever)));

        let stats = svc.stats(None).expect("stats");

        assert_eq!(stats.index.state, "ready");
        assert_eq!(stats.index.files_indexed, 7);
        assert_eq!(stats.index.last_scan_ms, 11);
    }

    #[test]
    fn store_succeeds_without_retriever() {
        let dir = tempdir().unwrap();
        let svc = MemoryService::new(dir.path(), None);
        let out = svc
            .store(StoreRequest {
                title: "No Index".into(),
                body: "ok".into(),
                aliases: vec!["x".into(), "y".into()],
                namespace: "default".into(),
                note_type: NoteType::Fact,
                tags: vec![],
                expires: None,
                mode: MergeMode::Append,
            })
            .unwrap();
        assert_eq!(out.action, StoreAction::Created);
    }
}
