//! Tool business logic: thin orchestration over store + retriever + pipeline.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::search::{
    apply_budget, merge_hits, path_to_handle, rank_hits, BudgetedSearch, MatchStage, NoteMeta,
    BUDGET_BYTES_DEFAULT, BUDGET_BYTES_MAX, MIN_RESULTS_FOR_FUZZY, SEARCH_LIMIT_DEFAULT,
};
use memory_core::{
    extract_wikilinks, AccessLog, AccessVia, GrepMode, IndexState, MemoryStore, MergeMode, Note,
    NoteFrontmatter, NoteType, Retriever, StoreAction, StoreInput, StoreOutcome,
};
use memory_core::{Error, Result};
use serde::{Deserialize, Serialize};
use time::Date;

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
        let query = opts.query.trim();
        let scope_owned = opts
            .namespace
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string());
        let scope = scope_owned.as_deref();
        let scope_label = scope.unwrap_or("").to_string();

        let mut stages = Vec::new();
        let mut file_raw: Vec<(PathBuf, f32)> = Vec::new();
        let mut plain_raw: Vec<(PathBuf, f32, String)> = Vec::new();
        let mut fuzzy_raw: Vec<(PathBuf, f32, String)> = Vec::new();

        let Some(retriever) = self.retriever.as_ref() else {
            stages.push(MatchStage::FindFiles);
            stages.push(MatchStage::GrepPlain);
            let budgeted =
                apply_budget(vec![], opts.limit, opts.budget_bytes, &stages, &scope_label);
            return Ok(budgeted.into());
        };

        if query.is_empty() {
            stages.push(MatchStage::FindFiles);
            stages.push(MatchStage::GrepPlain);
            let budgeted =
                apply_budget(vec![], opts.limit, opts.budget_bytes, &stages, &scope_label);
            return Ok(budgeted.into());
        }

        // Stage 2a: find_files
        stages.push(MatchStage::FindFiles);
        if let Ok(hits) = retriever.find_files(query, scope) {
            for h in hits {
                if keep_path(&h.path, opts.include_archived) {
                    file_raw.push((h.path, h.score));
                }
            }
        }

        // Stage 2b: plain grep
        stages.push(MatchStage::GrepPlain);
        if let Ok(hits) = retriever.grep(query, GrepMode::Plain, scope) {
            for h in hits {
                if keep_path(&h.path, opts.include_archived) {
                    plain_raw.push((h.path, h.score, h.snippet));
                }
            }
        }

        let mut merged = merge_hits(&file_raw, &plain_raw, &[]);

        // Stage 3: fuzzy escalate
        if merged.len() < MIN_RESULTS_FOR_FUZZY {
            stages.push(MatchStage::GrepFuzzy);
            if let Ok(hits) = retriever.grep(query, GrepMode::Fuzzy, scope) {
                for h in hits {
                    if keep_path(&h.path, opts.include_archived) {
                        fuzzy_raw.push((h.path, h.score, h.snippet));
                    }
                }
            }
            merged = merge_hits(&file_raw, &plain_raw, &fuzzy_raw);
        }

        // Load meta for ranking
        let mut meta = HashMap::new();
        for m in &merged {
            if let Some(nm) = self.load_meta(&m.path) {
                meta.insert(m.path.clone(), nm);
            }
        }

        let ranked = rank_hits(merged, &meta);

        // Soft search-hit access log for top results (best-effort)
        for hit in ranked
            .iter()
            .take(opts.limit.clamp(1, SEARCH_LIMIT_DEFAULT))
        {
            let _ = self.access.append(&hit.handle, AccessVia::SearchHit);
        }

        let budget_bytes = opts.budget_bytes.clamp(1, BUDGET_BYTES_MAX);
        let budgeted = apply_budget(ranked, opts.limit, budget_bytes, &stages, &scope_label);
        Ok(budgeted.into())
    }

    /// Read full note; records a read access event.
    pub fn read(&self, handle: &str) -> Result<ReadOutcome> {
        let (ns, slug) = parse_handle(handle)?;
        let note = self.store.read(&ns, &slug)?;
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

    /// Archive (default) or hard-delete a note.
    pub fn forget(&self, handle: &str, hard: bool) -> Result<ForgetOutcome> {
        let (ns, slug) = parse_handle(handle)?;
        let abs = self.store.absolute_path(&ns, &slug);
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
            if self.access.count_recent(&handle, 30) == 0 {
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

        let state = match self.retriever.as_ref().map(|r| r.index_state()) {
            Some(IndexState::Ready) => "ready",
            Some(IndexState::Cold) => "cold",
            Some(IndexState::Unavailable) | None => "unavailable",
        };

        Ok(StatsSnapshot {
            total_notes: total,
            by_namespace,
            by_type,
            disk_bytes,
            index: IndexStats {
                state: state.into(),
                access_log_events: self.access.len() as u64,
            },
            decay: DecayStats {
                never_accessed_30d,
                expiring_soon,
            },
        })
    }

    fn load_meta(&self, rel: &Path) -> Option<NoteMeta> {
        let abs = self.store.root().join(rel);
        let text = fs::read_to_string(abs).ok()?;
        let note = Note::parse(&text).ok()?;
        let handle = path_to_handle(rel);
        let access_count_30d = self.access.count_recent(&handle, 30);
        Some(NoteMeta {
            title: note.frontmatter.title,
            note_type: note.frontmatter.note_type,
            access_count_30d,
            handle,
        })
    }
}

fn keep_path(path: &Path, include_archived: bool) -> bool {
    let archived = path.components().any(|c| c.as_os_str() == ".archive");
    let index = path.components().any(|c| c.as_os_str() == ".index");
    if index {
        return false;
    }
    if archived && !include_archived {
        return false;
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
    use memory_core::{ContentHit, FileHit};
    use std::sync::Mutex;
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
