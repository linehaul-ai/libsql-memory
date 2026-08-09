//! Atomic store and write-path orchestration (spec 02).
//!
//! Invariants:
//! - No network I/O.
//! - No hard dependency on the index: dedup is best-effort via [`Retriever`].
//! - Disk writes are temp + fsync + rename in the target directory.

use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

use time::{Date, OffsetDateTime};

use crate::error::{Error, Result};
use crate::note::{Note, NoteFrontmatter, NoteType};
use crate::retriever::{GrepMode, IndexState, Retriever};
use crate::slugify::{slugify, validate_namespace};

/// How to merge body text when updating an existing note.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MergeMode {
    /// Append a dated `## Addendum YYYY-MM-DD` section (default).
    #[default]
    Append,
    /// Replace the body entirely with the new content.
    Replace,
}

/// What the store did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreAction {
    /// New file created.
    Created,
    /// Existing file updated (dedup hit or same-slug collision).
    Updated,
}

/// Input for [`MemoryStore::store`].
#[derive(Debug, Clone)]
pub struct StoreInput {
    /// Human-readable title (required, non-empty).
    pub title: String,
    /// Markdown body.
    pub body: String,
    /// Synonyms (min 2 after trim).
    pub aliases: Vec<String>,
    /// Relative namespace; empty string = root.
    pub namespace: String,
    /// Note kind.
    pub note_type: NoteType,
    /// Optional tags.
    pub tags: Vec<String>,
    /// Optional expiry date.
    pub expires: Option<Date>,
    /// Optional provenance.
    pub source: Option<String>,
    /// Body merge policy on update.
    pub mode: MergeMode,
}

/// Result of a successful store.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoreOutcome {
    /// Filename stem written or updated.
    pub slug: String,
    /// Namespace directory relative to root.
    pub namespace: String,
    /// Created vs updated.
    pub action: StoreAction,
    /// Slug of the note that dedup matched, if any (same as `slug` on update-via-dedup).
    pub dedup_hit: Option<String>,
}

/// On-disk memory root: namespaces are subdirectories; notes are `slug.md`.
#[derive(Debug, Clone)]
pub struct MemoryStore {
    root: PathBuf,
}

impl MemoryStore {
    /// Open a store rooted at `root` (created on first write if missing).
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Absolute path of the memory root.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Relative path of a note: `{namespace}/{slug}.md` or `{slug}.md` at root.
    pub fn relative_path(namespace: &str, slug: &str) -> PathBuf {
        let file = format!("{slug}.md");
        if namespace.is_empty() {
            PathBuf::from(file)
        } else {
            PathBuf::from(namespace).join(file)
        }
    }

    /// Absolute path for `namespace/slug`.
    pub fn absolute_path(&self, namespace: &str, slug: &str) -> PathBuf {
        self.root.join(Self::relative_path(namespace, slug))
    }

    /// `namespace/slug` handle string.
    pub fn handle(namespace: &str, slug: &str) -> String {
        if namespace.is_empty() {
            slug.to_string()
        } else {
            format!("{namespace}/{slug}")
        }
    }

    /// Read and parse a note by namespace + slug.
    pub fn read(&self, namespace: &str, slug: &str) -> Result<Note> {
        validate_namespace(namespace)?;
        let path = self.absolute_path(namespace, slug);
        let text = fs::read_to_string(&path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                Error::NotFound {
                    handle: Self::handle(namespace, slug),
                }
            } else {
                Error::io(&path, e)
            }
        })?;
        Note::parse(&text)
    }

    /// True when the note file exists on disk.
    pub fn exists(&self, namespace: &str, slug: &str) -> bool {
        self.absolute_path(namespace, slug).is_file()
    }

    /// Move a note to `.archive/{namespace}/{slug}.md`. Never hard-deletes.
    ///
    /// Used by `memory_forget` (soft) and `doctor --apply`. Recoverable by moving back.
    pub fn archive_note(&self, namespace: &str, slug: &str) -> Result<PathBuf> {
        validate_namespace(namespace)?;
        let abs = self.absolute_path(namespace, slug);
        if !abs.is_file() {
            return Err(Error::NotFound {
                handle: Self::handle(namespace, slug),
            });
        }
        let rel = Self::relative_path(namespace, slug);
        let dest = self.root.join(".archive").join(&rel);
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;
        }
        fs::rename(&abs, &dest).map_err(|e| Error::io(&abs, e))?;
        Ok(dest)
    }

    /// Parse `namespace/slug` or `slug` into (namespace, slug).
    pub fn parse_handle(handle: &str) -> Result<(String, String)> {
        let handle = handle.trim().trim_start_matches('/');
        if handle.is_empty() {
            return Err(Error::validation(
                "handle",
                "must be non-empty (namespace/slug or slug)",
            ));
        }
        if handle.contains("..") {
            return Err(Error::validation(
                "handle",
                "path traversal rejected; use a relative handle like 'proj/my-note'",
            ));
        }
        let handle = handle.strip_suffix(".md").unwrap_or(handle);
        if let Some((ns, slug)) = handle.rsplit_once('/') {
            if slug.is_empty() {
                return Err(Error::validation("handle", "slug segment is empty"));
            }
            if ns.split('/').any(|s| s.is_empty()) {
                return Err(Error::validation("handle", "namespace has empty segment"));
            }
            Ok((ns.to_string(), slug.to_string()))
        } else {
            Ok((String::new(), handle.to_string()))
        }
    }

    /// Store or update a note. Dedup uses `retriever` when `Some` and ready; never fails for index reasons.
    pub fn store(
        &self,
        input: StoreInput,
        retriever: Option<&dyn Retriever>,
    ) -> Result<StoreOutcome> {
        // 1. Validate
        validate_namespace(&input.namespace)?;
        reject_reserved_namespace(&input.namespace)?;
        if input.title.trim().is_empty() {
            return Err(Error::validation("title", "must be non-empty"));
        }

        // 2. Slugify
        let candidate_slug = slugify(&input.title)?;

        // Build a provisional frontmatter for alias validation (dates filled later)
        let today = today_utc();
        let probe_fm = NoteFrontmatter {
            title: input.title.trim().to_string(),
            aliases: input.aliases.clone(),
            tags: input.tags.clone(),
            note_type: input.note_type,
            created: today,
            updated: today,
            expires: input.expires,
            source: input.source.clone(),
        };
        probe_fm.validate()?;

        // 3–4. Resolve target: existing slug path, else dedup probe, else create
        let mut dedup_hit: Option<String> = None;
        let (target_namespace, target_slug, action) = {
            let path = self.absolute_path(&input.namespace, &candidate_slug);
            if path.is_file() {
                (
                    input.namespace.clone(),
                    candidate_slug.clone(),
                    StoreAction::Updated,
                )
            } else if let Some(hit_rel) = probe_dedup(
                self,
                retriever,
                &input.namespace,
                &input.title,
                &input.aliases,
            ) {
                let (ns, slug) = split_note_path(&hit_rel)?;
                dedup_hit = Some(slug.clone());
                (ns, slug, StoreAction::Updated)
            } else {
                (
                    input.namespace.clone(),
                    candidate_slug.clone(),
                    StoreAction::Created,
                )
            }
        };

        let abs = self.absolute_path(&target_namespace, &target_slug);

        let note = match action {
            StoreAction::Created => Note {
                frontmatter: NoteFrontmatter {
                    title: input.title.trim().to_string(),
                    aliases: normalize_aliases(&input.aliases),
                    tags: normalize_tags(&input.tags),
                    note_type: input.note_type,
                    created: today,
                    updated: today,
                    expires: input.expires,
                    source: input.source.clone(),
                },
                body: input.body,
            },
            StoreAction::Updated => {
                let existing = self.read(&target_namespace, &target_slug)?;
                merge_note(existing, &input, today)
            }
        };

        note.validate()?;
        let markdown = note.to_markdown()?;
        atomic_write(&abs, markdown.as_bytes())?;

        Ok(StoreOutcome {
            slug: target_slug,
            namespace: target_namespace,
            action,
            dedup_hit,
        })
    }
}

/// Reject namespaces that collide with reserved dirs (`.index`, `.archive`).
fn reject_reserved_namespace(namespace: &str) -> Result<()> {
    for segment in namespace.split('/').filter(|s| !s.is_empty()) {
        if segment == ".index" || segment == ".archive" || segment.starts_with('.') {
            return Err(Error::InvalidNamespace {
                path: namespace.to_string(),
                reason: format!(
                    "reserved or hidden segment {segment:?}; notes live outside .index/ and .archive/"
                ),
            });
        }
    }
    Ok(())
}

fn today_utc() -> Date {
    OffsetDateTime::now_utc().date()
}

fn normalize_aliases(aliases: &[String]) -> Vec<String> {
    aliases
        .iter()
        .map(|a| a.trim().to_string())
        .filter(|a| !a.is_empty())
        .collect()
}

fn normalize_tags(tags: &[String]) -> Vec<String> {
    let mut out: Vec<String> = tags
        .iter()
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
        .collect();
    out.sort();
    out.dedup();
    out
}

fn merge_note(mut existing: Note, input: &StoreInput, today: Date) -> Note {
    // Union aliases
    let mut aliases = existing.frontmatter.aliases.clone();
    for a in normalize_aliases(&input.aliases) {
        if !aliases.iter().any(|e| e.eq_ignore_ascii_case(&a)) {
            aliases.push(a);
        }
    }

    // Union tags
    let mut tags = existing.frontmatter.tags.clone();
    for t in normalize_tags(&input.tags) {
        if !tags.iter().any(|e| e == &t) {
            tags.push(t);
        }
    }
    tags.sort();
    tags.dedup();

    let body = match input.mode {
        MergeMode::Replace => input.body.clone(),
        MergeMode::Append => {
            if existing.body.trim().is_empty() {
                input.body.clone()
            } else if input.body.trim().is_empty() {
                existing.body.clone()
            } else {
                let mut b = existing.body.trim_end().to_string();
                b.push_str("\n\n## Addendum ");
                b.push_str(&format_date(today));
                b.push('\n');
                b.push_str(input.body.trim_end());
                if !input.body.ends_with('\n') {
                    // keep final newline via to_markdown
                }
                b.push('\n');
                b
            }
        }
    };

    // Prefer incoming title when non-empty (already validated)
    existing.frontmatter.title = input.title.trim().to_string();
    existing.frontmatter.aliases = aliases;
    existing.frontmatter.tags = tags;
    existing.frontmatter.note_type = input.note_type;
    existing.frontmatter.updated = today;
    if input.expires.is_some() {
        existing.frontmatter.expires = input.expires;
    }
    if input.source.is_some() {
        existing.frontmatter.source = input.source.clone();
    }
    // created stays
    existing.body = body;
    existing
}

fn format_date(d: Date) -> String {
    // YYYY-MM-DD without pulling format_description into hot path complexity
    format!("{:04}-{:02}-{:02}", d.year(), d.month() as u8, d.day())
}

/// Best-effort dedup: gather candidate paths from retriever, confirm on disk.
///
/// Conservative: only merge when an existing note's title equals (case-insensitive)
/// or shares an alias with the new input. Index errors → no hit.
fn probe_dedup(
    store: &MemoryStore,
    retriever: Option<&dyn Retriever>,
    namespace: &str,
    title: &str,
    aliases: &[String],
) -> Option<PathBuf> {
    let r = retriever?;
    if r.index_state() == IndexState::Unavailable {
        return None;
    }

    let scope = if namespace.is_empty() {
        None
    } else {
        Some(namespace)
    };

    let mut candidates: Vec<PathBuf> = Vec::new();

    // Path/slug search on title (errors abort the whole probe — index is unhealthy).
    match r.find_files(title, scope, false) {
        Ok(hits) => {
            for h in hits {
                push_unique(&mut candidates, h.path);
            }
        }
        Err(_) => return None,
    }

    // Content grep for title + aliases (frontmatter lives in file text).
    // Per-query errors are ignored; find_files already succeeded.
    let mut queries: Vec<&str> = vec![title.trim()];
    for alias in aliases {
        let a = alias.trim();
        if !a.is_empty() {
            queries.push(a);
        }
    }
    for q in queries {
        if let Ok(hits) = r.grep(q, GrepMode::Plain, scope, false) {
            for h in hits {
                push_unique(&mut candidates, h.path);
            }
        }
    }

    let title_l = title.trim().to_ascii_lowercase();
    let alias_set: Vec<String> = aliases
        .iter()
        .map(|a| a.trim().to_ascii_lowercase())
        .filter(|a| !a.is_empty())
        .collect();

    for rel in candidates {
        if is_reserved_path(&rel) {
            continue;
        }
        // Scope: if namespace set, path must live under it (segment boundary).
        if !namespace.is_empty() && !path_in_namespace(&rel, namespace) {
            continue;
        }

        let abs = store.root.join(&rel);
        let text = match fs::read_to_string(&abs) {
            Ok(t) => t,
            Err(_) => continue,
        };
        let note = match Note::parse(&text) {
            Ok(n) => n,
            Err(_) => continue,
        };

        let note_title = note.frontmatter.title.trim().to_ascii_lowercase();
        if note_title == title_l {
            return Some(rel);
        }
        for na in &note.frontmatter.aliases {
            let na = na.trim().to_ascii_lowercase();
            // existing alias == incoming title, or alias sets overlap
            if na == title_l || alias_set.iter().any(|a| a == &na) {
                return Some(rel);
            }
        }
    }

    None
}

/// True when `rel` is exactly under `namespace/` (not a prefix sibling like `proj` vs `project`).
fn path_in_namespace(rel: &Path, namespace: &str) -> bool {
    let p = rel.to_string_lossy().replace('\\', "/");
    p == namespace
        || p.starts_with(&format!("{namespace}/"))
        || p.strip_prefix(namespace)
            .is_some_and(|rest| rest.is_empty() || rest.starts_with('/'))
}

fn push_unique(out: &mut Vec<PathBuf>, path: PathBuf) {
    if !out.iter().any(|p| p == &path) {
        out.push(path);
    }
}

fn is_reserved_path(rel: &Path) -> bool {
    rel.components().any(|c| {
        let s = c.as_os_str().to_string_lossy();
        s == ".index" || s == ".archive"
    })
}

/// Split `namespace/slug.md` or `slug.md` into (namespace, slug).
fn split_note_path(rel: &Path) -> Result<(String, String)> {
    let stem = rel
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| Error::validation("path", format!("invalid note path {}", rel.display())))?;
    let parent = rel.parent().unwrap_or_else(|| Path::new(""));
    let ns = if parent.as_os_str().is_empty() {
        String::new()
    } else {
        parent.to_string_lossy().replace('\\', "/")
    };
    Ok((ns, stem.to_string()))
}

/// Write `bytes` to `path` via temp file in the same directory, fsync, rename.
pub fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(dir).map_err(|e| Error::io(dir, e))?;

    let mut tmp = tempfile::Builder::new()
        .prefix(".note-")
        .suffix(".tmp")
        .tempfile_in(dir)
        .map_err(|e| Error::io(dir, e))?;

    tmp.write_all(bytes).map_err(|e| Error::io(tmp.path(), e))?;
    tmp.as_file()
        .sync_all()
        .map_err(|e| Error::io(tmp.path(), e))?;

    tmp.persist(path).map_err(|e| {
        Error::io(
            path,
            std::io::Error::new(e.error.kind(), e.error.to_string()),
        )
    })?;

    // Best-effort fsync of the directory entry (rename durability) on Unix.
    #[cfg(unix)]
    {
        if let Ok(dir_file) = File::open(dir) {
            let _ = dir_file.sync_all();
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retriever::testing::FakeRetriever;
    use crate::retriever::{ContentHit, FileHit};
    use std::sync::Mutex;
    use tempfile::tempdir;

    fn base_input(title: &str, body: &str) -> StoreInput {
        StoreInput {
            title: title.into(),
            body: body.into(),
            aliases: vec!["alpha".into(), "beta".into()],
            namespace: "proj".into(),
            note_type: NoteType::Fact,
            tags: vec!["t1".into()],
            expires: None,
            source: None,
            mode: MergeMode::Append,
        }
    }

    #[test]
    fn create_writes_markdown_under_namespace() {
        let dir = tempdir().unwrap();
        let store = MemoryStore::new(dir.path());
        let out = store
            .store(base_input("Hello World", "The fact."), None)
            .unwrap();
        assert_eq!(out.action, StoreAction::Created);
        assert_eq!(out.slug, "hello-world");
        assert_eq!(out.namespace, "proj");
        assert_eq!(out.dedup_hit, None);

        let note = store.read("proj", "hello-world").unwrap();
        assert_eq!(note.frontmatter.title, "Hello World");
        assert!(note.body.contains("The fact."));
        assert!(dir.path().join("proj/hello-world.md").is_file());
    }

    #[test]
    fn same_slug_updates_without_retriever() {
        let dir = tempdir().unwrap();
        let store = MemoryStore::new(dir.path());
        store
            .store(base_input("Hello World", "First."), None)
            .unwrap();
        let out = store
            .store(
                StoreInput {
                    mode: MergeMode::Append,
                    body: "Second.".into(),
                    aliases: vec!["alpha".into(), "gamma".into()],
                    tags: vec!["t2".into()],
                    ..base_input("Hello World", "Second.")
                },
                None,
            )
            .unwrap();
        assert_eq!(out.action, StoreAction::Updated);
        let note = store.read("proj", "hello-world").unwrap();
        assert!(note.body.contains("First."));
        assert!(note.body.contains("## Addendum"));
        assert!(note.body.contains("Second."));
        assert!(note.frontmatter.aliases.iter().any(|a| a == "gamma"));
        assert!(note.frontmatter.tags.iter().any(|t| t == "t2"));
        assert_eq!(note.frontmatter.created, note.frontmatter.updated); // same day ok
    }

    #[test]
    fn replace_mode_overwrites_body() {
        let dir = tempdir().unwrap();
        let store = MemoryStore::new(dir.path());
        store
            .store(base_input("Replace Me", "Old body."), None)
            .unwrap();
        store
            .store(
                StoreInput {
                    mode: MergeMode::Replace,
                    body: "New body only.".into(),
                    ..base_input("Replace Me", "New body only.")
                },
                None,
            )
            .unwrap();
        let note = store.read("proj", "replace-me").unwrap();
        assert_eq!(note.body.trim(), "New body only.");
        assert!(!note.body.contains("Old body"));
    }

    #[test]
    fn store_succeeds_when_retriever_errors() {
        let dir = tempdir().unwrap();
        let store = MemoryStore::new(dir.path());
        let fake = FakeRetriever::ready();
        *fake.find_error.lock().unwrap() = Some("boom".into());
        let out = store
            .store(base_input("Resilient Write", "ok"), Some(&fake))
            .unwrap();
        assert_eq!(out.action, StoreAction::Created);
    }

    #[test]
    fn store_skips_dedup_when_index_unavailable() {
        let dir = tempdir().unwrap();
        let store = MemoryStore::new(dir.path());
        // Pre-create a different-slug note with same title on disk
        store
            .store(
                StoreInput {
                    title: "Unique Title".into(),
                    body: "A".into(),
                    aliases: vec!["x".into(), "y".into()],
                    namespace: "proj".into(),
                    ..base_input("Unique Title", "A")
                },
                None,
            )
            .unwrap();
        // Manually add a second path won't happen via same title+slug.
        // Unavailable index → create would collide on slug and update — use different title
        // to show Unavailable skips probe (no cross-slug merge).
        let fake = FakeRetriever {
            state: IndexState::Unavailable,
            ..Default::default()
        };
        // Put a file that would match if probed
        let other = dir.path().join("proj/other-note.md");
        fs::create_dir_all(other.parent().unwrap()).unwrap();
        let md = r#"---
title: Shipping Cadence
aliases: [release frequency, CI/CD]
type: fact
created: 2026-01-01
updated: 2026-01-01
---
Old.
"#;
        fs::write(&other, md).unwrap();
        *fake.files.lock().unwrap() = vec![FileHit {
            path: PathBuf::from("proj/other-note.md"),
            score: 10.0,
        }];

        let out = store
            .store(
                StoreInput {
                    title: "Shipping Cadence".into(),
                    body: "New.".into(),
                    aliases: vec!["release frequency".into(), "deploys".into()],
                    namespace: "proj".into(),
                    mode: MergeMode::Replace,
                    ..base_input("Shipping Cadence", "New.")
                },
                Some(&fake),
            )
            .unwrap();
        // Unavailable → no dedup → new slug path created (shipping-cadence.md)
        assert_eq!(out.action, StoreAction::Created);
        assert_eq!(out.slug, "shipping-cadence");
        assert!(dir.path().join("proj/shipping-cadence.md").is_file());
        // old note untouched
        let old = Note::parse(&fs::read_to_string(&other).unwrap()).unwrap();
        assert_eq!(old.body.trim(), "Old.");
    }

    #[test]
    fn dedup_merges_when_title_matches_via_retriever() {
        let dir = tempdir().unwrap();
        let store = MemoryStore::new(dir.path());
        let other = dir.path().join("proj/other-note.md");
        fs::create_dir_all(other.parent().unwrap()).unwrap();
        fs::write(
            &other,
            r#"---
title: Shipping Cadence
aliases: [release frequency, CI/CD]
type: fact
created: 2026-01-01
updated: 2026-01-01
---
Old fact.
"#,
        )
        .unwrap();

        // Path does not contain the title; content grep must surface the candidate.
        let fake = FakeRetriever {
            state: IndexState::Ready,
            files: Mutex::new(vec![]),
            contents: Mutex::new(vec![ContentHit {
                path: PathBuf::from("proj/other-note.md"),
                snippet: "title: Shipping Cadence".into(),
                line: 2,
                score: 5.0,
            }]),
            find_error: Mutex::new(None),
            grep_error: Mutex::new(None),
            reindex_calls: Mutex::new(0),
        };

        let out = store
            .store(
                StoreInput {
                    title: "Shipping Cadence".into(),
                    body: "New fact.".into(),
                    aliases: vec!["release frequency".into(), "deploys".into()],
                    namespace: "proj".into(),
                    mode: MergeMode::Append,
                    note_type: NoteType::Fact,
                    tags: vec![],
                    expires: None,
                    source: None,
                },
                Some(&fake),
            )
            .unwrap();

        assert_eq!(out.action, StoreAction::Updated);
        assert_eq!(out.slug, "other-note");
        assert_eq!(out.dedup_hit.as_deref(), Some("other-note"));
        assert!(!dir.path().join("proj/shipping-cadence.md").exists());
        let note = store.read("proj", "other-note").unwrap();
        assert!(note.body.contains("Old fact."));
        assert!(note.body.contains("New fact."));
        assert!(note.frontmatter.aliases.iter().any(|a| a == "deploys"));
        assert_eq!(
            note.frontmatter.created,
            time::Date::from_calendar_date(2026, time::Month::January, 1).unwrap()
        );
    }

    #[test]
    fn dedup_requires_strong_match_not_weak_path_hit() {
        let dir = tempdir().unwrap();
        let store = MemoryStore::new(dir.path());
        let weak = dir.path().join("proj/unrelated.md");
        fs::create_dir_all(weak.parent().unwrap()).unwrap();
        fs::write(
            &weak,
            r#"---
title: Completely Different
aliases: [one, two]
type: fact
created: 2026-01-01
updated: 2026-01-01
---
Nope.
"#,
        )
        .unwrap();

        let fake = FakeRetriever::with_files(vec![FileHit {
            path: PathBuf::from("proj/unrelated.md"),
            score: 99.0,
        }]);

        let out = store
            .store(
                StoreInput {
                    title: "Shipping Cadence".into(),
                    body: "Body.".into(),
                    aliases: vec!["release frequency".into(), "ci".into()],
                    namespace: "proj".into(),
                    ..base_input("Shipping Cadence", "Body.")
                },
                Some(&fake),
            )
            .unwrap();
        assert_eq!(out.action, StoreAction::Created);
        assert_eq!(out.slug, "shipping-cadence");
    }

    #[test]
    fn rejects_reserved_namespace() {
        let dir = tempdir().unwrap();
        let store = MemoryStore::new(dir.path());
        let err = store
            .store(
                StoreInput {
                    namespace: ".index".into(),
                    ..base_input("T", "b")
                },
                None,
            )
            .unwrap_err();
        assert!(matches!(err, Error::InvalidNamespace { .. }));
    }

    #[test]
    fn atomic_write_creates_parents_and_content() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("a/b/c.md");
        atomic_write(&path, b"hello\n").unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), "hello\n");
        // no leftover temps
        let leftovers: Vec<_> = fs::read_dir(dir.path().join("a/b"))
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().starts_with(".note-"))
            .collect();
        assert!(leftovers.is_empty());
    }

    #[test]
    fn path_in_namespace_uses_segment_boundary() {
        assert!(path_in_namespace(Path::new("proj/a.md"), "proj"));
        assert!(path_in_namespace(Path::new("proj/sub/a.md"), "proj"));
        assert!(!path_in_namespace(Path::new("project/a.md"), "proj"));
        assert!(!path_in_namespace(Path::new("pro/a.md"), "proj"));
    }

    #[test]
    fn read_missing_is_not_found() {
        let dir = tempdir().unwrap();
        let store = MemoryStore::new(dir.path());
        let err = store.read("proj", "nope").unwrap_err();
        assert!(matches!(err, Error::NotFound { .. }));
    }

    #[test]
    fn root_namespace_store() {
        let dir = tempdir().unwrap();
        let store = MemoryStore::new(dir.path());
        let out = store
            .store(
                StoreInput {
                    namespace: "".into(),
                    ..base_input("Root Note", "at root")
                },
                None,
            )
            .unwrap();
        assert!(dir.path().join("root-note.md").is_file());
        assert_eq!(out.namespace, "");
        let note = store.read("", "root-note").unwrap();
        assert!(note.body.contains("at root"));
    }

    // silence unused ContentHit in this module's tests if not used
    #[test]
    fn dedup_via_alias_grep_hit() {
        let dir = tempdir().unwrap();
        let store = MemoryStore::new(dir.path());
        let path = dir.path().join("proj/via-alias.md");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(
            &path,
            r#"---
title: Existing Name
aliases: [shipping cadence, release frequency]
type: fact
created: 2026-01-01
updated: 2026-01-01
---
Prior.
"#,
        )
        .unwrap();

        let fake = FakeRetriever {
            state: IndexState::Ready,
            files: Mutex::new(vec![]),
            contents: Mutex::new(vec![ContentHit {
                path: PathBuf::from("proj/via-alias.md"),
                snippet: "aliases: [shipping cadence, release frequency]".into(),
                line: 3,
                score: 1.0,
            }]),
            find_error: Mutex::new(None),
            grep_error: Mutex::new(None),
            reindex_calls: Mutex::new(0),
        };

        let out = store
            .store(
                StoreInput {
                    title: "Something New".into(),
                    body: "Add.".into(),
                    aliases: vec!["shipping cadence".into(), "deploys".into()],
                    namespace: "proj".into(),
                    mode: MergeMode::Append,
                    note_type: NoteType::Fact,
                    tags: vec![],
                    expires: None,
                    source: None,
                },
                Some(&fake),
            )
            .unwrap();
        assert_eq!(out.action, StoreAction::Updated);
        assert_eq!(out.slug, "via-alias");
    }
}
