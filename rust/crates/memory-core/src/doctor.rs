//! Doctor: decay candidates, quality warnings, compaction (spec 05).
//!
//! Advisory by default; `--apply` archives candidates and compacts the access log.
//! Never hard-deletes.

use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use time::OffsetDateTime;

use crate::access_log::{AccessLog, CompactionStats, COMPACT_MAX_AGE_DAYS};
use crate::error::{Error, Result};
use crate::links::extract_wikilinks;
use crate::note::{Note, NoteType};
use crate::retriever::{IndexState, Retriever};
use crate::store::MemoryStore;

/// Days after creation with zero access → never-accessed decay signal.
pub const NEVER_ACCESSED_DAYS: i64 = 30;
/// No access within this window (and note is at least this old) → stale signal.
pub const STALE_DAYS: i64 = 90;
/// Session-summary notes older than this are decay candidates.
pub const SESSION_SUMMARY_MAX_AGE_DAYS: i64 = 30;

/// Why a note was flagged for archive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchiveCandidate {
    /// `namespace/slug`.
    pub handle: String,
    /// Human-readable signal names.
    pub reasons: Vec<String>,
}

/// Alias-quality issue.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QualityWarning {
    /// Note handle.
    pub handle: String,
    /// What is wrong.
    pub message: String,
}

/// Unresolved `[[link]]` — a memory worth writing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnresolvedLink {
    /// Link target as written.
    pub target: String,
    /// Handle of the note that contains the link.
    pub found_in: String,
}

/// Full doctor report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DoctorReport {
    /// Notes matching decay / expiry signals.
    pub archive_candidates: Vec<ArchiveCandidate>,
    /// Thin or missing aliases.
    pub alias_warnings: Vec<QualityWarning>,
    /// Wikilink targets with no file.
    pub unresolved_links: Vec<UnresolvedLink>,
    /// Suggest reindex when index is cold/unavailable.
    pub index_hint: Option<String>,
    /// Compaction stats (set after apply, or dry-run estimate when none rolled).
    pub compaction: Option<CompactionStats>,
    /// Whether `--apply` ran.
    pub applied: bool,
    /// Suggested git commit message after apply.
    pub suggested_commit_message: Option<String>,
}

impl fmt::Display for DoctorReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "fff-memory doctor{}",
            if self.applied { " (applied)" } else { "" }
        )?;
        writeln!(f)?;

        if self.archive_candidates.is_empty() {
            writeln!(f, "Archive candidates: (none)")?;
        } else {
            writeln!(f, "Archive candidates ({}):", self.archive_candidates.len())?;
            for c in &self.archive_candidates {
                writeln!(f, "  - {}  [{}]", c.handle, c.reasons.join(", "))?;
            }
        }
        writeln!(f)?;

        if self.alias_warnings.is_empty() {
            writeln!(f, "Alias quality: ok")?;
        } else {
            writeln!(f, "Alias warnings ({}):", self.alias_warnings.len())?;
            for w in &self.alias_warnings {
                writeln!(f, "  - {}: {}", w.handle, w.message)?;
            }
        }
        writeln!(f)?;

        if self.unresolved_links.is_empty() {
            writeln!(f, "Unresolved links: (none)")?;
        } else {
            writeln!(
                f,
                "Memories worth writing ({} unresolved links):",
                self.unresolved_links.len()
            )?;
            for u in &self.unresolved_links {
                writeln!(f, "  - [[{}]] (from {})", u.target, u.found_in)?;
            }
        }
        writeln!(f)?;

        if let Some(hint) = &self.index_hint {
            writeln!(f, "Index: {hint}")?;
        } else {
            writeln!(f, "Index: ok")?;
        }

        if let Some(c) = &self.compaction {
            writeln!(
                f,
                "Access log compaction: kept={}, rolled={}, counters_updated={}",
                c.events_kept, c.events_rolled, c.counters_updated
            )?;
        }

        if let Some(msg) = &self.suggested_commit_message {
            writeln!(f)?;
            writeln!(f, "Suggested commit message:")?;
            writeln!(f, "{msg}")?;
        } else if !self.applied && !self.archive_candidates.is_empty() {
            writeln!(f)?;
            writeln!(
                f,
                "Run with --apply to archive candidates and compact the access log."
            )?;
        }

        Ok(())
    }
}

/// Options for [`run_doctor`].
#[derive(Debug, Clone, Default)]
pub struct DoctorOptions {
    /// When true, archive candidates + compact access log.
    pub apply: bool,
}

/// Scan the store, produce a report, optionally apply archives and compaction.
pub fn run_doctor(
    root: impl AsRef<Path>,
    retriever: Option<&dyn Retriever>,
    opts: DoctorOptions,
) -> Result<DoctorReport> {
    let root = root.as_ref();
    let store = MemoryStore::new(root);
    let access = AccessLog::open(root);
    let today = OffsetDateTime::now_utc().date();

    let mut archive_candidates = Vec::new();
    let mut alias_warnings = Vec::new();
    let mut unresolved_links = Vec::new();

    // Collect existing handles for link resolution
    let notes = walk_active_notes(root)?;
    let mut existing_handles = std::collections::HashSet::new();
    for (rel, _) in &notes {
        existing_handles.insert(rel_to_handle(rel));
    }

    for (rel, note) in &notes {
        let handle = rel_to_handle(rel);
        let age_days = (today - note.frontmatter.created).whole_days();

        let mut reasons = Vec::new();

        // Past expires
        if let Some(exp) = note.frontmatter.expires {
            if exp < today {
                reasons.push(format!("expired (expires {exp})"));
            }
        }

        // Session-summary older than 30d
        if note.frontmatter.note_type == NoteType::SessionSummary
            && age_days >= SESSION_SUMMARY_MAX_AGE_DAYS
        {
            reasons.push(format!(
                "session-summary older than {SESSION_SUMMARY_MAX_AGE_DAYS}d"
            ));
        }

        // Never accessed within 30 days of creation (age ≥ 30 and never accessed)
        if age_days >= NEVER_ACCESSED_DAYS && !access.ever_accessed(&handle) {
            reasons.push(format!(
                "never accessed within {NEVER_ACCESSED_DAYS}d of creation"
            ));
        }

        // No access in 90 days: note is old enough and has no recent JSONL events.
        // When never-accessed already fired, skip the redundant stale label.
        if age_days >= STALE_DAYS
            && !access.accessed_within(&handle, STALE_DAYS)
            && access.ever_accessed(&handle)
        {
            reasons.push(format!("no access in {STALE_DAYS}d"));
        }

        if !reasons.is_empty() {
            archive_candidates.push(ArchiveCandidate {
                handle: handle.clone(),
                reasons,
            });
        }

        // Alias quality
        let alias_count = note
            .frontmatter
            .aliases
            .iter()
            .map(|a| a.trim())
            .filter(|a| !a.is_empty())
            .count();
        if alias_count < 2 {
            alias_warnings.push(QualityWarning {
                handle: handle.clone(),
                message: format!(
                    "only {alias_count} non-empty alias(es); need at least 2 (aliases are load-bearing)"
                ),
            });
        }

        // Unresolved links
        for link in extract_wikilinks(&note.body) {
            let link_ns = link
                .namespace
                .clone()
                .unwrap_or_else(|| parent_namespace(&handle));
            let target_handle = MemoryStore::handle(&link_ns, &link.slug);
            let exists =
                existing_handles.contains(&target_handle) || store.exists(&link_ns, &link.slug);
            if !exists {
                unresolved_links.push(UnresolvedLink {
                    target: link.target,
                    found_in: handle.clone(),
                });
            }
        }
    }

    let index_hint = match retriever.map(|r| r.index_state()) {
        Some(IndexState::Ready) => None,
        Some(IndexState::Cold) => Some(
            "index is cold (scan in progress or incomplete); run `fff-memory reindex` if search is stale"
                .into(),
        ),
        Some(IndexState::Unavailable) | None => Some(
            "index unavailable; run `fff-memory reindex` after opening a retriever".into(),
        ),
    };

    let mut report = DoctorReport {
        archive_candidates,
        alias_warnings,
        unresolved_links,
        index_hint,
        compaction: None,
        applied: false,
        suggested_commit_message: None,
    };

    if opts.apply {
        let mut archived = 0usize;
        for c in &report.archive_candidates {
            let (ns, slug) = MemoryStore::parse_handle(&c.handle)?;
            match store.archive_note(&ns, &slug) {
                Ok(_) => archived += 1,
                Err(Error::NotFound { .. }) => {}
                Err(e) => return Err(e),
            }
        }
        let compaction = access.compact(COMPACT_MAX_AGE_DAYS)?;
        report.compaction = Some(compaction);
        report.applied = true;
        report.suggested_commit_message = Some(format!(
            "chore(memory): doctor archive {archived} note(s), compact access log (rolled {})",
            compaction.events_rolled
        ));
    }

    Ok(report)
}

fn parent_namespace(handle: &str) -> String {
    match handle.rsplit_once('/') {
        Some((ns, _)) => ns.to_string(),
        None => String::new(),
    }
}

fn rel_to_handle(rel: &Path) -> String {
    let s = rel.to_string_lossy().replace('\\', "/");
    s.strip_suffix(".md").unwrap_or(&s).to_string()
}

fn walk_active_notes(root: &Path) -> Result<Vec<(PathBuf, Note)>> {
    let mut out = Vec::new();
    if !root.exists() {
        return Ok(out);
    }
    walk_dir(root, root, &mut out)?;
    Ok(out)
}

fn walk_dir(root: &Path, dir: &Path, out: &mut Vec<(PathBuf, Note)>) -> Result<()> {
    let entries = fs::read_dir(dir).map_err(|e| Error::io(dir, e))?;
    for entry in entries {
        let entry = entry.map_err(|e| Error::io(dir, e))?;
        let path = entry.path();
        let name = entry.file_name();
        let name_s = name.to_string_lossy();
        if name_s == ".index" || name_s == ".archive" {
            continue;
        }
        if path.is_dir() {
            walk_dir(root, &path, out)?;
        } else if path.extension().and_then(|e| e.to_str()) == Some("md") {
            let text = fs::read_to_string(&path).map_err(|e| Error::io(&path, e))?;
            // Lenient: doctor must see hand-edited thin-alias notes
            if let Ok(note) = Note::parse_lenient(&text) {
                let rel = path.strip_prefix(root).unwrap_or(&path).to_path_buf();
                out.push((rel, note));
            }
        }
    }
    Ok(())
}

/// Helper for tests: shift a date string is awkward; use created age via Duration from now.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::access_log::{AccessEvent, AccessVia};
    use crate::retriever::testing::FakeRetriever;
    use tempfile::tempdir;
    use time::format_description::well_known::Rfc3339;
    use time::Duration;

    fn write_note(root: &Path, rel: &str, yaml_extra: &str, body: &str) {
        let path = root.join(rel);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        let md = format!(
            "---\ntitle: Test\naliases:\n  - one\n  - two\ntype: fact\ncreated: 2020-01-01\nupdated: 2020-01-01\n{yaml_extra}---\n{body}\n"
        );
        fs::write(path, md).unwrap();
    }

    struct NoteFix<'a> {
        rel: &'a str,
        title: &'a str,
        aliases: &'a str,
        typ: &'a str,
        created: &'a str,
        body: &'a str,
        extra: &'a str,
    }

    fn write_note_custom(root: &Path, fix: NoteFix<'_>) {
        let path = root.join(fix.rel);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        let md = format!(
            "---\ntitle: {title}\naliases:\n{aliases}\ntype: {typ}\ncreated: {created}\nupdated: {created}\n{extra}---\n{body}\n",
            title = fix.title,
            aliases = fix.aliases,
            typ = fix.typ,
            created = fix.created,
            extra = fix.extra,
            body = fix.body,
        );
        fs::write(path, md).unwrap();
    }

    fn days_ago_date(days: i64) -> String {
        let d = OffsetDateTime::now_utc().date() - Duration::days(days);
        format!("{:04}-{:02}-{:02}", d.year(), d.month() as u8, d.day())
    }

    #[test]
    fn expired_is_candidate() {
        let dir = tempdir().unwrap();
        let created = days_ago_date(10);
        write_note_custom(
            dir.path(),
            NoteFix {
                rel: "proj/exp.md",
                title: "Expired",
                aliases: "  - a\n  - b",
                typ: "fact",
                created: &created,
                body: "body",
                extra: "expires: 2020-01-01\n",
            },
        );
        let report = run_doctor(dir.path(), None, DoctorOptions::default()).unwrap();
        assert!(
            report
                .archive_candidates
                .iter()
                .any(|c| c.handle == "proj/exp" && c.reasons.iter().any(|r| r.contains("expired"))),
            "{:?}",
            report.archive_candidates
        );
    }

    #[test]
    fn session_summary_old_is_candidate() {
        let dir = tempdir().unwrap();
        let created = days_ago_date(40);
        write_note_custom(
            dir.path(),
            NoteFix {
                rel: "sess/sum.md",
                title: "Session",
                aliases: "  - a\n  - b",
                typ: "session-summary",
                created: &created,
                body: "body",
                extra: "",
            },
        );
        // Access so never-accessed doesn't also fire exclusively
        AccessLog::open(dir.path())
            .append("sess/sum", AccessVia::Read)
            .unwrap();
        let report = run_doctor(dir.path(), None, DoctorOptions::default()).unwrap();
        assert!(
            report.archive_candidates.iter().any(|c| {
                c.handle == "sess/sum" && c.reasons.iter().any(|r| r.contains("session-summary"))
            }),
            "{:?}",
            report.archive_candidates
        );
    }

    #[test]
    fn never_accessed_30d_is_candidate() {
        let dir = tempdir().unwrap();
        let created = days_ago_date(45);
        write_note_custom(
            dir.path(),
            NoteFix {
                rel: "old/note.md",
                title: "Old",
                aliases: "  - a\n  - b",
                typ: "fact",
                created: &created,
                body: "body",
                extra: "",
            },
        );
        let report = run_doctor(dir.path(), None, DoctorOptions::default()).unwrap();
        assert!(
            report.archive_candidates.iter().any(|c| {
                c.handle == "old/note" && c.reasons.iter().any(|r| r.contains("never accessed"))
            }),
            "{:?}",
            report.archive_candidates
        );
    }

    #[test]
    fn recent_access_skips_never_accessed() {
        let dir = tempdir().unwrap();
        let created = days_ago_date(45);
        write_note_custom(
            dir.path(),
            NoteFix {
                rel: "live/note.md",
                title: "Live",
                aliases: "  - a\n  - b",
                typ: "fact",
                created: &created,
                body: "body",
                extra: "",
            },
        );
        AccessLog::open(dir.path())
            .append("live/note", AccessVia::SearchHit)
            .unwrap();
        let report = run_doctor(dir.path(), None, DoctorOptions::default()).unwrap();
        assert!(
            !report
                .archive_candidates
                .iter()
                .any(|c| c.handle == "live/note"),
            "should not archive live note: {:?}",
            report.archive_candidates
        );
    }

    #[test]
    fn thin_aliases_warning() {
        let dir = tempdir().unwrap();
        write_note_custom(
            dir.path(),
            NoteFix {
                rel: "thin.md",
                title: "Thin",
                aliases: "  - only-one",
                typ: "fact",
                created: "2026-01-01",
                body: "body",
                extra: "",
            },
        );
        let report = run_doctor(dir.path(), None, DoctorOptions::default()).unwrap();
        assert!(
            report
                .alias_warnings
                .iter()
                .any(|w| w.handle == "thin" && w.message.contains("alias")),
            "{:?}",
            report.alias_warnings
        );
    }

    #[test]
    fn unresolved_links_listed() {
        let dir = tempdir().unwrap();
        write_note(
            dir.path(),
            "a.md",
            "",
            "See [[missing-target]] and [[proj/also-missing]].",
        );
        let report = run_doctor(dir.path(), None, DoctorOptions::default()).unwrap();
        assert!(
            report.unresolved_links.len() >= 2,
            "{:?}",
            report.unresolved_links
        );
    }

    #[test]
    fn apply_archives_and_never_hard_deletes() {
        let dir = tempdir().unwrap();
        let created = days_ago_date(100);
        write_note_custom(
            dir.path(),
            NoteFix {
                rel: "gone/x.md",
                title: "Gone",
                aliases: "  - a\n  - b",
                typ: "fact",
                created: &created,
                body: "body",
                extra: "expires: 2020-06-01\n",
            },
        );
        assert!(dir.path().join("gone/x.md").is_file());
        let report = run_doctor(dir.path(), None, DoctorOptions { apply: true }).unwrap();
        assert!(report.applied);
        assert!(!dir.path().join("gone/x.md").is_file());
        assert!(dir.path().join(".archive/gone/x.md").is_file());
        assert!(report.suggested_commit_message.is_some());
        assert!(report.compaction.is_some());
    }

    #[test]
    fn cold_index_hint() {
        let dir = tempdir().unwrap();
        let fake = FakeRetriever {
            state: IndexState::Cold,
            ..Default::default()
        };
        let report = run_doctor(dir.path(), Some(&fake), DoctorOptions::default()).unwrap();
        assert!(
            report
                .index_hint
                .as_ref()
                .is_some_and(|h| h.contains("reindex")),
            "{:?}",
            report.index_hint
        );
    }

    #[test]
    fn stale_90d_with_old_access_only_in_counters() {
        let dir = tempdir().unwrap();
        let created = days_ago_date(120);
        write_note_custom(
            dir.path(),
            NoteFix {
                rel: "stale/n.md",
                title: "Stale",
                aliases: "  - a\n  - b",
                typ: "fact",
                created: &created,
                body: "body",
                extra: "",
            },
        );
        let log = AccessLog::open(dir.path());
        let old_ts = (OffsetDateTime::now_utc() - Duration::days(100))
            .format(&Rfc3339)
            .unwrap();
        log.append_event(AccessEvent {
            ts: old_ts,
            handle: "stale/n".into(),
            via: AccessVia::Read,
        })
        .unwrap();
        log.compact(90).unwrap();
        assert!(log.ever_accessed("stale/n"));
        assert!(!log.accessed_within("stale/n", 90));

        let report = run_doctor(dir.path(), None, DoctorOptions::default()).unwrap();
        assert!(
            report.archive_candidates.iter().any(|c| {
                c.handle == "stale/n" && c.reasons.iter().any(|r| r.contains("no access in 90"))
            }),
            "{:?}",
            report.archive_candidates
        );
    }

    #[test]
    fn display_is_nonempty() {
        let dir = tempdir().unwrap();
        let report = run_doctor(dir.path(), None, DoctorOptions::default()).unwrap();
        let s = report.to_string();
        assert!(s.contains("doctor"));
        assert!(s.contains("Archive"));
    }
}
