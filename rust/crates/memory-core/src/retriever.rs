//! Backend-neutral retrieval surface (spec 06).
//!
//! `memory-index` will implement this with fff-search. `memory-core` only depends
//! on the trait so the write-path dedup probe stays best-effort and testable.

use std::path::{Path, PathBuf};

use crate::error::Result;

/// Path-ranked hit from file/title search (`fffind`-style).
#[derive(Debug, Clone, PartialEq)]
pub struct FileHit {
    /// Path relative to the memory root (e.g. `linehaul/tms/deploy.md`).
    pub path: PathBuf,
    /// Backend score; higher is better. Core uses this only for ordering probes.
    pub score: f32,
}

/// Content hit from grep over note bodies (including frontmatter).
#[derive(Debug, Clone, PartialEq)]
pub struct ContentHit {
    /// Path relative to the memory root.
    pub path: PathBuf,
    /// Matching line (trimmed) when available.
    pub snippet: String,
    /// 1-based line number when known; `0` if unknown.
    pub line: u32,
    /// Backend score; higher is better.
    pub score: f32,
    /// Exact note field responsible for the match.
    pub matched: ContentMatch,
}

/// Backend-neutral content-match provenance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContentMatch {
    /// The frontmatter title.
    Title,
    /// Exact frontmatter alias, or empty when only the alias field is known.
    Alias(String),
    /// A frontmatter tag.
    Tags,
    /// Markdown body content (or other non-retrieval frontmatter).
    Body,
}

/// Grep matching mode for [`Retriever::grep`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GrepMode {
    /// Literal substring match.
    #[default]
    Plain,
    /// Regex.
    Regex,
    /// Typo-tolerant fuzzy.
    Fuzzy,
}

/// Warmth of the search index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IndexState {
    /// Not open / not scanned; store skips the dedup probe.
    #[default]
    Unavailable,
    /// Scan in progress; probe may be incomplete — store still may try, treating errors as skip.
    Cold,
    /// Ready for queries.
    Ready,
}

/// Point-in-time state and live metrics for a retrieval index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct IndexSnapshot {
    /// Current readiness.
    pub state: IndexState,
    /// Live, non-tombstoned files in the active index.
    pub files_indexed: u64,
    /// Wall-clock duration of the most recently completed full scan.
    pub last_scan_ms: u64,
}

/// Search backend used by store (dedup) and MCP search tools.
///
/// Implementations must not be required on the write path: callers pass
/// `Option<&dyn Retriever>` and treat any error as “skip dedup, create note”.
///
/// `Send + Sync` so MCP servers can hold `Arc<dyn Retriever>`.
pub trait Retriever: Send + Sync {
    /// Fuzzy/path search over note paths and slugs (titles become slugs).
    fn find_files(
        &self,
        query: &str,
        scope: Option<&str>,
        include_archived: bool,
    ) -> Result<Vec<FileHit>>;

    /// Content search over note files (frontmatter included).
    fn grep(
        &self,
        query: &str,
        mode: GrepMode,
        scope: Option<&str>,
        include_archived: bool,
    ) -> Result<Vec<ContentHit>>;

    /// Current index readiness.
    fn index_state(&self) -> IndexState;

    /// Current index readiness and live scan metrics.
    fn index_snapshot(&self) -> IndexSnapshot {
        IndexSnapshot {
            state: self.index_state(),
            files_indexed: 0,
            last_scan_ms: 0,
        }
    }

    /// Live indexed paths relative to the memory root.
    fn indexed_paths(&self) -> Result<Vec<PathBuf>>;

    /// Reinforce a retrieved path in backend ranking; callers treat errors as best-effort.
    fn track_access(&self, _path: &Path) -> Result<()> {
        Ok(())
    }

    /// Drop and rebuild the index from disk. Optional for fakes.
    fn reindex(&self) -> Result<()>;
}

/// In-memory [`Retriever`] for tests in this crate and dependents.
pub mod testing {
    use super::*;
    use std::sync::Mutex;

    fn path_in_scope(path: &str, scope: &str) -> bool {
        path == scope || path.starts_with(&format!("{scope}/"))
    }

    /// In-memory retriever for unit and integration tests.
    #[derive(Debug, Default)]
    pub struct FakeRetriever {
        /// Index readiness reported to callers.
        pub state: IndexState,
        /// Path hits returned by [`Retriever::find_files`].
        pub files: Mutex<Vec<FileHit>>,
        /// Content hits returned by [`Retriever::grep`].
        pub contents: Mutex<Vec<ContentHit>>,
        /// When set, `find_files` fails with this message.
        pub find_error: Mutex<Option<String>>,
        /// When set, `grep` fails with this message.
        pub grep_error: Mutex<Option<String>>,
        /// Count of [`Retriever::reindex`] calls.
        pub reindex_calls: Mutex<u32>,
    }

    impl FakeRetriever {
        /// Ready index with empty hit lists.
        pub fn ready() -> Self {
            Self {
                state: IndexState::Ready,
                ..Default::default()
            }
        }

        /// Ready index seeded with path hits.
        pub fn with_files(hits: Vec<FileHit>) -> Self {
            Self {
                state: IndexState::Ready,
                files: Mutex::new(hits),
                ..Default::default()
            }
        }
    }

    impl Retriever for FakeRetriever {
        fn find_files(
            &self,
            query: &str,
            scope: Option<&str>,
            include_archived: bool,
        ) -> Result<Vec<FileHit>> {
            if let Some(msg) = self.find_error.lock().unwrap().clone() {
                return Err(crate::Error::Retriever(msg));
            }
            let q = query.to_ascii_lowercase();
            let files = self.files.lock().unwrap();
            Ok(files
                .iter()
                .filter(|h| {
                    let p = h.path.to_string_lossy().to_ascii_lowercase();
                    if !include_archived && (p == ".archive" || p.starts_with(".archive/")) {
                        return false;
                    }
                    let logical = p.strip_prefix(".archive/").unwrap_or(&p);
                    if let Some(s) = scope {
                        if !path_in_scope(logical, &s.to_ascii_lowercase()) {
                            return false;
                        }
                    }
                    p.contains(&q) || q.is_empty()
                })
                .cloned()
                .collect())
        }

        fn grep(
            &self,
            query: &str,
            _mode: GrepMode,
            scope: Option<&str>,
            include_archived: bool,
        ) -> Result<Vec<ContentHit>> {
            if let Some(msg) = self.grep_error.lock().unwrap().clone() {
                return Err(crate::Error::Retriever(msg));
            }
            let q = query.to_ascii_lowercase();
            let contents = self.contents.lock().unwrap();
            Ok(contents
                .iter()
                .filter(|h| {
                    let p = h.path.to_string_lossy().to_ascii_lowercase();
                    if !include_archived && (p == ".archive" || p.starts_with(".archive/")) {
                        return false;
                    }
                    let logical = p.strip_prefix(".archive/").unwrap_or(&p);
                    if let Some(s) = scope {
                        if !path_in_scope(logical, &s.to_ascii_lowercase()) {
                            return false;
                        }
                    }
                    h.snippet.to_ascii_lowercase().contains(&q) || p.contains(&q)
                })
                .cloned()
                .collect())
        }

        fn index_state(&self) -> IndexState {
            self.state
        }

        fn index_snapshot(&self) -> IndexSnapshot {
            IndexSnapshot {
                state: self.state,
                files_indexed: self.files.lock().unwrap().len() as u64,
                last_scan_ms: 0,
            }
        }

        fn indexed_paths(&self) -> Result<Vec<PathBuf>> {
            Ok(self
                .files
                .lock()
                .unwrap()
                .iter()
                .map(|hit| hit.path.clone())
                .collect())
        }

        fn reindex(&self) -> Result<()> {
            *self.reindex_calls.lock().unwrap() += 1;
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testing::FakeRetriever;
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn fake_filters_by_scope_and_query() {
        let fake = FakeRetriever::with_files(vec![
            FileHit {
                path: PathBuf::from("proj/a/foo.md"),
                score: 1.0,
            },
            FileHit {
                path: PathBuf::from("other/foo.md"),
                score: 0.9,
            },
        ]);
        let hits = fake.find_files("foo", Some("proj"), false).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].path, PathBuf::from("proj/a/foo.md"));
    }

    #[test]
    fn fake_find_error_surfaces_as_retriever() {
        let fake = FakeRetriever::ready();
        *fake.find_error.lock().unwrap() = Some("index corrupt".into());
        let err = fake.find_files("x", None, false).unwrap_err();
        assert!(matches!(err, crate::Error::Retriever(_)));
    }

    #[test]
    fn index_state_default_unavailable() {
        assert_eq!(IndexState::default(), IndexState::Unavailable);
    }
}
