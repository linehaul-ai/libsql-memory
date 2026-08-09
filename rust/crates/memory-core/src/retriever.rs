//! Backend-neutral retrieval surface (spec 06).
//!
//! `memory-index` will implement this with fff-search. `memory-core` only depends
//! on the trait so the write-path dedup probe stays best-effort and testable.

use std::path::PathBuf;

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

/// Search backend used by store (dedup) and later by MCP search tools.
///
/// Implementations must not be required on the write path: callers pass
/// `Option<&dyn Retriever>` and treat any error as “skip dedup, create note”.
pub trait Retriever {
    /// Fuzzy/path search over note paths and slugs (titles become slugs).
    fn find_files(&self, query: &str, scope: Option<&str>) -> Result<Vec<FileHit>>;

    /// Content search over note files (frontmatter included).
    fn grep(&self, query: &str, mode: GrepMode, scope: Option<&str>) -> Result<Vec<ContentHit>>;

    /// Current index readiness.
    fn index_state(&self) -> IndexState;

    /// Drop and rebuild the index from disk. Optional for fakes.
    fn reindex(&self) -> Result<()>;
}

#[cfg(test)]
pub(crate) mod fake {
    use super::*;
    use std::sync::Mutex;

    fn path_in_scope(path: &str, scope: &str) -> bool {
        path == scope || path.starts_with(&format!("{scope}/"))
    }

    /// In-memory retriever for unit tests.
    #[derive(Debug, Default)]
    pub struct FakeRetriever {
        pub state: IndexState,
        pub files: Mutex<Vec<FileHit>>,
        pub contents: Mutex<Vec<ContentHit>>,
        pub find_error: Mutex<Option<String>>,
        pub grep_error: Mutex<Option<String>>,
        pub reindex_calls: Mutex<u32>,
    }

    impl FakeRetriever {
        pub fn ready() -> Self {
            Self {
                state: IndexState::Ready,
                ..Default::default()
            }
        }

        pub fn with_files(hits: Vec<FileHit>) -> Self {
            Self {
                state: IndexState::Ready,
                files: Mutex::new(hits),
                ..Default::default()
            }
        }
    }

    impl Retriever for FakeRetriever {
        fn find_files(&self, query: &str, scope: Option<&str>) -> Result<Vec<FileHit>> {
            if let Some(msg) = self.find_error.lock().unwrap().clone() {
                return Err(crate::Error::Retriever(msg));
            }
            let q = query.to_ascii_lowercase();
            let files = self.files.lock().unwrap();
            Ok(files
                .iter()
                .filter(|h| {
                    let p = h.path.to_string_lossy().to_ascii_lowercase();
                    if let Some(s) = scope {
                        if !path_in_scope(&p, &s.to_ascii_lowercase()) {
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
                    if let Some(s) = scope {
                        if !path_in_scope(&p, &s.to_ascii_lowercase()) {
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

        fn reindex(&self) -> Result<()> {
            *self.reindex_calls.lock().unwrap() += 1;
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::fake::FakeRetriever;
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
        let hits = fake.find_files("foo", Some("proj")).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].path, PathBuf::from("proj/a/foo.md"));
    }

    #[test]
    fn fake_find_error_surfaces_as_retriever() {
        let fake = FakeRetriever::ready();
        *fake.find_error.lock().unwrap() = Some("index corrupt".into());
        let err = fake.find_files("x", None).unwrap_err();
        assert!(matches!(err, crate::Error::Retriever(_)));
    }

    #[test]
    fn index_state_default_unavailable() {
        assert_eq!(IndexState::default(), IndexState::Unavailable);
    }
}
