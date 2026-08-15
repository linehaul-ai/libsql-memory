//! Structured errors for note parse/validate, identity, and the write path.

use std::path::PathBuf;

use thiserror::Error;

/// Fallible operation result for `memory-core`.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors from note format, slugify, namespace validation, store, and retrieval.
#[derive(Debug, Error)]
pub enum Error {
    /// File text is missing a leading `---` / `---` YAML frontmatter block.
    #[error(
        "missing YAML frontmatter: expected a leading --- ... --- block (Obsidian-compatible)"
    )]
    MissingFrontmatter,

    /// YAML inside the frontmatter block failed to parse.
    #[error("invalid frontmatter YAML: {0}")]
    Yaml(#[from] serde_yaml::Error),

    /// A required field is missing or fails a domain rule.
    #[error("validation failed on `{field}`: {message}")]
    Validation {
        /// Frontmatter field name (or `"body"` / logical name).
        field: String,
        /// What is wrong and how to fix it.
        message: String,
    },

    /// Title could not be turned into a non-empty kebab-case slug.
    #[error(
        "cannot slugify title {title:?}: produce a title with at least one ASCII letter or digit"
    )]
    InvalidSlug {
        /// The title that failed slugification.
        title: String,
    },

    /// Namespace path is absolute, has `..`, or empty segments.
    #[error("invalid namespace {path:?}: {reason}")]
    InvalidNamespace {
        /// The rejected namespace string.
        path: String,
        /// Why it was rejected.
        reason: String,
    },

    /// Filesystem I/O failed while reading or writing a note.
    #[error("I/O error on {path}: {source}")]
    Io {
        /// Path involved in the failure (best-effort).
        path: PathBuf,
        /// Underlying OS error.
        #[source]
        source: std::io::Error,
    },

    /// Note file is missing on disk.
    #[error("note not found: {handle}")]
    NotFound {
        /// `namespace/slug` handle that was requested.
        handle: String,
    },

    /// Retrieval backend failed (store treats this as best-effort skip for dedup).
    #[error("retriever error: {0}")]
    Retriever(String),
}

impl Error {
    /// Build a [`Error::Validation`] for a named field.
    pub fn validation(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Validation {
            field: field.into(),
            message: message.into(),
        }
    }

    /// Build an [`Error::Io`] for a path.
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }
}
