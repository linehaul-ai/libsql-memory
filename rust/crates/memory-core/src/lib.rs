//! `memory-core` — note format, identity helpers, and (later) store/access-log/Retriever.
//!
//! Spec ownership: [`01-note-format`](../../../.specs/01-note-format.toml) for this
//! slice; write path, lifecycle, and the Retriever trait land from specs 02/05/06.
//!
//! **Invariant:** this crate never depends on `fff-search` or network I/O.
//! Files are the source of truth; search is a disposable cache owned by
//! `memory-index`.

#![deny(missing_docs)]

mod error;
mod links;
mod note;
mod slugify;

pub use error::{Error, Result};
pub use links::{extract_wikilinks, Wikilink};
pub use note::{Note, NoteFrontmatter, NoteType};
pub use slugify::{slugify, validate_namespace, SLUG_MAX_LEN};
