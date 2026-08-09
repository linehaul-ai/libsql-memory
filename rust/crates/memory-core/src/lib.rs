//! `memory-core` — note format, identity helpers, atomic store, and Retriever trait.
//!
//! Spec ownership: [`01-note-format`](../../../.specs/01-note-format.toml),
//! [`02-write-path`](../../../.specs/02-write-path.toml),
//! [`06-workspace`](../../../.specs/06-workspace.toml) (Retriever trait).
//!
//! **Invariant:** this crate never depends on `fff-search` or network I/O.
//! Files are the source of truth; search is a disposable cache owned by
//! `memory-index`. The write path has zero index dependencies: store succeeds
//! even when the retriever is cold, corrupt, or absent.

#![deny(missing_docs)]

mod error;
mod links;
mod note;
mod retriever;
mod slugify;
mod store;

pub use error::{Error, Result};
pub use links::{extract_wikilinks, Wikilink};
pub use note::{Note, NoteFrontmatter, NoteType};
pub use retriever::{testing, ContentHit, FileHit, GrepMode, IndexState, Retriever};
pub use slugify::{slugify, validate_namespace, SLUG_MAX_LEN};
pub use store::{MemoryStore, MergeMode, StoreAction, StoreInput, StoreOutcome};
