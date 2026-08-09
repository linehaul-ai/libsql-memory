//! `memory-core` — note format, identity helpers, atomic store, access log, doctor, Retriever trait.
//!
//! Spec ownership: [`01-note-format`](../../../.specs/01-note-format.toml),
//! [`02-write-path`](../../../.specs/02-write-path.toml),
//! [`05-lifecycle`](../../../.specs/05-lifecycle.toml),
//! [`06-workspace`](../../../.specs/06-workspace.toml) (Retriever trait).
//!
//! **Invariant:** this crate never depends on `fff-search` or network I/O.
//! Files are the source of truth; search is a disposable cache owned by
//! `memory-index`. The write path has zero index dependencies: store succeeds
//! even when the retriever is cold, corrupt, or absent.

#![deny(missing_docs)]

mod access_log;
mod doctor;
mod error;
mod links;
mod note;
mod retriever;
mod slugify;
mod store;

pub use access_log::{
    AccessCounters, AccessEvent, AccessLog, AccessVia, CompactionStats, ViaCounts,
    COMPACT_MAX_AGE_DAYS,
};
pub use doctor::{
    run_doctor, ArchiveCandidate, DoctorOptions, DoctorReport, QualityWarning, UnresolvedLink,
    NEVER_ACCESSED_DAYS, SESSION_SUMMARY_MAX_AGE_DAYS, STALE_DAYS,
};
pub use error::{Error, Result};
pub use links::{extract_wikilinks, Wikilink};
pub use note::{Note, NoteFrontmatter, NoteType};
pub use retriever::{testing, ContentHit, FileHit, GrepMode, IndexSnapshot, IndexState, Retriever};
pub use slugify::{slugify, validate_namespace, SLUG_MAX_LEN};
pub use store::{MemoryStore, MergeMode, StoreAction, StoreInput, StoreOutcome};
