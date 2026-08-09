//! `memory-index` — FffRetriever over fff-search.
//!
//! Spec ownership: [`06-workspace`](../../../.specs/06-workspace.toml) (FffRetriever),
//! search stages of [`03-retrieval`](../../../.specs/03-retrieval.toml).
//!
//! **Invariant:** this is the only crate that imports `fff-search`. Pin the version;
//! `memory-core` talks only to the [`memory_core::Retriever`] trait.

#![deny(missing_docs)]

mod fff;

pub use fff::FffRetriever;
