//! `memory-mcp` — five MCP tools, merge/rank/budget, thin adapters over memory-core.
//!
//! Spec ownership: [`04-mcp-interface`](../../../.specs/04-mcp-interface.toml),
//! pipeline steps 4–6 of [`03-retrieval`](../../../.specs/03-retrieval.toml).
//!
//! **Invariant:** does not import `fff-search`. Search stages 1–3 come through
//! [`memory_core::Retriever`]; this crate merges, ranks, budgets, and serves MCP.

#![deny(missing_docs)]

mod access_log;
mod search;
mod server;
mod service;

pub use access_log::{AccessEvent, AccessLog, AccessVia};
pub use search::{
    apply_budget, merge_hits, rank_hits, type_boost, BudgetedSearch, MatchStage, MergedHit,
    MoreHit, RankedHit, SearchHit, StageName, BOTH_STAGES_BONUS, BUDGET_BYTES_DEFAULT,
    BUDGET_BYTES_MAX, MIN_RESULTS_FOR_FUZZY, SEARCH_LIMIT_DEFAULT,
};
pub use server::{serve_stdio, MemoryServer};
pub use service::{
    parse_handle, ForgetAction, ForgetOutcome, MemoryService, ReadOutcome, SearchOptions,
    SearchResponse, StatsSnapshot, StoreRequest,
};
