# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Status: Rust Implementation Complete; Dogfooding In Progress

This repo has been rebuilt as **fff-memory** (working name): a persistent agent memory for
Claude Code, written in Rust around the [fff](https://github.com/dmtrKovalenko/fff) search
engine. The previous Go + LibSQL + embeddings implementation is abandoned.

**The specs in `.specs/*.toml` are the authoritative design.** Read the relevant spec
before implementing anything; update the spec when a decision changes.

| Spec | Covers |
|------|--------|
| `00-overview.toml` | Thesis, principles, non-goals, success metrics, predecessor lessons |
| `01-note-format.toml` | Memory file format (markdown + YAML frontmatter), aliases discipline |
| `02-write-path.toml` | Atomic writes, dedup probe, directory layout, concurrency |
| `03-retrieval.toml` | Layered search pipeline, ranking, token-budgeted output |
| `04-mcp-interface.toml` | The five MCP tools + CLI subcommands |
| `05-lifecycle.toml` | Access log, reinforcement, decay, `doctor` |
| `06-workspace.toml` | Crate layout, Retriever trait, testing policy |
| `07-hooks-automation.toml` | Thin hooks, skill, plugin packaging |
| `08-roadmap.toml` | Phases, risks, open questions |

## Core Thesis

**Write smart, search cheap.** The LLM storing a memory does the semantic work once at write
time (title, aliases, tags). Query time is lexical search via fff — microseconds, zero
external dependencies. Non-negotiable invariants:

- Files are the source of truth; the index is a disposable cache.
- The write path has **zero** network and **zero** index dependencies. Storing can never fail
  because search is unavailable.
- Five MCP tools, no more: `memory_store`, `memory_search`, `memory_read`, `memory_forget`,
  `memory_stats`.
- Search output is token-budgeted, snippet-first, and every result carries a "why".
- Empty results explain what was tried — never silent.

## Workspace Layout (per spec 06)

The Cargo workspace is the repository: `Cargo.toml` sits at the repo root alongside the plugin
wiring (`.mcp.json`, `hooks/`, `skills/`, `commands/`).

```
.
├── .specs/                  # authoritative design (TOML)
└── crates/
    ├── memory-core/         # note format, atomic store, access log, Retriever trait
    ├── memory-index/        # FffRetriever — the ONLY crate that imports fff-search
    ├── memory-mcp/          # rmcp stdio server, five tools, rank/budget stages
    └── fff-memory/          # the binary: serve | store | search | read | forget | stats | doctor | reindex
```

## Plugin Packaging

The repository is also the Claude Code plugin: `.claude-plugin/marketplace.json` declares one
plugin, `fff-memory`, with `"source": "./"`, so the plugin root and the repo root are the same
directory. `.claude-plugin/plugin.json` is **metadata-only** — it must never gain `skills`,
`commands`, `hooks`, or `mcpServers` keys, because their absence is what lets Claude Code
auto-discover `skills/`, `commands/`, `hooks/hooks.json`, and `.mcp.json` from the plugin root.
`crates/fff-memory/tests/plugin_package.rs` asserts exactly that, along with the byte contents
of all eight wiring files, so `cargo test --workspace` is the gate on plugin correctness.

Installing it locally (required for dogfooding — the tools, hooks, and skill are inert until
the plugin is both registered and enabled):

```bash
claude plugin marketplace add /Users/fakebizprez/Developer/projects/libsql-memory
# then enable fff-memory@linehaul-ai-fff-memory via /plugin
```

Tool names are plugin-scoped once installed, e.g.
`mcp__plugin_fff-memory_fff-memory__memory_store`. Pre-approve that one permission or the Stop
hook silently skips its write.

## Build & Development Commands

Run from the repo root:

```bash
cargo build --workspace
cargo test --workspace       # no test may require network or Docker
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all
```

## fff-search Integration Notes

A local skill exists: invoke `fff-search` (Skill tool) before writing code against the crate —
it carries the full API reference. Key facts:

- **Single dependency**: `fff-search = "0.10"` (grep is a module; `fff-query-parser` types are
  re-exported at the root). Pin the exact version; confine all imports to `memory-index`.
- **Frecency persists**: `FrecencyTracker` / `QueryTracker` are LMDB-backed databases we open
  at paths we choose — put them under `.index/`. This resolved spec OQ-1: fff's own frecency
  survives restarts. Our append-only access log remains the authority for lifecycle/decay
  decisions.
- **Init order**: open and `.init()` `SharedFrecency` (and `SharedQueryTracker`) *before*
  `FilePicker::new_with_shared_state`; then `wait_for_indexing_complete` before the first search.
- **Mode**: use `FFFMode::Ai`.
- **Lock discipline**: hold picker read guards only for the duration of a search; background
  rescan threads need the write lock.
- **Walker**: default `ripgrep` feature (the `zlob` alternative needs a Zig toolchain — don't).

## Conventions

- MCP protocol via the official `rmcp` SDK — never hand-rolled.
- Stdout is reserved for the MCP protocol; all logging to stderr.
- No committed binaries, ever. `.index/`, `target/`, `*.db` are gitignored.
- Hooks stay thin: one tool call each; all logic lives in the tools (spec 07).
- Errors are structured and actionable: name the failing input and the fix.

## Legacy Code

The Go + LibSQL + embeddings predecessor was removed on 2026-08-15; it remains recoverable
from `main` history. Its post-mortem is encoded in `00-overview.toml` `[lessons]` — read it
before relaxing any invariant above.
