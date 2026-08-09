# fff-memory

Persistent semantic memory for Claude Code — markdown files, lexical search, zero
infrastructure. Built in Rust on the [fff](https://github.com/dmtrKovalenko/fff) search engine.

> **Status: rewrite in progress.** This repo previously held a Go + LibSQL + embeddings
> implementation. It is being rebuilt from specs; see [`rust/.specs/`](rust/.specs/) for the
> authoritative design and [Why the rewrite](#why-the-rewrite) for the story.

## The idea: write smart, search cheap

The agent writing a memory is an LLM — the smartest text processor in the loop. It does the
semantic work **once, at write time**: a clear title, synonyms and alternate phrasings
(`aliases`), tags. Query time then needs only lexical search, which fff delivers in
microseconds from a warm in-memory index — no embedding model, no vector database, no network.

A memory is one markdown file:

```markdown
---
title: Deploy pipeline uses GitHub Actions + Docker
aliases: [release process, CI/CD, shipping, how we deploy]
tags: [infra, deployment]
type: fact
created: 2026-08-08
updated: 2026-08-08
---
The deployment pipeline builds in GitHub Actions and publishes Docker images.
Related: [[docker-registry-auth]]
```

Because memories are plain files: `cat` them, edit them, `grep` them, and put the whole store
under git — your agent's memory gets history, diffs, review, and sync for free.

## How it works

One Rust binary, both an MCP stdio server and a CLI, embedding the `fff-search` crate directly:

- **Store** — atomic file writes with zero network and zero index dependency. Storing can
  never fail because search is down (the failure that killed the predecessor).
- **Search** — layered lexical retrieval: fuzzy title/path search + content grep in parallel,
  escalating to typo-tolerant fuzzy grep. Results are ranked by match quality × **frecency**
  (how recently/often a memory is used) and returned as token-budgeted snippets, each with a
  "why it matched" line.
- **Lifecycle** — retrieval reinforces a memory's rank; a `doctor` command surfaces never-used
  and expired notes for archiving. Memory systems die of hoarding; usage data is the cure.

### MCP tools

`memory_store` · `memory_search` · `memory_read` · `memory_forget` · `memory_stats`

### Automation

Thin Claude Code hooks (session start, prompt submit, session end) plus a `memory-usage` skill
that teaches the alias discipline good recall depends on.

## Planned layout

```
rust/
├── .specs/              # design specs (TOML) — start here
└── crates/
    ├── memory-core/     # note format, write path, access log, Retriever trait
    ├── memory-index/    # fff-search integration
    ├── memory-mcp/      # MCP server (rmcp), retrieval pipeline
    └── fff-memory/      # binary: serve | store | search | read | forget | stats | doctor | reindex
```

## Why the rewrite

The predecessor required a live embedding server on the write path. When that server was
unreachable, nothing could ever be stored — the database was found months later with a perfect
schema and zero rows. The redesign makes that failure impossible by construction: writes are
plain file operations, search is a disposable cache over them, and the semantic burden moved to
write time where an LLM is always present. The full post-mortem is encoded as `[lessons]` in
[`rust/.specs/00-overview.toml`](rust/.specs/00-overview.toml).

## Building (once phase 1 lands)

```bash
cd rust
cargo build --workspace
cargo test --workspace
```

## License

MIT
