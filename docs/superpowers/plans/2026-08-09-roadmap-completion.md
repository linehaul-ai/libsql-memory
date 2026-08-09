# Roadmap Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete every automatable implementation item in `rust/.specs/00-overview.toml` through `08-roadmap.toml`, with red-green tests and explicit evidence for the remaining elapsed-use exit criteria.

**Architecture:** Keep files authoritative and the write path independent of search. Extend the existing `Retriever` seam only where the MCP contract needs backend metrics, archived search, and access reinforcement; keep all fff imports in `memory-index`. Reuse `MemoryService` for both rmcp and CLI, then replace only the root plugin wiring that still points at Go.

**Tech Stack:** Rust 2021, fff-search `=0.10.3`, rmcp `3.1.2`, clap 4, tokio 1, serde/serde_json, time, tempfile, proptest for mandated property tests.

## Global Constraints

- `rust/.specs/*.toml` are authoritative; update the relevant spec in the same task whenever a previously implicit decision becomes concrete.
- Files remain the source of truth; `.index/` is disposable and `.archive/` is reversible.
- `memory_store` performs no network I/O and succeeds when retrieval is unavailable.
- Exactly five MCP tools exist: `memory_store`, `memory_search`, `memory_read`, `memory_forget`, `memory_stats`.
- MCP uses official rmcp over stdio; stdout is protocol-only and diagnostics go to stderr.
- `fff-search` stays pinned to `=0.10.3`, imports stay in `memory-index`, mode is `FFFMode::Ai`, DBs initialize before the picker, and picker read guards stay search-scoped.
- New behavior follows TDD: add one focused failing test, run it and record the expected failure, implement the minimum, then rerun focused and affected tests.
- No test requires network or Docker; no binary, `.index/`, or database file is committed.
- Preserve the abandoned Go tree until OQ-5 receives an explicit destructive user decision; never extend or repair it.

---

### Task 1: Retrieval Contract and Live fff Index

**Files:**
- Modify: `rust/crates/memory-core/src/retriever.rs`
- Modify: `rust/crates/memory-core/src/store.rs`
- Modify: `rust/crates/memory-index/src/fff.rs`
- Modify: `rust/crates/memory-index/tests/fff_retriever.rs`
- Modify: `rust/crates/memory-mcp/src/service.rs`
- Modify: `rust/crates/memory-mcp/src/search.rs`
- Modify: `rust/.specs/03-retrieval.toml`
- Modify: `rust/.specs/05-lifecycle.toml`
- Modify: `rust/.specs/06-workspace.toml`

**Interfaces:**
- Produce `IndexSnapshot { state: IndexState, files_indexed: u64, last_scan_ms: u64 }` in `memory-core`.
- Extend `Retriever::find_files` and `Retriever::grep` with `include_archived: bool`.
- Add `Retriever::index_snapshot()` and best-effort `Retriever::track_access(path: &Path)`; fake defaults remain deterministic.
- `MemoryService::search` runs file-find and plain grep concurrently, propagates stage-labelled errors, and records only shaped results.

- [ ] **Step 1: Write failing contract tests**

Add focused tests proving: archived hits work with namespace scope; 60 out-of-scope notes cannot hide an in-scope hit; find/plain calls overlap using a barrier fake; a stage error names `find_files`, `grep_plain`, or `grep_fuzzy`; an external file edit becomes searchable without `reindex`; `IndexSnapshot` reports live file count and nonzero scan time; `reindex` preserves `access.jsonl` while recreating the fff DB directories; representative prose title/alias/body queries order the expected note first.

- [ ] **Step 2: Run red tests**

Run:

```bash
cd rust
cargo test -p memory-index --test fff_retriever -- --nocapture
cargo test -p memory-mcp service::tests:: -- --nocapture
```

Expected: failures demonstrate disabled watching, dead archived search, sequential stages, swallowed errors, missing index metrics, and incomplete reindex semantics.

- [ ] **Step 3: Implement the minimum contract**

Use `watch: true`; constrain fff queries before pagination; page grep via `next_file_offset` when needed. Search `.archive/` with a short-lived synchronous `FilePicker` rooted there only when `include_archived=true`, prefix returned paths with `.archive/`, and strip that prefix for logical namespace checks/handles. Normalize each backend result page to a stable `0.0..=1.0` component score before MCP merge. Time open/reindex with `Instant`, report `live_file_count`, destroy/reinitialize only `.index/frecency` and `.index/queries`, then trigger and await a rescan. Use `std::thread::scope` for stage 2 and attach the stage/query/scope to propagated errors.

- [ ] **Step 4: Run green and affected tests**

```bash
cd rust
cargo test -p memory-core
cargo test -p memory-index
cargo test -p memory-mcp
```

- [ ] **Step 5: Commit**

```bash
git add rust/crates/memory-core rust/crates/memory-index rust/crates/memory-mcp rust/.specs/03-retrieval.toml rust/.specs/05-lifecycle.toml rust/.specs/06-workspace.toml
git commit -m "fix(retrieval): complete live index contract"
```

### Task 2: Lifecycle Integrity and Decay Metrics

**Files:**
- Modify: `rust/Cargo.toml`
- Modify: `rust/crates/memory-core/Cargo.toml`
- Modify: `rust/crates/memory-core/src/access_log.rs`
- Modify: `rust/crates/memory-core/src/doctor.rs`
- Modify: `rust/crates/memory-mcp/src/search.rs`
- Modify: `rust/crates/memory-mcp/src/service.rs`
- Modify: `rust/.specs/05-lifecycle.toml`

**Interfaces:**
- Add a fallible access snapshot with per-via recent counts and `ever_accessed`.
- Access counters persist a `compacted_through` cutoff so interrupted compaction cannot double-count or lose events.
- Append and compaction serialize through a cross-process lock file using `fs2`.
- Ranking uses `read + 0.25 * search_hit`; why text reports `Nr+Ms/30d` instead of calling all events reads.

- [ ] **Step 1: Write failing lifecycle tests**

Add regression tests proving: malformed JSONL reports its path and line; `doctor --apply` refuses to archive when access state is corrupt; concurrent append/compact retains every event; simulated interruption after counters replacement is idempotently recovered by the cutoff; an old-but-ever-accessed note is not counted as `never_accessed_30d`; reads reinforce more than search hits; the why string distinguishes both via counts.

- [ ] **Step 2: Run red tests**

```bash
cd rust
cargo test -p memory-core access_log::tests:: doctor::tests:: -- --nocapture
cargo test -p memory-mcp search::tests:: service::tests:: -- --nocapture
```

Expected: corruption is currently ignored, compaction is unsynchronized, via counts are conflated, and decay stats mislabel old accesses.

- [ ] **Step 3: Implement the minimum safe lifecycle path**

Add `fs2 = "0.4"`; lock `.index/access.lock` around append/compact. Parse every nonblank JSONL line strictly. Write counters atomically before the compacted log and store the inclusive cutoff; readers ignore source events at or before that cutoff, making a crash between replacements safe and the next compaction idempotent. Make doctor obtain one fallible snapshot before decay decisions and stop `--apply` on error. Keep advisory read/search ranking best-effort, but lifecycle destruction fail-closed.

- [ ] **Step 4: Run green and affected tests**

```bash
cd rust
cargo test -p memory-core
cargo test -p memory-mcp
```

- [ ] **Step 5: Commit**

```bash
git add rust/Cargo.toml rust/Cargo.lock rust/crates/memory-core rust/crates/memory-mcp rust/.specs/05-lifecycle.toml
git commit -m "fix(lifecycle): make access history safe and truthful"
```

### Task 3: Search Budget and Explainable Provenance

**Files:**
- Modify: `rust/crates/memory-core/src/retriever.rs`
- Modify: `rust/crates/memory-index/src/fff.rs`
- Modify: `rust/crates/memory-mcp/src/search.rs`
- Modify: `rust/crates/memory-mcp/src/service.rs`
- Modify: `rust/.specs/03-retrieval.toml`

**Interfaces:**
- `ContentHit` carries matched-line provenance sufficient to distinguish title, alias, tag, and body.
- `apply_budget` serializes candidate `SearchHit` values with serde_json for byte accounting; an oversized first hit moves to `more`.
- `why` identifies the actually matched alias/token where available and accurately reports frecency components.

- [ ] **Step 1: Write failing shaping tests**

Add tests for a zero/default/max budget, an oversized first result, Unicode, long overflow, exact alias selection when the second alias matched, title/tag/body provenance, and logging only handles present in `results` rather than unshown ranked hits.

- [ ] **Step 2: Run red tests**

```bash
cd rust
cargo test -p memory-mcp search::tests:: service::tests:: -- --nocapture
```

Expected: the first oversized hit is admitted, why may name the wrong alias, and unshown hits are reinforced.

- [ ] **Step 3: Implement exact result budgeting and provenance**

Use `serde_json::to_vec(&candidate)?.len()` for each full result and never force-admit an oversized candidate. Keep `more` as the spec-defined bare handle/title overflow outside the snippet budget. Preserve the matching line plus query through merge so alias extraction selects the matching alias token; otherwise report the exact field family (`title`, `tags`, `content`, or path). Shape first, append search-hit events and call backend access tracking only for `results`.

- [ ] **Step 4: Run green tests**

```bash
cd rust
cargo test -p memory-core
cargo test -p memory-index
cargo test -p memory-mcp
```

- [ ] **Step 5: Commit**

```bash
git add rust/crates/memory-core rust/crates/memory-index rust/crates/memory-mcp rust/.specs/03-retrieval.toml
git commit -m "fix(search): enforce budget and truthful why output"
```

### Task 4: MCP Schema and Wire-Level Golden Tests

**Files:**
- Modify: `rust/crates/memory-mcp/src/server.rs`
- Create: `rust/crates/memory-mcp/tests/jsonrpc_golden.rs`
- Create: `rust/crates/memory-mcp/tests/golden/initialize.json`
- Create: `rust/crates/memory-mcp/tests/golden/tools-list.json`
- Modify: `rust/.specs/04-mcp-interface.toml`
- Modify: `rust/.specs/08-roadmap.toml`

**Interfaces:**
- Generated schema for `MemoryStoreArgs.aliases` has `minItems: 2`.
- A real rmcp in-memory transport proves initialize, exact five-tool listing, store, search, read, forget, and stats JSON-RPC shapes.

- [ ] **Step 1: Write failing schema and wire tests**

Inspect `MemoryServer::tool_router().list_all()` (or the public rmcp equivalent) and assert the exact five names, required arguments, no sixth tool, `aliases.minItems == 2`, and search budget maximum `16384`. Drive `MemoryServer` over `tokio::io::duplex`; compare normalized initialize and tools/list JSON values to checked-in golden files, then invoke all five tools through JSON-RPC.

- [ ] **Step 2: Run red tests**

```bash
cd rust
cargo test -p memory-mcp --test jsonrpc_golden -- --nocapture
```

Expected: the test file/API is absent and aliases lacks schema-level `minItems`.

- [ ] **Step 3: Implement the minimum schema fix**

Add `#[schemars(length(min = 2))]` to aliases and schema bounds for `limit`/`budget_bytes` where rmcp exposes them. Do not replace rmcp or hand-roll server protocol code; raw JSON exists only in the test client.

- [ ] **Step 4: Run green tests**

```bash
cd rust
cargo test -p memory-mcp
```

- [ ] **Step 5: Resolve OQ-3 and commit**

Record OQ-3 as resolved by the successful stdio/in-memory wire tests.

```bash
git add rust/crates/memory-mcp rust/.specs/04-mcp-interface.toml rust/.specs/08-roadmap.toml
git commit -m "test(mcp): prove five-tool JSON-RPC contract"
```

### Task 5: Complete Binary, Config Resolution, and Spawned E2E

**Files:**
- Modify: `rust/crates/fff-memory/Cargo.toml`
- Replace: `rust/crates/fff-memory/src/main.rs`
- Create: `rust/crates/fff-memory/src/config.rs`
- Create: `rust/crates/fff-memory/tests/cli.rs`
- Create: `rust/crates/fff-memory/tests/mcp_stdio.rs`
- Modify: `rust/crates/fff-memory/tests/cli_doctor.rs`
- Modify: `rust/.specs/02-write-path.toml`
- Modify: `rust/.specs/04-mcp-interface.toml`

**Interfaces:**
- The binary exposes exactly `serve|store|search|read|forget|stats|doctor|reindex`.
- Root precedence is `--root` > `FFF_MEMORY_ROOT` > config JSON > default.
- Config precedence inputs: `--config` > `FFF_MEMORY_CONFIG` > `${XDG_CONFIG_HOME:-$HOME/.config}/fff-memory/config.json`; schema is `{ "root": "/path" }`.
- Default root is `${XDG_DATA_HOME:-$HOME/.local/share}/fff-memory`; `--project [PATH]` selects `PATH/.memory` and conflicts with `--root`.
- CLI tool mirrors emit one JSON object to stdout; `serve` emits protocol only.

- [ ] **Step 1: Write failing CLI/config/E2E tests**

Spawn `CARGO_BIN_EXE_fff-memory` to test all eight help-visible commands, flag/env/config/XDG/project precedence, actionable invalid config, store success when retriever startup fails, and JSON outputs for store/search/read/forget/stats. Add a stdio test that sends `initialize`, `notifications/initialized`, `tools/list`, `memory_store`, polls `memory_search` until indexed, then `memory_read`, and terminates the child without leaving it running.

- [ ] **Step 2: Run red tests**

```bash
cd rust
cargo test -p fff-memory --test cli -- --nocapture
cargo test -p fff-memory --test mcp_stdio -- --nocapture
```

Expected: six commands and config resolution are absent; `serve` cannot start.

- [ ] **Step 3: Implement shared binary wiring**

Add direct dependencies on `memory-mcp`, `tokio`, `serde`, `serde_json`, and `time`. Use one async clap entry point and one root resolver. Create the root before opening fff. For `serve` and retrieval commands, try `FffRetriever::open`; log failure to stderr and continue with `None` so store stays available and search returns its explicit empty report. Reuse `MemoryService`; do not duplicate business logic in clap handlers.

- [ ] **Step 4: Run green tests**

```bash
cd rust
cargo test -p fff-memory
```

- [ ] **Step 5: Commit**

```bash
git add rust/crates/fff-memory rust/.specs/02-write-path.toml rust/.specs/04-mcp-interface.toml
git commit -m "feat(cli): expose the complete fff-memory binary"
```

### Task 6: Rust Plugin Automation and CI

**Files:**
- Modify: `.mcp.json`
- Modify: `.claude-plugin/plugin.json`
- Modify: `.claude-plugin/marketplace.json`
- Create: `hooks/hooks.json`
- Modify: `skills/memory-usage/SKILL.md`
- Modify: `commands/memory-status.md`
- Create: `commands/memory-doctor.md`
- Delete: `commands/memory-clear.md`
- Create: `.github/workflows/ci.yml`
- Create: `rust/crates/fff-memory/tests/plugin_package.rs`
- Modify: `rust/.specs/07-hooks-automation.toml`

**Interfaces:**
- `.mcp.json` starts `cargo run --quiet --manifest-path ${CLAUDE_PLUGIN_ROOT}/rust/Cargo.toml --bin fff-memory -- serve --project ${CLAUDE_PROJECT_DIR}` with no absolute path or embedding env.
- Hook server name is the official scoped `plugin:fff-memory:fff-memory`.
- Only `SessionStart`, `UserPromptSubmit`, and `Stop` hooks exist; no PreToolUse/PostToolUse.
- `UserPromptSubmit` uses one native `mcp_tool` call; `memory_search` performs a cheap empty response for prompts under 20 characters, slash commands, and common greetings to preserve the skip policy despite Claude hooks lacking non-tool conditional expressions.
- `Stop` uses one concise agent hook that either calls `memory_store` exactly once with substantive decisions/discoveries and 2–6 future-question aliases, or calls nothing.
- SessionStart uses one `mcp_tool` search and accepts the documented first-run non-blocking disconnected case.

- [ ] **Step 1: Write failing package tests**

Parse all JSON with serde_json and assert: no absolute user path, Go launcher, embedding env, `libsql`, or `vector`; exact server command/args; exact three hook events and one handler each; no Pre/PostToolUse; only the five valid tool names; current skill frontmatter and examples use title/body/aliases/type/tags/namespace; two current command files; CI runs fmt check, workspace tests, clippy with `-D warnings`, and workspace build.

- [ ] **Step 2: Run red test**

```bash
cd rust
cargo test -p fff-memory --test plugin_package -- --nocapture
```

Expected: stale absolute Go/vector wiring, obsolete skill schema/hooks, missing doctor command/hooks/CI, and invalid clear command are reported.

- [ ] **Step 3: Replace only the product surface**

Use official Claude plugin defaults: `.claude-plugin/plugin.json` contains metadata only; `skills/`, `commands/`, `hooks/`, and `.mcp.json` remain at plugin root. Keep hook prompts 1–3 lines and all logic in existing tools/skill. Do not touch `src/plugin/` or `bin/` pending OQ-5.

- [ ] **Step 4: Validate and run green tests**

```bash
claude plugin validate .
cd rust
cargo test -p fff-memory --test plugin_package
```

If the local Claude CLI lacks validation, record that exact external limitation; the Rust package test remains mandatory.

- [ ] **Step 5: Commit**

```bash
git add .mcp.json .claude-plugin hooks skills/memory-usage commands .github rust/crates/fff-memory/tests/plugin_package.rs rust/.specs/07-hooks-automation.toml
git commit -m "feat(plugin): ship Rust memory automation"
```

### Task 7: Mandated Properties, Dogfood Corpus, and Completion Audit

**Files:**
- Modify: `rust/Cargo.toml`
- Modify: `rust/crates/memory-core/Cargo.toml`
- Modify: `rust/crates/memory-core/src/slugify.rs`
- Modify: `rust/crates/memory-core/src/note.rs`
- Create: `rust/crates/fff-memory/tests/dogfood.rs`
- Modify: `rust/.specs/08-roadmap.toml`

**Interfaces:**
- Proptest covers slug invariants and Note markdown round-trip.
- A deterministic 20-note prose corpus covers title, alias, typo/fuzzy, namespace, archived exclusion/inclusion, frecency, type boost, empty explanation, and store-search-read behavior.
- The roadmap records implementation evidence separately from non-automatable elapsed-use acceptance.

- [ ] **Step 1: Write failing property and dogfood tests**

Add `proptest = "1"` as a dev dependency. Generate Unicode titles and valid notes; assert slug ASCII/kebab/max-60/idempotence and parse(serialize(note)) equality. Seed 20 distinct realistic markdown notes in a temp root and assert a table of future-question queries returns the intended handle first, including an alias-only query and one typo requiring fuzzy escalation.

- [ ] **Step 2: Run red tests**

```bash
cd rust
cargo test -p memory-core -- --nocapture
cargo test -p fff-memory --test dogfood -- --nocapture
```

Expected: property harness/corpus are absent; any ranking miss is treated as a real OQ-2 finding and fixed in the thin rerank layer, not by weakening expectations.

- [ ] **Step 3: Update roadmap evidence**

Resolve OQ-2 with the checked-in corpus evidence and OQ-3 with wire tests. Record OQ-4 as `NO for v1` unless the user explicitly selects migration; the nearly identical file format can be copied without product code. Leave OQ-5 explicitly awaiting the user because deleting committed legacy data is destructive. Mark implementation scope complete only where tests prove it; do not claim the Phase 1 subjective feel or Phase 2 real-week criterion without actual user use.

- [ ] **Step 4: Run full verification**

```bash
cd rust
cargo fmt --all -- --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo build --workspace
```

Then inspect `git diff --check`, `git status --short`, every roadmap scope bullet, all exit criteria, risks, and OQ statuses. Any missing evidence remains open.

- [ ] **Step 5: Commit**

```bash
git add rust/Cargo.toml rust/Cargo.lock rust/crates/memory-core rust/crates/fff-memory/tests/dogfood.rs rust/.specs/08-roadmap.toml
git commit -m "test: prove roadmap implementation on prose corpus"
```
