# Legacy Cleanup Plan — Removing the Pre-Refactor Go/LibSQL Tree

**Date:** 2026-08-15
**Branch:** `refactor/rust`
**Status:** PROPOSED — requires explicit user authorization (spec 08, OQ-5)

This plan removes the abandoned Go + LibSQL + embeddings implementation from the repository.
It is written to be executed in reviewable stages, each independently revertable, with the
riskiest change last.

---

## Summary

| Metric | Value |
|---|---|
| Tracked files today | 127 |
| Files deleted | 38 (30% of file count) |
| Bytes deleted | ~107.8 MB (~96% of tracked bytes) |
| Directories emptied | `src/`, `bin/`, `diagrams/`, `docs/plans/` |
| KEEP files needing edits | 4 (+1 new file) |
| Files that break on deletion | **0** |

The dominant number is 101.7 MB of committed Mach-O binaries — four near-identical copies of
the same Go executable plus one MCP server binary. The project's stated convention is
"No committed binaries, ever."

---

## Pre-flight checks

Run these before touching anything. Each must pass.

```bash
cd /Users/fakebizprez/Developer/projects/libsql-memory

# 1. Working tree is clean — nothing uncommitted to lose.
git status --porcelain          # expect: empty

# 2. The recovery point exists and is an ancestor of this branch.
git merge-base --is-ancestor main HEAD && echo "main is recoverable ancestor"

# 3. main really does hold the legacy tree (expect 33).
git ls-tree main -r --name-only | grep -cE '^(src/plugin|bin/|\.claude/memory\.db|docs/plans)'

# 4. Baseline: the test suite passes BEFORE deletion, so a later failure is attributable.
cd rust && cargo test --workspace && cargo clippy --workspace --all-targets -- -D warnings
```

---

## Recoverability

Every deletion below is recoverable. No history rewriting is involved: a plain `git rm` commit
leaves all blobs intact in the object database and on `origin`.

- **36 of 38 files** live on `main` (commit `71fd206`, also on `origin`), which is a verified
  ancestor of `refactor/rust`. Recover any of them with:
  ```bash
  git checkout main -- <path>      # restore
  git show main:<path>             # inspect without restoring
  ```
- **2 exceptions are NOT on `main`** and are recoverable only from `refactor/rust` history:
  - `diagrams/` (4 files) — added in commit `7599f0e` during the refactor.
  - `.DS_Store`
  Recover with `git show 7599f0e:diagrams/<file>`. These survive as long as
  `origin/refactor/rust` is not force-rewritten.

**Corollary worth stating plainly:** deleting the binaries does **not** shrink the clone. The
101.7 MB remains in the pack files. Reclaiming that disk space requires rewriting history
(`git filter-repo` / BFG) and force-pushing — a separate, far more invasive decision that this
plan deliberately does not take. See "Deferred decisions" below.

---

## Stage 1 — Committed binaries (101.7 MB, 5 files)

All five verified as `Mach-O 64-bit executable arm64`.

| Path | Bytes |
|---|---:|
| `bin/libsql-memory-darwin-arm64` | 25,864,498 |
| `src/plugin/build/libsql-memory` | 25,864,498 |
| `src/plugin/libsql-memory` | 25,863,762 |
| `src/plugin/bin/libsql-memory-darwin-arm64` | 25,863,634 |
| `src/plugin/mcp-server` | 3,168,210 |

```bash
git rm bin/libsql-memory-darwin-arm64 \
       src/plugin/build/libsql-memory \
       src/plugin/libsql-memory \
       src/plugin/bin/libsql-memory-darwin-arm64 \
       src/plugin/mcp-server
git commit -m "chore: remove committed Go binaries"
```

---

## Stage 2 — Legacy Go source and build config (17 files, 239 KB)

```bash
git rm src/plugin/go.mod src/plugin/go.sum src/plugin/Makefile \
       src/plugin/cmd/main.go \
       src/plugin/cmd/mcp-server/main.go \
       src/plugin/internal/config/config.go \
       src/plugin/internal/config/config_test.go \
       src/plugin/internal/db/libsql.go \
       src/plugin/internal/db/libsql_test.go \
       src/plugin/internal/db/integration_test.go \
       src/plugin/internal/embedding/embedder.go \
       src/plugin/internal/mcp/server.go \
       src/plugin/internal/mcp/server_test.go \
       src/plugin/internal/memory/store.go \
       src/plugin/internal/memory/store_integration_test.go \
       src/plugin/internal/testutil/containers.go \
       src/plugin/pkg/types/types.go
git commit -m "chore: remove abandoned Go implementation"
```

Notes:
- `go.mod` declares module `github.com/libsql-memory/plugin`, depending on
  `tursodatabase/go-libsql` and `testcontainers-go`.
- `internal/embedding/embedder.go` (39 KB) is the Nomic embedding path.
- `internal/testutil/containers.go` is a Docker/testcontainers harness, which directly
  contradicts the current rule that no test may require network or Docker.

---

## Stage 3 — Legacy plugin wiring, databases, and stale state (11 files, 138 KB)

**Plugin launcher** — `bin/run.sh` (504 B) hardcodes the Go binary path, `--embedding-provider
nomic`, and a LAN endpoint `http://192.168.128.10:1234/v1/embeddings`.

**Databases** — both carry the abandoned schema (a `memories` table with an `embedding BLOB`
column) and **both contain zero rows**, which is direct confirmation of the post-mortem
recorded in `rust/.specs/00-overview.toml`.

**Stale state** — `.DS_Store` is Finder metadata; the `src/plugin/.claude-*` files are frozen
at 2026-01-16 inside the dead tree.

```bash
git rm bin/run.sh \
       .claude/memory.db .claude/memory.db-shm .claude/memory.db-wal \
       src/plugin/memory.db src/plugin/memory.db-shm src/plugin/memory.db-wal \
       .DS_Store \
       src/plugin/.claude-session src/plugin/.claude-status \
       src/plugin/.claude/settings.json
git commit -m "chore: remove legacy launcher, databases, and stale state"
```

After this stage `src/` and `bin/` are empty and disappear from git.

---

## Stage 4 — Legacy documentation and diagrams (5 files, 832 KB)

- `docs/plans/2026-01-16-replace-nomic-model-name.md` — a plan to edit
  `internal/embedding/embedder.go`; every file it references is now gone.
- `diagrams/libsql-memory-agent-flow.{spec.json,excalidraw,png,gif}` — depict the abandoned
  architecture explicitly: "768-d Nomic vector", "Per-Project libSQL",
  "(.claude/memory.db in the agent working directory)". The GIF alone is 8.3 MB.

```bash
git rm docs/plans/2026-01-16-replace-nomic-model-name.md
git rm -r diagrams/
git commit -m "chore: remove documentation describing the abandoned architecture"
```

⚠️ **Confirm before running:** `diagrams/` is the one deletion not recoverable from `main`.
See "Open decisions" — you may prefer to redraw for the fff-memory architecture first.

---

## Stage 5 — Edits to files that stay

These four files (plus one new file) contain statements that become false once Stages 1–4 land.
This stage is what keeps the repo honest.

### 5a. `CLAUDE.md` — rewrite the "Legacy Code" section (lines 88–93)

Currently reads:

> `src/plugin/` (Go), `bin/`, and the old plugin wiring are the abandoned predecessor. Do not
> extend or fix them; they exist only for reference until removed (spec 08, OQ-5). The
> predecessor's post-mortem is encoded in `00-overview.toml` `[lessons]` — read it before
> relaxing any invariant above.

Replace with just the durable half — the post-mortem pointer keeps its value, the file
references do not:

> The Go + LibSQL + embeddings predecessor was removed on 2026-08-15; it remains recoverable
> from `main` history. Its post-mortem is encoded in `00-overview.toml` `[lessons]` — read it
> before relaxing any invariant above.

### 5b. `rust/.specs/08-roadmap.toml` — resolve OQ-5 (lines 108–111)

```toml
status = "AWAITING USER DECISION"
answer = "Keep the legacy Go tree and committed binaries untouched until the user explicitly authorizes their deletion after real Phase 1 dogfooding."
```

Becomes `RESOLVED 2026-08-15` with an answer recording that the tree was deleted, the commit
range, and that `main` is the recovery point.

### 5c. `docs/superpowers/plans/2026-08-09-roadmap-completion.md` line 21

> - Preserve the abandoned Go tree until OQ-5 receives an explicit destructive user decision; never extend or repair it.

Mark as superseded rather than editing the historical plan body.

### 5d. `.claude/settings.json` — disable Go tooling (lines 10–21)

Three plugins are enabled for a language the repo no longer contains:

- `gopls-lsp@claude-plugins-official`
- `jutsu-go@linehaulai-claude-marketplace`
- `testcontainers-go@testcontainers-claude-skills`

Set each to `false`.

### 5e. **NEW FILE** — create a root `.gitignore`

There is currently **no `.gitignore` at the repo root**; the only one is `rust/.gitignore`,
which protects `rust/` alone. This absence is the root cause of the committed `.DS_Store` and
`memory.db` files. Without this step, the same junk returns.

```gitignore
.DS_Store
*.db
*.db-wal
*.db-shm
.index/
.claude/settings.local.json
```

```bash
git add CLAUDE.md rust/.specs/08-roadmap.toml \
        docs/superpowers/plans/2026-08-09-roadmap-completion.md \
        .claude/settings.json .gitignore
git commit -m "docs: retire legacy references and add root gitignore"
```

---

## Verification

Run after every stage; treat any failure as a stop-and-revert signal.

```bash
cd rust
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all --check
```

`cargo test --workspace` is the meaningful gate here, because
`rust/crates/fff-memory/tests/plugin_package.rs` walks up to the repo root and reads the live
plugin files. If cleanup damaged the root wiring, that test fails.

Then confirm nothing dangles:

```bash
cd ..
grep -rn --exclude-dir=.git -E 'src/plugin|bin/run\.sh|libsql|nomic|embedding' \
  --include='*.md' --include='*.json' --include='*.toml' --include='*.yml' . \
  | grep -v '^\./rust/crates/fff-memory/tests/plugin_package.rs' \
  | grep -v '^\./rust/\.specs/00-overview\.toml'
```

Expected survivors are only: the negative test assertions, the `00-overview.toml` post-mortem,
and README history prose. Anything else is a stale reference to fix.

---

## Why nothing breaks

The reference audit found **zero KEEP files depending on any DELETE path**. The only mentions
are negative assertions and historical prose:

| Referencing file:line | Names | Effect of deletion |
|---|---|---|
| `rust/crates/fff-memory/tests/plugin_package.rs:38-44` | `bin/run.sh`, `src/plugin`, `embedding`, `libsql`, `vector`, `memory_list`, `memory_delete` | None — these assert the plugin files must **not** contain these strings. Deletion strengthens them. |
| `CLAUDE.md:91` | `src/plugin/`, `bin/` | Prose. Handled in Stage 5a. |
| `docs/superpowers/plans/…:21` | the Go tree | Prose. Handled in Stage 5c. |
| `rust/.specs/08-roadmap.toml:109-111` | OQ-5 | Prose. Handled in Stage 5b. |

Verified clean: `.github/workflows/ci.yml` has one job (`rust`, `working-directory: rust`, four
cargo steps) with no Go, no `bin/`, no `src/`, no Makefile. `.mcp.json`, `hooks/hooks.json`,
`.claude-plugin/*.json`, `commands/*.md`, and `skills/memory-usage/SKILL.md` contain zero
legacy references. `rust/Cargo.toml` has no external path dependencies.

---

## What explicitly stays

Not everything outside `rust/` is legacy. The current plugin wiring lives at the repo root and
is byte-asserted by a passing test:

| Path | Role |
|---|---|
| `.mcp.json` | Launches `cargo run --manifest-path ${CLAUDE_PLUGIN_ROOT}/rust/Cargo.toml --bin fff-memory -- serve` |
| `.claude-plugin/plugin.json`, `marketplace.json` | fff-memory manifest and marketplace entry |
| `hooks/hooks.json` | Three thin hooks calling `memory_search` / `memory_store` |
| `skills/memory-usage/SKILL.md` | Teaches the five-tool contract and alias discipline |
| `commands/memory-status.md`, `commands/memory-doctor.md` | Both invoke fff-memory |
| `.github/workflows/ci.yml` | Pure Rust CI |
| `README.md`, `LICENSE`, `CLAUDE.md`, `AGENTS.md` (symlink) | Repo infrastructure |
| `.claude/skills/fff-search/**`, `.codex/skills/fff-search/**` | Current search-engine reference |
| `.agents/skills/lanshu-…/**` + `skills-lock.json` | Vendored third-party diagram skill |

**Portability caveat:** `rust/` is not independently portable. `plugin_package.rs` reads eight
files from the repo root. All are in the KEEP set, so this cleanup is safe — but do not assume
the `rust/` directory can be lifted out on its own.

---

## Open decisions — need your answer before Stage 4 or 5

1. **`diagrams/` (4 files, 8.4 MB)** — content is unambiguously the old architecture, but it
   was authored *during* the Rust refactor and is the one deletion **not recoverable from
   `main`**. Delete outright, or redraw for fff-memory first?

2. **Root `.claude-session` / `.claude-status`** — live session-state files, tracked in git and
   rewritten every session. Churn, not legacy. Untrack and gitignore them?

3. **`.codex/skills/fff-search/**`** — byte-for-byte identical to `.claude/skills/fff-search/**`
   (verified with `diff -rq`). Replace with a symlink, the way `.claude/skills/lanshu-…`
   already points into `.agents/`?

4. **`docs/` after Stage 4** — only the completed 414-line roadmap plan remains. Keep as an
   execution record, or archive?

5. **`.agents/skills/lanshu-animated-architecture-diagram/**` (13 files, 15 MB)** — a vendored
   diagram-rendering skill including two GIF previews totaling 15.3 MB. Unrelated to Go, so
   KEEP by default. But if `diagrams/` goes away, is the skill still wanted?

---

## Deferred decisions (explicitly out of scope)

- **History rewrite to reclaim 101.7 MB.** Deleting the binaries does not shrink the clone.
  Reclaiming that space needs `git filter-repo` or BFG plus a force-push, which rewrites every
  SHA on the branch and breaks any existing clone or open PR. Decide separately, and only after
  `refactor/rust` has merged.
- **`rust/target/` is 10 GB on disk** — correctly gitignored, irrelevant to git, removable at
  any time with `cargo clean`.

---

## Execution order

```
Pre-flight  →  Stage 1 (binaries)      → verify
            →  Stage 2 (Go source)     → verify
            →  Stage 3 (wiring/data)   → verify
            →  [decide on diagrams]
            →  Stage 4 (docs/diagrams) → verify
            →  Stage 5 (edits + gitignore) → verify → push
```

Five separate commits, each revertable with `git revert`. Stage 5 comes last on purpose: the
documentation should not claim the tree is gone until it actually is.
