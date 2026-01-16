# Replace Nomic Embedding Model Name Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace all mentions of the Nomic embedding model from its old name to `text-embedding-qwen3-embedding-8b`

**Architecture:** This is a straightforward string replacement task across multiple files. The model name "nomic-embed-text-v1.5" appears in comments, and the default model constant `DefaultNomicModel` needs to be updated from whatever it currently is to `text-embedding-qwen3-embedding-8b`. No functional changes to the code logic are required - only documentation and constant values.

**Tech Stack:** Go codebase with documentation files (Markdown)

---

## Context

The Nomic embedding model reference appears in several locations:
1. **Code files**: `src/plugin/internal/embedding/embedder.go` - constants and comments
2. **Config files**: `src/plugin/internal/config/config.go` - comments
3. **Documentation**: `README.md` and `CLAUDE.md` - user-facing documentation
4. **Main entry point**: `src/plugin/cmd/main.go` - initialization logic

The current model reference in line 42 of embedder.go shows: `DefaultNomicDimension  = 768 // nomic-embed-text-v1.5 outputs 768 dimensions`

The current model constant in line 44 shows: `DefaultNomicModel = "text-embedding-qwen3-embedding-8b"` (already correct!)

So the task is primarily to:
1. Update the comment on line 42 that references "nomic-embed-text-v1.5"
2. Verify all other references use the correct model name
3. Update documentation to reflect the correct model name

---

## Task 1: Update embedder.go Comment

**Files:**
- Modify: `src/plugin/internal/embedding/embedder.go:42`

**Step 1: Read the current embedder.go file**

Already read above. Current line 42:
```go
DefaultNomicDimension  = 768 // nomic-embed-text-v1.5 outputs 768 dimensions
```

**Step 2: Update the comment to reference the correct model name**

```go
DefaultNomicDimension  = 768 // text-embedding-qwen3-embedding-8b outputs 768 dimensions
```

**Step 3: Verify the DefaultNomicModel constant is correct**

Line 44 is already correct:
```go
DefaultNomicModel      = "text-embedding-qwen3-embedding-8b"
```

No change needed.

**Step 4: Run tests to ensure no breakage**

Run: `cd src/plugin && make test`
Expected: All tests pass

**Step 5: Commit the change**

```bash
git add src/plugin/internal/embedding/embedder.go
git commit -m "docs: update Nomic model name in comment to text-embedding-qwen3-embedding-8b

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Update Package Documentation Comment

**Files:**
- Modify: `src/plugin/internal/embedding/embedder.go:1-4`

**Step 1: Read the current package comment**

Current lines 1-4:
```go
// Package embedding provides vector embedding generation and similarity search
// for the Claude memory plugin. It supports multiple embedding backends including
// OpenAI's text-embedding-3-small and a local TF-IDF/hash-based fallback.
package embedding
```

**Step 2: Update to mention the Nomic model**

```go
// Package embedding provides vector embedding generation and similarity search
// for the Claude memory plugin. It supports multiple embedding backends including
// Nomic's text-embedding-qwen3-embedding-8b, OpenAI's text-embedding-3-small,
// and a local TF-IDF/hash-based fallback.
package embedding
```

**Step 3: Run tests**

Run: `cd src/plugin && make test`
Expected: All tests pass

**Step 4: Commit**

```bash
git add src/plugin/internal/embedding/embedder.go
git commit -m "docs: add Nomic model to package documentation

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Update CLAUDE.md Documentation

**Files:**
- Modify: `CLAUDE.md:54`

**Step 1: Read current content**

Already read. Line 54:
```
- Nomic: Local model, 768 dimensions, default endpoint `http://192.168.128.10:1234/v1/embeddings`
```

**Step 2: Update to include model name**

```
- Nomic: text-embedding-qwen3-embedding-8b model, 768 dimensions, default endpoint `http://192.168.128.10:1234/v1/embeddings`
```

**Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: specify Nomic model name in CLAUDE.md

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Update README.md Documentation

**Files:**
- Modify: `README.md:23`

**Step 1: Read current content**

Already read. Line 23:
```bash
**Nomic embedding server** running at http://192.168.128.10:1234 (default)
```

**Step 2: Update to include model name**

```bash
**Nomic embedding server** (text-embedding-qwen3-embedding-8b) running at http://192.168.128.10:1234 (default)
```

**Step 3: Commit**

```bash
git add README.md
git commit -m "docs: specify Nomic model name in README

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Search for Any Other References

**Files:**
- Search across entire codebase

**Step 1: Search for old model reference**

Run: `grep -r "nomic-embed-text" . --exclude-dir=.git --exclude-dir=bin`
Expected: Should find any remaining references to the old model name

**Step 2: Search for generic "nomic" references**

Run: `grep -ri "nomic" . --exclude-dir=.git --exclude-dir=bin --exclude-dir=docs | grep -v "text-embedding-qwen3-embedding-8b"`
Expected: Should find any documentation or comments that mention Nomic without the model name

**Step 3: Update any additional findings**

If additional references are found, update them to use "text-embedding-qwen3-embedding-8b"

**Step 4: Commit any additional changes**

```bash
git add [files]
git commit -m "docs: update remaining Nomic model references

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Final Verification

**Files:**
- All modified files

**Step 1: Run all checks**

Run: `cd src/plugin && make verify`
Expected: All checks pass (fmt, vet, lint, test)

**Step 2: Review git diff**

Run: `git diff main`
Expected: Only documentation and comment changes, no functional code changes

**Step 3: Final commit if needed**

If make verify made any formatting changes:
```bash
git add -u
git commit -m "style: apply formatting from make verify

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Summary

This plan replaces all references to the old Nomic model name with `text-embedding-qwen3-embedding-8b`. The changes are purely documentation-related:

1. Update comment in embedder.go about model dimensions
2. Update package documentation to mention Nomic model
3. Update CLAUDE.md to specify model name
4. Update README.md to specify model name
5. Search and update any other references
6. Run full verification suite

No functional code changes are required since the `DefaultNomicModel` constant already has the correct value.
