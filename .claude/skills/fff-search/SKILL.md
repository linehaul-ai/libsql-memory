---
name: fff-search
description: |
  Embed the fff-search Rust crate — a fast, typo-resistant file and content search
  engine (SIMD fuzzy matching, LMDB-backed frecency ranking, live grep, filesystem
  watching) — as an in-process library instead of shelling out to ripgrep/fzf/find.
  Use when adding fff-search as a Cargo dependency, building a FilePicker-based
  file/directory/mixed search, wiring up frecency or query-history ranking,
  implementing live grep (plain/regex/fuzzy) with GrepSearchOptions, parsing fff's
  query syntax with QueryParser/FFFQuery/Constraint, or setting up filesystem watch
  subscriptions via SharedFilePicker::watch.
---

## What fff-search is

`fff-search` is the Rust core of [FFF](https://github.com/dmtrKovalenko/fff), a file-search
toolkit built for long-running processes (editors, AI agents, IDE integrations) rather than
one-shot CLI invocations. Unlike `ripgrep`/`fzf`/`find`, which fork a new process and rebuild
all filesystem state on every call, `fff-search` keeps an index and file-content cache resident
in memory: `.gitignore` parsing, directory walking, and metadata collection happen once at scan
time, and every subsequent search hits warm memory. It also has a stricter fuzzy-matching
algorithm than `fzf` — typo-resistant, with a query language for prefiltering
(`"*.rs !test/ shcema"` finds `.rs` files, excludes `test/`, fuzzy-matches "schema" despite the
typo).

Full upstream docs: https://docs.rs/fff-search/latest/fff_search/

## Add the dependency

```toml
[dependencies]
fff-search = "0.10"
```

Feature flags (7 total; `ripgrep` is the only one on by default):

| Feature | Default | What it does |
| --- | --- | --- |
| `ripgrep` | ✓ | Directory walking via `ignore` + glob matching via `globset` (the same walker ripgrep uses). |
| `zlob` | | Swaps the walker and glob matcher for [zlob](https://github.com/dmtrKovalenko/zlob)'s native parallel implementation. Requires the Zig toolchain (`v0.16.0`) at build time. Fastest option — the upstream README calls this out as the recommended build for production use. |
| `mimalloc` | | Use mimalloc as the global allocator. |
| `mimalloc-collect` | | mimalloc with heap-collection stats. |
| `definitions` | | Enables definition-line classification (backs `GrepMatch::is_definition` / `GrepSearchOptions::classify_definitions`). |
| `ffi` | | Enables the C-ABI surface (only relevant if you're also building `fff-c`; not needed for pure-Rust use). |
| `rescan-stats` | | Enables `rescan_stats` module — admitted-vs-throttled rescan accounting, exposed via `SharedFilePicker::rescan_stats()`. |

`ripgrep` and `zlob` are alternatives, not additive — pick one walker/glob backend per build.

## Quick Start

This is the canonical minimal working example (from the crate's own doctest) — the shape every
integration follows: create shared-state handles, open the persistence DBs, spin up the picker,
wait for the initial scan, then search.

```rust
use fff_search::file_picker::FilePicker;
use fff_search::frecency::FrecencyTracker;
use fff_search::query_tracker::QueryTracker;
use fff_search::{
    FFFMode, FilePickerOptions, FuzzySearchOptions, PaginationArgs, QueryParser,
    SharedFrecency, SharedFilePicker, SharedQueryTracker,
};

let shared_picker = SharedFilePicker::default();
let shared_frecency = SharedFrecency::default();
let shared_query_tracker = SharedQueryTracker::default();

// 1. Optionally initialize frecency and query tracker databases (LMDB-backed).
//    Do this BEFORE new_with_shared_state so frecency is available for post-scan warmup.
let frecency = FrecencyTracker::open(tmp.join("frecency"))?;
shared_frecency.init(frecency)?;

let query_tracker = QueryTracker::open(tmp.join("queries"))?;
shared_query_tracker.init(query_tracker)?;

// 2. Init the file picker (spawns background scan + watcher). Returns () — the picker
//    is reachable only through shared_picker afterwards.
FilePicker::new_with_shared_state(
    shared_picker.clone(),
    shared_frecency.clone(),
    FilePickerOptions {
        base_path: ".".into(),
        mode: FFFMode::Ai,
        ..Default::default()
    },
)?;

// 3. Wait for the initial scan — searching before this returns true hits a partial index.
shared_picker.wait_for_scan(std::time::Duration::from_secs(10));

// 4. Lock the picker and query tracker for the duration of the search only.
let picker_guard = shared_picker.read()?;
let picker = picker_guard.as_ref().unwrap();
let qt_guard = shared_query_tracker.read()?;

// 5. Parse the query, then search.
let parser = QueryParser::default(); // QueryParser<FileSearchConfig>
let query = parser.parse("lib.rs");

let results = picker.fuzzy_search(
    &query,
    qt_guard.as_ref(),
    FuzzySearchOptions {
        max_threads: 0,
        current_file: None,
        pagination: PaginationArgs { offset: 0, limit: 50 },
        ..Default::default()
    },
);
```

Two things that trip people up immediately — both covered in depth in
[references/file-picker.md](references/file-picker.md):

- `shared_picker.read()?.as_ref()` is `None` until `new_with_shared_state` finishes *and*
  `wait_for_scan` (or `wait_for_indexing_complete`) has returned. Bind the guard to a local
  before dereferencing — `shared.read()?.as_ref().unwrap()` inline does not compile.
- Hold read guards only for the duration of a search. Background scan/rescan threads need the
  write lock; a long-held read guard stalls indexing.

## Architecture

- **`file_picker`** — core engine: filesystem indexing, background watching, fuzzy search. Entry
  point is `FilePicker`; see [references/file-picker.md](references/file-picker.md).
- **`shared`** — thread-safe `Arc<RwLock<Option<T>>>` wrappers (`SharedFilePicker`,
  `SharedFrecency`, `SharedQueryTracker`) so the picker and DBs can be handed to background
  threads before they're initialized. Also in
  [references/file-picker.md](references/file-picker.md).
- **`dbs`** — LMDB-backed persistence: `FrecencyTracker` (access/modification ranking) and
  `QueryTracker` (query history + "combo-boost" scoring). See
  [references/dbs-and-watch.md](references/dbs-and-watch.md).
- **`grep`** — live content search: plain/regex/fuzzy modes, prefiltering, cursor pagination. See
  [references/grep.md](references/grep.md).
- **`watch`** — filesystem watch subscriptions (glob / exact path / directory subtree) with
  batched delivery. Also in [references/dbs-and-watch.md](references/dbs-and-watch.md).
- **`types`** — result and item types returned by every search method (`FileItem`, `Score`,
  `SearchResult`, …). See [references/result-types.md](references/result-types.md).
- **Query parsing** (from the `fff-query-parser` crate, re-exported at the `fff_search` root) —
  `QueryParser`, `FFFQuery`, `Constraint`, and the five `ParserConfig` implementations that give
  each search mode (file/dir/mixed/grep/AI-grep) different query syntax. See
  [references/query-parsing-and-configs.md](references/query-parsing-and-configs.md).
- **`git`** — git status caching/formatting, consumed by frecency scoring and `FileItem`.
- **`glob_detect`**, **`path_utils`**, **`location`**, **`log`**, **`constants`**,
  **`rescan_stats`** — supporting utilities; not covered in depth here, see docs.rs directly if
  needed.

## Reference index

| File | Read when you're... |
| --- | --- |
| [references/file-picker.md](references/file-picker.md) | Constructing a `FilePicker`, choosing `FilePickerOptions`/`FFFMode`, calling any search/grep/index-inspection method, working with `SharedFilePicker`/`SharedFrecency`/`SharedQueryTracker`, or handling `Error`. |
| [references/query-parsing-and-configs.md](references/query-parsing-and-configs.md) | Parsing query strings, choosing/writing a `ParserConfig`, or working with `Constraint`/`FuzzyQuery`/`Location` directly. |
| [references/grep.md](references/grep.md) | Implementing content search — `GrepSearchOptions`, `GrepMode`, reading `GrepMatch`/`GrepResult`. |
| [references/result-types.md](references/result-types.md) | Consuming search results — `FileItem`/`DirItem` accessors, `Score` breakdown, `SearchResult`/`DirSearchResult`/`MixedSearchResult`, pagination, `ContentCacheBudget`. |
| [references/dbs-and-watch.md](references/dbs-and-watch.md) | Opening/using `FrecencyTracker` or `QueryTracker`, or subscribing to filesystem changes via `SharedFilePicker::watch`. |

## Cross-cutting gotchas

- **Lock discipline.** `SharedFilePicker` guards are `parking_lot` locks (no poisoning);
  `SharedDb` (frecency/query-tracker) guards are `std::sync` locks. Never call a
  `SharedFilePicker` method that takes the write lock while already holding a read guard from
  the same handle — background rescan/watch threads need that write lock to make progress.
- **Init order matters.** Open and `.init()` `SharedFrecency` (and `SharedQueryTracker`, if used)
  *before* calling `FilePicker::new_with_shared_state` — the post-scan frecency warmup and
  query-tracker lookups both assume the DBs are already installed.
- **`QueryParser::default()` only means file-search semantics.** `Default` is implemented solely
  for `QueryParser<FileSearchConfig>`. For grep/dir/mixed/AI-grep queries, use
  `QueryParser::new(GrepConfig)` (etc.) or the `FFFQuery::parse(query, config)` shortcut.
  Full picture in [references/query-parsing-and-configs.md](references/query-parsing-and-configs.md).
- **`enable_fs_root_scanning` / `enable_home_dir_scanning` default off.** Indexing `/` or
  `$HOME` is expensive and almost always accidental — `FilePickerOptions` requires opting in
  explicitly, otherwise a root-resolving `base_path` returns `Error::FilesystemRoot`.
- **Items don't own their paths.** `FileItem`/`DirItem` store offsets into a chunk arena owned by
  the `FilePicker`; every path-producing method takes an `arena: impl FFFStringStorage` argument
  (usually `&FilePicker` itself). Details in
  [references/result-types.md](references/result-types.md).
