# FilePicker & Shared State

The index lives in a long-lived `FilePicker`. Normal usage: create it *into* a `SharedFilePicker`, let background threads scan and watch, then take short read guards to search.

## `FFFMode`

```rust
pub enum FFFMode { Neovim, Ai }
impl FFFMode { pub fn is_ai(self) -> bool; }
```

Selects both the scoring profile and how filesystem watcher events are handled.

- `Neovim` — editor-picker behavior; tuned for a human typing into a picker UI with buffer/current-file context.
- `Ai` — agent behavior; what you want when an LLM or tool is issuing the queries.

`Copy + Eq`. A `Default` impl exists but rustdoc does not show which variant it picks — set `mode` explicitly rather than relying on `..Default::default()` for it.

## `FilePickerOptions`

```rust
pub struct FilePickerOptions {
    pub base_path: String,                       // root of the indexed tree
    pub enable_mmap_cache: bool,                 // pre-populate mmap caches for top-frecency files after initial scan
    pub enable_content_indexing: bool,           // build content index after initial scan (faster content-aware filtering)
    pub mode: FFFMode,                           // scoring + watcher-event semantics
    pub cache_budget: Option<ContentCacheBudget>,// None => auto-computed from repo size after initial scan
    pub watch: bool,                             // false => no background watcher is created
    pub follow_symlinks: bool,
    pub enable_fs_root_scanning: bool,           // allow indexing `/`
    pub enable_home_dir_scanning: bool,          // allow indexing $HOME
}
```

`base_path` is a `String`, not a `PathBuf` — `".".into()` in the Quick Start. `FilePicker::base_path` (the resolved field/getter) is a `PathBuf`/`&Path`.

`Default` is hand-written. Documented defaults: `enable_fs_root_scanning` and `enable_home_dir_scanning` are **off** — indexing `/` or `$HOME` is an expensive, usually-accidental operation, so it must be opted into. rustdoc does not publish the remaining default values; the idiomatic call is `FilePickerOptions { base_path, mode, ..Default::default() }`.

Two hard guards derived from these flags: a `base_path` that resolves to the filesystem root yields `Error::FilesystemRoot(PathBuf)` unless `enable_fs_root_scanning` is set; an unusable path yields `Error::InvalidPath(PathBuf)`.

## `FilePicker`

```rust
pub struct FilePicker {
    pub mode: FFFMode,
    pub base_path: PathBuf,
    /* private */
}
```

Owns the file/dir tables, string arenas, bigram index, content caches, cancellation flag, and the background watcher handle. `Drop` tears down background threads. Not `Clone`.

### Construction

```rust
pub fn new(options: FilePickerOptions) -> Result<Self, Error>
```
Builds an **empty, inert** picker: no background watcher is spawned and the file tree is *not* walked. Follow with `collect_files()` or you will search an empty index. Use only for direct/synchronous ownership.

```rust
pub fn new_with_shared_state(
    shared_picker: SharedFilePicker,
    shared_frecency: SharedFrecency,
    options: FilePickerOptions,
) -> Result<(), Error>
```
The default entry point. Constructs the picker, installs it into `shared_picker`, and spawns background indexing plus the filesystem watcher. Returns `()` — the picker is reachable only through the shared handle afterwards. Initialize `shared_frecency` (via `SharedFrecency::init`) *before* this call so frecency is available to the post-scan warmup; use `SharedFrecency::noop()` if you don't want persistence.

```rust
pub fn collect_files(&mut self) -> Result<(), Error>
```
Synchronous filesystem walk that populates `self`. The `new` + `collect_files` path; blocks the caller for the whole scan.

### Search

```rust
pub fn fuzzy_search<'q>(
    &self,
    query: &'q FFFQuery<'q>,
    query_tracker: Option<&QueryTracker>,
    options: FuzzySearchOptions<'q>,
) -> SearchResult<'_>
```
Fuzzy file search over a **pre-parsed** query (`QueryParser::parse`). When a `QueryTracker` is passed, the search looks up the last file selected for this exact query and boosts it. `SearchResult` borrows the picker — the read guard must outlive it.

```rust
pub fn fuzzy_search_directories<'q>(
    &self,
    query: &'q FFFQuery<'q>,
    options: FuzzySearchOptions<'q>,
) -> DirSearchResult<'_>
```
Directories only, ranked by fuzzy match quality plus frecency. No query-tracker parameter.

```rust
pub fn fuzzy_search_mixed<'q>(
    &self,
    query: &'q FFFQuery<'q>,
    query_tracker: Option<&QueryTracker>,
    options: FuzzySearchOptions<'q>,
) -> MixedSearchResult<'_>
```
Files and directories interleaved into one flat list by total score, descending. Gotcha: if the raw query ends in `/`, files are skipped entirely and only directories are searched — and for that to work the caller must parse with `DirSearchConfig`, which keeps a trailing `/` as fuzzy text instead of turning it into a `PathSegment` constraint.

```rust
pub fn glob<'p>(&'p self, pattern: &'p str, options: FuzzySearchOptions<'p>) -> SearchResult<'p>
```
Literal glob filter (`*.rs`, `**/*.test.ts`), ranked by frecency and paginated. Bypasses the query parser entirely — no fuzzy matching, no multi-token constraint parsing. Ranking matches `fuzzy_search` with an empty fuzzy query. Invalid patterns surface as `Error::InvalidGlobPattern { pattern, reason }`.

```rust
pub fn grep(&self, query: &FFFQuery<'_>, options: &GrepSearchOptions) -> GrepResult<'_>
pub fn multi_grep(
    &self,
    patterns: &[&str],
    constraints: &[Constraint<'_>],
    options: &GrepSearchOptions,
) -> GrepResult<'_>
```
Live content grep across indexed files. `options.abort_signal`, when set, **overrides** the picker's internal cancellation flag, so the caller fully controls when the grep stops.

### Index inspection

```rust
pub fn base_path(&self) -> &Path
pub fn mode(&self) -> FFFMode
pub fn git_root(&self) -> Option<&Path>          // None for non-git bases
pub fn has_git_repo(&self) -> bool
pub fn get_files(&self) -> &[FileItem]           // sorted by PATH, not by score
pub fn get_overflow_files(&self) -> &[FileItem]  // paths added after the last full scan
pub fn get_dirs(&self) -> &[DirItem]             // sorted by path
pub fn live_file_count(&self) -> usize           // O(1), excludes tombstones
pub fn get_file_by_path(&self, path: impl AsRef<Path>) -> Option<&FileItem>
pub fn get_mut_file_by_path(&mut self, path: impl AsRef<Path>) -> Option<(ArenaPtr, &mut FileItem)>
pub fn get_file_mut(&mut self, index: usize) -> Option<(ArenaPtr, &mut FileItem)>
pub fn arena_bytes(&self) -> (usize, usize, usize) // (chunked_path_store, 0, 0); leaked overflow stores untracked
```
`get_files()` is path-ordered because the table is kept sorted for cheap insert/remove; anything frecency- or score-ordered comes from the search methods.

### Caches, capabilities, tracing

```rust
pub fn has_mmap_cache(&self) -> bool
pub fn has_content_indexing(&self) -> bool
pub fn has_watcher(&self) -> bool
pub fn is_watcher_ready(&self) -> bool
pub fn follows_symlinks(&self) -> bool
pub fn fs_root_scanning_enabled(&self) -> bool
pub fn home_dir_scanning_enabled(&self) -> bool
pub fn cache_budget(&self) -> &ContentCacheBudget
pub fn has_explicit_cache_budget(&self) -> bool
pub fn set_cache_budget(&mut self, budget: ContentCacheBudget)
pub fn bigram_index(&self) -> Option<&BigramFilter>
pub fn bigram_overlay(&self) -> Option<&RwLock<BigramOverlay>>   // parking_lot
pub fn trace_id(&self) -> &str
pub fn trace_span(&self) -> tracing::Span
```

### Incremental mutation

```rust
pub fn handle_create_or_modify(&mut self, path: impl AsRef<Path> + Debug) -> Option<&FileItem>
pub fn add_new_file(&mut self, path: &Path) -> Option<&FileItem>
pub fn remove_file_by_path(&mut self, path: impl AsRef<Path>) -> bool
pub fn remove_all_files_in_dir(&mut self, dir: impl AsRef<Path>) -> usize
pub fn update_single_file_frecency(
    &mut self,
    file_path: impl AsRef<Path>,
    frecency_tracker: &FrecencyTracker,
) -> Result<(), Error>
```
`None` from `handle_create_or_modify` / `add_new_file` is not "not found" — it means the picker is in an invalid state or the index capacity is exhausted, and the caller should trigger a full rescan (`SharedFilePicker::trigger_full_rescan_async`). `remove_file_by_path` only tombstones; if the file still exists on disk the watcher's internal machinery can revert the removal, so call it only when you know the file is going away or no watcher is installed.

### Lifecycle / progress

```rust
pub fn get_scan_progress(&self) -> ScanProgress
pub fn is_scan_active(&self) -> bool         // lock-free; prefer over get_scan_progress for polling
pub fn is_post_scan_active(&self) -> bool
pub fn cancel(&self)                         // stop background threads from grabbing locks
pub fn stop_background_monitor(&mut self)    // non-blocking watcher shutdown
pub fn watcher_signal(&self) -> Arc<AtomicBool>  // clone of the watcher-ready flag, pollable without a picker lock
```

`&FilePicker` implements `FFFStringStorage` (`arena_for`, `base_arena`, `overflow_arena`) — needed when resolving `FileItem` paths yourself, since scan-time and post-scan paths live in different arenas.

## `ScanProgress`

```rust
pub struct ScanProgress {
    pub scanned_files_count: usize,
    pub is_scanning: bool,
    pub is_watcher_ready: bool,
    pub is_warmup_complete: bool,
}
```
Point-in-time snapshot from `FilePicker::get_scan_progress`. `is_scanning == false` does not imply the index is warm: mmap/content warmup is tracked separately by `is_warmup_complete`.

## `FuzzySearchOptions<'a>`

```rust
pub struct FuzzySearchOptions<'a> {
    pub max_threads: usize,                    // 0 = let the crate choose
    pub current_file: Option<&'a str>,         // proximity boost relative to the file in focus
    pub project_path: Option<&'a Path>,        // project root for path-relative scoring
    pub combo_boost_score_multiplier: i32,     // weight of the query-tracker co-occurrence boost
    pub min_combo_count: u32,                  // minimum co-occurrences before that boost applies
    pub pagination: PaginationArgs,            // { offset, limit }
}
```
`Copy + Default`. The `'a` lifetime ties it to the query, so it must not outlive the parsed `FFFQuery`.

## Shared state

Four handles, all cheap-to-clone `Arc<RwLock<Option<T>>>` newtypes. `Option` because the payload is installed asynchronously: the handle exists before the picker/tracker does, so background threads can be handed a clone at startup. Every consumer clones the handle instead of passing `&FilePicker` around.

Note the two lock flavors: `SharedFilePicker` guards are **parking_lot** `RwLockReadGuard`/`RwLockWriteGuard` (no poisoning); `SharedDb` guards are **std::sync** ones.

### `SharedFilePicker`

```rust
pub fn read(&self) -> Result<RwLockReadGuard<'_, Option<FilePicker>>, Error>
pub fn write(&self) -> Result<RwLockWriteGuard<'_, Option<FilePicker>>, Error>
```
Its own inherent methods are the non-blocking control-plane operations (create, rescan, watch); anything that touches the index goes through `read()`/`write()` and `.as_ref()` / `.as_mut()`.

Gotchas:
- `read()?.as_ref()` is `None` until `new_with_shared_state` completes. The Quick Start's `.unwrap()` panics on an uninitialized handle — match instead, or expect `Error::FilePickerMissing` from methods that resolve it internally.
- `let picker = shared.read()?.as_ref().unwrap();` does not compile / dangles: bind the guard to a named local first, then borrow from it.
- Hold read guards for the duration of a search only. Background scan and rescan threads need the write lock; a long-held read guard stalls them. Never call a `SharedFilePicker` method that takes the write lock while holding a read guard from the same handle.

```rust
pub fn wait_for_scan(&self, timeout: Duration) -> bool             // true = finished, false = timed out
pub fn wait_for_watcher(&self, timeout: Duration) -> bool
pub fn wait_for_indexing_complete(&self, timeout: Duration) -> bool // scanning == false AND post-scan indexing == false
pub fn need_complex_rebuild(&self) -> bool   // this picker has a slow post-scan indexing/warmup job
pub fn cancel(&self)                          // non-blocking; threads bail at their next cancellation point
pub fn trigger_full_rescan_async(&self, shared_frecency: &SharedFrecency) -> Result<(), Error>
pub fn rescan_stats(&self) -> RescanStats     // admitted vs throttled requests, by reason
pub fn reset_rescan_stats(&self)
```
`wait_for_scan` is the required handshake after `new_with_shared_state` — searching before it returns `true` searches a partial index. `trigger_full_rescan_async` guarantees a single active rescan per picker and that the most recent request is the one that completes.

```rust
pub fn watch(
    &self,
    pattern: &str,
    options: WatchOptions,
    callback: impl Fn(WatchId, &[WatchEvent]) + Send + Sync + 'static,
) -> Result<WatchId, Error>
pub fn unwatch(&self, id: WatchId) -> bool          // true if the id was active
pub fn is_watch_active(&self, id: WatchId) -> bool
pub fn shutdown_watches(&self)                       // drops all subscriptions, does not wait for a running callback
pub fn shutdown_watches_and_wait(&self)              // waits; does not deadlock when called from inside a callback
```
Patterns: base-relative globs (`./` accepted), exact paths inside the indexed tree, or existing directories; an empty pattern watches the whole tree. Events are debounced into batches — at most 128 events per 100 ms window. Gitignored/ignored files never fire. Errors: `Error::WatcherDisabled` (`options.watch == false`), `Error::WatcherNotReady`, `Error::WatchBaseChanged`.

```rust
pub fn refresh_git_status(&self, shared_frecency: &SharedFrecency) -> Result<usize, Error>
pub fn update_git_status_for_paths(&self, paths: &[PathBuf], shared_frecency: &SharedFrecency) -> Result<(), Error>
```
Full refresh returns the number of files updated.

### `SharedDb<T>`, `SharedFrecency`, `SharedQueryTracker`

```rust
pub struct SharedDb<T: LmdbStore> { /* private */ }
pub type SharedFrecency     = SharedDb<FrecencyTracker>;
pub type SharedQueryTracker = SharedDb<QueryTracker>;
```
LMDB-backed persistence. `LmdbStore` is crate-private, so the type parameter is sealed to `FrecencyTracker` and `QueryTracker`.

```rust
pub fn noop() -> Self                       // disabled handle; silently ignores all writes
pub fn read(&self) -> Result<RwLockReadGuard<'_, Option<T>>, Error>
pub fn write(&self) -> Result<RwLockWriteGuard<'_, Option<T>>, Error>
pub fn init(&self, tracker: T) -> Result<(), Error>   // install + spawn GC thread; no-op when disabled
pub fn destroy(&self) -> Result<Option<PathBuf>, Error>
```

Order matters: `FrecencyTracker::open(path)? → shared_frecency.init(...)?` before `FilePicker::new_with_shared_state`; same for `QueryTracker::open` before the first search that passes a tracker. `destroy` takes the write lock so all readers (including live mmap access) are drained before the LMDB environment closes and the directory is deleted; it returns `Ok(Some(path))` for the deleted directory or `Ok(None)` if nothing was initialized.

`fuzzy_search` takes `Option<&QueryTracker>` — pass `qt_guard.as_ref()` from a `SharedQueryTracker` read guard, or `None` to skip last-selection boosting. Use `noop()` when you want the API shape without touching disk (tests, ephemeral sessions).

## `Error` and `Result`

```rust
pub type Result<T> = std::result::Result<T, Error>;
```
Crate-wide alias; `FilePicker`, grep, `dbs`, watch, and shared-handle methods all return it.

`Error` is `#[non_exhaustive]` — every `match` needs a wildcard arm. Implements `Display`, `std::error::Error` (with `source()`), and `From` for `std::io::Error`, `heed::Error`, `notify::Error`, `git2::Error`, and `std::path::StripPrefixError`.

Variants:

- Threading / state: `ThreadPanic`, `FilePickerMissing` (shared handle still holds `None`), `AcquireFrecencyLock`, `AcquireItemLock`, `AcquirePathCacheLock`
- Paths: `InvalidPath(PathBuf)`, `FilesystemRoot(PathBuf)`, `StripPrefixError(StripPrefixError)`, `CreateDir(io::Error)`
- Scanning / matching: `WalkFailed(String)`, `InvalidGlobPattern { pattern: String, reason: String }`, `Git(git2::Error)`
- Watching: `WatcherDisabled`, `WatcherNotReady`, `WatchBaseChanged`, `WatchDispatcherStart(io::Error)`, `FileSystemWatch(notify::Error)`
- LMDB, each carrying `{ db: &'static str, source: heed::Error }`: `EnvOpen`, `DbCreate`, `DbOpen`, `DbClearStaleReaders`, `DbStartReadTxn`, `DbStartWriteTxn`, `DbRead`, `DbWrite`, `DbCommit`; plus `GenericDbError(heed::Error)` and `RemoveDbDir { path: PathBuf, source: io::Error }`

The `db` field is a static name identifying which store failed (frecency vs query tracker) — useful in log messages without extra plumbing.
