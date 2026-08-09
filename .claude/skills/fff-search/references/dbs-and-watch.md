# fff-search: Persistence DBs & Filesystem Watch

Two LMDB (via `heed` 0.22) environments back ranking: `FrecencyTracker` (per-file access
history) and `QueryTracker` (query history + query→file associations). LMDB is used because
lookups sit on the hot path of every keystroke-driven ranking pass: memory-mapped reads are
zero-copy and lock-free for readers, and multi-process access (several editor instances
sharing one `~/.cache` DB) is safe without a server. Both types implement `DbHealthChecker`.

> Deprecation (0.7.0): `::new(path, use_unsafe_no_lock)` on both trackers is deprecated —
> LMDB unsafe no-lock mode was removed and the bool is ignored. Use `::open(path)`.

---

## `dbs::frecency::FrecencyTracker`

```rust
pub struct FrecencyTracker { /* private */ }
```

Tracks access counts and access timestamps per file path, and combines them with git
modification state to produce ranking scores. "Frecency" = frequency × recency decay.

```rust
pub fn open(db_path: impl AsRef<Path>) -> Result<Self>
pub fn db_path(&self) -> &Path
```

`open` takes the LMDB *environment directory* (not a file). `db_path` returns it back.

```rust
#[deprecated(since = "0.7.0", note = "use FrecencyTracker::open")]
pub fn new(db_path: impl AsRef<Path>, _use_unsafe_no_lock: bool) -> Result<Self>
```

### Recording

```rust
pub fn track_access(&self, path: &Path) -> Result<()>
```

Note `&self`, not `&mut self` — writes go through LMDB's own write txn, so a shared
reference suffices. Call on file open/selection.

### Reading raw counters

```rust
pub fn seconds_since_last_access(&self, path: &Path) -> Result<Option<u64>>
pub fn access_count(&self, path: &Path) -> Result<usize>
```

`seconds_since_last_access` returns `None` for a never-tracked file (distinct from
`Ok(Some(0))`). `access_count` returns `0` for untracked.

### Scoring

```rust
pub fn get_access_score(&self, file_path: &Path, mode: FFFMode) -> i64

pub fn get_modification_score(
    &self,
    modified_time: u64,
    git_status: Option<git2::Status>,
    mode: FFFMode,
) -> i64
```

Both return `i64` directly — no `Result`; DB errors degrade to a neutral score rather than
failing the search. `FFFMode` (`file_picker::FFFMode`) tunes weighting per picker mode.

`get_modification_score` is *not* keyed on a path: the caller supplies `modified_time`
(mtime as a unix timestamp) and the file's `git2::Status`. Score is only awarded when the
file is modified in the current git dir — pass `git_status: None` outside a repo or for
clean files and the modification component drops out.

### `DbHealthChecker` impl

```rust
fn get_env(&self) -> &heed::Env
fn is_healthy(&self) -> bool
fn count_entries(&self) -> Result<Vec<(&'static str, u64)>>
fn get_health(&self) -> Result<DbHealth>   // provided by the trait
```

`count_entries` yields one `(label, count)` per sub-database.

---

## `dbs::query_tracker::QueryTracker`

```rust
pub struct QueryTracker { /* private */ }
```

Stores (a) file-picker query history, (b) grep query history, and (c) query→file
associations used for **combo-boost**: when a user has typed query *Q* and then opened file
*F* before, *F* is boosted the next time *Q* is typed. The strength gate is the recorded
`open_count` for that (query, project, file) triple — repeat pairings score higher, so a
habitual "typed `hand` → opened `handlers.rs`" pattern promotes that file to the top even
when fuzzy-match score alone would not.

All methods return `Result<_, fff_search::Error>`. History and associations are scoped by
`project_path`, so the same query in different repos does not cross-contaminate.

```rust
pub fn open(db_path: impl AsRef<Path>) -> Result<Self, Error>
pub fn db_path(&self) -> &Path

#[deprecated(since = "0.7.0", note = "use QueryTracker::open")]
pub fn new(db_path: impl AsRef<Path>, _use_unsafe_no_lock: bool) -> Result<Self, Error>
```

### Recording (`&mut self`)

Unlike `FrecencyTracker`, the write methods take `&mut self`.

```rust
pub fn track_query_completion(
    &mut self,
    query: &str,
    project_path: &Path,
    file_path: &Path,
) -> Result<(), Error>
```

The combo-boost write path: records that `query` in `project_path` ended with the user
selecting `file_path`, appending to history and bumping the pair's `open_count` /
`last_opened`.

```rust
pub fn track_grep_query(&mut self, query: &str, project_path: &Path) -> Result<(), Error>
```

Grep history only — no file association is recorded, since a grep query resolves to many
matches rather than one selection.

### Reading associations

```rust
pub fn get_last_query_entry(
    &self,
    query: &str,
    project_path: &Path,
    min_combo_count: u32,
) -> Result<Option<QueryMatchEntry>, Error>
```

Returns the strongest association for `query`, or `None` if no pairing reached
`min_combo_count` opens. Raising `min_combo_count` is the knob for how much repetition is
required before history is trusted.

```rust
pub fn get_last_query_path(
    &self,
    query: &str,
    project_path: &Path,
    file_path: &Path,
    combo_boost: i32,
) -> Result<i32, Error>
```

The scoring-loop call: returns the boost to add for this candidate `file_path` — `0` when
the file has no association with `query`, otherwise a value derived from `combo_boost`
(the caller-supplied magnitude). Call per candidate during ranking.

### History replay

```rust
pub fn get_historical_query(&self, project_path: &Path, offset: usize)
    -> Result<Option<String>, Error>

pub fn get_historical_grep_query(&self, project_path: &Path, offset: usize)
    -> Result<Option<String>, Error>
```

`offset = 0` is the most recent, `1` the next, etc. — drive up-arrow history with an
incrementing offset until `None`.

### `QueryMatchEntry`

```rust
pub struct QueryMatchEntry {
    pub file_path: PathBuf,
    pub open_count: u32,   // combo strength
    pub last_opened: u64,  // unix seconds
}
```

`Clone + Debug + Serialize + Deserialize` (serde is how it is stored in LMDB).

---

## `dbs::db_healthcheck::DbHealth`

```rust
pub struct DbHealth {
    pub path: String,                          // DB file path
    pub disk_size: u64,                        // bytes
    pub entry_counts: Vec<(&'static str, u64)>,// per-table counts
    pub healthy: bool,                         // false if the write lock cannot be acquired
}
```

`healthy: false` specifically means write-lock acquisition failed (another process holds
it / stale lock) — it is not a corruption signal.

---

## Watch subsystem (`fff_search::watch`)

A single `BackgroundWatcher` owns the OS-level notifier; callers register interest and get
back a `WatchId`, which is the handle used to unregister. Events are normalized across
platforms and delivered in batches to subscribers.

### `WatchId`

```rust
#[repr(transparent)]
pub struct WatchId(pub u64);
```

`Copy + Clone + Debug + Eq + PartialEq + Hash` — usable directly as a `HashMap` key.
Returned at subscribe time; pass it back to unsubscribe. Field is public, so an id can be
round-tripped through FFI as a plain `u64`.

### `WatchOptions`

```rust
pub struct WatchOptions {
    pub ignore: Vec<String>,   // additional glob or path-prefix exclusions
}
```

Per-subscription options; `Default` gives an empty ignore list. `ignore` entries are
matched as either globs or path prefixes and are *additional* to the engine's global
ignore rules — they narrow a subscription, they cannot re-include something globally
ignored. The subscription target itself (glob pattern, exact path, or directory subtree)
is supplied at the subscribe call site, not in this struct.

### `WatchEventKind`

```rust
#[repr(u8)]
pub enum WatchEventKind {
    Created  = 0,
    Modified = 1,
    Removed  = 2,
    Rescan   = 3,
}

impl WatchEventKind {
    pub fn as_str(&self) -> &'static str
}
```

Kinds are normalized **best-effort**: editors and OSes express the same user action with
different native events (atomic-save often surfaces as create+remove rather than modify),
so do not treat a specific kind as authoritative for correctness. `Rescan` means individual
events were dropped (queue overflow) and the reported path must be re-walked — always
handle it, or the index silently diverges. `#[repr(u8)]` with explicit discriminants makes
the values stable across an FFI boundary.

### `WatchEvent`

```rust
pub struct WatchEvent {
    pub path: PathBuf,          // absolute; for Rescan, the indexed base path
    pub kind: WatchEventKind,
}
```

One change notification. For `Rescan`, `path` is the *indexed base path* to re-scan, not a
single changed file.

### `BackgroundWatcher`

```rust
pub struct BackgroundWatcher { /* private */ }

impl BackgroundWatcher {
    pub fn stop(&mut self)
}

impl Drop for BackgroundWatcher { /* joins worker threads */ }
```

Owns the filesystem watcher. `stop()` only *signals* shutdown and returns without blocking
on the worker threads — this is what makes it safe to call from any context, **including
while holding the `SharedFilePicker` write lock**. Calling a blocking teardown there would
deadlock, since the workers need that same lock to drain. All background threads are
joined in `Drop`, so drop the watcher outside the lock if you need the join to have
completed before continuing.

---

## `git::format_git_status`

```rust
pub fn format_git_status(status: Option<git2::Status>) -> &'static str
```

Renders a `git2::Status` bitflag set as a short display string for the picker's status
column. Returns a `&'static str` (no allocation, safe to call per row per frame); `None`
maps to the not-in-repo / clean rendering. Same `Option<git2::Status>` that
`FrecencyTracker::get_modification_score` consumes.
