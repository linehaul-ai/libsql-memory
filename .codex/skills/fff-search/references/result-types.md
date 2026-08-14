# fff-search: Result & Item Types

All types live in `fff_search::types` (`Location` is re-exported at `fff_search::Location`, originating in `fff-query-parser`).

Central design constraint: **items do not own their path strings.** Every `FileItem`/`DirItem` stores compact offsets into a shared chunk arena owned by the `FilePicker`. Any method that produces a path takes an `arena: impl FFFStringStorage` argument. Result structs borrow items from the picker (`Vec<&'a FileItem>`), so a `SearchResult` cannot outlive the picker's index snapshot.

---

## FileItem

```rust
pub struct FileItem {
    pub size: u64,                          // bytes
    pub modified: u64,                      // Unix seconds
    pub access_frecency_score: i16,
    pub modification_frecency_score: i16,
    pub git_status: Option<git2::Status>,
    /* private: path offsets into the chunk arena, flags byte, cached mmap */
}
```

`Clone`, `Debug`. No public path field — use the accessors below.

### Path accessors (all need the arena)

```rust
pub fn relative_path(&self, arena: impl FFFStringStorage) -> String;
pub fn absolute_path(&self, arena: impl FFFStringStorage, base_path: &Path) -> PathBuf;
pub fn dir_str(&self, arena: impl FFFStringStorage) -> String;   // parent dir portion
pub fn file_name(&self, arena: impl FFFStringStorage) -> String; // basename
pub fn relative_path_len(&self) -> usize;                        // no arena needed
pub fn filename_offset_in_relative_path(&self) -> usize;         // byte offset of basename
```

`relative_path_len` and `filename_offset_in_relative_path` read the stored offsets directly — cheap, no arena, useful for slicing `match_byte_offsets` into "path part" vs "filename part".

### Scoring / flag accessors

```rust
pub fn total_frecency_score(&self) -> i32;   // access + modification frecency combined
pub fn is_binary(&self) -> bool;
pub fn set_binary(&self, val: bool);         // &self — interior mutability (atomic flag byte)
pub fn is_deleted(&self) -> bool;
pub fn is_overflow(&self) -> bool;
pub fn set_overflow(&self, val: bool);       // &self
```

Setters take `&self`, not `&mut self`: flags are atomic so worker threads can mark items during a parallel scan.

### Construction (index-building; normally the picker's job)

```rust
pub fn new(path: PathBuf, base_path: &Path, git_status: Option<Status>) -> (Self, String);
pub fn new_from_walk(path: &Path, base_path: &Path, git_status: Option<Status>,
                     metadata: Option<&Metadata>) -> (Self, String);
pub fn new_from_walk_parts(path: &Path, base_path: &Path, git_status: Option<Status>,
                           size: u64, modified: u64) -> (Self, String);
pub fn new_from_walk_bytes(path: &Path, relative_path: &[u8], basename_offset: u16,
                           git_status: Option<Status>, size: u64, modified: u64) -> (Self, String);
pub fn new_raw(filename_start: u16, size: u64, modified: u64,
               git_status: Option<Status>, is_binary: bool) -> Self;
```

The `(Self, String)` return is deliberate: the item is created with an **empty** path reference, and the returned relative-path `String` must be kept alongside it until `build_chunked_path_store_and_assign` interns all paths into the arena and back-fills each item's path field. Constructing items without that step yields items whose `relative_path()` is empty.

- `new_from_walk_parts` — for walkers that already fetched size/mtime in bulk (zlob backend).
- `new_from_walk_bytes` — fast path taking the already-root-relative byte slice plus the byte offset where the basename starts; skips a `PathBuf` alloc and component walk.

### Content-cache maintenance

```rust
pub fn invalidate_mmap(&mut self, budget: &ContentCacheBudget);
pub fn update_metadata(&mut self, budget: &ContentCacheBudget,
                       modified_secs: Option<u64>, new_size: Option<u64>);
```

Call `invalidate_mmap` whenever the watcher reports the file changed. On Unix a mapped file truncated underneath you can raise SIGBUS; on Windows the stale buffer silently serves old bytes. Both fixed by invalidating so the next access re-reads. Both methods release the file's bytes back to the `ContentCacheBudget`.

---

## FileItemFlags

Unit struct namespace over the `u8` flags byte inside `FileItem`.

```rust
pub struct FileItemFlags;
impl FileItemFlags {
    pub const BINARY: u8;
    pub const DELETED: u8;
    pub const OVERFLOW: u8;
}
```

- `BINARY` — content search skips it.
- `DELETED` — tombstone. The slot stays allocated so bigram indices pointing at *other* items remain valid; deletion never compacts the item vector.
- `OVERFLOW` — added after the last full reindex; its string offsets resolve against the overflow builder arena, not the base arena. Resolving an overflow item against the base arena gives garbage, so use the picker's arena handle rather than a raw base-arena pointer.

Not git status. Git state lives in `FileItem::git_status: Option<git2::Status>` (a `git2` bitflag set: `INDEX_NEW`, `WT_MODIFIED`, `IGNORED`, …), and feeds `Score::git_status_boost`.

---

## DirItem

```rust
pub struct DirItem { /* private fields */ }
```

A directory in the index; shares the chunk arena with file paths. `Clone`, `Debug`.

```rust
pub fn relative_path(&self, arena: impl FFFStringStorage) -> String;
pub fn absolute_path(&self, arena: impl FFFStringStorage, base_path: &Path) -> PathBuf;
pub fn dir_name(&self, arena: impl FFFStringStorage) -> String;      // last segment only
pub fn write_dir_name(&self, arena: ArenaPtr, out: &mut String);     // alloc-free variant
pub fn last_segment_offset(&self) -> u16;                            // byte offset of last segment

pub fn max_access_frecency(&self) -> i32;
pub fn update_frecency_if_larger(&self, score: i32);  // atomic max, parallel-safe
pub fn reset_frecency(&self);                         // before a full recompute

pub fn is_overflow(&self) -> bool;
pub fn is_deleted(&self) -> bool;
```

A directory's frecency is the **max** over its files, not a sum — hence `update_frecency_if_larger`, which workers call concurrently while walking files. `reset_frecency` first, then re-drive the max, when recomputing wholesale. `write_dir_name` is the hot-loop version of `dir_name` (renders into a caller-owned `String`); note it takes a concrete `ArenaPtr` rather than `impl FFFStringStorage`.

---

## DirFlags

```rust
pub struct DirFlags;
impl DirFlags {
    pub const OVERFLOW: u8;
    pub const DELETED: u8;
}
```

Same semantics as the `FileItemFlags` counterparts. No `BINARY`.

---

## Score

Full breakdown of one item's rank. Parallel to `items` in every result struct.

```rust
pub struct Score {
    pub total: i32,                   // sum of all components; the sort key
    pub base_score: i32,              // raw fuzzy-match quality of the query vs the path
    pub filename_bonus: i32,          // match landed in the basename rather than a dir segment
    pub special_filename_bonus: i32,  // notable names (index/mod/main/README-class files)
    pub frecency_boost: i32,          // from access_frecency_score + modification_frecency_score
    pub git_status_boost: i32,        // derived from git_status — modified/staged files rank up
    pub distance_penalty: i32,        // path distance from project_path / cwd
    pub current_file_penalty: i32,    // demotes ScoringContext::current_file (you're already in it)
    pub combo_match_boost: i32,       // repeat query→selection history (see QueryTracker)
    pub path_alignment_bonus: i32,    // query segments aligning with path segment boundaries
    pub exact_match: bool,
    pub match_type: &'static str,     // e.g. exact / prefix / substring / fuzzy label
}
```

`Clone`, `Debug`, `Default`.

Penalty fields are stored as their signed contribution and are already folded into `total` — do not subtract them again. Sorting is by `total` descending; `exact_match` and `match_type` are for display/tie-explanation, not re-ranking.

---

## ScoringContext

Per-query knobs handed to the scorer.

```rust
pub struct ScoringContext<'a> {
    pub query: &'a FFFQuery<'a>,
    pub project_path: Option<&'a Path>,
    pub current_file: Option<&'a str>,
    pub max_typos: u16,
    pub max_threads: usize,
    pub last_same_query_match: Option<QueryMatchEntry>,
    pub combo_boost_score_multiplier: i32,
    pub min_combo_count: u32,
    pub pagination: PaginationArgs,
}

impl ScoringContext<'_> {
    pub fn effective_query(&self) -> &str;
}
```

`Clone`, `Debug`.

- `effective_query()` — the search text after the parser strips modifiers/constraints out of `FFFQuery`; use this, not the raw user string, when highlighting.
- `current_file` — relative path of the file the user is currently in; drives `Score::current_file_penalty`.
- `last_same_query_match` + `min_combo_count` + `combo_boost_score_multiplier` — the "you picked X last time you typed this" path, producing `Score::combo_match_boost`. `min_combo_count` is the number of prior selections required before the boost applies.
- `max_typos` — Levenshtein budget for the typo-tolerant matcher.
- `pagination` is part of the scoring context, not a separate call: the scorer scores everything but only materializes the requested window.

---

## SearchResult

```rust
pub struct SearchResult<'a> {
    pub items: Vec<&'a FileItem>,
    pub scores: Vec<Score>,
    pub match_byte_offsets: Vec<SmallVec<[(u32, u32); 4]>>,
    pub total_matched: usize,
    pub total_files: usize,
    pub location: Option<Location>,
}
```

`Clone`, `Debug`, `Default`.

`items`, `scores`, and `match_byte_offsets` are **index-parallel** — `scores[i]` and `match_byte_offsets[i]` describe `items[i]`. All three are already truncated to the pagination window.

- `match_byte_offsets[i]` — `(start, end)` byte ranges into that item's **relative path** string, for highlight rendering. `SmallVec<[_; 4]>` inline capacity 4, so typical results never heap-allocate.
- `total_matched` — matches before pagination; `items.len()` is the page. Use this for "showing N of M".
- `total_files` — total files in the index (the search corpus size), independent of the query.
- `location` — a `:line`/`:line:col` suffix parsed out of the query, e.g. `main.rs:42`:

```rust
pub enum Location {
    Line(i32),
    Range { start: (i32, i32), end: (i32, i32) },
    Position { line: i32, col: i32 },
}
```

Applies to the whole result (it came from the query), not per item; forward it to the editor when opening the selection.

---

## DirSearchResult

```rust
pub struct DirSearchResult<'a> {
    pub items: Vec<&'a DirItem>,
    pub scores: Vec<Score>,
    pub total_matched: usize,
    pub total_dirs: usize,
}
```

`Clone`, `Debug`, `Default`. Directory-only fuzzy search. No `match_byte_offsets` and no `location`.

---

## MixedSearchResult / MixedItemRef

```rust
pub enum MixedItemRef<'a> {
    File(&'a FileItem),
    Dir(&'a DirItem),
}

pub struct MixedSearchResult<'a> {
    pub items: Vec<MixedItemRef<'a>>,
    pub scores: Vec<Score>,
    pub total_matched: usize,
    pub total_files: usize,
    pub total_dirs: usize,
    pub location: Option<Location>,
}
```

Both `Clone`, `Debug`; `MixedSearchResult` and `MixedItemRef` both impl `Default`.

Files and directories are interleaved in one list ordered by `Score::total` descending — do not assume grouping by kind. `total_files` and `total_dirs` are corpus sizes; `total_matched` is the combined pre-pagination match count. No `match_byte_offsets` here, so highlight ranges must be recomputed if needed.

---

## PaginationArgs

```rust
pub struct PaginationArgs {
    pub offset: usize,
    pub limit: usize,
}
```

`Clone`, `Copy`, `Debug`, `Default` (non-trivial default — a preset window, not `0/0`; `limit: 0` would return nothing, so build from `Default::default()` and override rather than constructing zeroed). Page N is `offset = N * limit`; ranking is global, so paging is stable as long as the index and query are unchanged.

---

## ContentCacheBudget

Caps memory used by mmapped file contents during content/grep indexing.

```rust
pub struct ContentCacheBudget {
    pub max_files: usize,
    pub max_bytes: u64,
    pub max_file_size: u64,     // per-file cap; larger files are never cached
    pub cached_count: AtomicUsize,
    pub cached_bytes: AtomicU64,
}

impl ContentCacheBudget {
    pub fn unlimited() -> Self;
    pub fn zero() -> Self;                  // disables content caching
    pub fn new_for_repo(file_count: usize) -> Self;   // auto-size from repo size
    pub fn from_overrides(max_files: usize, max_bytes: u64, max_file_size: u64) -> Option<Self>;
    pub fn is_exhausted(&self) -> bool;
    pub fn reset(&self);
}
```

`Debug`, `Default` (= `new_for_repo(30_000)`).

The `cached_*` fields are live counters, mutated through `&self`; the struct is shared by reference across threads and is not `Clone`.

`from_overrides`: each argument is a cap where `0` means "inherit the library default for that cap". Returns `None` when **all three** are `0` — that is the signal for the picker to auto-size from the actual scanned file count instead of applying an override. Treat `None` as "no override", not as an error.

`reset()` zeroes the counters without changing caps; pair it with dropping cached content. `FileItem::invalidate_mmap` and `update_metadata` both take `&ContentCacheBudget` so freed bytes are returned to the counters.
