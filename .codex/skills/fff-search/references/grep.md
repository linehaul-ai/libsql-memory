# Grep (content search) — `fff_search::grep`

Live content search over the picker's file list. Three matching engines selected by `GrepMode`,
with two documented fallback paths (regex→literal, constrained→raw literal) surfaced on the result.

## `GrepMode`

```rust
pub enum GrepMode {
    PlainText, // default
    Regex,
    Fuzzy,
}
```

- **`PlainText`** — literal substring match (SIMD `memmem`); no regex machinery on the hot path. This is
  `Default::default()`.
- **`Regex`** — same matching engine as ripgrep. If the pattern fails to compile, the search does **not**
  error: it falls back to literal matching and reports the compile error in
  `GrepResult::regex_fallback_error`.
- **`Fuzzy`** — typo/gap-tolerant per-line scoring via `neo_frizbee`. Tolerates a couple of single-char
  typos or long gaps (`shcema` → `schema`, `UserController` → `UserAuthController`). Significantly slower
  than `PlainText`, especially on an unindexed `FilePicker`. Only mode that populates
  `GrepMatch::fuzzy_score`.

Derives: `Copy`, `Clone`, `Debug`, `Default` (→ `PlainText`), `PartialEq`/`Eq`.

Mode is an explicit input on `GrepSearchOptions::mode`. Callers that want ripgrep-like auto-detection
run `has_regex_metacharacters` on the raw query and set `Regex` when it returns `true`, else `PlainText`;
escalating to `Fuzzy` after a zero-hit plain pass is a caller-side policy, not something the option
struct does on its own.

## `GrepSearchOptions`

```rust
pub struct GrepSearchOptions {
    pub max_file_size: u64,
    pub max_matches_per_file: usize,
    pub smart_case: bool,
    pub file_offset: usize,
    pub page_limit: usize,
    pub mode: GrepMode,
    pub time_budget_ms: u64,
    pub before_context: usize,
    pub after_context: usize,
    pub classify_definitions: bool,
    pub trim_whitespace: bool,
    pub abort_signal: Option<Arc<AtomicBool>>,
}
```

| Field | Semantics |
| --- | --- |
| `max_file_size: u64` | Byte ceiling; larger files are skipped (they count against `filtered_file_count`, not `total_files_searched`). |
| `max_matches_per_file: usize` | Per-file cap on collected matches. |
| `smart_case: bool` | Case-insensitive unless the pattern itself contains uppercase. |
| `file_offset: usize` | File-based pagination cursor: index into the sorted/filtered file list to start from. `0` for the first page; thereafter pass `GrepResult::next_file_offset`. |
| `page_limit: usize` | Max matches to collect before stopping this call. |
| `mode: GrepMode` | How to interpret the pattern. Defaults to `PlainText`. |
| `time_budget_ms: u64` | Wall-clock ceiling; on expiry the search returns **partial** results rather than erroring. Guards against pathological queries freezing a UI. `0` = no limit. |
| `before_context: usize` | Context lines captured before each match. `0` disables (leaves `context_before` empty). |
| `after_context: usize` | Context lines captured after each match. `0` disables. |
| `classify_definitions: bool` | Compute `GrepMatch::is_definition` at match time. ~2% overhead on large repos; turn off for interactive grep that doesn't render definition badges. |
| `trim_whitespace: bool` | Strip leading whitespace from matched and context lines, adjusting highlight byte offsets to stay correct. Intended for AI/MCP consumers and UIs that don't render indentation. Default `false`. |
| `abort_signal: Option<Arc<AtomicBool>>` | External cancellation. When `Some`, it **overrides** the picker's internal cancellation flag. Store `true` to stop early and get partial results. Leave `None` (e.g. via `..Default::default()`) to let the picker manage cancellation. |

Derives `Clone`, `Debug`, `Default` — construct with `GrepSearchOptions { mode: GrepMode::Regex, ..Default::default() }`.

## `GrepMatch`

```rust
pub struct GrepMatch {
    pub file_index: usize,
    pub line_number: u64,
    pub col: usize,
    pub byte_offset: u64,
    pub line_content: String,
    pub match_byte_offsets: SmallVec<[(u32, u32); 4]>,
    pub fuzzy_score: Option<u16>,
    pub is_definition: bool,
    pub context_before: Vec<String>,
    pub context_after: Vec<String>,
}
```

- `file_index` — index into `GrepResult::files` (the deduplicated file vec), not into the picker's global
  file list. Always resolve paths through the result's `files`.
- `line_number` — **1-based**.
- `col` — **0-based byte** column of the first match start within the line (not a char index).
- `byte_offset` — absolute byte offset of the matched line from the start of the file; a preview pane can
  `seek` directly instead of scanning from the top.
- `line_content` — matched line text, truncated to `MAX_LINE_DISPLAY_LEN`.
- `match_byte_offsets` — `(start, end)` byte spans **within `line_content`**, one per match on the line;
  `SmallVec` inline for the common ≤4-span case. These are the highlight ranges; they stay valid after
  truncation and after whitespace trimming.
- `fuzzy_score` — `neo_frizbee` score, `Some` only in `GrepMode::Fuzzy`.
- `is_definition` — line looks like a definition (`struct`, `fn`, `class`, …). Computed at match time so
  output formatters never re-scan; only meaningful when `classify_definitions` was set.
- `context_before` / `context_after` — empty when the corresponding context option is `0`.

```rust
impl GrepMatch {
    pub fn trim_leading_whitespace(&mut self);
}
```

Strips leading whitespace from `line_content` and every context line, re-adjusting `col` and
`match_byte_offsets` so highlights remain correct. This is the same operation
`GrepSearchOptions::trim_whitespace` applies; call it manually to trim a subset of results post hoc.

## `GrepResult<'a>`

```rust
pub struct GrepResult<'a> {
    pub matches: Vec<GrepMatch>,
    pub files: Vec<&'a FileItem>,
    pub total_files_searched: usize,
    pub total_files: usize,
    pub filtered_file_count: usize,
    pub files_with_matches: usize,
    pub next_file_offset: usize,
    pub regex_fallback_error: Option<String>,
    pub literal_fallback: bool,
}
```

Borrows `FileItem`s from the picker for `'a`, so the result cannot outlive the index it was searched
against.

- `files` — deduplicated file references; `GrepMatch::file_index` indexes this.
- `total_files_searched` — files actually opened during **this call** (bounded by `page_limit` /
  `time_budget_ms` / abort).
- `total_files` — total indexed files, before filtering.
- `filtered_file_count` — searchable files after dropping binary, too-large, ignored, etc. The pagination
  space is this set, not `total_files`.
- `files_with_matches` — files with ≥1 match.
- `next_file_offset` — cursor to feed back as `GrepSearchOptions::file_offset` for the next page.
  **`0` means there are no more files** — it is a sentinel, not a "restart from the beginning", so test
  for `0` before issuing another page.
- `regex_fallback_error` — `Some(msg)` when `GrepMode::Regex` failed to compile and the search silently
  degraded to literal matching. Surface it so the user knows their regex was invalid; the `matches` are
  still valid literal hits.
- `literal_fallback` — `true` when the constrained query matched nothing and results come from retrying
  the whole **raw** query as literal text, ignoring all inferred constraints (see `parse_grep_query`).
  Useful for telling the user "no results for your filters; showing literal matches instead."

Derives `Clone`, `Debug`, `Default` (empty result).

## Free functions

```rust
pub fn parse_grep_query(query: &str) -> FFFQuery<'_>
```

Parses a raw grep query string into the structured `FFFQuery`, splitting the search term from inferred
constraints (path/extension/etc. filters embedded in the query). Zero-copy — the returned `FFFQuery`
borrows from `query`. When searching with the parsed constraints yields nothing, the engine retries the
untouched raw string as a literal and flags `GrepResult::literal_fallback`.

```rust
pub fn has_regex_metacharacters(text: &str) -> bool
```

Cheap predicate: does the string contain regex metacharacters? Use it to decide whether a user-typed
pattern should run as `GrepMode::Regex` or take the faster `PlainText` path. A `false` result means the
pattern is safe to treat as a literal.
