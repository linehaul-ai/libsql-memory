# Query Parsing & Parser Configs

Types live in the `fff-query-parser` crate (v0.10.3, module `fff_query_parser`) and are re-exported at the root of `fff_search`. Either import path works:

```rust
use fff_search::{QueryParser, FFFQuery, Constraint, FuzzyQuery, GitStatusFilter, Location, ParserConfig,
                 FileSearchConfig, DirSearchConfig, MixedSearchConfig, GrepConfig, AiGrepConfig};
// or: use fff_query_parser::{FFFQuery, FileSearchConfig};
```

The parser splits a raw query string into two halves: **constraints** (structured filters — extension, glob, exclude, git status, …) and a **fuzzy query** (the leftover text that gets typo-resistant matching). `"*.rs !test/ shcema"` yields `Constraint::Extension("rs")`, `Constraint::Exclude(["test"])`, and `FuzzyQuery::Text("shcema")`.

Parsing is zero-copy: every borrowed variant holds `&'a str` into the input query, so the `FFFQuery<'a>` cannot outlive the query string.

---

## `QueryParser<C>`

```rust
pub struct QueryParser<C>
where
    C: ParserConfig,
{ /* private fields */ }
```

Zero-cost wrapper around a `ParserConfig`. The config is a type parameter, so all the `enable_*` branches monomorphize away — the configs are unit structs with no runtime state.

```rust
impl<C> QueryParser<C>
where
    C: ParserConfig,
{
    pub fn new(config: C) -> QueryParser<C>;
    pub fn parse<'a>(&self, query: &'a str) -> FFFQuery<'a>;
}

impl Default for QueryParser<FileSearchConfig> { /* ... */ }
```

`Default` is implemented **only** for `QueryParser<FileSearchConfig>` — `QueryParser::default()` always means file-picker semantics. For any other mode you must call `QueryParser::new(GrepConfig)` etc., or use the `FFFQuery::parse` shortcut.

```rust
let q = QueryParser::default().parse("lib.rs");
let q = QueryParser::new(GrepConfig).parse("*.rs TODO");
```

---

## `FFFQuery<'a>`

```rust
pub struct FFFQuery<'a> {
    pub raw_query: &'a str,
    pub constraints: Vec<Constraint<'a>>,
    pub fuzzy_query: FuzzyQuery<'a>,
    pub location: Option<Location>,
}
```

- `raw_query` — the original string before parsing.
- `constraints` — parsed structured filters. Documented as "stack-allocated for ≤8 constraints" (the type alias is `ConstraintVec`; treat it as `Vec` for API purposes).
- `fuzzy_query` — the residual text tokens for fuzzy matching.
- `location` — a `file:12` / `file:12:4` suffix, when `ParserConfig::enable_location` is on.

```rust
impl<'a> FFFQuery<'a> {
    /// One-shot parse; equivalent to QueryParser::new(config).parse(query).
    pub fn parse(query: &'a str, config: impl ParserConfig) -> FFFQuery<'a>;

    /// Joins all non-constraint text tokens into the grep pattern.
    pub fn grep_text(&self) -> String;
}
```

`FFFQuery::parse` takes the config by value as `impl ParserConfig`, so it accepts a unit-struct config directly:

```rust
let query = FFFQuery::parse("file *.rs", FileSearchConfig);
```

### `grep_text()` semantics

Reconstructs the literal search pattern from `fuzzy_query`:

| `fuzzy_query` | `grep_text()` |
|---|---|
| `FuzzyQuery::Empty` | `""` |
| `FuzzyQuery::Text("foo")` | `"foo"` |
| `FuzzyQuery::Parts(["a", "\\*.rs", "b"])` | `"a *.rs b"` |

Backslash-escaped tokens (`\*.rs`) are the user's way of telling the parser "this is text, not a glob". `grep_text()` includes them with the leading `\` stripped, because the backslash is a parser signal and must not reach the final pattern.

---

## `ParserConfig`

```rust
pub trait ParserConfig {
    fn enable_glob(&self) -> bool { ... }
    fn enable_extension(&self) -> bool { ... }
    fn enable_exclude(&self) -> bool { ... }
    fn enable_path_segments(&self) -> bool { ... }
    fn enable_type_filter(&self) -> bool { ... }
    fn enable_git_status(&self) -> bool { ... }
    fn enable_location(&self) -> bool { ... }
    fn is_glob_pattern(&self, token: &str) -> bool { ... }
    fn treat_lone_path_as_text(&self) -> bool { ... }
    fn enable_filename_constraint(&self) -> bool { ... }
    fn parse_custom<'a>(&self, _input: &'a str) -> Option<Constraint<'a>> { ... }
}
```

Every method has a default body, so a new config is `struct MyConfig;` plus `impl ParserConfig for MyConfig {}` with only the deltas overridden. The trait **is dyn compatible** — `Box<dyn ParserConfig>` works if you need runtime dispatch, though the built-in configs are used generically.

Method semantics:

| Method | Governs |
|---|---|
| `enable_glob` | glob tokens → `Constraint::Glob` |
| `enable_extension` | extension shortcuts, `*.rs` → `Constraint::Extension("rs")` |
| `enable_exclude` | `!test` → `Constraint::Exclude` |
| `enable_path_segments` | `/src/` → `Constraint::PathSegment("src")` |
| `enable_type_filter` | `type:rust` → `Constraint::FileType("rust")` |
| `enable_git_status` | `status:modified` → `Constraint::GitStatus(Modified)` |
| `enable_location` | `file:12`, `file:12:4` suffixes → `FFFQuery::location` |
| `is_glob_pattern` | which tokens count as globs at all |
| `treat_lone_path_as_text` | demote a solitary `PathSegment` to fuzzy text |
| `enable_filename_constraint` | filename-shaped tokens → `Constraint::FilePath` |
| `parse_custom` | picker-specific constraint syntax; returns `None` by default |

Notable defaults and rationale, quoted from the docs:

- **`enable_location`** — "Disabled for grep modes where colon-number patterns like `localhost:8080` are search text, not file locations."
- **`is_glob_pattern`** — default delegates to `zlob::has_wildcards` with `RECOMMENDED` flags, recognising `*`, `?`, `[`, `{…}`. Override "in configs where some wildcard characters are common in search text (e.g. grep mode where `?` and `[` appear in code)."
- **`treat_lone_path_as_text`** — if `true`, a `PathSegment` constraint that is the *only* token in the query is demoted to fuzzy text, to avoid over-filtering.
- **`enable_filename_constraint`** — **off by default.** If `true`, tokens shaped like filenames (`score.rs`, `src/main.rs`) become `FilePath` constraints scoping the search to matching paths. It is off because "a partial filename like `vite.conf` would otherwise filter out `vite.config.ts` and yield zero results."
- **`parse_custom`** — returns `None` by default. Filename-token detection is *not* done here; the parser handles it separately via `enable_filename_constraint`.

### Why `AiGrepConfig` differs from `GrepConfig`

Both drive full-text content search, and both narrow `is_glob_pattern` the same way. The difference is `enable_filename_constraint`: `GrepConfig` leaves it off (trait default), `AiGrepConfig` turns it **on**. In AI mode, a bare `schema.rs` or a path-prefixed `libswscale/input.c` in the query is detected as a `Constraint::FilePath` so the grep is scoped to those files.

That is only safe because of a caller-side fallback the parser does not perform itself: **the caller validates the `FilePath` constraint against the index and drops it if no files match.** An agent-generated path guess therefore degrades to an unscoped search rather than returning zero results. If you implement a config with `enable_filename_constraint() -> true`, you must replicate that validate-and-drop step.

---

## The five built-in configs

All five are **unit structs** (`pub struct FileSearchConfig;`) deriving `Clone`, `Copy`, `Debug`, `Default`. There are no fields and no associated constants on any of them — every behavioral difference is a `ParserConfig` method override.

Override matrix (✓ = overridden by this config; blank = trait default):

| Method | `FileSearch` | `DirSearch` | `MixedSearch` | `Grep` | `AiGrep` |
|---|:--:|:--:|:--:|:--:|:--:|
| `enable_path_segments` | | ✓ off | ✓ off | ✓ | ✓ |
| `enable_extension` | | ✓ off | | | |
| `enable_type_filter` | | ✓ | | | |
| `enable_git_status` | | ✓ | | ✓ off | ✓ |
| `enable_location` | | | | ✓ off | ✓ |
| `enable_filename_constraint` | | | | | ✓ on |
| `is_glob_pattern` | | | | ✓ narrowed | ✓ narrowed |

### `FileSearchConfig`

```rust
pub struct FileSearchConfig;
```

Default configuration for the **file picker**, and the only config with a `Default` impl on `QueryParser`. Overrides nothing — it is the trait defaults verbatim. Filename-constraint detection is off (trait default) so partial filenames like `vite.conf` don't silently filter out fuzzy matches. The Neovim layer overrides `enable_filename_constraint` to opt in based on user config.

### `DirSearchConfig`

```rust
pub struct DirSearchConfig;
```

Directory search (and shared with mixed-search modes). The most heavily restricted config:

- `enable_path_segments` off — so a trailing `/` stays fuzzy text. `fff-core/` fuzzy-matches directory paths instead of becoming `PathSegment("fff-core")` with an empty residual query.
- `enable_extension` and `enable_type_filter` off — extensions don't apply to directories.
- `enable_git_status` overridden.
- Filename constraints are also inapplicable, though that needs no override (off by default).

### `MixedSearchConfig`

```rust
pub struct MixedSearchConfig;
```

Files **and** directories in one result set. Overrides only `enable_path_segments` (off), for the same trailing-`/` reason as `DirSearchConfig` — here a trailing `/` triggers dirs-only mode rather than becoming a constraint. Unlike `DirSearchConfig` it **keeps git status and extension filters enabled**, since files are part of the results.

### `GrepConfig`

```rust
pub struct GrepConfig;
```

Full-text content search. File constraints stay enabled (they select *which* files to search), but:

- `enable_git_status` off — not useful when searching file contents.
- `enable_location` off — `localhost:8080` in a grep query is search text, not a `file:line`.
- `enable_path_segments` overridden.
- `is_glob_pattern` narrowed: **only tokens containing a path separator (`/`) or brace expansion (`{…}`) count as globs.** `?` and `[` are extremely common in source code and must remain literal search text.

### `AiGrepConfig`

```rust
pub struct AiGrepConfig;
```

Grep for AI/agent callers. Same overrides as `GrepConfig` (`enable_path_segments`, `enable_git_status`, `enable_location`, narrowed `is_glob_pattern`) **plus `enable_filename_constraint` on** — see the section above for the required caller-side validate-and-drop fallback.

---

## `Constraint<'a>`

```rust
pub enum Constraint<'a> {
    Extension(&'a str),
    Glob(&'a str),
    Parts(&'a [&'a str]),
    Text(&'a str),
    Exclude(&'a [&'a str]),
    PathSegment(&'a str),
    FilePath(&'a str),
    FileType(&'a str),
    GitStatus(GitStatusFilter),
    Not(Box<Constraint<'a>>),
}
```

| Variant | Query syntax | Meaning |
|---|---|---|
| `Extension(&str)` | `*.rs` → `Extension("rs")` | match file extension |
| `Glob(&str)` | `**/*.rs` → `Glob("**/*.rs")` | glob pattern (kept whole) |
| `Parts(&[&str])` | — | multiple text search parts, e.g. `["src", "name"]`; a slice to avoid allocation |
| `Text(&str)` | — | single text token (optimized case of `Parts`) |
| `Exclude(&[&str])` | `!test` → `Exclude(&["test"])` | exclusion pattern |
| `PathSegment(&str)` | `/src/` → `PathSegment("src")` | path constraint |
| `FilePath(&str)` | `libswscale/input.c` (AI mode) | matches files whose relative path **ends with this suffix at a `/` boundary** |
| `FileType(&str)` | `type:rust` → `FileType("rust")` | file type constraint |
| `GitStatus(GitStatusFilter)` | `status:modified` | git status constraint |
| `Not(Box<Constraint<'a>>)` | `!extension:rs` → `Not(Extension("rs"))` | negates the inner constraint |

Note the two distinct negation forms: `Exclude` is the bare `!token` pattern (a list of excluded strings), while `Not` wraps a fully parsed inner constraint (`!extension:rs`). Match on both when implementing filtering.

Implements `Clone`, `PartialEq`, `Eq`.

```rust
impl Constraint<'_> {
    pub fn is_filename_constraint_token(token: &str) -> bool;
}
```

Associated (non-`self`) predicate — the same test `enable_filename_constraint` gates on. Use it to pre-check whether a token would become a `FilePath` constraint.

---

## `FuzzyQuery<'a>`

```rust
pub enum FuzzyQuery<'a> {
    Parts(Vec<&'a str>),
    Text(&'a str),
    Empty,
}
```

The residual text after constraints are stripped. `Text` is the single-token fast path, `Parts` holds multiple tokens, `Empty` means the query was all constraints (or blank). See `FFFQuery::grep_text` for how these flatten back to a pattern string. Implements `Clone`, `PartialEq`.

## `GitStatusFilter`

```rust
pub enum GitStatusFilter {
    Modified,
    Untracked,
    Staged,
    Unmodified,
}
```

Payload of `Constraint::GitStatus`, parsed from `status:<value>`. `Copy`, `PartialEq`, `Eq`.

## `Location`

```rust
pub enum Location {
    Line(i32),
    Range { start: (i32, i32), end: (i32, i32) },
    Position { line: i32, col: i32 },
}
```

Populated into `FFFQuery::location` from a trailing `file:line[:col]` suffix, gated by `ParserConfig::enable_location`. `Line(12)` for `file:12`; `Position { line: 12, col: 4 }` for `file:12:4`; `Range` carries `(line, col)` tuples for both endpoints. Coordinates are `i32`, not `usize`. `Copy`, `PartialEq`, `Eq`.

## `ConstraintVec<'a>`

```rust
pub type ConstraintVec<'a> = Vec<Constraint<'a>>;
```

The collection of parsed constraints produced by `QueryParser::parse`. `FFFQuery::constraints` has this type.
