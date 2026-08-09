//! Retrieval pipeline steps 4–6: merge, rank, budget (spec 03).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use memory_core::NoteType;
use serde::{Deserialize, Serialize};

/// Default `limit` for search results.
pub const SEARCH_LIMIT_DEFAULT: usize = 8;
/// Default token budget in bytes.
pub const BUDGET_BYTES_DEFAULT: usize = 4096;
/// Hard cap on `budget_bytes`.
pub const BUDGET_BYTES_MAX: usize = 16384;
/// If merged hits stay below this after plain grep, escalate to fuzzy.
pub const MIN_RESULTS_FOR_FUZZY: usize = 3;
/// Bonus when the same note matched path-find and content-grep.
pub const BOTH_STAGES_BONUS: f32 = 0.15;

/// Which pipeline stage produced a hit component.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MatchStage {
    /// Path/slug fuzzy find.
    FindFiles,
    /// Plain content grep.
    GrepPlain,
    /// Fuzzy content grep (escalation).
    GrepFuzzy,
}

impl MatchStage {
    /// Stable wire name for `stages_run`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::FindFiles => "find_files",
            Self::GrepPlain => "grep_plain",
            Self::GrepFuzzy => "grep_fuzzy",
        }
    }
}

/// Stage name for empty-result reporting (includes planned stages not run).
pub type StageName = &'static str;

/// Intermediate hit after merge (path-keyed).
#[derive(Debug, Clone, PartialEq)]
pub struct MergedHit {
    /// Path relative to memory root.
    pub path: PathBuf,
    /// Max component score plus optional both-stages bonus.
    pub match_score: f32,
    /// Stages that contributed.
    pub stages: Vec<MatchStage>,
    /// Best snippet (prefer content over empty).
    pub snippet: String,
    /// Compact why fragments before final join.
    pub why_parts: Vec<String>,
}

/// Ranked hit before budget shaping.
#[derive(Debug, Clone, PartialEq)]
pub struct RankedHit {
    /// Path relative to memory root.
    pub path: PathBuf,
    /// Final score after type boost and frecency.
    pub score: f32,
    /// Snippet text.
    pub snippet: String,
    /// Explainable why line.
    pub why: String,
    /// Title when known (filled by service from disk).
    pub title: String,
    /// Handle `namespace/slug` or `slug`.
    pub handle: String,
}

/// Full result row under budget.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchHit {
    /// `namespace/slug` for `memory_read`.
    pub handle: String,
    /// Note title.
    pub title: String,
    /// Matching line(s), not full body.
    pub snippet: String,
    /// Compact explanation of the match.
    pub why: String,
    /// Final ranking score.
    pub score: f32,
}

/// Overflow row: handle + title only.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MoreHit {
    /// Handle for follow-up read.
    pub handle: String,
    /// Note title.
    pub title: String,
}

/// Budgeted search output (spec 03 / 04).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BudgetedSearch {
    /// Snippet-full results that fit the budget.
    pub results: Vec<SearchHit>,
    /// Truncated overflow (handle + title).
    pub more: Vec<MoreHit>,
    /// Stages actually executed.
    pub stages_run: Vec<String>,
    /// Namespace scope used (`""` = whole root).
    pub scope: String,
    /// Present when `results` is empty — never silent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub empty_hint: Option<String>,
}

/// Merge path and content hits; dedupe by logical path; prefer active over archived.
pub fn merge_hits(
    file_hits: &[(PathBuf, f32)],
    plain_hits: &[(PathBuf, f32, String)],
    fuzzy_hits: &[(PathBuf, f32, String)],
) -> Vec<MergedHit> {
    let mut map: HashMap<PathBuf, MergedHit> = HashMap::new();

    for (path, score) in file_hits {
        upsert(
            &mut map,
            path.clone(),
            *score,
            MatchStage::FindFiles,
            String::new(),
            why_from_path(path),
        );
    }
    for (path, score, snippet) in plain_hits {
        upsert(
            &mut map,
            path.clone(),
            *score,
            MatchStage::GrepPlain,
            snippet.clone(),
            why_from_snippet(snippet),
        );
    }
    for (path, score, snippet) in fuzzy_hits {
        upsert(
            &mut map,
            path.clone(),
            *score,
            MatchStage::GrepFuzzy,
            snippet.clone(),
            why_from_snippet(snippet),
        );
    }

    let mut out: Vec<MergedHit> = map.into_values().collect();
    out.sort_by(|a, b| {
        b.match_score
            .partial_cmp(&a.match_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.path.cmp(&b.path))
    });
    out
}

fn upsert(
    map: &mut HashMap<PathBuf, MergedHit>,
    path: PathBuf,
    score: f32,
    stage: MatchStage,
    snippet: String,
    why: String,
) {
    use std::collections::hash_map::Entry;
    let logical = logical_path(&path);
    match map.entry(logical) {
        Entry::Occupied(mut e) => {
            let hit = e.get_mut();
            let hit_archived = is_archived(&hit.path);
            let incoming_archived = is_archived(&path);
            if !hit_archived && incoming_archived {
                return;
            }
            if hit_archived && !incoming_archived {
                *hit = MergedHit {
                    path,
                    match_score: score,
                    stages: vec![stage],
                    snippet,
                    why_parts: if why.is_empty() { vec![] } else { vec![why] },
                };
                return;
            }
            let had_file = hit.stages.contains(&MatchStage::FindFiles);
            let had_grep = hit
                .stages
                .iter()
                .any(|s| matches!(s, MatchStage::GrepPlain | MatchStage::GrepFuzzy));
            let is_file = stage == MatchStage::FindFiles;
            let is_grep = matches!(stage, MatchStage::GrepPlain | MatchStage::GrepFuzzy);

            if !hit.stages.contains(&stage) {
                hit.stages.push(stage);
            }
            hit.match_score = hit.match_score.max(score);
            if (had_file && is_grep) || (had_grep && is_file) {
                // Apply bonus once when both stage families are present.
                if !(had_file && had_grep) {
                    hit.match_score += BOTH_STAGES_BONUS;
                }
            }
            if hit.snippet.is_empty() && !snippet.is_empty() {
                hit.snippet = snippet;
            } else if snippet.len() > hit.snippet.len() && !snippet.is_empty() {
                // Prefer richer snippet
                hit.snippet = snippet;
            }
            if !why.is_empty() && !hit.why_parts.iter().any(|p| p == &why) {
                hit.why_parts.push(why);
            }
        }
        Entry::Vacant(e) => {
            let mut why_parts = Vec::new();
            if !why.is_empty() {
                why_parts.push(why);
            }
            e.insert(MergedHit {
                path,
                match_score: score,
                stages: vec![stage],
                snippet,
                why_parts,
            });
        }
    }
}

fn is_archived(path: &Path) -> bool {
    path.starts_with(".archive")
}

fn logical_path(path: &Path) -> PathBuf {
    path.strip_prefix(".archive").unwrap_or(path).to_path_buf()
}

fn why_from_path(path: &Path) -> String {
    let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("path");
    format!("path:{name}")
}

fn why_from_snippet(snippet: &str) -> String {
    let lower = snippet.to_ascii_lowercase();
    if lower.contains("aliases:") || lower.trim_start().starts_with("aliases") {
        // Pull a short token after aliases if present
        if let Some(rest) = snippet.split(':').nth(1) {
            let token = rest
                .trim()
                .trim_start_matches('[')
                .split([',', ']', '\n'])
                .next()
                .unwrap_or("")
                .trim();
            if !token.is_empty() {
                return format!("alias:{token}");
            }
        }
        return "alias".into();
    }
    if lower.contains("title:") {
        return "title".into();
    }
    if lower.contains("tags:") {
        return "tags".into();
    }
    "content".into()
}

/// Static type boosts (spec 03).
pub fn type_boost(note_type: NoteType) -> f32 {
    match note_type {
        NoteType::Preference => 1.15,
        NoteType::Decision => 1.10,
        NoteType::SessionSummary => 0.90,
        NoteType::Fact | NoteType::Lesson | NoteType::Reference => 1.0,
    }
}

/// Frecency multiplier from weighted access counts in the ranking window.
///
/// One read reinforces as much as four search-result hits; growth is capped.
pub fn frecency_multiplier(read_count_30d: u32, search_hit_count_30d: u32) -> f32 {
    let weighted_access = read_count_30d as f32 + search_hit_count_30d as f32 * 0.25;
    if weighted_access == 0.0 {
        return 1.0;
    }
    1.0 + (weighted_access * 0.05).min(0.5)
}

/// Apply type + frecency multipliers; fill handle/title from caller-supplied meta.
pub fn rank_hits(merged: Vec<MergedHit>, meta: &HashMap<PathBuf, NoteMeta>) -> Vec<RankedHit> {
    let mut ranked: Vec<RankedHit> = merged
        .into_iter()
        .map(|m| {
            let (title, note_type, read_count, search_hit_count, handle) = meta
                .get(&m.path)
                .map(|n| {
                    (
                        n.title.clone(),
                        n.note_type,
                        n.read_count_30d,
                        n.search_hit_count_30d,
                        n.handle.clone(),
                    )
                })
                .unwrap_or_else(|| {
                    let handle = path_to_handle(&m.path);
                    (handle.clone(), NoteType::Fact, 0, 0, handle)
                });

            let mult = type_boost(note_type) * frecency_multiplier(read_count, search_hit_count);
            let score = m.match_score * mult;

            let mut why = m.why_parts.join(" +");
            if why.is_empty() {
                why = m
                    .stages
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join("+");
            }
            if read_count > 0 || search_hit_count > 0 {
                why.push_str(&format!(
                    " +frecency({read_count}r+{search_hit_count}s/30d)"
                ));
            }

            let snippet = if m.snippet.is_empty() {
                title.clone()
            } else {
                m.snippet
            };

            RankedHit {
                path: m.path,
                score,
                snippet,
                why,
                title,
                handle,
            }
        })
        .collect();

    ranked.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.handle.cmp(&b.handle))
    });
    ranked
}

/// Note fields needed for ranking and output (loaded from disk by the service).
#[derive(Debug, Clone)]
pub struct NoteMeta {
    /// Display title.
    pub title: String,
    /// Note kind for type boost.
    pub note_type: NoteType,
    /// Full reads in the last 30 days (from access log).
    pub read_count_30d: u32,
    /// Search-result hits in the last 30 days (from access log).
    pub search_hit_count_30d: u32,
    /// `namespace/slug` handle.
    pub handle: String,
}

/// Convert `ns/slug.md` → `ns/slug`.
pub fn path_to_handle(path: &Path) -> String {
    let logical = logical_path(path);
    let s = logical.to_string_lossy().replace('\\', "/");
    s.strip_suffix(".md").unwrap_or(&s).to_string()
}

/// Shape ranked hits under limit and byte budget; empty results get a hint.
pub fn apply_budget(
    ranked: Vec<RankedHit>,
    limit: usize,
    budget_bytes: usize,
    stages_run: &[MatchStage],
    scope: &str,
) -> BudgetedSearch {
    let budget_bytes = budget_bytes.min(BUDGET_BYTES_MAX);
    let limit = if limit == 0 {
        SEARCH_LIMIT_DEFAULT
    } else {
        limit
    };

    let stages_run: Vec<String> = stages_run.iter().map(|s| s.as_str().to_string()).collect();
    let scope = scope.to_string();

    if ranked.is_empty() {
        let stages_label = if stages_run.is_empty() {
            "(none)".to_string()
        } else {
            stages_run.join(", ")
        };
        let scope_label = if scope.is_empty() {
            "all"
        } else {
            scope.as_str()
        };
        return BudgetedSearch {
            results: vec![],
            more: vec![],
            stages_run,
            scope: scope.clone(),
            empty_hint: Some(format!(
                "no matches; stages tried: {stages_label}; scope={scope_label:?}; try broader terms or namespace=\"\""
            )),
        };
    }

    let mut results = Vec::new();
    let mut more = Vec::new();
    let mut used = 0usize;

    for (i, hit) in ranked.into_iter().enumerate() {
        if i >= limit {
            more.push(MoreHit {
                handle: hit.handle,
                title: hit.title,
            });
            continue;
        }

        let candidate = SearchHit {
            handle: hit.handle.clone(),
            title: hit.title.clone(),
            snippet: hit.snippet.clone(),
            why: hit.why.clone(),
            score: hit.score,
        };
        let size = estimate_bytes(&candidate);

        if !results.is_empty() && used + size > budget_bytes {
            more.push(MoreHit {
                handle: hit.handle,
                title: hit.title,
            });
            continue;
        }

        used += size;
        results.push(candidate);
    }

    BudgetedSearch {
        results,
        more,
        stages_run,
        scope,
        empty_hint: None,
    }
}

fn estimate_bytes(hit: &SearchHit) -> usize {
    // Approximate JSON-ish payload size for budgeting.
    hit.handle.len() + hit.title.len() + hit.snippet.len() + hit.why.len() + 32
}

#[cfg(test)]
mod tests {
    use super::*;
    use memory_core::NoteType;

    #[test]
    fn merge_dedupes_and_applies_both_stages_bonus() {
        let file = vec![(PathBuf::from("proj/a.md"), 1.0)];
        let plain = vec![(PathBuf::from("proj/a.md"), 0.8, "aliases: [ship]".into())];
        let merged = merge_hits(&file, &plain, &[]);
        assert_eq!(merged.len(), 1);
        assert!((merged[0].match_score - (1.0 + BOTH_STAGES_BONUS)).abs() < 1e-5);
        assert!(merged[0].stages.contains(&MatchStage::FindFiles));
        assert!(merged[0].stages.contains(&MatchStage::GrepPlain));
        assert!(merged[0].why_parts.iter().any(|w| w.starts_with("alias:")));
    }

    #[test]
    fn merge_keeps_distinct_paths() {
        let file = vec![(PathBuf::from("a.md"), 1.0), (PathBuf::from("b.md"), 0.5)];
        let merged = merge_hits(&file, &[], &[]);
        assert_eq!(merged.len(), 2);
    }

    #[test]
    fn type_boosts_preference_over_session_summary() {
        assert!(type_boost(NoteType::Preference) > type_boost(NoteType::SessionSummary));
        assert!(type_boost(NoteType::Decision) > type_boost(NoteType::Fact));
    }

    #[test]
    fn rank_orders_by_final_score() {
        let merged = vec![
            MergedHit {
                path: PathBuf::from("low.md"),
                match_score: 1.0,
                stages: vec![MatchStage::FindFiles],
                snippet: "x".into(),
                why_parts: vec!["path:low".into()],
            },
            MergedHit {
                path: PathBuf::from("high.md"),
                match_score: 1.0,
                stages: vec![MatchStage::FindFiles],
                snippet: "y".into(),
                why_parts: vec!["path:high".into()],
            },
        ];
        let mut meta = HashMap::new();
        meta.insert(
            PathBuf::from("low.md"),
            NoteMeta {
                title: "Low".into(),
                note_type: NoteType::SessionSummary,
                read_count_30d: 0,
                search_hit_count_30d: 0,
                handle: "low".into(),
            },
        );
        meta.insert(
            PathBuf::from("high.md"),
            NoteMeta {
                title: "High".into(),
                note_type: NoteType::Preference,
                read_count_30d: 10,
                search_hit_count_30d: 0,
                handle: "high".into(),
            },
        );
        let ranked = rank_hits(merged, &meta);
        assert_eq!(ranked[0].handle, "high");
        assert!(ranked[0].score > ranked[1].score);
        assert!(ranked[0].why.contains("frecency"));
    }

    #[test]
    fn budget_puts_overflow_in_more() {
        let ranked: Vec<RankedHit> = (0..5)
            .map(|i| RankedHit {
                path: PathBuf::from(format!("{i}.md")),
                score: 10.0 - i as f32,
                snippet: "s".into(),
                why: "w".into(),
                title: format!("T{i}"),
                handle: format!("h{i}"),
            })
            .collect();
        let out = apply_budget(
            ranked,
            2,
            BUDGET_BYTES_DEFAULT,
            &[MatchStage::FindFiles],
            "proj",
        );
        assert_eq!(out.results.len(), 2);
        assert_eq!(out.more.len(), 3);
        assert_eq!(out.scope, "proj");
        assert!(out.empty_hint.is_none());
    }

    #[test]
    fn budget_tiny_byte_limit_spills_to_more() {
        let ranked = vec![
            RankedHit {
                path: PathBuf::from("a.md"),
                score: 2.0,
                snippet: "short".into(),
                why: "w".into(),
                title: "A".into(),
                handle: "a".into(),
            },
            RankedHit {
                path: PathBuf::from("b.md"),
                score: 1.0,
                snippet: "x".repeat(200),
                why: "w".into(),
                title: "B".into(),
                handle: "b".into(),
            },
        ];
        let out = apply_budget(ranked, 8, 80, &[MatchStage::GrepPlain], "");
        assert_eq!(out.results.len(), 1);
        assert_eq!(out.more.len(), 1);
        assert_eq!(out.more[0].handle, "b");
    }

    #[test]
    fn empty_results_include_hint_and_stages() {
        let out = apply_budget(
            vec![],
            8,
            4096,
            &[MatchStage::FindFiles, MatchStage::GrepPlain],
            "",
        );
        assert!(out.results.is_empty());
        let hint = out.empty_hint.expect("hint required");
        assert!(hint.contains("find_files"));
        assert!(hint.contains("grep_plain"));
        assert!(hint.contains("broader"));
    }

    #[test]
    fn path_to_handle_strips_md() {
        assert_eq!(path_to_handle(Path::new("ns/slug.md")), "ns/slug");
        assert_eq!(path_to_handle(Path::new("slug.md")), "slug");
    }

    #[test]
    fn frecency_is_one_when_zero_accesses() {
        assert!((frecency_multiplier(0, 0) - 1.0).abs() < 1e-6);
        assert!(frecency_multiplier(20, 0) > frecency_multiplier(1, 0));
        assert!((frecency_multiplier(100, 0) - 1.5).abs() < 1e-5);
    }

    #[test]
    fn reads_reinforce_more_than_search_hits() {
        assert!(frecency_multiplier(1, 0) > frecency_multiplier(0, 1));
        assert!((frecency_multiplier(1, 0) - frecency_multiplier(0, 4)).abs() < 1e-6);
    }

    #[test]
    fn why_distinguishes_read_and_search_hit_counts() {
        let path = PathBuf::from("note.md");
        let merged = vec![MergedHit {
            path: path.clone(),
            match_score: 1.0,
            stages: vec![MatchStage::FindFiles],
            snippet: "hit".into(),
            why_parts: vec!["path:note".into()],
        }];
        let meta = HashMap::from([(
            path,
            NoteMeta {
                title: "Note".into(),
                note_type: NoteType::Fact,
                read_count_30d: 3,
                search_hit_count_30d: 8,
                handle: "note".into(),
            },
        )]);

        let ranked = rank_hits(merged, &meta);

        assert!(ranked[0].why.contains("3r+8s/30d"), "{}", ranked[0].why);
        assert!(!ranked[0].why.contains("reads"), "{}", ranked[0].why);
    }
}
