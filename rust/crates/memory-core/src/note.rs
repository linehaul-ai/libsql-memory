//! Note type: YAML frontmatter + markdown body (spec 01).

use serde::{Deserialize, Serialize};
use time::format_description::FormatItem;
use time::macros::format_description;
use time::Date;

use crate::error::{Error, Result};

/// Compile-time `YYYY-MM-DD` format for note dates.
const DATE_FORMAT: &[FormatItem<'static>] = format_description!("[year]-[month]-[day]");

/// Kind of memory; drives retrieval boosts and doctor policies later.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum NoteType {
    /// Discrete factual knowledge.
    Fact,
    /// A choice that was made (and usually why).
    Decision,
    /// User or project preference.
    Preference,
    /// Hard-won learning / post-mortem style note.
    Lesson,
    /// Pointer to external material.
    Reference,
    /// Condensed session recap (decays faster under lifecycle rules).
    SessionSummary,
}

impl NoteType {
    /// All allowed wire values, for error messages.
    pub const ALL: &'static [&'static str] = &[
        "fact",
        "decision",
        "preference",
        "lesson",
        "reference",
        "session-summary",
    ];
}

/// YAML frontmatter fields for a memory note.
///
/// Unknown YAML keys are ignored so hand-edited Obsidian notes remain readable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NoteFrontmatter {
    /// Human-readable one-line summary.
    pub title: String,
    /// Synonyms / alternate phrasings — the semantic-search replacement (min 2).
    pub aliases: Vec<String>,
    /// Categorical labels; omit or empty when unused.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
    /// What kind of memory this is.
    #[serde(rename = "type")]
    pub note_type: NoteType,
    /// Creation date (`YYYY-MM-DD`).
    #[serde(with = "date_serde")]
    pub created: Date,
    /// Last content update (`YYYY-MM-DD`). Never bumped on read.
    #[serde(with = "date_serde")]
    pub updated: Date,
    /// Optional TTL; doctor archives after this date.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "optional_date_serde"
    )]
    pub expires: Option<Date>,
    /// Provenance (session id, conversation ref, or `"manual"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

impl NoteFrontmatter {
    /// Enforce required fields and domain rules from spec 01.
    pub fn validate(&self) -> Result<()> {
        if self.title.trim().is_empty() {
            return Err(Error::validation("title", "must be non-empty"));
        }

        let aliases: Vec<&str> = self
            .aliases
            .iter()
            .map(|a| a.trim())
            .filter(|a| !a.is_empty())
            .collect();

        if aliases.len() < 2 {
            return Err(Error::validation(
                "aliases",
                "require at least 2 non-empty aliases (semantic-search replacement; aim 3–6)",
            ));
        }

        if self.aliases.iter().any(|a| a.trim().is_empty()) {
            return Err(Error::validation(
                "aliases",
                "must not contain empty strings",
            ));
        }

        if let Some(src) = &self.source {
            if src.trim().is_empty() {
                return Err(Error::validation(
                    "source",
                    "when present, must be non-empty",
                ));
            }
        }

        Ok(())
    }
}

/// One memory: frontmatter + markdown body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Note {
    /// Structured metadata (YAML block).
    pub frontmatter: NoteFrontmatter,
    /// Markdown prose after the closing `---`. Lead with the fact itself.
    pub body: String,
}

impl Note {
    /// Parse a full note file (frontmatter + body).
    ///
    /// Expects Obsidian-style delimiters: a leading `---` line, YAML, a closing
    /// `---` line, then the body.
    /// Parse frontmatter + body and enforce domain validation (store / normal reads).
    pub fn parse(text: &str) -> Result<Self> {
        let note = Self::parse_lenient(text)?;
        note.validate()?;
        Ok(note)
    }

    /// Parse without domain validation — for doctor/hand-edited notes that may
    /// violate alias rules but still have readable frontmatter.
    pub fn parse_lenient(text: &str) -> Result<Self> {
        let (yaml, body) = split_frontmatter(text)?;
        let frontmatter: NoteFrontmatter = serde_yaml::from_str(yaml)?;
        Ok(Self {
            frontmatter,
            body: body.to_string(),
        })
    }

    /// Serialize to a markdown file with YAML frontmatter.
    pub fn to_markdown(&self) -> Result<String> {
        self.frontmatter.validate()?;
        let yaml = serde_yaml::to_string(&self.frontmatter)?;
        // serde_yaml adds a trailing newline; keep a single blank line after ---
        let yaml = yaml.trim_end();
        let mut out = String::with_capacity(yaml.len() + self.body.len() + 16);
        out.push_str("---\n");
        out.push_str(yaml);
        out.push('\n');
        out.push_str("---\n");
        if !self.body.is_empty() {
            // Body as stored (may or may not end with newline)
            out.push_str(&self.body);
            if !self.body.ends_with('\n') {
                out.push('\n');
            }
        }
        Ok(out)
    }

    /// Re-run frontmatter validation (e.g. after mutating fields in memory).
    pub fn validate(&self) -> Result<()> {
        self.frontmatter.validate()
    }
}

/// Split `---\nyaml\n---\nbody` into `(yaml, body)`.
fn split_frontmatter(text: &str) -> Result<(&str, &str)> {
    // Accept optional UTF-8 BOM
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);

    let rest = if let Some(r) = text.strip_prefix("---\n") {
        r
    } else if let Some(r) = text.strip_prefix("---\r\n") {
        r
    } else {
        return Err(Error::MissingFrontmatter);
    };

    // Find closing --- on its own line
    let mut search_from = 0;
    loop {
        let slice = &rest[search_from..];
        let Some(rel) = slice.find("---") else {
            return Err(Error::MissingFrontmatter);
        };
        let abs = search_from + rel;

        // Must be at line start
        let at_line_start = abs == 0 || rest.as_bytes().get(abs - 1) == Some(&b'\n');
        if !at_line_start {
            search_from = abs + 3;
            continue;
        }

        // After --- must be EOL or EOF (optional \r)
        let after = abs + 3;
        let ok_end = match rest.as_bytes().get(after) {
            None => true,
            Some(b'\n') => true,
            Some(b'\r') => rest.as_bytes().get(after + 1) == Some(&b'\n'),
            _ => false,
        };
        if !ok_end {
            search_from = abs + 3;
            continue;
        }

        let yaml = &rest[..abs];
        let yaml = yaml.strip_suffix('\n').unwrap_or(yaml);
        let yaml = yaml.strip_suffix('\r').unwrap_or(yaml);

        let body = if rest.as_bytes().get(after) == Some(&b'\r') {
            rest.get(after + 2..).unwrap_or("")
        } else if rest.as_bytes().get(after) == Some(&b'\n') {
            rest.get(after + 1..).unwrap_or("")
        } else {
            ""
        };

        return Ok((yaml, body));
    }
}

/// Serde helpers for required `Date` fields as `YYYY-MM-DD`.
mod date_serde {
    use super::{Date, DATE_FORMAT};
    use serde::{self, Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(date: &Date, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let s = date
            .format(DATE_FORMAT)
            .map_err(serde::ser::Error::custom)?;
        serializer.serialize_str(&s)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Date, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Date::parse(&s, DATE_FORMAT).map_err(|_| {
            serde::de::Error::custom(format!("invalid date {s:?}, expected YYYY-MM-DD"))
        })
    }
}

/// Serde helpers for optional `Date` fields.
mod optional_date_serde {
    use super::{Date, DATE_FORMAT};
    use serde::{self, Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(date: &Option<Date>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match date {
            Some(d) => {
                let s = d.format(DATE_FORMAT).map_err(serde::ser::Error::custom)?;
                serializer.serialize_some(&s)
            }
            None => serializer.serialize_none(),
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<Date>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let opt = Option::<String>::deserialize(deserializer)?;
        match opt {
            None => Ok(None),
            Some(s) if s.trim().is_empty() => Ok(None),
            Some(s) => Date::parse(&s, DATE_FORMAT).map(Some).map_err(|_| {
                serde::de::Error::custom(format!("invalid date {s:?}, expected YYYY-MM-DD"))
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use time::Month;

    const SPEC_EXAMPLE: &str = r#"---
title: Deploy pipeline uses GitHub Actions + Docker
aliases: [release process, CI/CD, shipping, how we deploy]
tags: [infra, deployment]
type: fact
created: 2026-08-08
updated: 2026-08-08
source: session e32af3d7
---
The deployment pipeline builds in GitHub Actions and publishes Docker images.
Rollbacks are re-deploys of the previous tag.
Related: [[docker-registry-auth]]
"#;

    fn date(y: i32, m: u8, d: u8) -> Date {
        Date::from_calendar_date(y, Month::try_from(m).unwrap(), d).unwrap()
    }

    #[test]
    fn parse_spec_example_round_trip() {
        let note = Note::parse(SPEC_EXAMPLE).unwrap();
        assert_eq!(
            note.frontmatter.title,
            "Deploy pipeline uses GitHub Actions + Docker"
        );
        assert_eq!(
            note.frontmatter.aliases,
            vec!["release process", "CI/CD", "shipping", "how we deploy"]
        );
        assert_eq!(note.frontmatter.tags, vec!["infra", "deployment"]);
        assert_eq!(note.frontmatter.note_type, NoteType::Fact);
        assert_eq!(note.frontmatter.created, date(2026, 8, 8));
        assert_eq!(note.frontmatter.updated, date(2026, 8, 8));
        assert_eq!(note.frontmatter.expires, None);
        assert_eq!(note.frontmatter.source.as_deref(), Some("session e32af3d7"));
        assert!(note.body.starts_with("The deployment pipeline"));
        assert!(note.body.contains("[[docker-registry-auth]]"));

        let md = note.to_markdown().unwrap();
        let again = Note::parse(&md).unwrap();
        assert_eq!(again.frontmatter, note.frontmatter);
        assert_eq!(again.body.trim(), note.body.trim());
    }

    #[test]
    fn aliases_min_two() {
        let one = r#"---
title: Only one alias
aliases: [solo]
type: fact
created: 2026-08-08
updated: 2026-08-08
---
body
"#;
        let err = Note::parse(one).unwrap_err();
        assert!(matches!(err, Error::Validation { ref field, .. } if field == "aliases"));

        let zero = r#"---
title: No aliases
aliases: []
type: fact
created: 2026-08-08
updated: 2026-08-08
---
body
"#;
        assert!(Note::parse(zero).is_err());
    }

    #[test]
    fn empty_title_rejected() {
        let text = r#"---
title: "   "
aliases: [a, b]
type: fact
created: 2026-08-08
updated: 2026-08-08
---
x
"#;
        assert!(Note::parse(text).is_err());
    }

    #[test]
    fn note_type_all_variants() {
        for (wire, expected) in [
            ("fact", NoteType::Fact),
            ("decision", NoteType::Decision),
            ("preference", NoteType::Preference),
            ("lesson", NoteType::Lesson),
            ("reference", NoteType::Reference),
            ("session-summary", NoteType::SessionSummary),
        ] {
            let text = format!(
                r#"---
title: T
aliases: [one, two]
type: {wire}
created: 2026-01-01
updated: 2026-01-01
---
body
"#
            );
            let note = Note::parse(&text).unwrap();
            assert_eq!(note.frontmatter.note_type, expected);
            let md = note.to_markdown().unwrap();
            assert!(md.contains(&format!("type: {wire}")));
        }
    }

    #[test]
    fn note_type_rejects_unknown() {
        let text = r#"---
title: T
aliases: [one, two]
type: foo
created: 2026-01-01
updated: 2026-01-01
---
body
"#;
        assert!(Note::parse(text).is_err());
    }

    #[test]
    fn dates_yyyy_mm_dd() {
        let bad = r#"---
title: T
aliases: [one, two]
type: fact
created: 08/08/2026
updated: 2026-08-08
---
body
"#;
        assert!(Note::parse(bad).is_err());
    }

    #[test]
    fn optional_fields_absent() {
        let text = r#"---
title: Minimal
aliases: [alpha, beta]
type: lesson
created: 2026-08-08
updated: 2026-08-09
---
The fact comes first.
"#;
        let note = Note::parse(text).unwrap();
        assert!(note.frontmatter.tags.is_empty());
        assert_eq!(note.frontmatter.expires, None);
        assert_eq!(note.frontmatter.source, None);
        assert_eq!(note.frontmatter.note_type, NoteType::Lesson);
        assert_eq!(note.body, "The fact comes first.\n");
    }

    #[test]
    fn expires_round_trip() {
        let text = r#"---
title: Temporary
aliases: [temp, short-lived]
type: fact
created: 2026-08-08
updated: 2026-08-08
expires: 2026-12-31
---
Expires at year end.
"#;
        let note = Note::parse(text).unwrap();
        assert_eq!(note.frontmatter.expires, Some(date(2026, 12, 31)));
        let again = Note::parse(&note.to_markdown().unwrap()).unwrap();
        assert_eq!(again.frontmatter.expires, Some(date(2026, 12, 31)));
    }

    #[test]
    fn missing_frontmatter() {
        assert!(matches!(
            Note::parse("just prose"),
            Err(Error::MissingFrontmatter)
        ));
    }

    #[test]
    fn leading_fact_body_preserved() {
        let text = r#"---
title: T
aliases: [a, b]
type: fact
created: 2026-08-08
updated: 2026-08-08
---
First sentence is the fact.

Why: because snippets show the top.
"#;
        let note = Note::parse(text).unwrap();
        assert!(note.body.starts_with("First sentence is the fact."));
        assert!(note.body.contains("Why: because snippets"));
    }

    #[test]
    fn empty_body_allowed() {
        let text = r#"---
title: T
aliases: [a, b]
type: fact
created: 2026-08-08
updated: 2026-08-08
---
"#;
        let note = Note::parse(text).unwrap();
        assert!(note.body.is_empty() || note.body == "\n" || note.body.trim().is_empty());
    }
}
