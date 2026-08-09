//! Append-only access log (spec 05) — authority for ranking reinforcement.

use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use time::OffsetDateTime;

use memory_core::{Error, Result};

/// How the note was accessed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AccessVia {
    /// Surfaced in a search result (lighter weight).
    SearchHit,
    /// Full note fetched via `memory_read`.
    Read,
}

/// One JSONL line in `.index/access.jsonl`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AccessEvent {
    /// RFC3339 timestamp.
    pub ts: String,
    /// `namespace/slug` handle.
    pub handle: String,
    /// Channel that caused the access.
    pub via: AccessVia,
}

/// Append-only JSONL access log under the memory root.
#[derive(Debug, Clone)]
pub struct AccessLog {
    path: PathBuf,
}

impl AccessLog {
    /// Log path is `{root}/.index/access.jsonl`.
    pub fn open(root: impl AsRef<Path>) -> Self {
        Self {
            path: root.as_ref().join(".index").join("access.jsonl"),
        }
    }

    /// Path of the log file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Append one event (creates parent dirs). Failures are I/O errors.
    pub fn append(&self, handle: &str, via: AccessVia) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;
        }
        let event = AccessEvent {
            ts: OffsetDateTime::now_utc()
                .format(&time::format_description::well_known::Rfc3339)
                .unwrap_or_else(|_| "1970-01-01T00:00:00Z".into()),
            handle: handle.to_string(),
            via,
        };
        let line = serde_json::to_string(&event)
            .map_err(|e| Error::validation("access_log", format!("serialize event: {e}")))?;
        let mut f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .map_err(|e| Error::io(&self.path, e))?;
        writeln!(f, "{line}").map_err(|e| Error::io(&self.path, e))?;
        Ok(())
    }

    /// Count events for `handle` with `ts` within the last `days` days.
    pub fn count_recent(&self, handle: &str, days: i64) -> u32 {
        let events = self.read_all().unwrap_or_default();
        let cutoff = OffsetDateTime::now_utc() - time::Duration::days(days);
        events
            .iter()
            .filter(|e| e.handle == handle)
            .filter(|e| parse_ts(&e.ts).is_some_and(|t| t >= cutoff))
            .count() as u32
    }

    /// Total event count (for stats).
    pub fn len(&self) -> usize {
        self.read_all().map(|v| v.len()).unwrap_or(0)
    }

    /// Whether the log file is missing or empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Read all events; missing file → empty.
    pub fn read_all(&self) -> Result<Vec<AccessEvent>> {
        if !self.path.is_file() {
            return Ok(vec![]);
        }
        let f = fs::File::open(&self.path).map_err(|e| Error::io(&self.path, e))?;
        let mut out = Vec::new();
        for line in BufReader::new(f).lines() {
            let line = line.map_err(|e| Error::io(&self.path, e))?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Ok(ev) = serde_json::from_str::<AccessEvent>(line) {
                out.push(ev);
            }
        }
        Ok(out)
    }
}

fn parse_ts(s: &str) -> Option<OffsetDateTime> {
    OffsetDateTime::parse(s, &time::format_description::well_known::Rfc3339).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn append_and_count() {
        let dir = tempdir().unwrap();
        let log = AccessLog::open(dir.path());
        log.append("proj/a", AccessVia::Read).unwrap();
        log.append("proj/a", AccessVia::SearchHit).unwrap();
        log.append("proj/b", AccessVia::Read).unwrap();
        assert_eq!(log.count_recent("proj/a", 30), 2);
        assert_eq!(log.count_recent("proj/b", 30), 1);
        assert_eq!(log.count_recent("missing", 30), 0);
        assert_eq!(log.len(), 3);
    }

    #[test]
    fn missing_file_is_empty() {
        let dir = tempdir().unwrap();
        let log = AccessLog::open(dir.path());
        assert!(log.is_empty());
        assert!(log.read_all().unwrap().is_empty());
    }
}
