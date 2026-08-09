//! Append-only access log (spec 05) — authority for ranking reinforcement and decay.
//!
//! Path: `{root}/.index/access.jsonl` lines `{ ts, handle, via }`.
//! Compaction rolls events older than 90 days into `{root}/.index/access_counts.json`.

use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use time::OffsetDateTime;

use crate::error::{Error, Result};

/// Default age after which doctor compaction rolls events into counters.
pub const COMPACT_MAX_AGE_DAYS: i64 = 90;

/// How the note was accessed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
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

/// Rolled totals for events removed from the JSONL by compaction.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AccessCounters {
    /// Map handle → per-via counts.
    #[serde(default)]
    pub by_handle: HashMap<String, ViaCounts>,
}

/// Per-via access counts for one handle.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ViaCounts {
    /// Search-hit total from compacted events.
    #[serde(default)]
    pub search_hit: u64,
    /// Read total from compacted events.
    #[serde(default)]
    pub read: u64,
}

impl ViaCounts {
    fn add(&mut self, via: AccessVia) {
        match via {
            AccessVia::SearchHit => self.search_hit += 1,
            AccessVia::Read => self.read += 1,
        }
    }

    /// Total accesses of any kind.
    pub fn total(&self) -> u64 {
        self.search_hit + self.read
    }
}

/// Result of [`AccessLog::compact`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CompactionStats {
    /// Events still in the JSONL after compact.
    pub events_kept: usize,
    /// Events rolled into counters.
    pub events_rolled: usize,
    /// Distinct handles whose counters changed.
    pub counters_updated: usize,
}

/// Append-only JSONL access log under the memory root.
#[derive(Debug, Clone)]
pub struct AccessLog {
    path: PathBuf,
    counts_path: PathBuf,
}

impl AccessLog {
    /// Log paths are `{root}/.index/access.jsonl` and `access_counts.json`.
    pub fn open(root: impl AsRef<Path>) -> Self {
        let index = root.as_ref().join(".index");
        Self {
            path: index.join("access.jsonl"),
            counts_path: index.join("access_counts.json"),
        }
    }

    /// Path of the JSONL log file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Path of the rolled counters file.
    pub fn counts_path(&self) -> &Path {
        &self.counts_path
    }

    /// Append one event at the current UTC time (creates parent dirs).
    pub fn append(&self, handle: &str, via: AccessVia) -> Result<()> {
        let ts = OffsetDateTime::now_utc()
            .format(&time::format_description::well_known::Rfc3339)
            .unwrap_or_else(|_| "1970-01-01T00:00:00Z".into());
        self.append_event(AccessEvent {
            ts,
            handle: handle.to_string(),
            via,
        })
    }

    /// Append a fully specified event (used by tests and compaction rewrites).
    pub fn append_event(&self, event: AccessEvent) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;
        }
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

    /// Count JSONL events for `handle` with `ts` within the last `days` days.
    ///
    /// Does not include compacted counters (those are older than the compact window).
    pub fn count_recent(&self, handle: &str, days: i64) -> u32 {
        let events = self.read_all().unwrap_or_default();
        let cutoff = OffsetDateTime::now_utc() - time::Duration::days(days);
        events
            .iter()
            .filter(|e| e.handle == handle)
            .filter(|e| parse_ts(&e.ts).is_some_and(|t| t >= cutoff))
            .count() as u32
    }

    /// Whether this handle has any recorded access (recent JSONL **or** compacted counters).
    pub fn ever_accessed(&self, handle: &str) -> bool {
        if self
            .read_all()
            .unwrap_or_default()
            .iter()
            .any(|e| e.handle == handle)
        {
            return true;
        }
        self.load_counters()
            .ok()
            .and_then(|c| c.by_handle.get(handle).map(|v| v.total() > 0))
            .unwrap_or(false)
    }

    /// True when there is at least one JSONL event for `handle` with `ts` within `days`.
    pub fn accessed_within(&self, handle: &str, days: i64) -> bool {
        self.count_recent(handle, days) > 0
    }

    /// Total JSONL event count (for stats).
    pub fn len(&self) -> usize {
        self.read_all().map(|v| v.len()).unwrap_or(0)
    }

    /// Whether the log file is missing or empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Read all JSONL events; missing file → empty.
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

    /// Load compacted counters; missing file → empty.
    pub fn load_counters(&self) -> Result<AccessCounters> {
        if !self.counts_path.is_file() {
            return Ok(AccessCounters::default());
        }
        let text =
            fs::read_to_string(&self.counts_path).map_err(|e| Error::io(&self.counts_path, e))?;
        if text.trim().is_empty() {
            return Ok(AccessCounters::default());
        }
        serde_json::from_str(&text)
            .map_err(|e| Error::validation("access_counts", format!("parse counters: {e}")))
    }

    /// Roll events older than `max_age_days` into counters; rewrite JSONL with kept events.
    ///
    /// Idempotent when run again with no new old events.
    pub fn compact(&self, max_age_days: i64) -> Result<CompactionStats> {
        let events = self.read_all()?;
        let cutoff = OffsetDateTime::now_utc() - time::Duration::days(max_age_days);
        let mut counters = self.load_counters()?;
        let mut kept = Vec::new();
        let mut rolled = 0usize;
        let mut touched: HashMap<String, ()> = HashMap::new();

        for ev in events {
            let old = parse_ts(&ev.ts).is_some_and(|t| t < cutoff);
            if old {
                counters
                    .by_handle
                    .entry(ev.handle.clone())
                    .or_default()
                    .add(ev.via);
                touched.insert(ev.handle.clone(), ());
                rolled += 1;
            } else {
                kept.push(ev);
            }
        }

        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;
        }

        // Rewrite JSONL atomically via temp + rename in the index dir.
        let dir = self.path.parent().unwrap_or_else(|| Path::new("."));
        let mut tmp = tempfile::Builder::new()
            .prefix(".access-")
            .suffix(".tmp")
            .tempfile_in(dir)
            .map_err(|e| Error::io(dir, e))?;
        for ev in &kept {
            let line = serde_json::to_string(ev)
                .map_err(|e| Error::validation("access_log", format!("serialize: {e}")))?;
            writeln!(tmp, "{line}").map_err(|e| Error::io(tmp.path(), e))?;
        }
        tmp.as_file()
            .sync_all()
            .map_err(|e| Error::io(tmp.path(), e))?;
        tmp.persist(&self.path).map_err(|e| {
            Error::io(
                &self.path,
                std::io::Error::new(e.error.kind(), e.error.to_string()),
            )
        })?;

        let counts_json = serde_json::to_string_pretty(&counters)
            .map_err(|e| Error::validation("access_counts", format!("serialize: {e}")))?;
        crate::store::atomic_write(&self.counts_path, counts_json.as_bytes())?;

        Ok(CompactionStats {
            events_kept: kept.len(),
            events_rolled: rolled,
            counters_updated: touched.len(),
        })
    }
}

fn parse_ts(s: &str) -> Option<OffsetDateTime> {
    OffsetDateTime::parse(s, &time::format_description::well_known::Rfc3339).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use time::format_description::well_known::Rfc3339;

    fn ts_days_ago(days: i64) -> String {
        (OffsetDateTime::now_utc() - time::Duration::days(days))
            .format(&Rfc3339)
            .unwrap()
    }

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
        assert!(!log.ever_accessed("x"));
    }

    #[test]
    fn compact_rolls_old_keeps_young() {
        let dir = tempdir().unwrap();
        let log = AccessLog::open(dir.path());
        log.append_event(AccessEvent {
            ts: ts_days_ago(100),
            handle: "ns/old".into(),
            via: AccessVia::Read,
        })
        .unwrap();
        log.append_event(AccessEvent {
            ts: ts_days_ago(100),
            handle: "ns/old".into(),
            via: AccessVia::SearchHit,
        })
        .unwrap();
        log.append_event(AccessEvent {
            ts: ts_days_ago(5),
            handle: "ns/new".into(),
            via: AccessVia::Read,
        })
        .unwrap();

        let stats = log.compact(COMPACT_MAX_AGE_DAYS).unwrap();
        assert_eq!(stats.events_rolled, 2);
        assert_eq!(stats.events_kept, 1);
        assert_eq!(stats.counters_updated, 1);
        assert_eq!(log.len(), 1);
        assert_eq!(log.read_all().unwrap()[0].handle, "ns/new");

        let counters = log.load_counters().unwrap();
        let c = counters.by_handle.get("ns/old").unwrap();
        assert_eq!(c.read, 1);
        assert_eq!(c.search_hit, 1);
        assert!(log.ever_accessed("ns/old"));
        assert!(log.ever_accessed("ns/new"));
        assert!(!log.accessed_within("ns/old", 90));
        assert!(log.accessed_within("ns/new", 90));
    }

    #[test]
    fn compact_idempotent() {
        let dir = tempdir().unwrap();
        let log = AccessLog::open(dir.path());
        log.append_event(AccessEvent {
            ts: ts_days_ago(120),
            handle: "a/b".into(),
            via: AccessVia::Read,
        })
        .unwrap();
        let s1 = log.compact(90).unwrap();
        assert_eq!(s1.events_rolled, 1);
        let s2 = log.compact(90).unwrap();
        assert_eq!(s2.events_rolled, 0);
        assert_eq!(s2.events_kept, 0);
        assert_eq!(log.load_counters().unwrap().by_handle["a/b"].read, 1);
    }
}
