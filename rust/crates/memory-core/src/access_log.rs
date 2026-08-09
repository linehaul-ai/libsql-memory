//! Append-only access log (spec 05) — authority for ranking reinforcement and decay.
//!
//! Path: `{root}/.index/access.jsonl` lines `{ ts, handle, via }`.
//! Compaction rolls events older than 90 days into `{root}/.index/access_counts.json`.

use std::collections::{HashMap, HashSet};
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use fs2::FileExt;
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
    /// Inclusive timestamp through which source events have already been rolled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compacted_through: Option<String>,
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

/// One coherent, validated view of access history.
#[derive(Debug, Clone, Default)]
pub struct AccessSnapshot {
    events: Vec<SnapshotEvent>,
    ever_accessed: HashSet<String>,
}

#[derive(Debug, Clone)]
struct SnapshotEvent {
    ts: OffsetDateTime,
    event: AccessEvent,
}

impl AccessSnapshot {
    /// Per-via source-event counts for `handle` within the last `days` days.
    pub fn recent_counts(&self, handle: &str, days: i64) -> ViaCounts {
        let cutoff = OffsetDateTime::now_utc() - time::Duration::days(days);
        let mut counts = ViaCounts::default();
        for event in self
            .events
            .iter()
            .filter(|event| event.event.handle == handle && event.ts >= cutoff)
        {
            counts.add(event.event.via);
        }
        counts
    }

    /// Whether `handle` has any source or compacted access event.
    pub fn ever_accessed(&self, handle: &str) -> bool {
        self.ever_accessed.contains(handle)
    }

    /// Whether `handle` has a source event within the last `days` days.
    pub fn accessed_within(&self, handle: &str, days: i64) -> bool {
        self.recent_counts(handle, days).total() > 0
    }

    /// Effective JSONL event count after ignoring already-compacted source events.
    pub fn event_count(&self) -> usize {
        self.events.len()
    }
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
    lock_path: PathBuf,
}

struct AccessLock(fs::File);

impl Drop for AccessLock {
    fn drop(&mut self) {
        let _ = FileExt::unlock(&self.0);
    }
}

impl AccessLog {
    /// Log paths are `{root}/.index/access.jsonl` and `access_counts.json`.
    pub fn open(root: impl AsRef<Path>) -> Self {
        let index = root.as_ref().join(".index");
        Self {
            path: index.join("access.jsonl"),
            counts_path: index.join("access_counts.json"),
            lock_path: index.join("access.lock"),
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
        let _lock = self.lock_exclusive()?;
        let ts = OffsetDateTime::now_utc()
            .format(&time::format_description::well_known::Rfc3339)
            .unwrap_or_else(|_| "1970-01-01T00:00:00Z".into());
        self.append_event_locked(AccessEvent {
            ts,
            handle: handle.to_string(),
            via,
        })
    }

    /// Append a fully specified event (used by tests and compaction rewrites).
    pub fn append_event(&self, event: AccessEvent) -> Result<()> {
        let _lock = self.lock_exclusive()?;
        self.append_event_locked(event)
    }

    fn append_event_locked(&self, event: AccessEvent) -> Result<()> {
        let event_ts = parse_ts(&event.ts).ok_or_else(|| {
            Error::validation(
                "access_log",
                format!("invalid RFC3339 timestamp {:?}", event.ts),
            )
        })?;
        let mut counters = self.load_counters_unlocked()?;
        if self
            .parse_compacted_through(&counters)?
            .is_some_and(|cutoff| event_ts <= cutoff)
        {
            counters
                .by_handle
                .entry(event.handle)
                .or_default()
                .add(event.via);
            return self.write_counters_unlocked(&counters);
        }
        self.append_event_unlocked(&event)
    }

    fn append_event_unlocked(&self, event: &AccessEvent) -> Result<()> {
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
        self.snapshot()
            .map(|snapshot| snapshot.recent_counts(handle, days).total() as u32)
            .unwrap_or(0)
    }

    /// Whether this handle has any recorded access (recent JSONL **or** compacted counters).
    pub fn ever_accessed(&self, handle: &str) -> bool {
        self.snapshot()
            .map(|snapshot| snapshot.ever_accessed(handle))
            .unwrap_or(false)
    }

    /// True when there is at least one JSONL event for `handle` with `ts` within `days`.
    pub fn accessed_within(&self, handle: &str, days: i64) -> bool {
        self.snapshot()
            .map(|snapshot| snapshot.accessed_within(handle, days))
            .unwrap_or(false)
    }

    /// Total JSONL event count (for stats).
    pub fn len(&self) -> usize {
        self.snapshot()
            .map(|snapshot| snapshot.event_count())
            .unwrap_or(0)
    }

    /// Whether the log file is missing or empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Read all JSONL events; missing file → empty.
    pub fn read_all(&self) -> Result<Vec<AccessEvent>> {
        Ok(self
            .snapshot()?
            .events
            .into_iter()
            .map(|event| event.event)
            .collect())
    }

    fn read_source_unlocked(&self) -> Result<Vec<SnapshotEvent>> {
        if !self.path.is_file() {
            return Ok(vec![]);
        }
        let f = fs::File::open(&self.path).map_err(|e| Error::io(&self.path, e))?;
        let mut out = Vec::new();
        for (index, line) in BufReader::new(f).lines().enumerate() {
            let line = line.map_err(|e| {
                Error::validation(
                    "access_log",
                    format!("{}:{}: read event: {e}", self.path.display(), index + 1),
                )
            })?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let event = serde_json::from_str::<AccessEvent>(line).map_err(|e| {
                Error::validation(
                    "access_log",
                    format!("{}:{}: parse event: {e}", self.path.display(), index + 1),
                )
            })?;
            let ts = parse_ts(&event.ts).ok_or_else(|| {
                Error::validation(
                    "access_log",
                    format!(
                        "{}:{}: invalid RFC3339 timestamp {:?}",
                        self.path.display(),
                        index + 1,
                        event.ts
                    ),
                )
            })?;
            out.push(SnapshotEvent { ts, event });
        }
        Ok(out)
    }

    /// Load compacted counters; missing file → empty.
    pub fn load_counters(&self) -> Result<AccessCounters> {
        let _lock = self.lock_shared()?;
        self.load_counters_unlocked()
    }

    fn load_counters_unlocked(&self) -> Result<AccessCounters> {
        if !self.counts_path.is_file() {
            return Ok(AccessCounters::default());
        }
        let text =
            fs::read_to_string(&self.counts_path).map_err(|e| Error::io(&self.counts_path, e))?;
        if text.trim().is_empty() {
            return Ok(AccessCounters::default());
        }
        serde_json::from_str(&text).map_err(|e| {
            Error::validation(
                "access_counts",
                format!("{}: parse counters: {e}", self.counts_path.display()),
            )
        })
    }

    fn write_counters_unlocked(&self, counters: &AccessCounters) -> Result<()> {
        let counts_json = serde_json::to_string_pretty(counters)
            .map_err(|e| Error::validation("access_counts", format!("serialize: {e}")))?;
        crate::store::atomic_write(&self.counts_path, counts_json.as_bytes())
    }

    /// Load a coherent, strict snapshot of source events and compacted history.
    pub fn snapshot(&self) -> Result<AccessSnapshot> {
        let _lock = self.lock_shared()?;
        self.snapshot_unlocked()
    }

    fn snapshot_unlocked(&self) -> Result<AccessSnapshot> {
        let counters = self.load_counters_unlocked()?;
        let compacted_through = self.parse_compacted_through(&counters)?;
        let events = self
            .read_source_unlocked()?
            .into_iter()
            .filter(|event| compacted_through.is_none_or(|cutoff| event.ts > cutoff))
            .collect::<Vec<_>>();
        let mut ever_accessed = counters
            .by_handle
            .iter()
            .filter(|(_, counts)| counts.total() > 0)
            .map(|(handle, _)| handle.clone())
            .collect::<HashSet<_>>();
        ever_accessed.extend(events.iter().map(|event| event.event.handle.clone()));
        Ok(AccessSnapshot {
            events,
            ever_accessed,
        })
    }

    /// Roll events older than `max_age_days` into counters; rewrite JSONL with kept events.
    ///
    /// Idempotent when run again with no new old events.
    pub fn compact(&self, max_age_days: i64) -> Result<CompactionStats> {
        let _lock = self.lock_exclusive()?;
        let events = self.read_source_unlocked()?;
        let requested_cutoff = OffsetDateTime::now_utc() - time::Duration::days(max_age_days);
        let mut counters = self.load_counters_unlocked()?;
        let previous_cutoff = self.parse_compacted_through(&counters)?;
        let cutoff = previous_cutoff
            .map(|previous| previous.max(requested_cutoff))
            .unwrap_or(requested_cutoff);
        let mut kept = Vec::new();
        let mut rolled = 0usize;
        let mut touched: HashMap<String, ()> = HashMap::new();

        for snapshot_event in events {
            let ev = snapshot_event.event;
            if previous_cutoff.is_some_and(|previous| snapshot_event.ts <= previous) {
                continue;
            }
            if snapshot_event.ts <= cutoff {
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

        counters.compacted_through = Some(
            cutoff
                .format(&time::format_description::well_known::Rfc3339)
                .map_err(|e| Error::validation("access_counts", format!("format cutoff: {e}")))?,
        );
        self.write_counters_unlocked(&counters)?;

        // Counters go first: their cutoff makes a stale source log safe after interruption.
        let mut source = Vec::new();
        for ev in &kept {
            serde_json::to_writer(&mut source, ev)
                .map_err(|e| Error::validation("access_log", format!("serialize: {e}")))?;
            source.push(b'\n');
        }
        crate::store::atomic_write(&self.path, &source)?;

        Ok(CompactionStats {
            events_kept: kept.len(),
            events_rolled: rolled,
            counters_updated: touched.len(),
        })
    }

    fn parse_compacted_through(&self, counters: &AccessCounters) -> Result<Option<OffsetDateTime>> {
        counters
            .compacted_through
            .as_deref()
            .map(|cutoff| {
                parse_ts(cutoff).ok_or_else(|| {
                    Error::validation(
                        "access_counts",
                        format!(
                            "{}: invalid compacted_through RFC3339 timestamp {cutoff:?}",
                            self.counts_path.display()
                        ),
                    )
                })
            })
            .transpose()
    }

    fn lock_shared(&self) -> Result<AccessLock> {
        self.lock(false)
    }

    fn lock_exclusive(&self) -> Result<AccessLock> {
        self.lock(true)
    }

    fn lock(&self, exclusive: bool) -> Result<AccessLock> {
        let parent = self.lock_path.parent().unwrap_or_else(|| Path::new("."));
        if let Some(root) = parent.parent() {
            crate::store::ensure_index_ignored(root)?;
        }
        fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&self.lock_path)
            .map_err(|e| Error::io(&self.lock_path, e))?;
        if exclusive {
            file.lock_exclusive()
        } else {
            file.lock_shared()
        }
        .map_err(|e| Error::io(&self.lock_path, e))?;
        Ok(AccessLock(file))
    }
}

fn parse_ts(s: &str) -> Option<OffsetDateTime> {
    OffsetDateTime::parse(s, &time::format_description::well_known::Rfc3339).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;
    use std::sync::{Arc, Barrier};
    use std::thread;
    use std::time::{Duration as StdDuration, Instant};
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
        assert_eq!(
            fs::read_to_string(dir.path().join(".gitignore")).unwrap(),
            "/.index/\n"
        );
        assert_eq!(log.count_recent("proj/a", 30), 2);
        assert_eq!(log.count_recent("proj/b", 30), 1);
        assert_eq!(log.count_recent("missing", 30), 0);
        assert_eq!(log.len(), 3);
        let snapshot = log.snapshot().unwrap();
        assert_eq!(
            snapshot.recent_counts("proj/a", 30),
            ViaCounts {
                read: 1,
                search_hit: 1,
            }
        );
        assert!(snapshot.ever_accessed("proj/a"));
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
    fn malformed_jsonl_reports_path_and_line() {
        let dir = tempdir().unwrap();
        let log = AccessLog::open(dir.path());
        fs::create_dir_all(log.path().parent().unwrap()).unwrap();
        fs::write(
            log.path(),
            concat!(
                "{\"ts\":\"2026-01-01T00:00:00Z\",\"handle\":\"ok/note\",\"via\":\"read\"}\n",
                "{not-json}\n"
            ),
        )
        .unwrap();

        let error = log.read_all().unwrap_err().to_string();
        assert!(error.contains(&log.path().display().to_string()), "{error}");
        assert!(error.contains(":2"), "{error}");
    }

    #[test]
    fn invalid_utf8_reports_path_and_line() {
        let dir = tempdir().unwrap();
        let log = AccessLog::open(dir.path());
        fs::create_dir_all(log.path().parent().unwrap()).unwrap();
        let mut bytes =
            b"{\"ts\":\"2026-01-01T00:00:00Z\",\"handle\":\"ok/note\",\"via\":\"read\"}\n".to_vec();
        bytes.extend_from_slice(&[0xff, b'\n']);
        fs::write(log.path(), bytes).unwrap();

        let error = log.read_all().unwrap_err().to_string();
        assert!(error.contains(&log.path().display().to_string()), "{error}");
        assert!(error.contains(":2"), "{error}");
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

    #[test]
    fn interrupted_compaction_recovers_idempotently_from_cutoff() {
        let dir = tempdir().unwrap();
        let log = AccessLog::open(dir.path());
        log.append_event(AccessEvent {
            ts: ts_days_ago(100),
            handle: "ns/old".into(),
            via: AccessVia::Read,
        })
        .unwrap();

        let counters = AccessCounters {
            compacted_through: Some(ts_days_ago(90)),
            by_handle: HashMap::from([(
                "ns/old".into(),
                ViaCounts {
                    read: 1,
                    search_hit: 0,
                },
            )]),
        };
        fs::write(
            log.counts_path(),
            serde_json::to_vec_pretty(&counters).unwrap(),
        )
        .unwrap();

        assert_eq!(log.snapshot().unwrap().event_count(), 0);
        assert!(log.snapshot().unwrap().ever_accessed("ns/old"));

        let first = log.compact(90).unwrap();
        assert_eq!(first.events_rolled, 0);
        assert_eq!(log.load_counters().unwrap().by_handle["ns/old"].read, 1);
        assert!(log.read_all().unwrap().is_empty());

        let second = log.compact(90).unwrap();
        assert_eq!(second.events_rolled, 0);
        assert_eq!(log.load_counters().unwrap().by_handle["ns/old"].read, 1);
    }

    #[test]
    fn append_retains_event_when_clock_is_not_after_cutoff() {
        let dir = tempdir().unwrap();
        let log = AccessLog::open(dir.path());
        fs::create_dir_all(log.counts_path().parent().unwrap()).unwrap();
        let counters = AccessCounters {
            compacted_through: Some(
                (OffsetDateTime::now_utc() + time::Duration::days(1))
                    .format(&Rfc3339)
                    .unwrap(),
            ),
            by_handle: HashMap::new(),
        };
        fs::write(
            log.counts_path(),
            serde_json::to_vec_pretty(&counters).unwrap(),
        )
        .unwrap();

        log.append("clock/rollback", AccessVia::Read).unwrap();

        assert!(log.snapshot().unwrap().ever_accessed("clock/rollback"));
        assert_eq!(
            log.load_counters().unwrap().by_handle["clock/rollback"].read,
            1
        );
    }

    #[test]
    fn concurrent_append_and_compact_retains_every_event() {
        const WRITERS: usize = 4;
        const EVENTS_PER_WRITER: usize = 200;

        let dir = tempdir().unwrap();
        let log = AccessLog::open(dir.path());
        let barrier = Arc::new(Barrier::new(WRITERS + 1));
        let mut threads = Vec::new();

        for writer in 0..WRITERS {
            let log = log.clone();
            let barrier = barrier.clone();
            threads.push(thread::spawn(move || {
                barrier.wait();
                for event in 0..EVENTS_PER_WRITER {
                    log.append(
                        &format!("writer/{writer}"),
                        if event % 2 == 0 {
                            AccessVia::Read
                        } else {
                            AccessVia::SearchHit
                        },
                    )
                    .unwrap();
                }
            }));
        }

        let compact_log = log.clone();
        let compact_barrier = barrier.clone();
        let compactor = thread::spawn(move || {
            compact_barrier.wait();
            for _ in 0..50 {
                compact_log.compact(0).unwrap();
            }
        });

        for thread in threads {
            thread.join().unwrap();
        }
        compactor.join().unwrap();
        log.compact(0).unwrap();

        let retained = log
            .load_counters()
            .unwrap()
            .by_handle
            .values()
            .map(ViaCounts::total)
            .sum::<u64>()
            + log.read_all().unwrap().len() as u64;
        assert_eq!(retained, (WRITERS * EVENTS_PER_WRITER) as u64);
    }

    #[test]
    fn append_waits_for_cross_process_lock() {
        let dir = tempdir().unwrap();
        let log = AccessLog::open(dir.path());
        fs::create_dir_all(log.lock_path.parent().unwrap()).unwrap();
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&log.lock_path)
            .unwrap();
        lock.lock_exclusive().unwrap();

        let ready = dir.path().join("child-ready");
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args([
                "--ignored",
                "--exact",
                "access_log::tests::cross_process_append_helper",
            ])
            .env("FFF_MEMORY_TEST_ROOT", dir.path())
            .env("FFF_MEMORY_TEST_READY", &ready)
            .spawn()
            .unwrap();

        let deadline = Instant::now() + StdDuration::from_secs(2);
        while !ready.is_file() && Instant::now() < deadline {
            thread::sleep(StdDuration::from_millis(5));
        }
        assert!(ready.is_file(), "child did not reach append");
        thread::sleep(StdDuration::from_millis(25));
        assert!(child.try_wait().unwrap().is_none(), "append bypassed lock");

        FileExt::unlock(&lock).unwrap();
        assert!(child.wait().unwrap().success());
        assert_eq!(log.read_all().unwrap().len(), 1);
    }

    #[test]
    #[ignore = "subprocess helper"]
    fn cross_process_append_helper() {
        let Ok(root) = std::env::var("FFF_MEMORY_TEST_ROOT") else {
            return;
        };
        let ready = std::env::var("FFF_MEMORY_TEST_READY").unwrap();
        fs::write(ready, b"ready").unwrap();
        AccessLog::open(root)
            .append("child/note", AccessVia::Read)
            .unwrap();
    }
}
