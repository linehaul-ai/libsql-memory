use std::fs;
use std::sync::Arc;
use std::time::{Duration, Instant};

use memory_index::FffRetriever;
use memory_mcp::{MemoryService, SearchOptions};
use tempfile::tempdir;

#[test]
#[ignore = "release-mode performance acceptance gate"]
fn low_thousands_cold_start_and_warm_search_meet_targets() {
    assert!(
        !cfg!(debug_assertions),
        "run this acceptance gate with --release"
    );
    let dir = tempdir().unwrap();
    for index in 0..2_000 {
        fs::write(
            dir.path().join(format!("note-{index:04}.md")),
            format!(
                "---\ntitle: Performance Note {index}\naliases: [lookup phrase {index}, alternate phrase {index}]\ntype: fact\ncreated: 2026-08-09\nupdated: 2026-08-09\n---\nFixture body number {index}.\n"
            ),
        )
        .unwrap();
    }

    let started = Instant::now();
    let retriever = Arc::new(FffRetriever::open(dir.path()).unwrap());
    let cold_start = started.elapsed();
    assert!(
        cold_start < Duration::from_secs(1),
        "2,000-note cold start was {cold_start:?}"
    );

    let service = MemoryService::new(dir.path(), Some(retriever));
    let mut samples = Vec::with_capacity(101);
    for _ in 0..101 {
        let started = Instant::now();
        let response = service
            .search(SearchOptions {
                query: "lookup phrase 1234".into(),
                ..SearchOptions::default()
            })
            .unwrap();
        samples.push(started.elapsed());
        assert_eq!(response.results[0].handle, "note-1234");
    }
    samples.sort_unstable();
    let p50 = samples[samples.len() / 2];
    eprintln!(
        "2,000-note cold start: {cold_start:?}; warm search min/p50/max: {:?}/{p50:?}/{:?}",
        samples[0],
        samples[samples.len() - 1]
    );
    assert!(
        p50 < Duration::from_millis(10),
        "warm search p50 was {p50:?}"
    );
}
