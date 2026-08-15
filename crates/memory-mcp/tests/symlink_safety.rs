#![cfg(unix)]

use std::fs;
use std::os::unix::fs::symlink;
use std::path::PathBuf;
use std::sync::Arc;

use memory_core::{testing::FakeRetriever, FileHit};
use memory_mcp::{MemoryService, SearchOptions};
use tempfile::tempdir;

#[test]
fn hard_forget_rejects_symlinked_namespace_without_deleting_external_note() {
    let root = tempdir().unwrap();
    let outside = tempdir().unwrap();
    let external_note = outside.path().join("keep.md");
    fs::write(
        &external_note,
        "---\ntitle: Keep\naliases: [outside note, symlink target]\ntype: fact\ncreated: 2026-01-01\nupdated: 2026-01-01\n---\noutside\n",
    )
    .unwrap();
    symlink(outside.path(), root.path().join("escaped")).unwrap();

    let error = MemoryService::new(root.path(), None)
        .forget("escaped/keep", true)
        .expect_err("hard forget must reject symlinked namespaces")
        .to_string();

    assert!(error.contains("symlink"), "error={error}");
    assert!(error.contains("escaped"), "error={error}");
    assert!(external_note.is_file());
}

#[test]
fn read_rejects_symlinked_namespace_without_exposing_external_note() {
    let root = tempdir().unwrap();
    let outside = tempdir().unwrap();
    fs::write(
        outside.path().join("private.md"),
        "---\ntitle: Private\naliases: [outside note, symlink target]\ntype: fact\ncreated: 2026-01-01\nupdated: 2026-01-01\n---\nsecret outside body\n",
    )
    .unwrap();
    symlink(outside.path(), root.path().join("escaped")).unwrap();

    let error = MemoryService::new(root.path(), None)
        .read("escaped/private")
        .expect_err("read must reject symlinked namespaces")
        .to_string();

    assert!(error.contains("symlink"), "error={error}");
    assert!(!error.contains("secret outside body"), "error={error}");
}

#[test]
fn stats_and_search_skip_notes_reached_through_symlinked_directories() {
    let root = tempdir().unwrap();
    let outside = tempdir().unwrap();
    fs::write(
        outside.path().join("private.md"),
        "---\ntitle: Private\naliases: [outside note, symlink target]\ntype: fact\ncreated: 2026-01-01\nupdated: 2026-01-01\n---\nsecret outside body\n",
    )
    .unwrap();
    symlink(outside.path(), root.path().join("escaped")).unwrap();
    let retriever = Arc::new(FakeRetriever::with_files(vec![FileHit {
        path: PathBuf::from("escaped/private.md"),
        score: 1.0,
    }]));
    let service = MemoryService::new(root.path(), Some(retriever));

    assert_eq!(service.stats(None).unwrap().total_notes, 0);
    let response = service
        .search(SearchOptions {
            query: "private".into(),
            ..SearchOptions::default()
        })
        .unwrap();
    assert!(response.results.is_empty(), "{:?}", response.results);
    assert!(response.more.is_empty(), "{:?}", response.more);
}

#[test]
fn read_reports_symlinked_access_log_instead_of_claiming_recorded_access() {
    let root = tempdir().unwrap();
    let outside = tempdir().unwrap();
    fs::create_dir(root.path().join("proj")).unwrap();
    fs::write(
        root.path().join("proj/note.md"),
        "---\ntitle: Note\naliases: [first alias, second alias]\ntype: fact\ncreated: 2026-01-01\nupdated: 2026-01-01\n---\ninside\n",
    )
    .unwrap();
    fs::create_dir(root.path().join(".index")).unwrap();
    let external_log = outside.path().join("access.jsonl");
    fs::write(&external_log, "sentinel\n").unwrap();
    symlink(&external_log, root.path().join(".index/access.jsonl")).unwrap();

    let error = MemoryService::new(root.path(), None)
        .read("proj/note")
        .expect_err("read must report failure to record authoritative access")
        .to_string();

    assert!(error.contains("symlink"), "error={error}");
    assert_eq!(fs::read_to_string(external_log).unwrap(), "sentinel\n");
}

#[test]
fn search_reports_symlinked_access_log_instead_of_claiming_recorded_hits() {
    let root = tempdir().unwrap();
    let outside = tempdir().unwrap();
    fs::create_dir(root.path().join("proj")).unwrap();
    fs::write(
        root.path().join("proj/note.md"),
        "---\ntitle: Note\naliases: [first alias, second alias]\ntype: fact\ncreated: 2026-01-01\nupdated: 2026-01-01\n---\ninside\n",
    )
    .unwrap();
    fs::create_dir(root.path().join(".index")).unwrap();
    let external_log = outside.path().join("access.jsonl");
    fs::write(&external_log, "sentinel\n").unwrap();
    symlink(&external_log, root.path().join(".index/access.jsonl")).unwrap();
    let retriever = Arc::new(FakeRetriever::with_files(vec![FileHit {
        path: PathBuf::from("proj/note.md"),
        score: 1.0,
    }]));

    let error = MemoryService::new(root.path(), Some(retriever))
        .search(SearchOptions {
            query: "note".into(),
            ..SearchOptions::default()
        })
        .expect_err("search must report failure to record authoritative hits")
        .to_string();

    assert!(error.contains("symlink"), "error={error}");
    assert_eq!(fs::read_to_string(external_log).unwrap(), "sentinel\n");
}
