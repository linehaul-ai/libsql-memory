//! Integration tests: real fff-search against a temp markdown corpus (spec 03/06).

use std::fs;
use std::path::Path;
use std::time::Duration;

use memory_core::{GrepMode, IndexState, Retriever};
use memory_index::FffRetriever;
use tempfile::tempdir;

fn write_note(root: &Path, rel: &str, body: &str) {
    let path = root.join(rel);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(path, body).unwrap();
}

fn seed_corpus(root: &Path) {
    write_note(
        root,
        "linehaul/deploy.md",
        r#"---
title: Deploy Linehaul TMS
aliases: [shipping deploy, tms rollout]
tags: [ops]
type: decision
created: 2026-01-01
updated: 2026-01-01
---

Use blue-green for shipping production deploys.
"#,
    );
    write_note(
        root,
        "linehaul/rates.md",
        r#"---
title: Spot Rate Lookup
aliases: [DAT rates, market rates]
tags: [freight]
type: preference
created: 2026-01-02
updated: 2026-01-02
---

Prefer contract rates when L2T is tight.
"#,
    );
    write_note(
        root,
        "other/session.md",
        r#"---
title: Unrelated Session
aliases: [chat log, scratch]
tags: []
type: session-summary
created: 2026-01-03
updated: 2026-01-03
---

Nothing about freight here.
"#,
    );
    write_note(
        root,
        ".archive/old.md",
        r#"---
title: Archived Deploy
aliases: [shipping deploy old, legacy deploy]
tags: []
type: decision
created: 2025-01-01
updated: 2025-01-01
---

Old shipping deploy notes should not surface.
"#,
    );
}

fn open_ready(root: &Path) -> FffRetriever {
    let r = FffRetriever::open(root).expect("open FffRetriever");
    // Poll briefly if scan races past wait.
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while r.index_state() != IndexState::Ready && std::time::Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(50));
    }
    assert_eq!(r.index_state(), IndexState::Ready);
    r
}

#[test]
fn open_reports_ready() {
    let dir = tempdir().unwrap();
    seed_corpus(dir.path());
    let r = open_ready(dir.path());
    assert_eq!(r.index_state(), IndexState::Ready);
    assert!(
        r.root().ends_with(dir.path().file_name().unwrap())
            || r.root() == dir.path().canonicalize().unwrap()
    );
}

#[test]
fn find_files_matches_slug_paths() {
    let dir = tempdir().unwrap();
    seed_corpus(dir.path());
    let r = open_ready(dir.path());

    let hits = r.find_files("deploy", None).expect("find_files");
    assert!(
        hits.iter()
            .any(|h| h.path.to_string_lossy().contains("deploy")),
        "expected deploy path in {hits:?}"
    );
    // Archived path must never appear.
    assert!(
        hits.iter()
            .all(|h| !h.path.to_string_lossy().contains(".archive")),
        "archive leaked: {hits:?}"
    );
}

#[test]
fn find_files_respects_namespace_scope() {
    let dir = tempdir().unwrap();
    seed_corpus(dir.path());
    let r = open_ready(dir.path());

    let hits = r.find_files("md", Some("linehaul")).expect("scoped find");
    assert!(!hits.is_empty(), "expected hits under linehaul");
    for h in &hits {
        let p = h.path.to_string_lossy();
        assert!(
            p == "linehaul" || p.starts_with("linehaul/"),
            "out of scope: {p}"
        );
    }
    assert!(
        hits.iter()
            .all(|h| !h.path.to_string_lossy().starts_with("other/")),
        "other/ should be excluded"
    );
}

#[test]
fn grep_plain_matches_frontmatter_aliases() {
    let dir = tempdir().unwrap();
    seed_corpus(dir.path());
    let r = open_ready(dir.path());

    // Alias lives only in YAML frontmatter — must be greppable.
    let hits = r
        .grep("shipping deploy", GrepMode::Plain, None)
        .expect("grep");
    assert!(
        hits.iter()
            .any(|h| h.path.to_string_lossy().contains("deploy")),
        "alias 'shipping deploy' should hit deploy.md: {hits:?}"
    );
    assert!(
        hits.iter()
            .all(|h| !h.path.to_string_lossy().contains(".archive")),
        "archive must be filtered: {hits:?}"
    );
    assert!(hits.iter().all(|h| h.line > 0 || !h.snippet.is_empty()));
}

#[test]
fn grep_body_content() {
    let dir = tempdir().unwrap();
    seed_corpus(dir.path());
    let r = open_ready(dir.path());

    let hits = r
        .grep("blue-green", GrepMode::Plain, None)
        .expect("grep body");
    assert!(
        hits.iter()
            .any(|h| h.snippet.to_ascii_lowercase().contains("blue-green")),
        "body match missing: {hits:?}"
    );
}

#[test]
fn grep_scoped_to_namespace() {
    let dir = tempdir().unwrap();
    seed_corpus(dir.path());
    let r = open_ready(dir.path());

    let hits = r
        .grep("rates", GrepMode::Plain, Some("linehaul"))
        .expect("scoped grep");
    for h in &hits {
        assert!(
            h.path.to_string_lossy().starts_with("linehaul/"),
            "out of scope: {:?}",
            h.path
        );
    }
}

#[test]
fn reindex_after_new_file() {
    let dir = tempdir().unwrap();
    seed_corpus(dir.path());
    let r = open_ready(dir.path());

    write_note(
        dir.path(),
        "linehaul/new-note.md",
        r#"---
title: Brand New Note
aliases: [fresh memory, brand new]
tags: []
type: fact
created: 2026-06-01
updated: 2026-06-01
---

Unique zebra token for reindex test.
"#,
    );

    r.reindex().expect("reindex");
    assert_eq!(r.index_state(), IndexState::Ready);

    let hits = r
        .grep("zebra token", GrepMode::Plain, None)
        .expect("grep after reindex");
    assert!(
        hits.iter()
            .any(|h| h.path.to_string_lossy().contains("new-note")),
        "new file should be greppable after reindex: {hits:?}"
    );
}

#[test]
fn store_dedup_probe_via_retriever() {
    // Cross-crate smoke: MemoryStore can use FffRetriever for dedup.
    use memory_core::{MemoryStore, NoteType, StoreInput};

    let dir = tempdir().unwrap();
    seed_corpus(dir.path());
    let r = open_ready(dir.path());
    let store = MemoryStore::new(dir.path());

    let out = store
        .store(
            StoreInput {
                title: "Deploy Linehaul TMS".into(),
                body: "Addendum about deploys.".into(),
                aliases: vec!["shipping deploy".into(), "tms rollout".into()],
                namespace: "linehaul".into(),
                note_type: NoteType::Decision,
                tags: vec![],
                expires: None,
                source: None,
                mode: Default::default(),
            },
            Some(&r as &dyn Retriever),
        )
        .expect("store with dedup");

    // Either updates existing deploy.md or creates — must not fail.
    assert!(!out.slug.is_empty());
}
