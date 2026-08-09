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
fn index_snapshot_reports_live_files_and_scan_time() {
    let dir = tempdir().unwrap();
    seed_corpus(dir.path());
    let r = open_ready(dir.path());

    let snapshot = r.index_snapshot();

    assert_eq!(snapshot.state, IndexState::Ready);
    assert_eq!(snapshot.files_indexed, 3);
    assert!(snapshot.last_scan_ms > 0, "snapshot={snapshot:?}");
}

#[test]
fn access_tracking_is_best_effort_for_indexed_paths() {
    let dir = tempdir().unwrap();
    seed_corpus(dir.path());
    let r = open_ready(dir.path());

    r.track_access(Path::new("linehaul/deploy.md"))
        .expect("track access");
}

#[test]
fn find_files_matches_slug_paths() {
    let dir = tempdir().unwrap();
    seed_corpus(dir.path());
    let r = open_ready(dir.path());

    let hits = r.find_files("deploy", None, false).expect("find_files");
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

    let hits = r
        .find_files("md", Some("linehaul"), false)
        .expect("scoped find");
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
fn scope_is_applied_before_find_pagination() {
    let dir = tempdir().unwrap();
    write_note(dir.path(), "wanted/needel.md", "needle");
    for i in 0..60 {
        write_note(dir.path(), &format!("wanted-noise-{i}/needle.md"), "needle");
    }
    let r = open_ready(dir.path());

    let hits = r
        .find_files("needle", Some("wanted"), false)
        .expect("scoped find");

    assert_eq!(hits.len(), 1, "scope page lost the target: {hits:?}");
    assert_eq!(hits[0].path, Path::new("wanted/needel.md"));
}

#[test]
fn grep_plain_matches_frontmatter_aliases() {
    let dir = tempdir().unwrap();
    seed_corpus(dir.path());
    let r = open_ready(dir.path());

    // Alias lives only in YAML frontmatter — must be greppable.
    let hits = r
        .grep("shipping deploy", GrepMode::Plain, None, false)
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
        .grep("blue-green", GrepMode::Plain, None, false)
        .expect("grep body");
    assert!(
        hits.iter()
            .any(|h| h.snippet.to_ascii_lowercase().contains("blue-green")),
        "body match missing: {hits:?}"
    );
}

#[test]
fn backend_scores_are_normalized_per_page() {
    let dir = tempdir().unwrap();
    seed_corpus(dir.path());
    let r = open_ready(dir.path());

    let files = r.find_files("md", None, false).expect("find scores");
    let contents = r
        .grep("raets", GrepMode::Fuzzy, None, false)
        .expect("grep scores");

    assert!(!files.is_empty());
    assert!(!contents.is_empty());
    assert!(files.iter().all(|h| (0.0..=1.0).contains(&h.score)));
    assert!(contents.iter().all(|h| (0.0..=1.0).contains(&h.score)));
}

#[test]
fn prose_title_alias_and_body_queries_rank_expected_note_first() {
    let dir = tempdir().unwrap();
    seed_corpus(dir.path());
    let r = open_ready(dir.path());

    let title = r.find_files("deploy", None, false).expect("title search");
    let alias = r
        .grep("DAT rates", GrepMode::Plain, None, false)
        .expect("alias search");
    let body = r
        .grep("blue-green", GrepMode::Plain, None, false)
        .expect("body search");

    assert_eq!(
        title.first().map(|h| h.path.as_path()),
        Some(Path::new("linehaul/deploy.md"))
    );
    assert_eq!(
        alias.first().map(|h| h.path.as_path()),
        Some(Path::new("linehaul/rates.md"))
    );
    assert_eq!(
        body.first().map(|h| h.path.as_path()),
        Some(Path::new("linehaul/deploy.md"))
    );
}

#[test]
fn grep_scoped_to_namespace() {
    let dir = tempdir().unwrap();
    seed_corpus(dir.path());
    let r = open_ready(dir.path());

    let hits = r
        .grep("rates", GrepMode::Plain, Some("linehaul"), false)
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
fn scope_is_applied_before_grep_pagination() {
    let dir = tempdir().unwrap();
    write_note(dir.path(), "z-wanted/hit.md", "pagination needle");
    for i in 0..60 {
        write_note(
            dir.path(),
            &format!("a-noise-{i}/hit.md"),
            "pagination needle",
        );
    }
    let r = open_ready(dir.path());

    let hits = r
        .grep(
            "pagination needle",
            GrepMode::Plain,
            Some("z-wanted"),
            false,
        )
        .expect("scoped grep");

    assert_eq!(hits.len(), 1, "scope page lost the target: {hits:?}");
    assert_eq!(hits[0].path, Path::new("z-wanted/hit.md"));
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
        .grep("zebra token", GrepMode::Plain, None, false)
        .expect("grep after reindex");
    assert!(
        hits.iter()
            .any(|h| h.path.to_string_lossy().contains("new-note")),
        "new file should be greppable after reindex: {hits:?}"
    );
}

#[test]
fn reindex_recreates_fff_dbs_but_preserves_access_log() {
    let dir = tempdir().unwrap();
    seed_corpus(dir.path());
    let r = open_ready(dir.path());
    let index = dir.path().join(".index");
    let access = index.join("access.jsonl");
    fs::write(&access, "{\"handle\":\"linehaul/deploy\"}\n").unwrap();
    fs::write(index.join("frecency/stale-marker"), "stale").unwrap();
    fs::write(index.join("queries/stale-marker"), "stale").unwrap();

    r.reindex().expect("reindex");

    assert_eq!(
        fs::read_to_string(access).unwrap(),
        "{\"handle\":\"linehaul/deploy\"}\n"
    );
    assert!(index.join("frecency").is_dir());
    assert!(index.join("queries").is_dir());
    assert!(!index.join("frecency/stale-marker").exists());
    assert!(!index.join("queries/stale-marker").exists());
}

#[test]
fn external_file_edit_becomes_searchable_without_reindex() {
    let dir = tempdir().unwrap();
    seed_corpus(dir.path());
    let r = open_ready(dir.path());

    write_note(
        dir.path(),
        "linehaul/watched-note.md",
        "---\ntitle: Watched Note\naliases: [live edit, watcher test]\ntype: fact\ncreated: 2026-06-01\nupdated: 2026-06-01\n---\nwatcher-only-needle\n",
    );

    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    loop {
        let hits = r
            .grep("watcher-only-needle", GrepMode::Plain, None, false)
            .expect("live grep");
        if hits.iter().any(|h| h.path.ends_with("watched-note.md")) {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "watcher did not index the external edit: {hits:?}"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
}

#[test]
fn archived_hits_respect_namespace_scope() {
    let dir = tempdir().unwrap();
    seed_corpus(dir.path());
    write_note(
        dir.path(),
        ".archive/linehaul/old.md",
        "---\ntitle: Linehaul Archive\naliases: [retired linehaul, old linehaul]\ntype: fact\ncreated: 2025-01-01\nupdated: 2025-01-01\n---\narchived needle\n",
    );
    write_note(
        dir.path(),
        ".archive/other/old.md",
        "---\ntitle: Other Archive\naliases: [retired note, old note]\ntype: fact\ncreated: 2025-01-01\nupdated: 2025-01-01\n---\narchived needle\n",
    );
    let r = open_ready(dir.path());

    let hits = r
        .grep("archived needle", GrepMode::Plain, Some("linehaul"), true)
        .expect("archived grep");

    assert_eq!(hits.len(), 1, "unexpected archived hits: {hits:?}");
    assert_eq!(hits[0].path, Path::new(".archive/linehaul/old.md"));
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
