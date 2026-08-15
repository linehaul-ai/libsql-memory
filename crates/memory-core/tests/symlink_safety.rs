#![cfg(unix)]

use std::fs;
use std::os::unix::fs::symlink;

use memory_core::{
    run_doctor, AccessLog, AccessVia, DoctorOptions, MemoryStore, MergeMode, NoteType, StoreInput,
};
use tempfile::tempdir;

fn input(namespace: &str) -> StoreInput {
    StoreInput {
        title: "External Escape".into(),
        body: "must stay inside the configured memory root".into(),
        aliases: vec!["outside path".into(), "symlink escape".into()],
        namespace: namespace.into(),
        note_type: NoteType::Fact,
        tags: Vec::new(),
        expires: None,
        source: None,
        mode: MergeMode::Append,
    }
}

#[test]
fn store_rejects_symlinked_namespace_without_writing_outside_root() {
    let root = tempdir().unwrap();
    let outside = tempdir().unwrap();
    symlink(outside.path(), root.path().join("escaped")).unwrap();

    let error = MemoryStore::new(root.path())
        .store(input("escaped"), None)
        .expect_err("symlinked namespace must be rejected")
        .to_string();

    assert!(error.contains("symlink"), "error={error}");
    assert!(error.contains("escaped"), "error={error}");
    assert!(!outside.path().join("external-escape.md").exists());
}

#[test]
fn doctor_apply_skips_symlinked_namespace_without_moving_external_note() {
    let root = tempdir().unwrap();
    let outside = tempdir().unwrap();
    let external_note = outside.path().join("expired.md");
    fs::write(
        &external_note,
        "---\ntitle: External\naliases: [outside note, symlink target]\ntype: fact\ncreated: 2020-01-01\nupdated: 2020-01-01\nexpires: 2020-01-02\n---\noutside\n",
    )
    .unwrap();
    symlink(outside.path(), root.path().join("escaped")).unwrap();

    let report = run_doctor(root.path(), None, DoctorOptions { apply: true }).unwrap();

    assert!(external_note.is_file());
    assert!(fs::read_to_string(&external_note)
        .unwrap()
        .contains("outside"));
    assert!(report.archive_candidates.is_empty(), "{report:?}");
    assert!(!root.path().join(".archive/escaped/expired.md").exists());
}

#[test]
fn access_log_rejects_symlinked_index_without_writing_outside_root() {
    let root = tempdir().unwrap();
    let outside = tempdir().unwrap();
    symlink(outside.path(), root.path().join(".index")).unwrap();

    let error = AccessLog::open(root.path())
        .append("proj/note", AccessVia::Read)
        .expect_err("access log must reject a symlinked index")
        .to_string();

    assert!(error.contains("symlink"), "error={error}");
    assert!(fs::read_dir(outside.path()).unwrap().next().is_none());
}

#[test]
fn access_log_rejects_symlinked_data_file_without_modifying_its_target() {
    let root = tempdir().unwrap();
    let outside = tempdir().unwrap();
    let external_log = outside.path().join("external.jsonl");
    fs::write(&external_log, "sentinel\n").unwrap();
    fs::create_dir(root.path().join(".index")).unwrap();
    symlink(&external_log, root.path().join(".index/access.jsonl")).unwrap();

    let error = AccessLog::open(root.path())
        .append("proj/note", AccessVia::Read)
        .expect_err("access log must reject a symlinked data file")
        .to_string();

    assert!(error.contains("symlink"), "error={error}");
    assert_eq!(fs::read_to_string(external_log).unwrap(), "sentinel\n");
}

#[test]
fn archive_rejects_symlinked_destination_without_moving_active_note() {
    let root = tempdir().unwrap();
    let outside = tempdir().unwrap();
    let store = MemoryStore::new(root.path());
    store.store(input("active"), None).unwrap();
    symlink(outside.path(), root.path().join(".archive")).unwrap();

    let error = store
        .archive_note("active", "external-escape")
        .expect_err("archive must reject a symlinked destination")
        .to_string();

    assert!(error.contains("symlink"), "error={error}");
    assert!(root.path().join("active/external-escape.md").is_file());
    assert!(fs::read_dir(outside.path()).unwrap().next().is_none());
}
