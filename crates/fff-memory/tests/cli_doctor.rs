//! CLI smoke tests for `fff-memory doctor` and `reindex` (spec 05).

use std::fs;
use std::process::Command;

use tempfile::tempdir;

fn bin() -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_fff-memory"));
    for key in [
        "FFF_MEMORY_ROOT",
        "FFF_MEMORY_CONFIG",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "HOME",
    ] {
        c.env_remove(key);
    }
    c
}

fn write_expired(root: &std::path::Path) {
    let path = root.join("proj/exp.md");
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(
        &path,
        r#"---
title: Expired Note
aliases:
  - a
  - b
type: fact
created: 2020-01-01
updated: 2020-01-01
expires: 2020-06-01
---
Body.
"#,
    )
    .unwrap();
}

#[test]
fn doctor_report_lists_candidates() {
    let dir = tempdir().unwrap();
    write_expired(dir.path());
    let out = bin()
        .args(["doctor", "--root"])
        .arg(dir.path())
        .output()
        .expect("run doctor");
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("proj/exp") || stdout.contains("expired"),
        "stdout={stdout}"
    );
}

#[test]
fn doctor_uses_the_healthy_index_by_default() {
    let dir = tempdir().unwrap();
    write_expired(dir.path());
    let reindex = bin()
        .args(["reindex", "--root"])
        .arg(dir.path())
        .output()
        .expect("seed healthy index");
    assert!(reindex.status.success());

    let out = bin()
        .args(["doctor", "--root"])
        .arg(dir.path())
        .output()
        .expect("run doctor");
    assert!(
        out.status.success(),
        "stderr={} stdout={}",
        String::from_utf8_lossy(&out.stderr),
        String::from_utf8_lossy(&out.stdout)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("Index: ok"), "stdout={stdout}");
    assert!(!stdout.contains("index unavailable"), "stdout={stdout}");
    assert!(!stdout.contains("fff-memory reindex"), "stdout={stdout}");
}

#[test]
fn doctor_does_not_call_a_live_indexed_non_note_orphaned() {
    let dir = tempdir().unwrap();
    write_expired(dir.path());
    fs::write(dir.path().join("orphan.txt"), "not a memory note").unwrap();
    let reindex = bin()
        .args(["reindex", "--root"])
        .arg(dir.path())
        .output()
        .expect("seed index with an orphan");
    assert!(reindex.status.success());

    let out = bin()
        .args(["doctor", "--root"])
        .arg(dir.path())
        .output()
        .expect("run doctor");
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("Index: ok"), "stdout={stdout}");
    assert!(!stdout.contains("orphaned index"), "stdout={stdout}");
    assert!(!stdout.contains("fff-memory reindex"), "stdout={stdout}");
}

#[test]
fn doctor_inspects_files_when_the_index_cannot_open() {
    let dir = tempdir().unwrap();
    write_expired(dir.path());
    let index = dir.path().join(".index");
    fs::create_dir(&index).unwrap();
    fs::write(index.join("frecency"), "not an LMDB directory").unwrap();

    let out = bin()
        .args(["doctor", "--root"])
        .arg(dir.path())
        .output()
        .expect("run doctor with broken index");
    assert!(
        out.status.success(),
        "stderr={} stdout={}",
        String::from_utf8_lossy(&out.stderr),
        String::from_utf8_lossy(&out.stdout)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("proj/exp") || stdout.contains("expired"));
    assert!(stdout.contains("index unavailable"), "stdout={stdout}");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains(&index.display().to_string()),
        "stderr={stderr}"
    );
    assert!(stderr.contains("fff-memory reindex"), "stderr={stderr}");
}

#[test]
fn doctor_help_has_no_undocumented_index_opt_in() {
    let out = bin()
        .args(["doctor", "--help"])
        .output()
        .expect("run doctor help");
    assert!(out.status.success());
    assert!(!String::from_utf8_lossy(&out.stdout).contains("--with-index"));
}

#[test]
fn doctor_apply_archives() {
    let dir = tempdir().unwrap();
    write_expired(dir.path());
    assert!(dir.path().join("proj/exp.md").is_file());
    let out = bin()
        .args(["doctor", "--apply", "--root"])
        .arg(dir.path())
        .output()
        .expect("run doctor --apply");
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(!dir.path().join("proj/exp.md").is_file());
    assert!(dir.path().join(".archive/proj/exp.md").is_file());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("applied") || stdout.contains("Suggested commit"));
}

#[test]
fn reindex_on_empty_root() {
    let dir = tempdir().unwrap();
    // Seed a note so open/scan has something
    write_expired(dir.path());
    let out = bin()
        .args(["reindex", "--root"])
        .arg(dir.path())
        .output()
        .expect("run reindex");
    assert!(
        out.status.success(),
        "stderr={} stdout={}",
        String::from_utf8_lossy(&out.stderr),
        String::from_utf8_lossy(&out.stdout)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("reindex complete"), "stdout={stdout}");
}

#[test]
fn reindex_recovers_unopenable_cache_and_preserves_access_log() {
    let dir = tempdir().unwrap();
    write_expired(dir.path());
    let index = dir.path().join(".index");
    fs::create_dir(&index).unwrap();
    fs::write(index.join("frecency"), "not an LMDB directory").unwrap();
    let access = "{\"ts\":\"2026-08-09T00:00:00Z\",\"handle\":\"proj/exp\",\"via\":\"read\"}\n";
    fs::write(index.join("access.jsonl"), access).unwrap();

    let out = bin()
        .args(["reindex", "--root"])
        .arg(dir.path())
        .output()
        .expect("run recovery reindex");
    assert!(
        out.status.success(),
        "stderr={} stdout={}",
        String::from_utf8_lossy(&out.stderr),
        String::from_utf8_lossy(&out.stdout)
    );
    assert!(index.join("frecency").is_dir());
    assert!(index.join("queries").is_dir());
    assert_eq!(
        fs::read_to_string(index.join("access.jsonl")).unwrap(),
        access
    );
}
