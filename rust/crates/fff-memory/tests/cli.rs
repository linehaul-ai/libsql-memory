use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::Value;
use tempfile::tempdir;

const HANDLE: &str = "test/resolver-note";

fn bin() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_fff-memory"));
    for key in [
        "FFF_MEMORY_ROOT",
        "FFF_MEMORY_CONFIG",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "HOME",
    ] {
        command.env_remove(key);
    }
    command
}

fn output(args: &[&str]) -> Output {
    bin().args(args).output().expect("spawn fff-memory")
}

fn assert_success(out: &Output) {
    assert!(
        out.status.success(),
        "status={}\nstdout={}\nstderr={}",
        out.status,
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}

fn json_stdout(out: &Output) -> Value {
    assert_success(out);
    let stdout = std::str::from_utf8(&out.stdout).unwrap();
    assert!(stdout.ends_with('\n'), "stdout lacks newline: {stdout:?}");
    assert_eq!(stdout.lines().count(), 1, "stdout={stdout:?}");
    serde_json::from_str(stdout).expect("stdout is one JSON object")
}

fn seed(root: &Path, body: &str) {
    let path = root.join("test/resolver-note.md");
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(
        path,
        format!(
            "---\ntitle: Resolver Note\naliases:\n  - resolver choice\n  - root precedence\ntype: fact\ncreated: 2026-08-09\nupdated: 2026-08-09\n---\n{body}\n"
        ),
    )
    .unwrap();
}

fn write_config(path: &Path, root: &Path) {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, serde_json::json!({ "root": root }).to_string()).unwrap();
}

fn read_body(command: &mut Command) -> String {
    let out = command.output().expect("spawn read");
    json_stdout(&out)["body"]
        .as_str()
        .unwrap()
        .trim()
        .to_string()
}

#[test]
fn help_lists_exactly_the_eight_commands() {
    let out = output(&["--help"]);
    assert_success(&out);
    let stdout = String::from_utf8(out.stdout).unwrap();
    let expected = BTreeSet::from([
        "serve", "store", "search", "read", "forget", "stats", "doctor", "reindex",
    ]);
    let commands: BTreeSet<_> = stdout
        .split("Commands:\n")
        .nth(1)
        .unwrap()
        .split("\nOptions:")
        .next()
        .unwrap()
        .lines()
        .filter_map(|line| line.split_whitespace().next())
        .collect();
    assert_eq!(commands, expected);
    for command in expected {
        let sub = output(&[command, "--help"]);
        assert_success(&sub);
    }
}

#[test]
fn required_cli_arguments_are_enforced() {
    for args in [vec!["store"], vec!["search"], vec!["read"], vec!["forget"]] {
        let out = output(&args);
        assert!(!out.status.success(), "unexpected success for {args:?}");
        assert!(
            String::from_utf8_lossy(&out.stderr).contains("required"),
            "args={args:?} stderr={}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    let dir = tempdir().unwrap();
    let out = bin()
        .args(["store", "--root"])
        .arg(dir.path())
        .args([
            "--title",
            "One Alias",
            "--body",
            "body",
            "--alias",
            "only one",
            "--type",
            "fact",
        ])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("at least 2"));
}

#[test]
fn tool_mirrors_emit_one_json_object() {
    let dir = tempdir().unwrap();
    let root = dir.path().join("memory");

    let stored = json_stdout(
        &bin()
            .args([
                "--root",
                root.to_str().unwrap(),
                "store",
                "--title",
                "Deploy Policy",
                "--body",
                "Prefer blue-green releases.",
                "--alias",
                "release process",
                "--alias",
                "deployment rules",
                "--namespace",
                "ops",
                "--type",
                "decision",
                "--tag",
                "deploy",
            ])
            .output()
            .unwrap(),
    );
    assert_eq!(stored["slug"], "deploy-policy");
    assert_eq!(stored["action"], "created");

    let searched = json_stdout(
        &bin()
            .args(["search", "deploy", "--root", root.to_str().unwrap()])
            .output()
            .unwrap(),
    );
    assert!(searched["results"].is_array());
    assert!(searched["stages_run"].is_array());

    let read = json_stdout(
        &bin()
            .args([
                "read",
                "ops/deploy-policy",
                "--root",
                root.to_str().unwrap(),
            ])
            .output()
            .unwrap(),
    );
    assert_eq!(read["frontmatter"]["title"], "Deploy Policy");

    let stats = json_stdout(
        &bin()
            .args(["stats", "--root", root.to_str().unwrap()])
            .output()
            .unwrap(),
    );
    assert_eq!(stats["total_notes"], 1);

    let forgot = json_stdout(
        &bin()
            .args([
                "forget",
                "ops/deploy-policy",
                "--hard",
                "--root",
                root.to_str().unwrap(),
            ])
            .output()
            .unwrap(),
    );
    assert_eq!(forgot["action"], "deleted");
}

#[test]
fn root_precedence_is_flag_then_env_then_config_then_xdg_default() {
    let dir = tempdir().unwrap();
    let flag = dir.path().join("flag");
    let env = dir.path().join("env");
    let configured = dir.path().join("configured");
    let xdg_data = dir.path().join("xdg-data");
    let default = xdg_data.join("fff-memory");
    let config = dir.path().join("config.json");
    for (root, body) in [
        (&flag, "flag"),
        (&env, "env"),
        (&configured, "config"),
        (&default, "default"),
    ] {
        seed(root, body);
    }
    write_config(&config, &configured);

    let mut command = bin();
    command
        .env("FFF_MEMORY_ROOT", &env)
        .env("FFF_MEMORY_CONFIG", &config)
        .env("XDG_DATA_HOME", &xdg_data)
        .args(["read", HANDLE, "--root"])
        .arg(&flag);
    assert_eq!(read_body(&mut command), "flag");

    let mut command = bin();
    command
        .env("FFF_MEMORY_ROOT", &env)
        .env("FFF_MEMORY_CONFIG", &config)
        .env("XDG_DATA_HOME", &xdg_data)
        .args(["read", HANDLE]);
    assert_eq!(read_body(&mut command), "env");

    let mut command = bin();
    command
        .env("FFF_MEMORY_CONFIG", &config)
        .env("XDG_DATA_HOME", &xdg_data)
        .args(["read", HANDLE]);
    assert_eq!(read_body(&mut command), "config");

    let mut command = bin();
    command
        .env("XDG_DATA_HOME", &xdg_data)
        .env("HOME", dir.path())
        .args(["read", HANDLE]);
    assert_eq!(read_body(&mut command), "default");

    let home = dir.path().join("home-default");
    seed(
        &home.join(".local/share/fff-memory"),
        "home fallback default",
    );
    let mut command = bin();
    command.env("HOME", &home).args(["read", HANDLE]);
    assert_eq!(read_body(&mut command), "home fallback default");
}

#[test]
fn config_path_precedence_is_flag_then_env_then_xdg_or_home_default() {
    let dir = tempdir().unwrap();
    let roots: Vec<PathBuf> = ["flag", "env", "xdg", "home"]
        .into_iter()
        .map(|name| dir.path().join(format!("root-{name}")))
        .collect();
    for (root, body) in roots.iter().zip(["flag", "env", "xdg", "home"]) {
        seed(root, body);
    }
    let flag_config = dir.path().join("flag.json");
    let env_config = dir.path().join("env.json");
    let xdg_home = dir.path().join("xdg-config");
    let xdg_config = xdg_home.join("fff-memory/config.json");
    let home = dir.path().join("home");
    let home_config = home.join(".config/fff-memory/config.json");
    for (path, root) in [
        (&flag_config, &roots[0]),
        (&env_config, &roots[1]),
        (&xdg_config, &roots[2]),
        (&home_config, &roots[3]),
    ] {
        write_config(path, root);
    }

    let mut command = bin();
    command
        .env("FFF_MEMORY_CONFIG", &env_config)
        .env("XDG_CONFIG_HOME", &xdg_home)
        .args(["read", HANDLE, "--config"])
        .arg(&flag_config);
    assert_eq!(read_body(&mut command), "flag");

    let mut command = bin();
    command
        .env("FFF_MEMORY_CONFIG", &env_config)
        .env("XDG_CONFIG_HOME", &xdg_home)
        .args(["read", HANDLE]);
    assert_eq!(read_body(&mut command), "env");

    let mut command = bin();
    command
        .env("XDG_CONFIG_HOME", &xdg_home)
        .args(["read", HANDLE]);
    assert_eq!(read_body(&mut command), "xdg");

    let mut command = bin();
    command
        .env_remove("XDG_CONFIG_HOME")
        .env("HOME", &home)
        .args(["read", HANDLE]);
    assert_eq!(read_body(&mut command), "home");
}

#[test]
fn project_selects_dot_memory_and_conflicts_only_with_explicit_root() {
    let dir = tempdir().unwrap();
    let project = dir.path().join("project");
    let project_root = project.join(".memory");
    let env_root = dir.path().join("env-root");
    seed(&project_root, "project");
    seed(&env_root, "env");

    let mut command = bin();
    command
        .env("FFF_MEMORY_ROOT", &env_root)
        .env("FFF_MEMORY_CONFIG", dir.path().join("missing-config.json"))
        .args(["read", HANDLE, "--project"])
        .arg(&project);
    let out = command.output().unwrap();
    assert!(
        !out.status.success(),
        "an explicitly selected missing config is invalid"
    );

    let config_root = dir.path().join("config-root");
    seed(&config_root, "config");
    let config = dir.path().join("config.json");
    write_config(&config, &config_root);
    let mut command = bin();
    command
        .env("FFF_MEMORY_ROOT", &env_root)
        .env("FFF_MEMORY_CONFIG", &config)
        .args(["read", HANDLE, "--project"])
        .arg(&project);
    assert_eq!(read_body(&mut command), "project");

    let mut command = bin();
    command
        .args(["--project"])
        .arg(&project)
        .args(["read", HANDLE]);
    assert_eq!(read_body(&mut command), "project");

    let out = bin()
        .args(["read", HANDLE, "--project"])
        .arg(&project)
        .args(["--root"])
        .arg(&env_root)
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("cannot be used with"));

    let cwd_project = dir.path().join("cwd-project");
    seed(&cwd_project.join(".memory"), "cwd-project");
    let mut command = bin();
    command
        .current_dir(&cwd_project)
        .args(["read", HANDLE, "--project"]);
    assert_eq!(read_body(&mut command), "cwd-project");
}

#[test]
fn config_errors_name_the_path_and_the_fix() {
    let dir = tempdir().unwrap();
    let cases = [
        ("malformed.json", "{"),
        ("wrong-type.json", r#"{"root": 42}"#),
        ("missing-root.json", r#"{}"#),
        ("unknown.json", r#"{"root":"ok","extra":true}"#),
    ];
    for (name, contents) in cases {
        let path = dir.path().join(name);
        fs::write(&path, contents).unwrap();
        let out = bin()
            .args(["read", HANDLE, "--config"])
            .arg(&path)
            .output()
            .unwrap();
        assert!(!out.status.success(), "case={name}");
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(stderr.contains(path.to_str().unwrap()), "stderr={stderr}");
        assert!(stderr.contains("{\"root\":\"/path\"}"), "stderr={stderr}");
    }

    let missing = dir.path().join("missing.json");
    for selector in ["flag", "env"] {
        let mut command = bin();
        command.args(["read", HANDLE]);
        if selector == "flag" {
            command.arg("--config").arg(&missing);
        } else {
            command.env("FFF_MEMORY_CONFIG", &missing);
        }
        let out = command.output().unwrap();
        assert!(!out.status.success());
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(stderr.contains(missing.to_str().unwrap()));
        assert!(
            stderr.contains("create the file or choose"),
            "stderr={stderr}"
        );
    }

    let out = bin()
        .args(["--root"])
        .arg(dir.path().join("root"))
        .args(["--config"])
        .arg(&missing)
        .arg("stats")
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains(missing.to_str().unwrap()));
}

#[test]
fn missing_default_config_is_harmless() {
    let dir = tempdir().unwrap();
    let root = dir.path().join("data/fff-memory");
    seed(&root, "default without config");
    let mut command = bin();
    command
        .env("HOME", dir.path().join("empty-home"))
        .env("XDG_DATA_HOME", dir.path().join("data"))
        .env("XDG_CONFIG_HOME", dir.path().join("empty-config"))
        .args(["read", HANDLE]);
    assert_eq!(read_body(&mut command), "default without config");
}

#[test]
fn unavailable_index_does_not_block_store_or_truthful_empty_search() {
    let dir = tempdir().unwrap();
    let root = dir.path().join("memory");
    fs::create_dir_all(&root).unwrap();
    fs::create_dir(root.join(".index")).unwrap();
    fs::write(root.join(".index/frecency"), "not a directory").unwrap();

    let out = bin()
        .args(["--root"])
        .arg(&root)
        .args([
            "store",
            "--title",
            "Index Independent",
            "--body",
            "Files remain authoritative.",
            "--alias",
            "store without index",
            "--alias",
            "retrieval unavailable",
            "--type",
            "fact",
        ])
        .output()
        .unwrap();
    let stored = json_stdout(&out);
    assert_eq!(stored["action"], "created");
    assert_eq!(
        String::from_utf8_lossy(&out.stderr)
            .matches("retrieval unavailable")
            .count(),
        1
    );

    let out = bin()
        .args(["search", "index", "--root"])
        .arg(&root)
        .output()
        .unwrap();
    let searched = json_stdout(&out);
    assert_eq!(searched["results"], serde_json::json!([]));
    assert_eq!(searched["stages_run"], serde_json::json!([]));
    assert!(searched["empty_hint"]
        .as_str()
        .unwrap()
        .contains("index unavailable"));
    assert_eq!(
        String::from_utf8_lossy(&out.stderr)
            .matches("retrieval unavailable")
            .count(),
        1
    );

    let out = bin().args(["stats", "--root"]).arg(&root).output().unwrap();
    let stats = json_stdout(&out);
    assert_eq!(stats["index"]["state"], "unavailable");
    assert_eq!(
        String::from_utf8_lossy(&out.stderr)
            .matches("retrieval unavailable")
            .count(),
        1
    );
}
