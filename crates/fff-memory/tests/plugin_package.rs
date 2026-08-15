use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::{json, Value};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .unwrap()
        .to_path_buf()
}

fn read(rel: &str) -> String {
    fs::read_to_string(repo_root().join(rel)).unwrap_or_else(|e| panic!("read {rel}: {e}"))
}

fn json_file(rel: &str) -> Value {
    serde_json::from_str(&read(rel)).unwrap_or_else(|e| panic!("parse {rel}: {e}"))
}

fn memory_tool_names(text: &str) -> BTreeSet<&str> {
    text.split(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, '_' | '-')))
        .filter(|word| {
            word.strip_prefix("memory_")
                .is_some_and(|name| name.starts_with(char::is_alphanumeric))
        })
        .collect()
}

fn assert_current_text(rel: &str) {
    const ORIGIN: &str = "https://github.com/linehaul-ai/libsql-memory.git";
    let text = read(rel).replace(ORIGIN, "").to_ascii_lowercase();
    for stale in [
        "/users/",
        "bin/run.sh",
        "src/plugin",
        "embedding",
        "libsql",
        "vector",
        "memory_list",
        "memory_delete",
    ] {
        assert!(!text.contains(stale), "{rel} contains obsolete {stale:?}");
    }
}

#[test]
fn package_is_portable_rust_fff_memory() {
    let mcp = json_file(".mcp.json");
    assert_eq!(
        mcp,
        json!({
            "mcpServers": {
                "fff-memory": {
                    "command": "cargo",
                    "args": [
                        "run",
                        "--quiet",
                        "--manifest-path",
                        "${CLAUDE_PLUGIN_ROOT}/Cargo.toml",
                        "--bin",
                        "fff-memory",
                        "--",
                        "serve",
                        "--project",
                        "${CLAUDE_PROJECT_DIR}"
                    ]
                }
            }
        })
    );

    let manifest = json_file(".claude-plugin/plugin.json");
    assert_eq!(manifest["name"], "fff-memory");
    assert_eq!(
        manifest["repository"],
        "https://github.com/linehaul-ai/libsql-memory.git"
    );
    assert!(manifest["description"]
        .as_str()
        .unwrap()
        .contains("lexical"));
    for component in ["skills", "commands", "hooks", "mcpServers"] {
        assert!(manifest.get(component).is_none(), "metadata-only manifest");
    }

    let marketplace = json_file(".claude-plugin/marketplace.json");
    let entries = marketplace["plugins"].as_array().unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0]["name"], "fff-memory");
    assert_eq!(entries[0]["source"], "./");
    assert!(entries[0]["description"]
        .as_str()
        .unwrap()
        .contains("lexical"));

    for rel in [
        ".mcp.json",
        ".claude-plugin/plugin.json",
        ".claude-plugin/marketplace.json",
        "hooks/hooks.json",
        "skills/memory-usage/SKILL.md",
        "commands/memory-status.md",
        "commands/memory-doctor.md",
    ] {
        assert_current_text(rel);
    }
}

#[test]
fn hooks_are_exactly_three_thin_handlers() {
    let config = json_file("hooks/hooks.json");
    let hooks = config["hooks"].as_object().unwrap();
    assert_eq!(
        hooks.keys().map(String::as_str).collect::<BTreeSet<_>>(),
        BTreeSet::from(["SessionStart", "Stop", "UserPromptSubmit"])
    );

    for event in ["SessionStart", "UserPromptSubmit", "Stop"] {
        let groups = hooks[event].as_array().unwrap();
        assert_eq!(groups.len(), 1, "{event} has one matcher group");
        let handlers = groups[0]["hooks"].as_array().unwrap();
        assert_eq!(handlers.len(), 1, "{event} has one handler");
    }

    for event in ["SessionStart", "UserPromptSubmit"] {
        let handler = &hooks[event][0]["hooks"][0];
        assert_eq!(handler["type"], "mcp_tool");
        assert_eq!(handler["server"], "plugin:fff-memory:fff-memory");
        assert_eq!(handler["tool"], "memory_search");
        assert_eq!(handler["timeout"], 5);
    }
    let prompt = &hooks["UserPromptSubmit"][0]["hooks"][0];
    assert_eq!(
        prompt["input"]["query"],
        "__fff_memory_user_prompt__:${prompt}"
    );
    assert_eq!(prompt["input"]["budget_bytes"], 2048);

    let stop = &hooks["Stop"][0]["hooks"][0];
    assert_eq!(stop["type"], "agent");
    let stop_prompt = stop["prompt"].as_str().unwrap();
    for required in [
        "memory_store",
        "mcp__plugin_fff-memory_fff-memory__memory_store",
        "already allowed",
        "at most once",
        "session-summary",
        "2–6",
        "skip",
    ] {
        assert!(
            stop_prompt.contains(required),
            "Stop prompt lacks {required:?}"
        );
    }
    assert!(stop_prompt.lines().count() <= 3);
}

#[test]
fn skill_and_commands_expose_only_the_five_tool_contract() {
    let skill = read("skills/memory-usage/SKILL.md");
    assert!(skill.starts_with("---\nname: memory-usage\ndescription:"));
    let skill_lower = skill.to_ascii_lowercase();
    for required in [
        "title",
        "body",
        "aliases",
        "type",
        "tags",
        "namespace",
        "2–6",
        "future-question",
        "one fact per note",
        "update",
    ] {
        assert!(skill_lower.contains(required), "skill lacks {required:?}");
    }
    let valid = BTreeSet::from([
        "memory_store",
        "memory_search",
        "memory_read",
        "memory_forget",
        "memory_stats",
    ]);
    assert_eq!(memory_tool_names(&skill), valid);
    assert!(skill.contains("mcp__plugin_fff-memory_fff-memory__memory_store"));
    assert!(skill_lower.contains("pre-approve"));
    assert!(skill_lower.contains("stop"));

    let command_names: BTreeSet<_> = fs::read_dir(repo_root().join("commands"))
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .collect();
    assert_eq!(
        command_names,
        BTreeSet::from(["memory-doctor.md".into(), "memory-status.md".into()])
    );
    let status = read("commands/memory-status.md");
    assert!(status.contains("mcp__plugin_fff-memory_fff-memory__memory_stats"));
    assert!(!status.contains("mcp__fff-memory__memory_stats"));
    assert!(!status.contains("memory_search"));
    let doctor = read("commands/memory-doctor.md");
    assert!(doctor.contains("disable-model-invocation: true"));
    assert!(doctor.contains(
        "Bash(cargo run --quiet --manifest-path \"${CLAUDE_PLUGIN_ROOT}/Cargo.toml\" --bin fff-memory -- doctor --project \"${CLAUDE_PROJECT_DIR}\")"
    ));
    assert!(!doctor.contains("manifest-path *)"));
    assert!(doctor.contains("cargo run"));
    assert!(doctor.contains("fff-memory"));
    assert!(doctor.contains("doctor"));
    assert!(doctor.contains("report"));

    for rel in [
        "hooks/hooks.json",
        "commands/memory-status.md",
        "commands/memory-doctor.md",
    ] {
        let text = read(rel);
        let invalid: Vec<_> = memory_tool_names(&text)
            .into_iter()
            .filter(|name| !valid.contains(name))
            .collect();
        assert!(invalid.is_empty(), "{rel} has invalid tools: {invalid:?}");
    }
}

#[test]
fn ci_runs_all_four_workspace_gates_from_the_repo_root() {
    let ci = read(".github/workflows/ci.yml");
    assert!(
        !ci.contains("working-directory"),
        "workspace is at the repo root; CI needs no working-directory"
    );
    for gate in [
        "cargo fmt --all -- --check",
        "cargo test --workspace",
        "cargo clippy --workspace --all-targets -- -D warnings",
        "cargo build --workspace",
    ] {
        assert!(ci.contains(gate), "CI lacks {gate:?}");
    }
}

#[test]
fn local_claude_validates_the_plugin_when_available() {
    let available = Command::new("claude").arg("--version").output();
    if !available.is_ok_and(|output| output.status.success()) {
        return;
    }

    let output = Command::new("claude")
        .args(["plugin", "validate", "--strict", "."])
        .current_dir(repo_root())
        .output()
        .expect("run local Claude plugin validator");
    assert!(
        output.status.success(),
        "validator failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}
