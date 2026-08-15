---
description: Run the report-only fff-memory doctor.
disable-model-invocation: true
allowed-tools:
  - Bash(cargo run --quiet --manifest-path "${CLAUDE_PLUGIN_ROOT}/Cargo.toml" --bin fff-memory -- doctor --project "${CLAUDE_PROJECT_DIR}")
---

Run `cargo run --quiet --manifest-path "${CLAUDE_PLUGIN_ROOT}/Cargo.toml" --bin fff-memory -- doctor --project "${CLAUDE_PROJECT_DIR}"` and report its findings. Do not repair, archive, or delete anything.
