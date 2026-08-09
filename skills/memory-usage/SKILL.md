---
name: memory-usage
description: Store and retrieve durable project knowledge with fff-memory.
---

# Memory usage

Search with `memory_search` before relying on prior decisions, preferences, conventions, or hard-won fixes. Try 2–3 concise reformulations before concluding no memory exists, then use `memory_read` for the full note behind a result handle.

Store only durable knowledge with `memory_store`: one fact per note, never secrets, routine activity, or facts already obvious from maintained documentation. Supply `title`, `body`, `type`, and 2–6 `aliases`; optional `tags` and `namespace` improve organization. Aliases must use future-question vocabulary, not restate the title.

```text
memory_store(
  title: "Deployments use blue-green releases",
  body: "Production deploys switch traffic after health checks. Why: rollback stays immediate.",
  aliases: ["release process", "how production ships", "rollback strategy"],
  type: "decision",
  tags: ["deployment", "production"],
  namespace: "project/architecture"
)
```

Search before storing. Update the existing note when the same fact changed or gained useful detail; create a new note only for a distinct fact. Use a stable project or repository slug as the namespace, with narrow subtrees such as `project/preferences`, `project/architecture`, or `project/sessions`.

Use `memory_stats` for store health. Use `memory_forget` to archive stale knowledge by default; hard deletion is only for explicitly requested permanent removal.
