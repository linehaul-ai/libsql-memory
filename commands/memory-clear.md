---
allowed-tools:
  - mcp__libsql-memory__memory_list
  - mcp__libsql-memory__memory_delete
  - AskUserQuestion
---

Clear memories from a namespace.

Arguments: $ARGUMENTS (namespace name, defaults to "default" if empty)

Steps:
1. Parse namespace from $ARGUMENTS, use "default" if not provided
2. Use memory_list to show entries in the namespace
3. Ask user to confirm deletion using AskUserQuestion
4. If confirmed, delete all entries in that namespace using memory_delete
5. Confirm completion
