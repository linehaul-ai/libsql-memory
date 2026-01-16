---
description: Guide for effectively using persistent memory in Claude Code sessions
triggers:
  - how to use memory
  - when to store memories
  - memory best practices
  - what should I remember
---

# Memory Usage Best Practices

This skill guides effective use of the libsql-memory persistent storage.

## When to Store Memories

**DO store:**
- Key architectural decisions and their rationale
- User preferences discovered during conversation
- Important patterns or conventions for this project
- Solutions to complex problems that took investigation
- Project-specific context not in documentation

**DON'T store:**
- Information already in code comments or README
- Temporary debugging steps
- Trivial facts or obvious information
- Sensitive data (credentials, API keys)

## When to Search Memories

**Search before:**
- Starting a new task (check for relevant context)
- Making assumptions about project conventions
- Implementing features similar to past work
- Answering questions about project history

## Namespace Organization

Use namespaces to organize memories:
- Project name for project-specific context
- Language name (go, typescript, python) for language patterns
- "preferences" for user preferences
- "decisions" for architectural decisions

## Example Usage

Storing:
```
memory_store(
  content: "User prefers functional React components over class components",
  namespace: "preferences",
  metadata: {"project": "frontend-app"}
)
```

Searching:
```
memory_search(
  query: "React component conventions",
  namespace: "preferences",
  limit: 5
)
```

## Automated Memory Retrieval

The plugin includes automation hooks that handle memory operations automatically:

### SessionStart
- Detects project type and loads relevant context
- Searches memory for stored project preferences and conventions
- Provides context summary at session start

### UserPromptSubmit
- Proactively searches memory for context relevant to your query
- Extracts technical terms and searches for related patterns
- Injects relevant memories before processing your request

### PreToolUse (Write/Edit)
- Retrieves file-specific patterns before modifying code
- Searches for conventions related to the package/module
- Ensures consistency with established patterns

### PostToolUse (Write/Edit)
- Evaluates if new patterns or decisions should be stored
- Automatically captures architectural decisions
- Stores non-obvious solutions for future reference

### Stop
- Reviews the session for valuable learnings
- Stores key decisions, patterns, and user preferences
- Creates session summaries for complex work

## Namespace Strategy

Organize memories by purpose:
- `{project}/preferences` - User preferences for this project
- `{project}/conventions` - Coding patterns and style decisions
- `{project}/architecture` - Structural decisions and rationale
- `{project}/errors` - Error patterns and solutions
- `{project}/sessions` - Session summaries
- `global` - Cross-project preferences

## Key Principles

1. **Quality over quantity** - Store meaningful insights, not everything
2. **Search before storing** - Avoid duplicates
3. **Use specific namespaces** - Makes retrieval more accurate
4. **Include context** - Add metadata for better organization
5. **Trust automation** - Hooks handle routine memory operations
