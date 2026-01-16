# libsql-memory Plugin for Claude Code

Persistent semantic memory for Claude Code sessions using LibSQL with vector search.

## Prerequisites

1. **Build the MCP server binary:**
   ```bash
   cd src/plugin
   make build
   ```

2. **Nomic embedding server** running at http://192.168.128.10:1234
   - Or configure a different embedding provider in `.mcp.json`

## Installation

Load the plugin in Claude Code:
```bash
claude --plugin-dir /path/to/libsql-memory/.claude-plugin
```

Or add to your Claude Code settings.

## Components

### MCP Server
Provides memory tools:
- `memory_store` - Store a memory with content and namespace
- `memory_retrieve` - Get a specific memory by ID
- `memory_search` - Semantic search across memories
- `memory_list` - List memories in a namespace
- `memory_delete` - Delete a memory
- `memory_stats` - Get storage statistics

### Hooks

| Hook | Purpose |
|------|---------|
| **SessionStart** | Loads project context and preferences at session start |
| **UserPromptSubmit** | Searches memory for context relevant to user queries |
| **PreToolUse** (Write/Edit) | Retrieves file-specific patterns before code changes |
| **PostToolUse** (Write/Edit) | Stores architectural decisions and patterns after changes |
| **Stop** | Saves session summaries and important learnings |

### Commands
- `/memory-status` - View memory statistics and recent entries
- `/memory-clear [namespace]` - Clear memories from a namespace

### Skills
- **memory-usage** - Best practices for when and how to use memory effectively

## Configuration

Database is stored at `.claude/memory.db` relative to where Claude Code runs.

To change embedding provider, edit `.mcp.json`:
```json
{
  "mcpServers": {
    "libsql-memory": {
      "args": [
        "--embedding-provider", "local"
      ]
    }
  }
}
```

Available providers: `nomic`, `openai`, `local`

## Troubleshooting

1. **MCP server not starting**: Ensure binary is built at `src/plugin/build/libsql-memory`
2. **Embedding errors**: Check that Nomic server is running or switch to `local` provider
3. **Database errors**: Ensure `.claude/` directory exists and is writable
