# libsql-memory Plugin for Claude Code

Persistent semantic memory for Claude Code sessions using LibSQL with vector search.

## Quick Start

```bash
# Clone the repo
git clone https://github.com/linehaul-ai/libsql-memory.git
cd libsql-memory

# Run Claude Code with the plugin
claude --plugin-dir .
```

That's it! The plugin will automatically:
- Start the MCP server with memory tools
- Load automation hooks for context retrieval
- Enable `/memory-status` and `/memory-clear` commands

## Prerequisites

**Nomic embedding server** running at http://192.168.128.10:1234 (default)

Or use local embeddings (no external dependencies):
```bash
export LIBSQL_MEMORY_EMBEDDING_PROVIDER=local
claude --plugin-dir /path/to/libsql-memory
```

Or use OpenAI:
```bash
export LIBSQL_MEMORY_EMBEDDING_PROVIDER=openai
export OPENAI_API_KEY=your-key
claude --plugin-dir /path/to/libsql-memory
```

## Alternative Installation

Copy to your project as a local plugin:
```bash
# Copy entire plugin directory
cp -r /path/to/libsql-memory /your/project/libsql-memory

# Then run Claude with the plugin directory
cd /your/project && claude --plugin-dir ./libsql-memory
```

Or install globally:
```bash
claude plugins add https://github.com/linehaul-ai/libsql-memory.git
```

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

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LIBSQL_MEMORY_EMBEDDING_PROVIDER` | `nomic` | Embedding provider: `nomic`, `openai`, `local` |
| `LIBSQL_MEMORY_EMBEDDING_ENDPOINT` | `http://192.168.128.10:1234/v1/embeddings` | Nomic server endpoint |
| `OPENAI_API_KEY` | - | Required if using OpenAI provider |

### Database
Stored at `.claude/memory.db` relative to where Claude Code runs.

## Supported Platforms

Pre-built binaries are included for:
- macOS (Apple Silicon / arm64)

For other platforms, build from source:
```bash
cd src/plugin
make build
```

## Building from Source

```bash
cd src/plugin
make build          # Build for current platform
make test           # Run tests
make verify         # Run all checks
```

## Troubleshooting

1. **MCP server not starting**: Check binary exists in `bin/` directory
2. **Embedding errors**: Ensure Nomic server is running or switch to `local` provider
3. **Database errors**: Ensure `.claude/` directory exists and is writable
4. **Platform not supported**: Build from source using `make build`

## License

MIT
