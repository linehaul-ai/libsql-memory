# LibSQL Memory Plugin for Claude

A persistent memory plugin for Claude that uses LibSQL (SQLite fork with vector support) for storage and semantic search capabilities. Implements the Model Context Protocol (MCP) for seamless integration with Claude.

## Features

- **Persistent Memory Storage**: Store and retrieve memories with key-value pairs
- **Semantic Search**: Vector-based similarity search using embeddings
- **Namespace Support**: Organize memories into logical namespaces
- **Multiple Embedding Providers**:
  - **Nomic** (default): Local `text-embedding-nomic-embed-text-v1.5@q8_0` model
  - **OpenAI**: `text-embedding-3-small` API
  - **Local**: TF-IDF/hash-based fallback (no external dependencies)
- **LibSQL Backend**: SQLite-compatible with native vector indexing
- **MCP Protocol**: JSON-RPC 2.0 over stdio for Claude integration
- **Metadata & Tags**: Rich metadata and tagging support for memories

## Building

```bash
cd src/plugin
go build -o libsql-memory ./cmd
```

## Configuration

Configuration can be set via command-line flags, environment variables, or config files (JSON/YAML).

### Command-Line Flags

| Flag | Description | Default |
|------|-------------|---------|
| `-database-path` | Database file path or Turso URL | (required) |
| `-embedding-provider` | Embedding provider: `nomic`, `openai`, or `local` | `nomic` |
| `-embedding-endpoint` | Embedding API endpoint (for nomic) | `http://192.168.128.10:1234/v1/embeddings` |
| `-embedding-dimension` | Vector dimension | `768` |
| `-openai-api-key` | OpenAI API key (required if provider is openai) | - |
| `-default-namespace` | Default namespace for operations | `default` |
| `-max-connections` | Database connection pool size | `10` |
| `-log-level` | Log level: `debug`, `info`, `warn`, `error` | `info` |
| `-config` | Path to config file (JSON or YAML) | - |

### Environment Variables

All flags can be set via environment variables with the `LIBSQL_MEMORY_` prefix:

```bash
export LIBSQL_MEMORY_DATABASE_PATH=./memory.db
export LIBSQL_MEMORY_EMBEDDING_PROVIDER=nomic
export LIBSQL_MEMORY_EMBEDDING_ENDPOINT=http://192.168.128.10:1234/v1/embeddings
export LIBSQL_MEMORY_EMBEDDING_DIMENSION=768
export LIBSQL_MEMORY_OPENAI_API_KEY=sk-...  # if using openai provider
export LIBSQL_MEMORY_DEFAULT_NAMESPACE=default
export LIBSQL_MEMORY_LOG_LEVEL=info
```

### Config File (YAML)

```yaml
database_path: ./memory.db
embedding_provider: nomic
embedding_endpoint: http://192.168.128.10:1234/v1/embeddings
embedding_dimension: 768
default_namespace: default
max_connections: 10
log_level: info
```

### Config File (JSON)

```json
{
  "database_path": "./memory.db",
  "embedding_provider": "nomic",
  "embedding_endpoint": "http://192.168.128.10:1234/v1/embeddings",
  "embedding_dimension": 768,
  "default_namespace": "default",
  "max_connections": 10,
  "log_level": "info"
}
```

## Usage

### Running the Server

```bash
# With local Nomic model (default)
./libsql-memory -database-path ./memory.db

# With OpenAI embeddings
./libsql-memory -database-path ./memory.db \
    -embedding-provider openai \
    -openai-api-key sk-your-key \
    -embedding-dimension 1536

# With local TF-IDF fallback (no external API)
./libsql-memory -database-path ./memory.db \
    -embedding-provider local \
    -embedding-dimension 384

# With config file
./libsql-memory -config ./config.yaml
```

### Claude Code Integration

Add the plugin to Claude Code:

```bash
claude mcp add libsql-memory -- /path/to/libsql-memory -database-path /path/to/memory.db
```

### Claude Desktop Integration

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "libsql-memory": {
      "command": "/path/to/libsql-memory",
      "args": ["-database-path", "/path/to/memory.db"]
    }
  }
}
```

## MCP Tools

The plugin exposes the following tools via MCP:

### `memory_store`

Store a memory with optional metadata and tags.

```json
{
  "key": "user-preference",
  "value": "User prefers dark mode and compact layouts",
  "namespace": "preferences",
  "metadata": {"source": "conversation", "confidence": "high"},
  "tags": ["ui", "settings"]
}
```

### `memory_retrieve`

Retrieve a specific memory by key.

```json
{
  "key": "user-preference",
  "namespace": "preferences"
}
```

### `memory_search`

Semantic search across memories.

```json
{
  "query": "user interface preferences",
  "namespace": "",
  "limit": 10,
  "threshold": 0.5
}
```

### `memory_list`

List memories in a namespace.

```json
{
  "namespace": "preferences",
  "limit": 50
}
```

### `memory_delete`

Delete a memory by key.

```json
{
  "key": "user-preference",
  "namespace": "preferences"
}
```

### `memory_stats`

Get statistics about stored memories.

```json
{
  "namespace": ""
}
```

## Embedding Providers

### Nomic (Default)

Uses a locally running Nomic embedding model with OpenAI-compatible API:

- **Model**: `text-embedding-nomic-embed-text-v1.5@q8_0`
- **Dimension**: 768
- **Endpoint**: `http://192.168.128.10:1234/v1/embeddings`

Requires a local server running the Nomic model (e.g., via LM Studio, Ollama, or similar).

### OpenAI

Uses OpenAI's embedding API:

- **Model**: `text-embedding-3-small`
- **Dimension**: 1536 (configurable)
- **Requires**: `LIBSQL_MEMORY_OPENAI_API_KEY`

### Local

TF-IDF and hash-based embeddings with no external dependencies:

- **Dimension**: 384 (configurable)
- **No API required**
- **Lower quality** but works offline

## Architecture

```
src/plugin/
├── cmd/
│   └── main.go              # Entry point and MCP server setup
├── internal/
│   ├── config/              # Configuration management
│   ├── db/                  # LibSQL database layer
│   ├── embedding/           # Embedding providers (Nomic, OpenAI, Local)
│   ├── mcp/                 # MCP protocol implementation
│   └── memory/              # Memory store operations
└── pkg/
    └── types/               # Shared types
```

## Development

### Running Tests

```bash
go test ./... -v
```

### Building with Version Info

```bash
go build -ldflags "-X main.version=1.0.0 -X main.commit=$(git rev-parse HEAD) -X main.buildDate=$(date -u +%Y-%m-%dT%H:%M:%SZ)" -o libsql-memory ./cmd
```

## License

MIT
