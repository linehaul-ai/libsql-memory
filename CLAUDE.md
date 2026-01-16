# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

libsql-memory is a Go-based persistent memory plugin for Claude using LibSQL (SQLite fork with vector support). It implements the Model Context Protocol (MCP) for integration with Claude, enabling long-term memory with vector-based semantic search.

## Build & Development Commands

All commands run from `src/plugin/`:

```bash
make build          # Build main plugin binary
make build-mcp      # Build MCP server binary
make test           # Run all tests with race detection
make test-verbose   # Run tests with verbose output
make test-coverage  # Generate HTML coverage report
make lint           # Run golangci-lint
make fmt            # Format code with goimports
make vet            # Run go vet
make verify         # Run all checks (fmt, vet, lint, test) - run before commits
make clean          # Remove build artifacts
make dist           # Build multi-platform binaries
```

Build from source:
```bash
cd src/plugin
go build -o libsql-memory ./cmd
```

## Architecture

```
src/plugin/
├── cmd/                    # Entry points
│   ├── main.go             # Primary MCP server binary
│   └── mcp-server/main.go  # Secondary MCP server (development)
├── internal/
│   ├── config/             # Configuration (flags, env, files with priority ordering)
│   ├── db/                 # LibSQL backend with vector storage and cosine similarity search
│   ├── embedding/          # Embedding providers (Nomic, OpenAI, Local TF-IDF)
│   ├── mcp/                # MCP JSON-RPC 2.0 server over stdio
│   └── memory/             # High-level memory store operations
└── pkg/types/              # Shared request/response types
```

## Key Technical Details

**Configuration Priority** (lowest to highest): defaults → config file → environment variables (`LIBSQL_MEMORY_*` prefix) → command-line flags

**Embedding Providers**:
- Nomic: text-embedding-qwen3-embedding-8b model, 768 dimensions, default endpoint `http://192.168.128.10:1234/v1/embeddings`
- OpenAI: `text-embedding-3-small`, 1536 dimensions
- Local: TF-IDF + hash-based, 384 dimensions, zero external dependencies

**Vector Storage**: Float32 arrays serialized as BLOBs with little-endian byte order. Search uses in-memory cosine similarity calculation (O(n) brute-force).

**MCP Protocol**: JSON-RPC 2.0 over stdio. Exposed tools: `memory_store`, `memory_retrieve`, `memory_search`, `memory_list`, `memory_delete`, `memory_stats`

**Namespaces**: All memories organized by namespace. Empty namespace searches across all namespaces.

**TTL Support**: Background cleanup goroutine runs every minute.

## Conventions

- Go 1.22.0 minimum
- Interface-based design for extensibility (Embedder interface)
- Functional options pattern for configuration
- Sync.RWMutex for database access protection
- Stdout reserved for MCP protocol; all logging to stderr
