#!/bin/bash
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)
[[ "$ARCH" == "x86_64" ]] && ARCH="amd64"
[[ "$ARCH" == "aarch64" ]] && ARCH="arm64"

BINARY="${CLAUDE_PLUGIN_ROOT}/bin/libsql-memory-${OS}-${ARCH}"
[[ "$OS" == "windows" ]] && BINARY="${BINARY}.exe"

exec "$BINARY" \
  --database-path ".claude/memory.db" \
  --embedding-provider "${LIBSQL_MEMORY_EMBEDDING_PROVIDER:-nomic}" \
  --embedding-endpoint "${LIBSQL_MEMORY_EMBEDDING_ENDPOINT:-http://192.168.128.10:1234/v1/embeddings}"
