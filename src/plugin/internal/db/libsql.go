// Package db provides the LibSQL database layer for the Claude memory plugin.
// It implements connection pooling, vector search capabilities, and graceful shutdown.
// Uses LibSQL (SQLite fork) with native vector indexing support.
package db

import (
	"context"
	"database/sql"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"math"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	_ "github.com/tursodatabase/go-libsql" // LibSQL driver
)

// Common errors returned by the database layer.
var (
	ErrNotFound      = errors.New("memory not found")
	ErrDuplicateKey  = errors.New("duplicate key in namespace")
	ErrInvalidConfig = errors.New("invalid database configuration")
	ErrClosed        = errors.New("database connection closed")
	ErrInvalidVector = errors.New("invalid vector dimensions")
	ErrContextClosed = errors.New("context cancelled or deadline exceeded")
)

// serializeVector converts a []float32 to bytes for storage.
func serializeVector(vec []float32) []byte {
	if vec == nil {
		return nil
	}
	buf := make([]byte, len(vec)*4)
	for i, v := range vec {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}
	return buf
}

// deserializeVector converts bytes back to []float32.
func deserializeVector(data []byte) ([]float32, error) {
	if len(data) == 0 {
		return nil, nil
	}
	if len(data)%4 != 0 {
		return nil, fmt.Errorf("invalid vector data length: %d", len(data))
	}
	vec := make([]float32, len(data)/4)
	for i := range vec {
		vec[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
	}
	return vec, nil
}

// cosineSimilarity calculates the cosine similarity between two vectors.
// Returns a value between -1 and 1, where 1 means identical direction.
func cosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}

	var dotProduct, normA, normB float64
	for i := range a {
		dotProduct += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
}

// Config holds the database configuration options.
type Config struct {
	// Path is the local file path for the database.
	Path string

	// MaxOpenConns sets the maximum number of open connections in the pool.
	MaxOpenConns int

	// MaxIdleConns sets the maximum number of idle connections in the pool.
	MaxIdleConns int

	// ConnMaxLifetime sets the maximum lifetime of a connection.
	ConnMaxLifetime time.Duration

	// ConnMaxIdleTime sets the maximum idle time for a connection.
	ConnMaxIdleTime time.Duration

	// VectorDimensions specifies the embedding vector size (default: 1536 for OpenAI).
	VectorDimensions int
}

// DefaultConfig returns a Config with sensible defaults.
func DefaultConfig() Config {
	return Config{
		MaxOpenConns:     25,
		MaxIdleConns:     10,
		ConnMaxLifetime:  30 * time.Minute,
		ConnMaxIdleTime:  5 * time.Minute,
		VectorDimensions: 1536,
	}
}

// Memory represents a single memory entry in the database.
type Memory struct {
	ID        string         `json:"id"`
	Namespace string         `json:"namespace"`
	Key       string         `json:"key"`
	Value     string         `json:"value"`
	Embedding []float32      `json:"embedding,omitempty"`
	Metadata  map[string]any `json:"metadata,omitempty"`
	Tags      []string       `json:"tags,omitempty"`
	CreatedAt time.Time      `json:"created_at"`
	UpdatedAt time.Time      `json:"updated_at"`
	TTL       *time.Duration `json:"ttl,omitempty"`
	ExpiresAt *time.Time     `json:"expires_at,omitempty"`
}

// SearchResult represents a memory with its similarity score.
type SearchResult struct {
	Memory     Memory  `json:"memory"`
	Similarity float64 `json:"similarity"`
}

// DB is the main database interface for memory operations.
type DB struct {
	db        *sql.DB
	config    Config
	mu        sync.RWMutex
	closed    bool
	closeChan chan struct{}
	wg        sync.WaitGroup
}

// New creates a new DB instance with the given configuration.
func New(ctx context.Context, cfg Config) (*DB, error) {
	if err := validateConfig(cfg); err != nil {
		return nil, fmt.Errorf("config validation failed: %w", err)
	}

	// Apply defaults for unset values
	cfg = applyDefaults(cfg)

	// Build connection string based on path type
	connStr := cfg.Path
	if !strings.HasPrefix(cfg.Path, "file:") && !strings.HasPrefix(cfg.Path, "http") && cfg.Path != ":memory:" {
		connStr = "file:" + cfg.Path
	}

	// Open LibSQL database
	sqlDB, err := sql.Open("libsql", connStr)
	if err != nil {
		return nil, fmt.Errorf("failed to open database: %w", err)
	}

	// Configure connection pool
	sqlDB.SetMaxOpenConns(cfg.MaxOpenConns)
	sqlDB.SetMaxIdleConns(cfg.MaxIdleConns)
	sqlDB.SetConnMaxLifetime(cfg.ConnMaxLifetime)
	sqlDB.SetConnMaxIdleTime(cfg.ConnMaxIdleTime)

	// Enable WAL mode for better concurrent access
	// Use QueryRowContext since PRAGMA returns a row
	var journalMode string
	if err := sqlDB.QueryRowContext(ctx, "PRAGMA journal_mode=WAL").Scan(&journalMode); err != nil {
		sqlDB.Close()
		return nil, fmt.Errorf("failed to enable WAL mode: %w", err)
	}

	// Enable foreign keys
	var fkEnabled int
	if err := sqlDB.QueryRowContext(ctx, "PRAGMA foreign_keys=ON").Scan(&fkEnabled); err != nil {
		// foreign_keys pragma might not return a value, try without scan
		if _, err := sqlDB.ExecContext(ctx, "PRAGMA foreign_keys=ON"); err != nil {
			sqlDB.Close()
			return nil, fmt.Errorf("failed to enable foreign keys: %w", err)
		}
	}

	// Verify connection
	if err := sqlDB.PingContext(ctx); err != nil {
		sqlDB.Close()
		return nil, fmt.Errorf("failed to ping database: %w", err)
	}

	d := &DB{
		db:        sqlDB,
		config:    cfg,
		closeChan: make(chan struct{}),
	}

	// Initialize schema
	if err := d.initSchema(ctx); err != nil {
		d.Close()
		return nil, fmt.Errorf("failed to initialize schema: %w", err)
	}

	// Start background TTL cleanup
	d.startTTLCleanup(ctx)

	return d, nil
}

// validateConfig validates the database configuration.
func validateConfig(cfg Config) error {
	if cfg.Path == "" {
		return fmt.Errorf("%w: Path must be specified", ErrInvalidConfig)
	}
	if cfg.VectorDimensions < 0 {
		return fmt.Errorf("%w: VectorDimensions must be non-negative", ErrInvalidConfig)
	}
	return nil
}

// applyDefaults applies default values to unset configuration fields.
func applyDefaults(cfg Config) Config {
	defaults := DefaultConfig()
	if cfg.MaxOpenConns == 0 {
		cfg.MaxOpenConns = defaults.MaxOpenConns
	}
	if cfg.MaxIdleConns == 0 {
		cfg.MaxIdleConns = defaults.MaxIdleConns
	}
	if cfg.ConnMaxLifetime == 0 {
		cfg.ConnMaxLifetime = defaults.ConnMaxLifetime
	}
	if cfg.ConnMaxIdleTime == 0 {
		cfg.ConnMaxIdleTime = defaults.ConnMaxIdleTime
	}
	if cfg.VectorDimensions == 0 {
		cfg.VectorDimensions = defaults.VectorDimensions
	}
	return cfg
}

// initSchema creates the necessary tables and indices.
func (d *DB) initSchema(ctx context.Context) error {
	// Create memories table - vectors stored as BLOB
	createTableSQL := `
		CREATE TABLE IF NOT EXISTS memories (
			id TEXT PRIMARY KEY,
			namespace TEXT NOT NULL DEFAULT 'default',
			key TEXT NOT NULL,
			value TEXT NOT NULL,
			embedding BLOB,
			metadata TEXT DEFAULT '{}',
			tags TEXT DEFAULT '[]',
			created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
			updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
			ttl INTEGER,
			expires_at DATETIME,
			UNIQUE(namespace, key)
		)
	`

	if _, err := d.db.ExecContext(ctx, createTableSQL); err != nil {
		return fmt.Errorf("failed to create memories table: %w", err)
	}

	// Create indices for efficient queries
	indices := []string{
		`CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(namespace)`,
		`CREATE INDEX IF NOT EXISTS idx_memories_namespace_key ON memories(namespace, key)`,
		`CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at)`,
		`CREATE INDEX IF NOT EXISTS idx_memories_expires_at ON memories(expires_at) WHERE expires_at IS NOT NULL`,
	}

	for _, idx := range indices {
		if _, err := d.db.ExecContext(ctx, idx); err != nil {
			return fmt.Errorf("failed to create index: %w", err)
		}
	}

	return nil
}

// startTTLCleanup starts a background goroutine to clean up expired memories.
func (d *DB) startTTLCleanup(ctx context.Context) {
	d.wg.Add(1)
	go func() {
		defer d.wg.Done()
		ticker := time.NewTicker(time.Minute)
		defer ticker.Stop()

		for {
			select {
			case <-d.closeChan:
				return
			case <-ctx.Done():
				return
			case <-ticker.C:
				cleanupCtx, cleanupCancel := context.WithTimeout(context.Background(), 30*time.Second)
				if err := d.cleanupExpired(cleanupCtx); err != nil {
					log.Printf("[WARN] TTL cleanup failed: %v", err)
				}
				cleanupCancel()
			}
		}
	}()
}

// cleanupExpired removes memories that have passed their expiration time.
func (d *DB) cleanupExpired(ctx context.Context) error {
	d.mu.RLock()
	if d.closed {
		d.mu.RUnlock()
		return ErrClosed
	}
	d.mu.RUnlock()

	_, err := d.db.ExecContext(ctx,
		`DELETE FROM memories WHERE expires_at IS NOT NULL AND expires_at < datetime('now')`,
	)
	return err
}

// Store saves a memory to the database.
func (d *DB) Store(ctx context.Context, m *Memory) error {
	if err := d.checkClosed(); err != nil {
		return err
	}

	if err := ctx.Err(); err != nil {
		return fmt.Errorf("%w: %v", ErrContextClosed, err)
	}

	// Validate embedding dimensions if provided
	if m.Embedding != nil && d.config.VectorDimensions > 0 && len(m.Embedding) != d.config.VectorDimensions {
		return fmt.Errorf("%w: expected %d, got %d", ErrInvalidVector, d.config.VectorDimensions, len(m.Embedding))
	}

	// Marshal metadata to JSON
	metadataJSON, err := json.Marshal(m.Metadata)
	if err != nil {
		return fmt.Errorf("failed to marshal metadata: %w", err)
	}

	// Marshal tags to JSON
	tagsJSON, err := json.Marshal(m.Tags)
	if err != nil {
		return fmt.Errorf("failed to marshal tags: %w", err)
	}

	// Calculate expires_at from TTL if provided
	var expiresAt *time.Time
	if m.TTL != nil {
		t := time.Now().Add(*m.TTL)
		expiresAt = &t
	}

	now := time.Now()
	if m.ID == "" {
		m.ID = generateID()
	}

	// Serialize embedding to binary
	embeddingBlob := serializeVector(m.Embedding)

	query := `
		INSERT INTO memories (id, namespace, key, value, embedding, metadata, tags, created_at, updated_at, ttl, expires_at)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
		ON CONFLICT(namespace, key) DO UPDATE SET
			value = excluded.value,
			embedding = excluded.embedding,
			metadata = excluded.metadata,
			tags = excluded.tags,
			updated_at = excluded.updated_at,
			ttl = excluded.ttl,
			expires_at = excluded.expires_at
	`

	var ttlSeconds *int64
	if m.TTL != nil {
		secs := int64(m.TTL.Seconds())
		ttlSeconds = &secs
	}

	_, err = d.db.ExecContext(ctx, query,
		m.ID,
		m.Namespace,
		m.Key,
		m.Value,
		embeddingBlob,
		string(metadataJSON),
		string(tagsJSON),
		now,
		now,
		ttlSeconds,
		expiresAt,
	)
	if err != nil {
		return fmt.Errorf("failed to store memory: %w", err)
	}

	m.CreatedAt = now
	m.UpdatedAt = now
	m.ExpiresAt = expiresAt

	return nil
}

// Retrieve fetches a memory by namespace and key.
func (d *DB) Retrieve(ctx context.Context, namespace, key string) (*Memory, error) {
	if err := d.checkClosed(); err != nil {
		return nil, err
	}

	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrContextClosed, err)
	}

	query := `
		SELECT id, namespace, key, value, embedding, metadata, tags, created_at, updated_at, ttl, expires_at
		FROM memories
		WHERE namespace = ? AND key = ?
		AND (expires_at IS NULL OR expires_at > datetime('now'))
	`

	row := d.db.QueryRowContext(ctx, query, namespace, key)
	return d.scanMemory(row)
}

// RetrieveByID fetches a memory by its ID.
func (d *DB) RetrieveByID(ctx context.Context, id string) (*Memory, error) {
	if err := d.checkClosed(); err != nil {
		return nil, err
	}

	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrContextClosed, err)
	}

	query := `
		SELECT id, namespace, key, value, embedding, metadata, tags, created_at, updated_at, ttl, expires_at
		FROM memories
		WHERE id = ?
		AND (expires_at IS NULL OR expires_at > datetime('now'))
	`

	row := d.db.QueryRowContext(ctx, query, id)
	return d.scanMemory(row)
}

// scanMemory scans a database row into a Memory struct.
func (d *DB) scanMemory(row *sql.Row) (*Memory, error) {
	var m Memory
	var metadataJSON string
	var tagsJSON string
	var embeddingBlob []byte
	var ttlSeconds sql.NullInt64

	err := row.Scan(
		&m.ID,
		&m.Namespace,
		&m.Key,
		&m.Value,
		&embeddingBlob,
		&metadataJSON,
		&tagsJSON,
		&m.CreatedAt,
		&m.UpdatedAt,
		&ttlSeconds,
		&m.ExpiresAt,
	)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return nil, ErrNotFound
		}
		return nil, fmt.Errorf("failed to scan memory: %w", err)
	}

	// Parse metadata JSON
	if metadataJSON != "" && metadataJSON != "{}" {
		if err := json.Unmarshal([]byte(metadataJSON), &m.Metadata); err != nil {
			return nil, fmt.Errorf("failed to unmarshal metadata: %w", err)
		}
	}

	// Parse tags JSON
	if tagsJSON != "" && tagsJSON != "[]" {
		if err := json.Unmarshal([]byte(tagsJSON), &m.Tags); err != nil {
			return nil, fmt.Errorf("failed to unmarshal tags: %w", err)
		}
	}

	// Parse embedding if present
	if len(embeddingBlob) > 0 {
		m.Embedding, err = deserializeVector(embeddingBlob)
		if err != nil {
			return nil, fmt.Errorf("failed to deserialize embedding: %w", err)
		}
	}

	// Convert TTL seconds back to duration
	if ttlSeconds.Valid {
		ttl := time.Duration(ttlSeconds.Int64) * time.Second
		m.TTL = &ttl
	}

	return &m, nil
}

// Delete removes a memory by namespace and key.
func (d *DB) Delete(ctx context.Context, namespace, key string) error {
	if err := d.checkClosed(); err != nil {
		return err
	}

	if err := ctx.Err(); err != nil {
		return fmt.Errorf("%w: %v", ErrContextClosed, err)
	}

	result, err := d.db.ExecContext(ctx,
		`DELETE FROM memories WHERE namespace = ? AND key = ?`,
		namespace, key,
	)
	if err != nil {
		return fmt.Errorf("failed to delete memory: %w", err)
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get rows affected: %w", err)
	}

	if rowsAffected == 0 {
		return ErrNotFound
	}

	return nil
}

// DeleteByID removes a memory by its ID.
func (d *DB) DeleteByID(ctx context.Context, id string) error {
	if err := d.checkClosed(); err != nil {
		return err
	}

	if err := ctx.Err(); err != nil {
		return fmt.Errorf("%w: %v", ErrContextClosed, err)
	}

	result, err := d.db.ExecContext(ctx, `DELETE FROM memories WHERE id = ?`, id)
	if err != nil {
		return fmt.Errorf("failed to delete memory: %w", err)
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get rows affected: %w", err)
	}

	if rowsAffected == 0 {
		return ErrNotFound
	}

	return nil
}

// List retrieves all memories in a namespace with pagination.
func (d *DB) List(ctx context.Context, namespace string, limit, offset int) ([]*Memory, error) {
	if err := d.checkClosed(); err != nil {
		return nil, err
	}

	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrContextClosed, err)
	}

	if limit <= 0 {
		limit = 100
	}
	if offset < 0 {
		offset = 0
	}

	query := `
		SELECT id, namespace, key, value, embedding, metadata, tags, created_at, updated_at, ttl, expires_at
		FROM memories
		WHERE namespace = ?
		AND (expires_at IS NULL OR expires_at > datetime('now'))
		ORDER BY created_at DESC
		LIMIT ? OFFSET ?
	`

	rows, err := d.db.QueryContext(ctx, query, namespace, limit, offset)
	if err != nil {
		return nil, fmt.Errorf("failed to list memories: %w", err)
	}
	defer rows.Close()

	return d.scanMemories(rows)
}

// scanMemories scans multiple rows into a slice of Memory structs.
func (d *DB) scanMemories(rows *sql.Rows) ([]*Memory, error) {
	var memories []*Memory

	for rows.Next() {
		var m Memory
		var metadataJSON string
		var tagsJSON string
		var embeddingBlob []byte
		var ttlSeconds sql.NullInt64

		err := rows.Scan(
			&m.ID,
			&m.Namespace,
			&m.Key,
			&m.Value,
			&embeddingBlob,
			&metadataJSON,
			&tagsJSON,
			&m.CreatedAt,
			&m.UpdatedAt,
			&ttlSeconds,
			&m.ExpiresAt,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to scan memory row: %w", err)
		}

		// Parse metadata JSON
		if metadataJSON != "" && metadataJSON != "{}" {
			if err := json.Unmarshal([]byte(metadataJSON), &m.Metadata); err != nil {
				return nil, fmt.Errorf("failed to unmarshal metadata: %w", err)
			}
		}

		// Parse tags JSON
		if tagsJSON != "" && tagsJSON != "[]" {
			if err := json.Unmarshal([]byte(tagsJSON), &m.Tags); err != nil {
				return nil, fmt.Errorf("failed to unmarshal tags: %w", err)
			}
		}

		// Parse embedding if present
		if len(embeddingBlob) > 0 {
			var err error
			m.Embedding, err = deserializeVector(embeddingBlob)
			if err != nil {
				return nil, fmt.Errorf("failed to deserialize embedding: %w", err)
			}
		}

		// Convert TTL seconds back to duration
		if ttlSeconds.Valid {
			ttl := time.Duration(ttlSeconds.Int64) * time.Second
			m.TTL = &ttl
		}

		memories = append(memories, &m)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error iterating rows: %w", err)
	}

	return memories, nil
}

// Search performs a vector similarity search using cosine similarity.
// Similarity is calculated in Go after fetching candidates from the database.
func (d *DB) Search(ctx context.Context, namespace string, queryEmbedding []float32, limit int, threshold float64) ([]SearchResult, error) {
	if err := d.checkClosed(); err != nil {
		return nil, err
	}

	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrContextClosed, err)
	}

	if d.config.VectorDimensions > 0 && len(queryEmbedding) != d.config.VectorDimensions {
		return nil, fmt.Errorf("%w: expected %d, got %d", ErrInvalidVector, d.config.VectorDimensions, len(queryEmbedding))
	}

	if limit <= 0 {
		limit = 10
	}
	if threshold < 0 || threshold > 1 {
		threshold = 0.7
	}

	// Fetch all memories with embeddings in the namespace
	query := `
		SELECT id, namespace, key, value, embedding, metadata, tags, created_at, updated_at, ttl, expires_at
		FROM memories
		WHERE namespace = ?
		AND embedding IS NOT NULL
		AND (expires_at IS NULL OR expires_at > datetime('now'))
	`

	rows, err := d.db.QueryContext(ctx, query, namespace)
	if err != nil {
		return nil, fmt.Errorf("failed to execute search query: %w", err)
	}
	defer rows.Close()

	memories, err := d.scanMemories(rows)
	if err != nil {
		return nil, err
	}

	// Calculate similarity for each memory
	var results []SearchResult
	for _, m := range memories {
		if m.Embedding == nil {
			continue
		}

		similarity := cosineSimilarity(queryEmbedding, m.Embedding)
		if similarity >= threshold {
			results = append(results, SearchResult{
				Memory:     *m,
				Similarity: similarity,
			})
		}
	}

	// Sort by similarity (descending)
	sort.Slice(results, func(i, j int) bool {
		return results[i].Similarity > results[j].Similarity
	})

	// Apply limit
	if len(results) > limit {
		results = results[:limit]
	}

	return results, nil
}

// SearchAll performs a vector similarity search across all namespaces.
func (d *DB) SearchAll(ctx context.Context, queryEmbedding []float32, limit int, threshold float64) ([]SearchResult, error) {
	if err := d.checkClosed(); err != nil {
		return nil, err
	}

	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrContextClosed, err)
	}

	if d.config.VectorDimensions > 0 && len(queryEmbedding) != d.config.VectorDimensions {
		return nil, fmt.Errorf("%w: expected %d, got %d", ErrInvalidVector, d.config.VectorDimensions, len(queryEmbedding))
	}

	if limit <= 0 {
		limit = 10
	}
	if threshold < 0 || threshold > 1 {
		threshold = 0.7
	}

	// Fetch all memories with embeddings
	query := `
		SELECT id, namespace, key, value, embedding, metadata, tags, created_at, updated_at, ttl, expires_at
		FROM memories
		WHERE embedding IS NOT NULL
		AND (expires_at IS NULL OR expires_at > datetime('now'))
	`

	rows, err := d.db.QueryContext(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to execute search query: %w", err)
	}
	defer rows.Close()

	memories, err := d.scanMemories(rows)
	if err != nil {
		return nil, err
	}

	// Calculate similarity for each memory
	var results []SearchResult
	for _, m := range memories {
		if m.Embedding == nil {
			continue
		}

		similarity := cosineSimilarity(queryEmbedding, m.Embedding)
		if similarity >= threshold {
			results = append(results, SearchResult{
				Memory:     *m,
				Similarity: similarity,
			})
		}
	}

	// Sort by similarity (descending)
	sort.Slice(results, func(i, j int) bool {
		return results[i].Similarity > results[j].Similarity
	})

	// Apply limit
	if len(results) > limit {
		results = results[:limit]
	}

	return results, nil
}

// Count returns the number of memories in a namespace.
func (d *DB) Count(ctx context.Context, namespace string) (int64, error) {
	if err := d.checkClosed(); err != nil {
		return 0, err
	}

	if err := ctx.Err(); err != nil {
		return 0, fmt.Errorf("%w: %v", ErrContextClosed, err)
	}

	var count int64
	err := d.db.QueryRowContext(ctx,
		`SELECT COUNT(*) FROM memories WHERE namespace = ? AND (expires_at IS NULL OR expires_at > datetime('now'))`,
		namespace,
	).Scan(&count)
	if err != nil {
		return 0, fmt.Errorf("failed to count memories: %w", err)
	}

	return count, nil
}

// CountAll returns the total number of memories across all namespaces.
func (d *DB) CountAll(ctx context.Context) (int64, error) {
	if err := d.checkClosed(); err != nil {
		return 0, err
	}

	if err := ctx.Err(); err != nil {
		return 0, fmt.Errorf("%w: %v", ErrContextClosed, err)
	}

	var count int64
	err := d.db.QueryRowContext(ctx,
		`SELECT COUNT(*) FROM memories WHERE expires_at IS NULL OR expires_at > datetime('now')`,
	).Scan(&count)
	if err != nil {
		return 0, fmt.Errorf("failed to count memories: %w", err)
	}

	return count, nil
}

// ListNamespaces returns all unique namespaces.
func (d *DB) ListNamespaces(ctx context.Context) ([]string, error) {
	if err := d.checkClosed(); err != nil {
		return nil, err
	}

	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrContextClosed, err)
	}

	rows, err := d.db.QueryContext(ctx, `SELECT DISTINCT namespace FROM memories ORDER BY namespace`)
	if err != nil {
		return nil, fmt.Errorf("failed to list namespaces: %w", err)
	}
	defer rows.Close()

	var namespaces []string
	for rows.Next() {
		var ns string
		if err := rows.Scan(&ns); err != nil {
			return nil, fmt.Errorf("failed to scan namespace: %w", err)
		}
		namespaces = append(namespaces, ns)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error iterating namespaces: %w", err)
	}

	return namespaces, nil
}

// DeleteNamespace removes all memories in a namespace.
func (d *DB) DeleteNamespace(ctx context.Context, namespace string) (int64, error) {
	if err := d.checkClosed(); err != nil {
		return 0, err
	}

	if err := ctx.Err(); err != nil {
		return 0, fmt.Errorf("%w: %v", ErrContextClosed, err)
	}

	result, err := d.db.ExecContext(ctx, `DELETE FROM memories WHERE namespace = ?`, namespace)
	if err != nil {
		return 0, fmt.Errorf("failed to delete namespace: %w", err)
	}

	return result.RowsAffected()
}

// Stats returns database statistics.
type Stats struct {
	TotalMemories   int64            `json:"total_memories"`
	NamespaceCounts map[string]int64 `json:"namespace_counts"`
	OldestMemory    *time.Time       `json:"oldest_memory,omitempty"`
	NewestMemory    *time.Time       `json:"newest_memory,omitempty"`
	ExpiringCount   int64            `json:"expiring_count"`
	PoolStats       sql.DBStats      `json:"pool_stats"`
}

// GetStats returns statistics about the database.
func (d *DB) GetStats(ctx context.Context) (*Stats, error) {
	if err := d.checkClosed(); err != nil {
		return nil, err
	}

	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrContextClosed, err)
	}

	stats := &Stats{
		NamespaceCounts: make(map[string]int64),
		PoolStats:       d.db.Stats(),
	}

	// Get total count
	if err := d.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM memories`).Scan(&stats.TotalMemories); err != nil {
		return nil, fmt.Errorf("failed to get total count: %w", err)
	}

	// Get counts per namespace
	rows, err := d.db.QueryContext(ctx, `SELECT namespace, COUNT(*) FROM memories GROUP BY namespace`)
	if err != nil {
		return nil, fmt.Errorf("failed to get namespace counts: %w", err)
	}
	defer rows.Close()

	for rows.Next() {
		var ns string
		var count int64
		if err := rows.Scan(&ns, &count); err != nil {
			return nil, fmt.Errorf("failed to scan namespace count: %w", err)
		}
		stats.NamespaceCounts[ns] = count
	}

	// Get oldest memory
	var oldest sql.NullTime
	if err := d.db.QueryRowContext(ctx, `SELECT MIN(created_at) FROM memories`).Scan(&oldest); err != nil && !errors.Is(err, sql.ErrNoRows) {
		return nil, fmt.Errorf("failed to get oldest memory: %w", err)
	}
	if oldest.Valid {
		stats.OldestMemory = &oldest.Time
	}

	// Get newest memory
	var newest sql.NullTime
	if err := d.db.QueryRowContext(ctx, `SELECT MAX(created_at) FROM memories`).Scan(&newest); err != nil && !errors.Is(err, sql.ErrNoRows) {
		return nil, fmt.Errorf("failed to get newest memory: %w", err)
	}
	if newest.Valid {
		stats.NewestMemory = &newest.Time
	}

	// Get count of expiring memories
	if err := d.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM memories WHERE expires_at IS NOT NULL`).Scan(&stats.ExpiringCount); err != nil {
		return nil, fmt.Errorf("failed to get expiring count: %w", err)
	}

	return stats, nil
}

// checkClosed returns an error if the database is closed.
func (d *DB) checkClosed() error {
	d.mu.RLock()
	defer d.mu.RUnlock()
	if d.closed {
		return ErrClosed
	}
	return nil
}

// Close gracefully shuts down the database connection.
func (d *DB) Close() error {
	d.mu.Lock()
	if d.closed {
		d.mu.Unlock()
		return ErrClosed
	}
	d.closed = true
	d.mu.Unlock()

	// Signal background goroutines to stop
	close(d.closeChan)

	// Wait for background goroutines to finish
	d.wg.Wait()

	// Close database connection
	if err := d.db.Close(); err != nil {
		return fmt.Errorf("failed to close database: %w", err)
	}

	return nil
}

// Ping checks if the database connection is alive.
func (d *DB) Ping(ctx context.Context) error {
	if err := d.checkClosed(); err != nil {
		return err
	}
	return d.db.PingContext(ctx)
}

// generateID generates a unique ID for a memory.
func generateID() string {
	return uuid.New().String()
}

// Transaction support

// Tx wraps a database transaction.
type Tx struct {
	tx *sql.Tx
	db *DB
}

// BeginTx starts a new transaction.
func (d *DB) BeginTx(ctx context.Context, opts *sql.TxOptions) (*Tx, error) {
	if err := d.checkClosed(); err != nil {
		return nil, err
	}

	tx, err := d.db.BeginTx(ctx, opts)
	if err != nil {
		return nil, fmt.Errorf("failed to begin transaction: %w", err)
	}

	return &Tx{tx: tx, db: d}, nil
}

// Commit commits the transaction.
func (t *Tx) Commit() error {
	return t.tx.Commit()
}

// Rollback aborts the transaction.
func (t *Tx) Rollback() error {
	return t.tx.Rollback()
}

// StoreTx stores a memory within a transaction.
func (t *Tx) StoreTx(ctx context.Context, m *Memory) error {
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("%w: %v", ErrContextClosed, err)
	}

	// Validate embedding dimensions if provided
	if m.Embedding != nil && t.db.config.VectorDimensions > 0 && len(m.Embedding) != t.db.config.VectorDimensions {
		return fmt.Errorf("%w: expected %d, got %d", ErrInvalidVector, t.db.config.VectorDimensions, len(m.Embedding))
	}

	// Marshal metadata to JSON
	metadataJSON, err := json.Marshal(m.Metadata)
	if err != nil {
		return fmt.Errorf("failed to marshal metadata: %w", err)
	}

	// Marshal tags to JSON
	tagsJSON, err := json.Marshal(m.Tags)
	if err != nil {
		return fmt.Errorf("failed to marshal tags: %w", err)
	}

	// Calculate expires_at from TTL if provided
	var expiresAt *time.Time
	if m.TTL != nil {
		exp := time.Now().Add(*m.TTL)
		expiresAt = &exp
	}

	now := time.Now()
	if m.ID == "" {
		m.ID = generateID()
	}

	// Serialize embedding to binary
	embeddingBlob := serializeVector(m.Embedding)

	query := `
		INSERT INTO memories (id, namespace, key, value, embedding, metadata, tags, created_at, updated_at, ttl, expires_at)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
		ON CONFLICT(namespace, key) DO UPDATE SET
			value = excluded.value,
			embedding = excluded.embedding,
			metadata = excluded.metadata,
			tags = excluded.tags,
			updated_at = excluded.updated_at,
			ttl = excluded.ttl,
			expires_at = excluded.expires_at
	`

	var ttlSeconds *int64
	if m.TTL != nil {
		secs := int64(m.TTL.Seconds())
		ttlSeconds = &secs
	}

	_, err = t.tx.ExecContext(ctx, query,
		m.ID,
		m.Namespace,
		m.Key,
		m.Value,
		embeddingBlob,
		string(metadataJSON),
		string(tagsJSON),
		now,
		now,
		ttlSeconds,
		expiresAt,
	)
	if err != nil {
		return fmt.Errorf("failed to store memory in transaction: %w", err)
	}

	m.CreatedAt = now
	m.UpdatedAt = now
	m.ExpiresAt = expiresAt

	return nil
}

// DeleteTx deletes a memory within a transaction.
func (t *Tx) DeleteTx(ctx context.Context, namespace, key string) error {
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("%w: %v", ErrContextClosed, err)
	}

	result, err := t.tx.ExecContext(ctx,
		`DELETE FROM memories WHERE namespace = ? AND key = ?`,
		namespace, key,
	)
	if err != nil {
		return fmt.Errorf("failed to delete memory in transaction: %w", err)
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get rows affected: %w", err)
	}

	if rowsAffected == 0 {
		return ErrNotFound
	}

	return nil
}
