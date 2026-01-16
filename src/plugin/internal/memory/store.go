// Package memory provides memory storage and retrieval operations for the Claude memory plugin.
// It wraps the underlying database layer and provides a clean interface for memory operations
// with support for TTL expiration, tags, and semantic search.
package memory

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync/atomic"
	"time"

	"github.com/google/uuid"

	"github.com/libsql-memory/plugin/internal/db"
	"github.com/libsql-memory/plugin/internal/embedding"
)

// Common errors returned by the memory store.
var (
	ErrNotFound     = errors.New("memory not found")
	ErrExpired      = errors.New("memory has expired")
	ErrInvalidKey   = errors.New("invalid key")
	ErrInvalidNS    = errors.New("invalid namespace")
	ErrEmptyValue   = errors.New("value cannot be empty")
	ErrDuplicateKey = errors.New("key already exists in namespace")
	ErrStoreClosed  = errors.New("memory store is closed")
)

// Memory represents a stored memory entry with all associated data.
type Memory struct {
	ID        string         `json:"id"`
	Namespace string         `json:"namespace"`
	Key       string         `json:"key"`
	Value     string         `json:"value"`
	Embedding []float32      `json:"embedding,omitempty"`
	Metadata  map[string]any `json:"metadata,omitempty"`
	CreatedAt time.Time      `json:"created_at"`
	UpdatedAt time.Time      `json:"updated_at"`
	TTL       *time.Duration `json:"ttl,omitempty"`
	Tags      []string       `json:"tags,omitempty"`
}

// SearchResult represents a memory with its similarity score.
type SearchResult struct {
	Memory *Memory `json:"memory"`
	Score  float64 `json:"score"`
}

// IsExpired checks if the memory has exceeded its TTL.
func (m *Memory) IsExpired() bool {
	if m.TTL == nil {
		return false
	}
	expiresAt := m.CreatedAt.Add(*m.TTL)
	return time.Now().After(expiresAt)
}

// ExpiresAt returns the expiration time, or nil if no TTL is set.
func (m *Memory) ExpiresAt() *time.Time {
	if m.TTL == nil {
		return nil
	}
	t := m.CreatedAt.Add(*m.TTL)
	return &t
}

// HasTag checks if the memory has a specific tag (case-insensitive).
func (m *Memory) HasTag(tag string) bool {
	for _, t := range m.Tags {
		if strings.EqualFold(t, tag) {
			return true
		}
	}
	return false
}

// HasAllTags checks if the memory has all specified tags.
func (m *Memory) HasAllTags(tags []string) bool {
	for _, tag := range tags {
		if !m.HasTag(tag) {
			return false
		}
	}
	return true
}

// HasAnyTag checks if the memory has any of the specified tags.
func (m *Memory) HasAnyTag(tags []string) bool {
	for _, tag := range tags {
		if m.HasTag(tag) {
			return true
		}
	}
	return len(tags) == 0
}

// MemoryStore defines the interface for memory storage operations.
type MemoryStore interface {
	// Store saves a memory entry with the given namespace, key, value, and metadata.
	Store(ctx context.Context, namespace, key, value string, metadata map[string]any) error

	// Retrieve fetches a memory entry by namespace and key.
	Retrieve(ctx context.Context, namespace, key string) (*Memory, error)

	// Delete removes a memory entry by namespace and key.
	Delete(ctx context.Context, namespace, key string) error

	// List returns memory entries in a namespace, up to the specified limit.
	List(ctx context.Context, namespace string, limit int) ([]*Memory, error)

	// Search performs a semantic search using embeddings and returns matching memories with scores.
	Search(ctx context.Context, query string, namespace string, limit int, threshold float64) ([]SearchResult, error)

	// Close releases any resources held by the store.
	Close() error
}

// StoreOptions configures optional parameters for storing memories.
type StoreOptions struct {
	TTL       *time.Duration
	Tags      []string
	Embedding []float32
}

// StoreOption is a functional option for configuring store operations.
type StoreOption func(*StoreOptions)

// WithTTL sets a time-to-live for the memory entry.
func WithTTL(ttl time.Duration) StoreOption {
	return func(o *StoreOptions) {
		o.TTL = &ttl
	}
}

// WithTags sets tags for the memory entry.
func WithTags(tags ...string) StoreOption {
	return func(o *StoreOptions) {
		o.Tags = tags
	}
}

// WithEmbedding sets a pre-computed embedding for the memory entry.
func WithEmbedding(emb []float32) StoreOption {
	return func(o *StoreOptions) {
		o.Embedding = emb
	}
}

// Store implements the MemoryStore interface using the database layer.
type Store struct {
	database *db.DB
	embedder embedding.Embedder
	closed   atomic.Bool
}

// NewStore creates a new memory store with the given database connection and embedder.
func NewStore(database *db.DB, embedder embedding.Embedder) *Store {
	return &Store{
		database: database,
		embedder: embedder,
	}
}

// Store saves a memory entry to the database.
func (s *Store) Store(ctx context.Context, namespace, key, value string, metadata map[string]any, opts ...StoreOption) error {
	if s.closed.Load() {
		return ErrStoreClosed
	}

	if err := validateNamespace(namespace); err != nil {
		return err
	}
	if err := validateKey(key); err != nil {
		return err
	}
	if value == "" {
		return ErrEmptyValue
	}

	options := &StoreOptions{}
	for _, opt := range opts {
		opt(options)
	}

	// Generate embedding if not provided and embedder is available
	var emb []float32
	if options.Embedding != nil {
		emb = options.Embedding
	} else if s.embedder != nil {
		vec, err := s.embedder.Embed(ctx, value)
		if err != nil {
			return fmt.Errorf("failed to generate embedding: %w", err)
		}
		emb = []float32(vec)
	}

	// Merge tags into metadata for storage
	if metadata == nil {
		metadata = make(map[string]any)
	}
	if len(options.Tags) > 0 {
		metadata["_tags"] = options.Tags
	}

	// Create the db.Memory struct
	dbMemory := &db.Memory{
		ID:        uuid.New().String(),
		Namespace: namespace,
		Key:       key,
		Value:     value,
		Embedding: emb,
		Metadata:  metadata,
		TTL:       options.TTL,
	}

	if err := s.database.Store(ctx, dbMemory); err != nil {
		return fmt.Errorf("failed to store memory: %w", err)
	}

	return nil
}

// Retrieve fetches a memory entry by namespace and key.
func (s *Store) Retrieve(ctx context.Context, namespace, key string) (*Memory, error) {
	if s.closed.Load() {
		return nil, ErrStoreClosed
	}

	if err := validateNamespace(namespace); err != nil {
		return nil, err
	}
	if err := validateKey(key); err != nil {
		return nil, err
	}

	dbMemory, err := s.database.Retrieve(ctx, namespace, key)
	if err != nil {
		if errors.Is(err, db.ErrNotFound) {
			return nil, ErrNotFound
		}
		return nil, fmt.Errorf("failed to retrieve memory: %w", err)
	}

	return convertDBMemory(dbMemory), nil
}

// Delete removes a memory entry by namespace and key.
func (s *Store) Delete(ctx context.Context, namespace, key string) error {
	if s.closed.Load() {
		return ErrStoreClosed
	}

	if err := validateNamespace(namespace); err != nil {
		return err
	}
	if err := validateKey(key); err != nil {
		return err
	}

	if err := s.database.Delete(ctx, namespace, key); err != nil {
		if errors.Is(err, db.ErrNotFound) {
			return ErrNotFound
		}
		return fmt.Errorf("failed to delete memory: %w", err)
	}

	return nil
}

// List returns memory entries in a namespace, ordered by creation time descending.
func (s *Store) List(ctx context.Context, namespace string, limit int) ([]*Memory, error) {
	if s.closed.Load() {
		return nil, ErrStoreClosed
	}

	if err := validateNamespace(namespace); err != nil {
		return nil, err
	}

	if limit <= 0 {
		limit = 100
	}
	if limit > 1000 {
		limit = 1000
	}

	dbMemories, err := s.database.List(ctx, namespace, limit, 0)
	if err != nil {
		return nil, fmt.Errorf("failed to list memories: %w", err)
	}

	memories := make([]*Memory, 0, len(dbMemories))
	for _, dbMem := range dbMemories {
		memories = append(memories, convertDBMemory(dbMem))
	}

	return memories, nil
}

// Search performs a semantic search using embeddings.
func (s *Store) Search(ctx context.Context, query string, namespace string, limit int, threshold float64) ([]SearchResult, error) {
	if s.closed.Load() {
		return nil, ErrStoreClosed
	}

	if query == "" {
		return nil, errors.New("search query cannot be empty")
	}

	if namespace != "" {
		if err := validateNamespace(namespace); err != nil {
			return nil, err
		}
	}

	if limit <= 0 {
		limit = 10
	}
	if limit > 100 {
		limit = 100
	}
	if threshold <= 0 {
		threshold = 0.7
	}
	if threshold > 1 {
		threshold = 1.0
	}

	// Generate embedding for query
	if s.embedder == nil {
		return nil, errors.New("embedder not configured for search")
	}

	queryVec, err := s.embedder.Embed(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to generate query embedding: %w", err)
	}

	// Use namespace search if specified, otherwise search all
	var dbResults []db.SearchResult
	if namespace != "" {
		dbResults, err = s.database.Search(ctx, namespace, []float32(queryVec), limit, threshold)
	} else {
		dbResults, err = s.database.SearchAll(ctx, []float32(queryVec), limit, threshold)
	}
	if err != nil {
		return nil, fmt.Errorf("failed to search memories: %w", err)
	}

	results := make([]SearchResult, 0, len(dbResults))
	for _, r := range dbResults {
		results = append(results, SearchResult{
			Memory: convertDBMemory(&r.Memory),
			Score:  r.Similarity,
		})
	}

	return results, nil
}

// ListByTags returns memories that have all specified tags.
func (s *Store) ListByTags(ctx context.Context, namespace string, tags []string, limit int) ([]*Memory, error) {
	if s.closed.Load() {
		return nil, ErrStoreClosed
	}

	if len(tags) == 0 {
		return s.List(ctx, namespace, limit)
	}

	// Fetch more entries to account for filtering
	memories, err := s.List(ctx, namespace, limit*3)
	if err != nil {
		return nil, err
	}

	filtered := make([]*Memory, 0)
	for _, m := range memories {
		if m.HasAllTags(tags) {
			filtered = append(filtered, m)
			if len(filtered) >= limit {
				break
			}
		}
	}

	return filtered, nil
}

// ListByAnyTag returns memories that have any of the specified tags.
func (s *Store) ListByAnyTag(ctx context.Context, namespace string, tags []string, limit int) ([]*Memory, error) {
	if s.closed.Load() {
		return nil, ErrStoreClosed
	}

	if len(tags) == 0 {
		return s.List(ctx, namespace, limit)
	}

	memories, err := s.List(ctx, namespace, limit*3)
	if err != nil {
		return nil, err
	}

	filtered := make([]*Memory, 0)
	for _, m := range memories {
		if m.HasAnyTag(tags) {
			filtered = append(filtered, m)
			if len(filtered) >= limit {
				break
			}
		}
	}

	return filtered, nil
}

// DeleteNamespace removes all memories in a namespace.
func (s *Store) DeleteNamespace(ctx context.Context, namespace string) (int64, error) {
	if s.closed.Load() {
		return 0, ErrStoreClosed
	}

	if err := validateNamespace(namespace); err != nil {
		return 0, err
	}

	return s.database.DeleteNamespace(ctx, namespace)
}

// Count returns the number of memories in a namespace.
func (s *Store) Count(ctx context.Context, namespace string) (int64, error) {
	if s.closed.Load() {
		return 0, ErrStoreClosed
	}

	if namespace != "" {
		if err := validateNamespace(namespace); err != nil {
			return 0, err
		}
		return s.database.Count(ctx, namespace)
	}

	return s.database.CountAll(ctx)
}

// ListNamespaces returns all unique namespaces.
func (s *Store) ListNamespaces(ctx context.Context) ([]string, error) {
	if s.closed.Load() {
		return nil, ErrStoreClosed
	}

	return s.database.ListNamespaces(ctx)
}

// Stats returns database statistics.
func (s *Store) Stats(ctx context.Context) (*db.Stats, error) {
	if s.closed.Load() {
		return nil, ErrStoreClosed
	}

	return s.database.GetStats(ctx)
}

// Close releases resources held by the store.
func (s *Store) Close() error {
	if s.closed.Load() {
		return ErrStoreClosed
	}

	s.closed.Store(true)

	var errs []error

	if s.embedder != nil {
		if err := s.embedder.Close(); err != nil {
			errs = append(errs, fmt.Errorf("failed to close embedder: %w", err))
		}
	}

	if s.database != nil {
		if err := s.database.Close(); err != nil {
			errs = append(errs, fmt.Errorf("failed to close database: %w", err))
		}
	}

	if len(errs) > 0 {
		return errors.Join(errs...)
	}

	return nil
}

// Helper functions

func validateNamespace(namespace string) error {
	if namespace == "" {
		return ErrInvalidNS
	}
	if len(namespace) > 255 {
		return fmt.Errorf("%w: namespace too long (max 255 characters)", ErrInvalidNS)
	}
	return nil
}

func validateKey(key string) error {
	if key == "" {
		return ErrInvalidKey
	}
	if len(key) > 512 {
		return fmt.Errorf("%w: key too long (max 512 characters)", ErrInvalidKey)
	}
	return nil
}

// convertDBMemory converts a db.Memory to our Memory type with tags support.
func convertDBMemory(dbMem *db.Memory) *Memory {
	mem := &Memory{
		ID:        dbMem.ID,
		Namespace: dbMem.Namespace,
		Key:       dbMem.Key,
		Value:     dbMem.Value,
		Embedding: dbMem.Embedding,
		Metadata:  dbMem.Metadata,
		CreatedAt: dbMem.CreatedAt,
		UpdatedAt: dbMem.UpdatedAt,
		TTL:       dbMem.TTL,
	}

	// Extract tags from metadata if present
	if dbMem.Metadata != nil {
		if tags, ok := dbMem.Metadata["_tags"]; ok {
			switch t := tags.(type) {
			case []string:
				mem.Tags = t
			case []interface{}:
				mem.Tags = make([]string, 0, len(t))
				for _, v := range t {
					if s, ok := v.(string); ok {
						mem.Tags = append(mem.Tags, s)
					}
				}
			case string:
				// Handle JSON-encoded tags
				var parsedTags []string
				if err := json.Unmarshal([]byte(t), &parsedTags); err == nil {
					mem.Tags = parsedTags
				}
			}

			// Remove internal _tags from public metadata
			delete(mem.Metadata, "_tags")
			if len(mem.Metadata) == 0 {
				mem.Metadata = nil
			}
		}
	}

	return mem
}

// BatchStore stores multiple memories in a single transaction.
func (s *Store) BatchStore(ctx context.Context, memories []*Memory) error {
	if s.closed.Load() {
		return ErrStoreClosed
	}

	if len(memories) == 0 {
		return nil
	}

	tx, err := s.database.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
	}

	for _, mem := range memories {
		if err := validateNamespace(mem.Namespace); err != nil {
			_ = tx.Rollback()
			return err
		}
		if err := validateKey(mem.Key); err != nil {
			_ = tx.Rollback()
			return err
		}
		if mem.Value == "" {
			_ = tx.Rollback()
			return ErrEmptyValue
		}

		// Generate embedding if not provided
		var emb []float32
		if mem.Embedding != nil {
			emb = mem.Embedding
		} else if s.embedder != nil {
			vec, err := s.embedder.Embed(ctx, mem.Value)
			if err != nil {
				_ = tx.Rollback()
				return fmt.Errorf("failed to generate embedding: %w", err)
			}
			emb = []float32(vec)
		}

		// Merge tags into metadata
		metadata := mem.Metadata
		if metadata == nil {
			metadata = make(map[string]any)
		}
		if len(mem.Tags) > 0 {
			metadata["_tags"] = mem.Tags
		}

		dbMem := &db.Memory{
			ID:        mem.ID,
			Namespace: mem.Namespace,
			Key:       mem.Key,
			Value:     mem.Value,
			Embedding: emb,
			Metadata:  metadata,
			TTL:       mem.TTL,
		}

		if dbMem.ID == "" {
			dbMem.ID = uuid.New().String()
		}

		if err := tx.StoreTx(ctx, dbMem); err != nil {
			_ = tx.Rollback()
			return fmt.Errorf("failed to store memory %s/%s: %w", mem.Namespace, mem.Key, err)
		}
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("failed to commit transaction: %w", err)
	}

	return nil
}
