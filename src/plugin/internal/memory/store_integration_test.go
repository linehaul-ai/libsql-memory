//go:build integration

package memory

import (
	"context"
	"fmt"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/testcontainers/testcontainers-go"
	"github.com/testcontainers/testcontainers-go/wait"

	"github.com/libsql-memory/plugin/internal/db"
	"github.com/libsql-memory/plugin/internal/embedding"
)

// TestContainerEnv holds the test container environment.
type TestContainerEnv struct {
	Container testcontainers.Container
	DBPath    string
}

// SetupTestContainer creates a LibSQL container for integration testing.
func SetupTestContainer(ctx context.Context) (*TestContainerEnv, error) {
	req := testcontainers.ContainerRequest{
		Image:        "ghcr.io/tursodatabase/libsql-server:latest",
		ExposedPorts: []string{"8080/tcp"},
		Env: map[string]string{
			"SQLD_NODE": "primary",
		},
		WaitingFor: wait.ForHTTP("/health").WithPort("8080/tcp").WithStartupTimeout(60 * time.Second),
	}

	container, err := testcontainers.GenericContainer(ctx, testcontainers.GenericContainerRequest{
		ContainerRequest: req,
		Started:          true,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to start container: %w", err)
	}

	host, err := container.Host(ctx)
	if err != nil {
		container.Terminate(ctx)
		return nil, fmt.Errorf("failed to get container host: %w", err)
	}

	port, err := container.MappedPort(ctx, "8080")
	if err != nil {
		container.Terminate(ctx)
		return nil, fmt.Errorf("failed to get mapped port: %w", err)
	}

	dbPath := fmt.Sprintf("http://%s:%s", host, port.Port())

	return &TestContainerEnv{
		Container: container,
		DBPath:    dbPath,
	}, nil
}

// Cleanup terminates the container.
func (e *TestContainerEnv) Cleanup(ctx context.Context) error {
	if e.Container != nil {
		return e.Container.Terminate(ctx)
	}
	return nil
}

// MockEmbedder is a simple embedder for testing that generates deterministic embeddings.
type MockEmbedder struct {
	dimensions int
	mu         sync.Mutex
	callCount  int
}

func NewMockEmbedder(dimensions int) *MockEmbedder {
	return &MockEmbedder{dimensions: dimensions}
}

func (m *MockEmbedder) Embed(ctx context.Context, text string) (embedding.Vector, error) {
	m.mu.Lock()
	m.callCount++
	m.mu.Unlock()

	vec := make(embedding.Vector, m.dimensions)
	for i := range vec {
		// Create deterministic embedding based on text hash
		vec[i] = float32(i%100)/100.0 + float32(len(text)%10)/1000.0
	}
	return vec, nil
}

func (m *MockEmbedder) EmbedBatch(ctx context.Context, texts []string) ([]embedding.Vector, error) {
	vectors := make([]embedding.Vector, len(texts))
	for i, text := range texts {
		vec, err := m.Embed(ctx, text)
		if err != nil {
			return nil, err
		}
		vectors[i] = vec
	}
	return vectors, nil
}

func (m *MockEmbedder) Dimension() int {
	return m.dimensions
}

func (m *MockEmbedder) Close() error {
	return nil
}

func (m *MockEmbedder) CallCount() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.callCount
}

// createTestStore creates a memory store for testing with a file-based database.
func createTestStore(t *testing.T, dimensions int) (*Store, func()) {
	t.Helper()

	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "test-memory.db")

	ctx := context.Background()

	cfg := db.Config{
		Path:             dbPath,
		VectorDimensions: dimensions,
		MaxOpenConns:     5,
		MaxIdleConns:     2,
	}

	database, err := db.New(ctx, cfg)
	require.NoError(t, err)

	embedder := NewMockEmbedder(dimensions)
	store := NewStore(database, embedder)

	cleanup := func() {
		store.Close()
	}

	return store, cleanup
}

func TestIntegration_StoreBasicOperations(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test in short mode")
	}

	store, cleanup := createTestStore(t, 384)
	defer cleanup()

	ctx := context.Background()

	t.Run("StoreAndRetrieve", func(t *testing.T) {
		err := store.Store(ctx, "test-ns", "key-1", "This is a test value", map[string]any{
			"source": "test",
		})
		require.NoError(t, err)

		mem, err := store.Retrieve(ctx, "test-ns", "key-1")
		require.NoError(t, err)

		assert.Equal(t, "key-1", mem.Key)
		assert.Equal(t, "This is a test value", mem.Value)
		assert.Equal(t, "test-ns", mem.Namespace)
		assert.NotNil(t, mem.Embedding)
		assert.Len(t, mem.Embedding, 384)
	})

	t.Run("StoreWithTags", func(t *testing.T) {
		err := store.Store(ctx, "test-ns", "tagged-key", "Tagged memory value", nil,
			WithTags("important", "work", "project-a"),
		)
		require.NoError(t, err)

		mem, err := store.Retrieve(ctx, "test-ns", "tagged-key")
		require.NoError(t, err)

		assert.True(t, mem.HasTag("important"))
		assert.True(t, mem.HasTag("work"))
		assert.True(t, mem.HasTag("project-a"))
		assert.False(t, mem.HasTag("nonexistent"))
	})

	t.Run("StoreWithTTL", func(t *testing.T) {
		ttl := 1 * time.Hour
		err := store.Store(ctx, "test-ns", "ttl-key", "Memory with TTL", nil,
			WithTTL(ttl),
		)
		require.NoError(t, err)

		mem, err := store.Retrieve(ctx, "test-ns", "ttl-key")
		require.NoError(t, err)

		assert.NotNil(t, mem.TTL)
		assert.Equal(t, ttl, *mem.TTL)
		assert.False(t, mem.IsExpired())
	})

	t.Run("Delete", func(t *testing.T) {
		err := store.Store(ctx, "test-ns", "delete-key", "To be deleted", nil)
		require.NoError(t, err)

		err = store.Delete(ctx, "test-ns", "delete-key")
		require.NoError(t, err)

		_, err = store.Retrieve(ctx, "test-ns", "delete-key")
		assert.ErrorIs(t, err, ErrNotFound)
	})

	t.Run("UpdateExisting", func(t *testing.T) {
		err := store.Store(ctx, "test-ns", "update-key", "Original value", nil)
		require.NoError(t, err)

		err = store.Store(ctx, "test-ns", "update-key", "Updated value", nil)
		require.NoError(t, err)

		mem, err := store.Retrieve(ctx, "test-ns", "update-key")
		require.NoError(t, err)

		assert.Equal(t, "Updated value", mem.Value)
	})
}

func TestIntegration_StoreListOperations(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test in short mode")
	}

	store, cleanup := createTestStore(t, 384)
	defer cleanup()

	ctx := context.Background()

	// Store multiple memories
	for i := 0; i < 10; i++ {
		err := store.Store(ctx, "list-ns", fmt.Sprintf("list-key-%d", i),
			fmt.Sprintf("List value %d", i), nil,
			WithTags(fmt.Sprintf("group-%d", i%3)),
		)
		require.NoError(t, err)
	}

	t.Run("ListAll", func(t *testing.T) {
		memories, err := store.List(ctx, "list-ns", 100)
		require.NoError(t, err)
		assert.Len(t, memories, 10)
	})

	t.Run("ListWithLimit", func(t *testing.T) {
		memories, err := store.List(ctx, "list-ns", 5)
		require.NoError(t, err)
		assert.Len(t, memories, 5)
	})

	t.Run("ListByTags", func(t *testing.T) {
		memories, err := store.ListByTags(ctx, "list-ns", []string{"group-0"}, 100)
		require.NoError(t, err)
		assert.NotEmpty(t, memories)

		for _, m := range memories {
			assert.True(t, m.HasTag("group-0"))
		}
	})

	t.Run("ListByAnyTag", func(t *testing.T) {
		memories, err := store.ListByAnyTag(ctx, "list-ns", []string{"group-0", "group-1"}, 100)
		require.NoError(t, err)
		assert.NotEmpty(t, memories)
	})

	t.Run("Count", func(t *testing.T) {
		count, err := store.Count(ctx, "list-ns")
		require.NoError(t, err)
		assert.Equal(t, int64(10), count)
	})

	t.Run("ListNamespaces", func(t *testing.T) {
		// Store in another namespace
		err := store.Store(ctx, "another-ns", "key", "value", nil)
		require.NoError(t, err)

		namespaces, err := store.ListNamespaces(ctx)
		require.NoError(t, err)
		assert.Contains(t, namespaces, "list-ns")
		assert.Contains(t, namespaces, "another-ns")
	})
}

func TestIntegration_StoreSearch(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test in short mode")
	}

	store, cleanup := createTestStore(t, 384)
	defer cleanup()

	ctx := context.Background()

	// Store memories for search
	testData := []struct {
		key   string
		value string
	}{
		{"golang", "Go is a statically typed, compiled programming language"},
		{"python", "Python is a high-level, general-purpose programming language"},
		{"rust", "Rust is a multi-paradigm, memory-safe programming language"},
		{"javascript", "JavaScript is a dynamic programming language for web development"},
		{"database", "LibSQL is a fork of SQLite with vector search capabilities"},
	}

	for _, td := range testData {
		err := store.Store(ctx, "search-ns", td.key, td.value, nil)
		require.NoError(t, err)
	}

	t.Run("SearchWithNamespace", func(t *testing.T) {
		results, err := store.Search(ctx, "programming language features", "search-ns", 3, 0.0)
		require.NoError(t, err)
		assert.NotEmpty(t, results)
		assert.LessOrEqual(t, len(results), 3)

		// Results should have scores
		for _, r := range results {
			assert.NotNil(t, r.Memory)
			assert.GreaterOrEqual(t, r.Score, 0.0)
		}
	})

	t.Run("SearchAllNamespaces", func(t *testing.T) {
		// Store in another namespace
		err := store.Store(ctx, "other-ns", "java", "Java is an object-oriented programming language", nil)
		require.NoError(t, err)

		results, err := store.Search(ctx, "programming", "", 10, 0.0)
		require.NoError(t, err)
		assert.NotEmpty(t, results)
	})

	t.Run("SearchWithThreshold", func(t *testing.T) {
		results, err := store.Search(ctx, "programming", "search-ns", 10, 0.9)
		require.NoError(t, err)

		// High threshold may return fewer or no results
		for _, r := range results {
			assert.GreaterOrEqual(t, r.Score, 0.9)
		}
	})
}

func TestIntegration_StoreBatchOperations(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test in short mode")
	}

	store, cleanup := createTestStore(t, 384)
	defer cleanup()

	ctx := context.Background()

	t.Run("BatchStore", func(t *testing.T) {
		memories := make([]*Memory, 20)
		for i := range memories {
			memories[i] = &Memory{
				Namespace: "batch-ns",
				Key:       fmt.Sprintf("batch-key-%d", i),
				Value:     fmt.Sprintf("Batch value %d", i),
				Tags:      []string{"batch"},
			}
		}

		err := store.BatchStore(ctx, memories)
		require.NoError(t, err)

		count, err := store.Count(ctx, "batch-ns")
		require.NoError(t, err)
		assert.Equal(t, int64(20), count)
	})

	t.Run("BatchStoreRollback", func(t *testing.T) {
		memories := []*Memory{
			{
				Namespace: "rollback-ns",
				Key:       "valid-key",
				Value:     "Valid value",
			},
			{
				Namespace: "rollback-ns",
				Key:       "", // Invalid - should cause rollback
				Value:     "Invalid key",
			},
		}

		err := store.BatchStore(ctx, memories)
		assert.Error(t, err)

		// First entry should not exist due to rollback
		_, err = store.Retrieve(ctx, "rollback-ns", "valid-key")
		assert.ErrorIs(t, err, ErrNotFound)
	})
}

func TestIntegration_StoreNamespaceOperations(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test in short mode")
	}

	store, cleanup := createTestStore(t, 384)
	defer cleanup()

	ctx := context.Background()

	// Store in namespace
	for i := 0; i < 5; i++ {
		err := store.Store(ctx, "delete-all-ns", fmt.Sprintf("key-%d", i),
			fmt.Sprintf("value %d", i), nil)
		require.NoError(t, err)
	}

	t.Run("DeleteNamespace", func(t *testing.T) {
		count, err := store.DeleteNamespace(ctx, "delete-all-ns")
		require.NoError(t, err)
		assert.Equal(t, int64(5), count)

		remaining, err := store.Count(ctx, "delete-all-ns")
		require.NoError(t, err)
		assert.Equal(t, int64(0), remaining)
	})
}

func TestIntegration_StoreConcurrency(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test in short mode")
	}

	store, cleanup := createTestStore(t, 384)
	defer cleanup()

	ctx := context.Background()

	// SQLite has limited concurrent write support (single writer at a time)
	// So we test sequential writes followed by concurrent reads
	const numEntries = 20

	// Sequential writes
	for i := 0; i < numEntries; i++ {
		key := fmt.Sprintf("concurrent-%d", i)
		err := store.Store(ctx, "concurrent-ns", key, fmt.Sprintf("Value %d", i), nil)
		require.NoError(t, err)
	}

	// Concurrent reads
	const numReaders = 5
	var wg sync.WaitGroup
	errChan := make(chan error, numReaders*numEntries)

	for r := 0; r < numReaders; r++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < numEntries; i++ {
				key := fmt.Sprintf("concurrent-%d", i)
				_, err := store.Retrieve(ctx, "concurrent-ns", key)
				if err != nil {
					errChan <- fmt.Errorf("retrieve failed for %s: %w", key, err)
					return
				}
			}
		}()
	}

	wg.Wait()
	close(errChan)

	for err := range errChan {
		t.Errorf("concurrent operation error: %v", err)
	}

	count, err := store.Count(ctx, "concurrent-ns")
	require.NoError(t, err)
	assert.Equal(t, int64(numEntries), count)
}

func TestIntegration_StoreValidation(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test in short mode")
	}

	store, cleanup := createTestStore(t, 384)
	defer cleanup()

	ctx := context.Background()

	t.Run("EmptyNamespace", func(t *testing.T) {
		err := store.Store(ctx, "", "key", "value", nil)
		assert.ErrorIs(t, err, ErrInvalidNS)
	})

	t.Run("EmptyKey", func(t *testing.T) {
		err := store.Store(ctx, "ns", "", "value", nil)
		assert.ErrorIs(t, err, ErrInvalidKey)
	})

	t.Run("EmptyValue", func(t *testing.T) {
		err := store.Store(ctx, "ns", "key", "", nil)
		assert.ErrorIs(t, err, ErrEmptyValue)
	})

	t.Run("TooLongNamespace", func(t *testing.T) {
		longNS := make([]byte, 300)
		for i := range longNS {
			longNS[i] = 'a'
		}
		err := store.Store(ctx, string(longNS), "key", "value", nil)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "too long")
	})

	t.Run("TooLongKey", func(t *testing.T) {
		longKey := make([]byte, 600)
		for i := range longKey {
			longKey[i] = 'a'
		}
		err := store.Store(ctx, "ns", string(longKey), "value", nil)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "too long")
	})
}

func TestIntegration_StoreStats(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test in short mode")
	}

	store, cleanup := createTestStore(t, 384)
	defer cleanup()

	ctx := context.Background()

	// Store some data
	for ns := range []string{"stats-ns-1", "stats-ns-2"} {
		for i := 0; i < 3; i++ {
			err := store.Store(ctx, fmt.Sprintf("stats-ns-%d", ns+1),
				fmt.Sprintf("key-%d", i), fmt.Sprintf("value %d", i), nil)
			require.NoError(t, err)
		}
	}

	t.Run("GetStats", func(t *testing.T) {
		stats, err := store.Stats(ctx)
		require.NoError(t, err)

		assert.Greater(t, stats.TotalMemories, int64(0))
		assert.NotEmpty(t, stats.NamespaceCounts)
	})
}

func TestIntegration_StoreClosedOperations(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test in short mode")
	}

	store, cleanup := createTestStore(t, 384)
	cleanup() // Close immediately

	ctx := context.Background()

	t.Run("StoreAfterClose", func(t *testing.T) {
		err := store.Store(ctx, "ns", "key", "value", nil)
		assert.ErrorIs(t, err, ErrStoreClosed)
	})

	t.Run("RetrieveAfterClose", func(t *testing.T) {
		_, err := store.Retrieve(ctx, "ns", "key")
		assert.ErrorIs(t, err, ErrStoreClosed)
	})

	t.Run("DeleteAfterClose", func(t *testing.T) {
		err := store.Delete(ctx, "ns", "key")
		assert.ErrorIs(t, err, ErrStoreClosed)
	})

	t.Run("ListAfterClose", func(t *testing.T) {
		_, err := store.List(ctx, "ns", 10)
		assert.ErrorIs(t, err, ErrStoreClosed)
	})

	t.Run("SearchAfterClose", func(t *testing.T) {
		_, err := store.Search(ctx, "query", "ns", 10, 0.5)
		assert.ErrorIs(t, err, ErrStoreClosed)
	})
}

// BenchmarkIntegration_StoreStore benchmarks store operations.
func BenchmarkIntegration_StoreStore(b *testing.B) {
	if testing.Short() {
		b.Skip("skipping integration benchmark in short mode")
	}

	tmpDir := b.TempDir()
	dbPath := filepath.Join(tmpDir, "bench-memory.db")

	ctx := context.Background()

	cfg := db.Config{
		Path:             dbPath,
		VectorDimensions: 384,
	}

	database, err := db.New(ctx, cfg)
	if err != nil {
		b.Fatalf("failed to create DB: %v", err)
	}
	defer database.Close()

	embedder := NewMockEmbedder(384)
	store := NewStore(database, embedder)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		err := store.Store(ctx, "benchmark", fmt.Sprintf("key-%d", i), "Benchmark test value", nil)
		if err != nil {
			b.Fatalf("store failed: %v", err)
		}
	}
}

// BenchmarkIntegration_StoreSearch benchmarks search operations.
func BenchmarkIntegration_StoreSearch(b *testing.B) {
	if testing.Short() {
		b.Skip("skipping integration benchmark in short mode")
	}

	tmpDir := b.TempDir()
	dbPath := filepath.Join(tmpDir, "bench-search.db")

	ctx := context.Background()

	cfg := db.Config{
		Path:             dbPath,
		VectorDimensions: 384,
	}

	database, err := db.New(ctx, cfg)
	if err != nil {
		b.Fatalf("failed to create DB: %v", err)
	}
	defer database.Close()

	embedder := NewMockEmbedder(384)
	store := NewStore(database, embedder)

	// Pre-populate
	for i := 0; i < 500; i++ {
		err := store.Store(ctx, "benchmark", fmt.Sprintf("key-%d", i), fmt.Sprintf("Search benchmark value %d", i), nil)
		if err != nil {
			b.Fatalf("store failed: %v", err)
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := store.Search(ctx, "benchmark search query", "benchmark", 10, 0.5)
		if err != nil {
			b.Fatalf("search failed: %v", err)
		}
	}
}
