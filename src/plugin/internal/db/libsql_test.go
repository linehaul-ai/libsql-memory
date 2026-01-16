package db

import (
	"context"
	"os"
	"testing"
)

func TestDBOperations(t *testing.T) {
	// Create temp database
	tmpFile, err := os.CreateTemp("", "test-memory-*.db")
	if err != nil {
		t.Fatalf("failed to create temp file: %v", err)
	}
	tmpPath := tmpFile.Name()
	tmpFile.Close()
	defer os.Remove(tmpPath)
	defer os.Remove(tmpPath + "-shm")
	defer os.Remove(tmpPath + "-wal")

	ctx := context.Background()

	// Create DB instance
	cfg := Config{
		Path:             tmpPath,
		VectorDimensions: 384,
	}
	db, err := New(ctx, cfg)
	if err != nil {
		t.Fatalf("failed to create DB: %v", err)
	}
	defer db.Close()

	// Test Ping
	t.Run("Ping", func(t *testing.T) {
		if err := db.Ping(ctx); err != nil {
			t.Errorf("ping failed: %v", err)
		}
	})

	// Test Store and Retrieve
	t.Run("StoreAndRetrieve", func(t *testing.T) {
		mem := &Memory{
			ID:        "test-id-1",
			Namespace: "default",
			Key:       "test-key-1",
			Value:     "This is a test memory value",
			Metadata:  map[string]any{"source": "test"},
			Tags:      []string{"test", "unit"},
			Embedding: make([]float32, 384),
		}

		// Store
		if err := db.Store(ctx, mem); err != nil {
			t.Fatalf("store failed: %v", err)
		}

		// Retrieve
		retrieved, err := db.Retrieve(ctx, "default", "test-key-1")
		if err != nil {
			t.Fatalf("retrieve failed: %v", err)
		}

		if retrieved.Key != mem.Key {
			t.Errorf("key mismatch: got %s, want %s", retrieved.Key, mem.Key)
		}
		if retrieved.Value != mem.Value {
			t.Errorf("value mismatch: got %s, want %s", retrieved.Value, mem.Value)
		}
		if len(retrieved.Tags) != len(mem.Tags) {
			t.Errorf("tags mismatch: got %v, want %v", retrieved.Tags, mem.Tags)
		}
	})

	// Test List
	t.Run("List", func(t *testing.T) {
		memories, err := db.List(ctx, "default", 10, 0)
		if err != nil {
			t.Fatalf("list failed: %v", err)
		}
		if len(memories) == 0 {
			t.Error("expected at least one memory")
		}
	})

	// Test Count
	t.Run("Count", func(t *testing.T) {
		count, err := db.Count(ctx, "default")
		if err != nil {
			t.Fatalf("count failed: %v", err)
		}
		if count == 0 {
			t.Error("expected count > 0")
		}
	})

	// Test Search (vector similarity)
	t.Run("Search", func(t *testing.T) {
		queryVec := make([]float32, 384)
		results, err := db.Search(ctx, "default", queryVec, 5, 0.0)
		if err != nil {
			t.Fatalf("search failed: %v", err)
		}
		if len(results) == 0 {
			t.Error("expected at least one search result")
		}
	})

	// Test Update (store with same key)
	t.Run("Update", func(t *testing.T) {
		mem := &Memory{
			ID:        "test-id-1-updated",
			Namespace: "default",
			Key:       "test-key-1",
			Value:     "Updated value",
			Metadata:  map[string]any{"source": "test", "updated": true},
			Tags:      []string{"test", "updated"},
			Embedding: make([]float32, 384),
		}

		if err := db.Store(ctx, mem); err != nil {
			t.Fatalf("update (store) failed: %v", err)
		}

		retrieved, err := db.Retrieve(ctx, "default", "test-key-1")
		if err != nil {
			t.Fatalf("retrieve after update failed: %v", err)
		}

		if retrieved.Value != "Updated value" {
			t.Errorf("value not updated: got %s", retrieved.Value)
		}
	})

	// Test Delete
	t.Run("Delete", func(t *testing.T) {
		if err := db.Delete(ctx, "default", "test-key-1"); err != nil {
			t.Fatalf("delete failed: %v", err)
		}

		_, err := db.Retrieve(ctx, "default", "test-key-1")
		if err == nil {
			t.Error("expected error after delete, got nil")
		}
		if err != ErrNotFound {
			t.Errorf("expected ErrNotFound, got: %v", err)
		}
	})
}

func TestVectorSerialization(t *testing.T) {
	original := []float32{0.1, 0.2, 0.3, 0.4, 0.5}

	serialized := serializeVector(original)
	if len(serialized) != len(original)*4 {
		t.Errorf("serialized length mismatch: got %d, want %d", len(serialized), len(original)*4)
	}

	deserialized, err := deserializeVector(serialized)
	if err != nil {
		t.Fatalf("deserialize failed: %v", err)
	}

	if len(deserialized) != len(original) {
		t.Fatalf("length mismatch: got %d, want %d", len(deserialized), len(original))
	}

	for i := range original {
		if deserialized[i] != original[i] {
			t.Errorf("value mismatch at %d: got %f, want %f", i, deserialized[i], original[i])
		}
	}
}

func TestCosineSimilarity(t *testing.T) {
	tests := []struct {
		name     string
		a, b     []float32
		expected float64
	}{
		{
			name:     "identical vectors",
			a:        []float32{1, 0, 0},
			b:        []float32{1, 0, 0},
			expected: 1.0,
		},
		{
			name:     "orthogonal vectors",
			a:        []float32{1, 0, 0},
			b:        []float32{0, 1, 0},
			expected: 0.0,
		},
		{
			name:     "opposite vectors",
			a:        []float32{1, 0, 0},
			b:        []float32{-1, 0, 0},
			expected: -1.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sim := cosineSimilarity(tt.a, tt.b)
			if sim != tt.expected {
				t.Errorf("cosineSimilarity(%v, %v) = %f, want %f", tt.a, tt.b, sim, tt.expected)
			}
		})
	}
}

func TestTTLCleanup(t *testing.T) {
	tmpFile, err := os.CreateTemp("", "test-ttl-*.db")
	if err != nil {
		t.Fatalf("failed to create temp file: %v", err)
	}
	tmpPath := tmpFile.Name()
	tmpFile.Close()
	defer os.Remove(tmpPath)
	defer os.Remove(tmpPath + "-shm")
	defer os.Remove(tmpPath + "-wal")

	ctx := context.Background()

	cfg := Config{
		Path:             tmpPath,
		VectorDimensions: 384,
	}
	db, err := New(ctx, cfg)
	if err != nil {
		t.Fatalf("failed to create DB: %v", err)
	}
	defer db.Close()

	// Store a regular memory (no TTL)
	mem := &Memory{
		ID:        "no-ttl-test",
		Namespace: "default",
		Key:       "permanent-key",
		Value:     "This will not expire",
		Embedding: make([]float32, 384),
	}

	if err := db.Store(ctx, mem); err != nil {
		t.Fatalf("store failed: %v", err)
	}

	// Should exist
	retrieved, err := db.Retrieve(ctx, "default", "permanent-key")
	if err != nil {
		t.Fatalf("retrieve failed: %v", err)
	}
	if retrieved.Value != mem.Value {
		t.Errorf("value mismatch: got %s, want %s", retrieved.Value, mem.Value)
	}

	// Cleanup should not affect non-expiring entries
	if err := db.cleanupExpired(ctx); err != nil {
		t.Fatalf("cleanup failed: %v", err)
	}

	// Should still exist
	_, err = db.Retrieve(ctx, "default", "permanent-key")
	if err != nil {
		t.Errorf("non-expiring memory was deleted: %v", err)
	}
}

func TestListNamespaces(t *testing.T) {
	tmpFile, err := os.CreateTemp("", "test-ns-*.db")
	if err != nil {
		t.Fatalf("failed to create temp file: %v", err)
	}
	tmpPath := tmpFile.Name()
	tmpFile.Close()
	defer os.Remove(tmpPath)
	defer os.Remove(tmpPath + "-shm")
	defer os.Remove(tmpPath + "-wal")

	ctx := context.Background()

	cfg := Config{
		Path:             tmpPath,
		VectorDimensions: 384,
	}
	db, err := New(ctx, cfg)
	if err != nil {
		t.Fatalf("failed to create DB: %v", err)
	}
	defer db.Close()

	// Store in multiple namespaces
	namespaces := []string{"ns1", "ns2", "ns3"}
	for _, ns := range namespaces {
		mem := &Memory{
			ID:        "id-" + ns,
			Namespace: ns,
			Key:       "key-" + ns,
			Value:     "value-" + ns,
			Embedding: make([]float32, 384),
		}
		if err := db.Store(ctx, mem); err != nil {
			t.Fatalf("store in %s failed: %v", ns, err)
		}
	}

	// List namespaces
	nsList, err := db.ListNamespaces(ctx)
	if err != nil {
		t.Fatalf("list namespaces failed: %v", err)
	}

	if len(nsList) != len(namespaces) {
		t.Errorf("namespace count mismatch: got %d, want %d", len(nsList), len(namespaces))
	}
}

func TestSearchAll(t *testing.T) {
	tmpFile, err := os.CreateTemp("", "test-searchall-*.db")
	if err != nil {
		t.Fatalf("failed to create temp file: %v", err)
	}
	tmpPath := tmpFile.Name()
	tmpFile.Close()
	defer os.Remove(tmpPath)
	defer os.Remove(tmpPath + "-shm")
	defer os.Remove(tmpPath + "-wal")

	ctx := context.Background()

	cfg := Config{
		Path:             tmpPath,
		VectorDimensions: 384,
	}
	db, err := New(ctx, cfg)
	if err != nil {
		t.Fatalf("failed to create DB: %v", err)
	}
	defer db.Close()

	// Store in multiple namespaces
	for _, ns := range []string{"ns1", "ns2"} {
		mem := &Memory{
			ID:        "id-" + ns,
			Namespace: ns,
			Key:       "key-" + ns,
			Value:     "value-" + ns,
			Embedding: make([]float32, 384),
		}
		if err := db.Store(ctx, mem); err != nil {
			t.Fatalf("store in %s failed: %v", ns, err)
		}
	}

	// Search across all namespaces
	queryVec := make([]float32, 384)
	results, err := db.SearchAll(ctx, queryVec, 10, 0.0)
	if err != nil {
		t.Fatalf("search all failed: %v", err)
	}

	if len(results) != 2 {
		t.Errorf("expected 2 results from SearchAll, got %d", len(results))
	}
}

func TestCount(t *testing.T) {
	tmpFile, err := os.CreateTemp("", "test-count-*.db")
	if err != nil {
		t.Fatalf("failed to create temp file: %v", err)
	}
	tmpPath := tmpFile.Name()
	tmpFile.Close()
	defer os.Remove(tmpPath)
	defer os.Remove(tmpPath + "-shm")
	defer os.Remove(tmpPath + "-wal")

	ctx := context.Background()

	cfg := Config{
		Path:             tmpPath,
		VectorDimensions: 384,
	}
	db, err := New(ctx, cfg)
	if err != nil {
		t.Fatalf("failed to create DB: %v", err)
	}
	defer db.Close()

	// Store some data
	for i := 0; i < 5; i++ {
		mem := &Memory{
			ID:        "id-" + string(rune('0'+i)),
			Namespace: "default",
			Key:       "key-" + string(rune('0'+i)),
			Value:     "value",
			Embedding: make([]float32, 384),
		}
		if err := db.Store(ctx, mem); err != nil {
			t.Fatalf("store failed: %v", err)
		}
	}

	count, err := db.Count(ctx, "default")
	if err != nil {
		t.Fatalf("count failed: %v", err)
	}

	if count != 5 {
		t.Errorf("expected 5 memories, got %d", count)
	}

	// Test CountAll
	countAll, err := db.CountAll(ctx)
	if err != nil {
		t.Fatalf("count all failed: %v", err)
	}

	if countAll != 5 {
		t.Errorf("expected 5 total memories, got %d", countAll)
	}
}
