package db

import (
	"context"
	"os"
	"testing"
	"time"
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

func TestExpirationBehavior(t *testing.T) {
	// Create temp database
	tmpFile, err := os.CreateTemp("", "test-repro-*.db")
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

	// Store a memory that has already expired
	negTTL := -1 * time.Hour
	mem := &Memory{
		ID:        "expired-mem",
		Namespace: "default",
		Key:       "expired-key",
		Value:     "This should be expired",
		TTL:       &negTTL,
		Embedding: make([]float32, 384),
	}

	if err := db.Store(ctx, mem); err != nil {
		t.Fatalf("store failed: %v", err)
	}

	// Try to Retrieve it. It should NOT be returned.
	_, err = db.Retrieve(ctx, "default", "expired-key")
	if err != ErrNotFound {
		t.Errorf("expected ErrNotFound for expired memory, got: %v", err)
	}

	// Verify it exists in the DB if we query without expiration check (manual query)
	var count int
	err = db.db.QueryRowContext(ctx, "SELECT COUNT(*) FROM memories WHERE key = 'expired-key'").Scan(&count)
	if err != nil {
		t.Fatalf("manual count failed: %v", err)
	}
	if count != 1 {
		t.Errorf("memory should exist in DB but be hidden, count: %d", count)
	}

	// Run cleanup
	if err := db.cleanupExpired(ctx); err != nil {
		t.Fatalf("cleanup failed: %v", err)
	}

	// Verify it is GONE from the DB
	err = db.db.QueryRowContext(ctx, "SELECT COUNT(*) FROM memories WHERE key = 'expired-key'").Scan(&count)
	if err != nil {
		t.Fatalf("manual count failed: %v", err)
	}
	if count != 0 {
		t.Errorf("memory should have been cleaned up, count: %d", count)
	}
}

func TestRetrieveByID(t *testing.T) {
	tmpFile, err := os.CreateTemp("", "test-retrieve-id-*.db")
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

	// Store a memory
	mem := &Memory{
		ID:        "test-retrieve-by-id",
		Namespace: "default",
		Key:       "test-key",
		Value:     "test value",
		Embedding: make([]float32, 384),
	}

	if err := db.Store(ctx, mem); err != nil {
		t.Fatalf("store failed: %v", err)
	}

	// Retrieve by ID
	retrieved, err := db.RetrieveByID(ctx, "test-retrieve-by-id")
	if err != nil {
		t.Fatalf("retrieve by ID failed: %v", err)
	}

	if retrieved.ID != mem.ID {
		t.Errorf("ID mismatch: got %s, want %s", retrieved.ID, mem.ID)
	}
	if retrieved.Key != mem.Key {
		t.Errorf("Key mismatch: got %s, want %s", retrieved.Key, mem.Key)
	}
	if retrieved.Value != mem.Value {
		t.Errorf("Value mismatch: got %s, want %s", retrieved.Value, mem.Value)
	}

	// Try to retrieve non-existent ID
	_, err = db.RetrieveByID(ctx, "non-existent-id")
	if err != ErrNotFound {
		t.Errorf("expected ErrNotFound for non-existent ID, got: %v", err)
	}
}

func TestDeleteByID(t *testing.T) {
	tmpFile, err := os.CreateTemp("", "test-delete-id-*.db")
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

	// Store a memory
	mem := &Memory{
		ID:        "test-delete-by-id",
		Namespace: "default",
		Key:       "test-key",
		Value:     "test value",
		Embedding: make([]float32, 384),
	}

	if err := db.Store(ctx, mem); err != nil {
		t.Fatalf("store failed: %v", err)
	}

	// Delete by ID
	if err := db.DeleteByID(ctx, "test-delete-by-id"); err != nil {
		t.Fatalf("delete by ID failed: %v", err)
	}

	// Verify it's gone
	_, err = db.RetrieveByID(ctx, "test-delete-by-id")
	if err != ErrNotFound {
		t.Errorf("expected ErrNotFound after delete, got: %v", err)
	}

	// Try to delete non-existent ID
	err = db.DeleteByID(ctx, "non-existent-id")
	if err != ErrNotFound {
		t.Errorf("expected ErrNotFound for non-existent ID, got: %v", err)
	}
}

func TestDeleteNamespace(t *testing.T) {
	tmpFile, err := os.CreateTemp("", "test-delete-ns-*.db")
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

	// Store memories in different namespaces
	for i := 0; i < 3; i++ {
		mem := &Memory{
			ID:        "test-id-ns1-" + string(rune('0'+i)),
			Namespace: "ns1",
			Key:       "key-" + string(rune('0'+i)),
			Value:     "value",
			Embedding: make([]float32, 384),
		}
		if err := db.Store(ctx, mem); err != nil {
			t.Fatalf("store in ns1 failed: %v", err)
		}
	}

	for i := 0; i < 2; i++ {
		mem := &Memory{
			ID:        "test-id-ns2-" + string(rune('0'+i)),
			Namespace: "ns2",
			Key:       "key-" + string(rune('0'+i)),
			Value:     "value",
			Embedding: make([]float32, 384),
		}
		if err := db.Store(ctx, mem); err != nil {
			t.Fatalf("store in ns2 failed: %v", err)
		}
	}

	// Delete ns1
	rowsAffected, err := db.DeleteNamespace(ctx, "ns1")
	if err != nil {
		t.Fatalf("delete namespace failed: %v", err)
	}
	if rowsAffected != 3 {
		t.Errorf("expected 3 rows deleted, got %d", rowsAffected)
	}

	// Verify ns1 is empty
	count, err := db.Count(ctx, "ns1")
	if err != nil {
		t.Fatalf("count ns1 failed: %v", err)
	}
	if count != 0 {
		t.Errorf("expected ns1 to be empty, got %d memories", count)
	}

	// Verify ns2 is intact
	count, err = db.Count(ctx, "ns2")
	if err != nil {
		t.Fatalf("count ns2 failed: %v", err)
	}
	if count != 2 {
		t.Errorf("expected ns2 to have 2 memories, got %d", count)
	}
}

func TestGetStats(t *testing.T) {
	tmpFile, err := os.CreateTemp("", "test-stats-*.db")
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

	// Initially empty
	stats, err := db.GetStats(ctx)
	if err != nil {
		t.Fatalf("get stats failed: %v", err)
	}
	if stats.TotalMemories != 0 {
		t.Errorf("expected 0 total memories initially, got %d", stats.TotalMemories)
	}

	// Store memories in different namespaces
	for i := 0; i < 3; i++ {
		mem := &Memory{
			ID:        "stats-id-ns1-" + string(rune('0'+i)),
			Namespace: "stats-ns1",
			Key:       "key-" + string(rune('0'+i)),
			Value:     "value",
			Embedding: make([]float32, 384),
		}
		if err := db.Store(ctx, mem); err != nil {
			t.Fatalf("store failed: %v", err)
		}
	}

	ttl := 1 * time.Hour
	mem := &Memory{
		ID:        "stats-id-ns2",
		Namespace: "stats-ns2",
		Key:       "key-expiring",
		Value:     "value",
		TTL:       &ttl,
		Embedding: make([]float32, 384),
	}
	if err := db.Store(ctx, mem); err != nil {
		t.Fatalf("store with TTL failed: %v", err)
	}

	// Get stats again
	stats, err = db.GetStats(ctx)
	if err != nil {
		t.Fatalf("get stats failed: %v", err)
	}

	if stats.TotalMemories != 4 {
		t.Errorf("expected 4 total memories, got %d", stats.TotalMemories)
	}

	if stats.NamespaceCounts["stats-ns1"] != 3 {
		t.Errorf("expected 3 memories in stats-ns1, got %d", stats.NamespaceCounts["stats-ns1"])
	}

	if stats.NamespaceCounts["stats-ns2"] != 1 {
		t.Errorf("expected 1 memory in stats-ns2, got %d", stats.NamespaceCounts["stats-ns2"])
	}

	if stats.ExpiringCount != 1 {
		t.Errorf("expected 1 expiring memory, got %d", stats.ExpiringCount)
	}

	if stats.OldestMemory == nil || stats.NewestMemory == nil {
		t.Error("expected oldest and newest memory timestamps to be set")
	}
}

func TestTransactions(t *testing.T) {
	tmpFile, err := os.CreateTemp("", "test-tx-*.db")
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

	t.Run("CommitTransaction", func(t *testing.T) {
		tx, err := db.BeginTx(ctx, nil)
		if err != nil {
			t.Fatalf("begin tx failed: %v", err)
		}

		mem := &Memory{
			ID:        "tx-commit-id",
			Namespace: "default",
			Key:       "tx-commit-key",
			Value:     "committed value",
			Embedding: make([]float32, 384),
		}

		if err := tx.StoreTx(ctx, mem); err != nil {
			tx.Rollback()
			t.Fatalf("store tx failed: %v", err)
		}

		if err := tx.Commit(); err != nil {
			t.Fatalf("commit failed: %v", err)
		}

		// Verify it's stored
		retrieved, err := db.Retrieve(ctx, "default", "tx-commit-key")
		if err != nil {
			t.Fatalf("retrieve after commit failed: %v", err)
		}
		if retrieved.Value != "committed value" {
			t.Errorf("value mismatch: got %s", retrieved.Value)
		}
	})

	t.Run("RollbackTransaction", func(t *testing.T) {
		tx, err := db.BeginTx(ctx, nil)
		if err != nil {
			t.Fatalf("begin tx failed: %v", err)
		}

		mem := &Memory{
			ID:        "tx-rollback-id",
			Namespace: "default",
			Key:       "tx-rollback-key",
			Value:     "rolled back value",
			Embedding: make([]float32, 384),
		}

		if err := tx.StoreTx(ctx, mem); err != nil {
			tx.Rollback()
			t.Fatalf("store tx failed: %v", err)
		}

		if err := tx.Rollback(); err != nil {
			t.Fatalf("rollback failed: %v", err)
		}

		// Verify it's NOT stored
		_, err = db.Retrieve(ctx, "default", "tx-rollback-key")
		if err != ErrNotFound {
			t.Errorf("expected ErrNotFound after rollback, got: %v", err)
		}
	})

	t.Run("DeleteInTransaction", func(t *testing.T) {
		// Store a memory first
		mem := &Memory{
			ID:        "tx-delete-id",
			Namespace: "default",
			Key:       "tx-delete-key",
			Value:     "to be deleted",
			Embedding: make([]float32, 384),
		}
		if err := db.Store(ctx, mem); err != nil {
			t.Fatalf("store failed: %v", err)
		}

		// Delete in transaction and commit
		tx, err := db.BeginTx(ctx, nil)
		if err != nil {
			t.Fatalf("begin tx failed: %v", err)
		}

		if err := tx.DeleteTx(ctx, "default", "tx-delete-key"); err != nil {
			tx.Rollback()
			t.Fatalf("delete tx failed: %v", err)
		}

		if err := tx.Commit(); err != nil {
			t.Fatalf("commit failed: %v", err)
		}

		// Verify it's deleted
		_, err = db.Retrieve(ctx, "default", "tx-delete-key")
		if err != ErrNotFound {
			t.Errorf("expected ErrNotFound after delete, got: %v", err)
		}
	})
}

func TestConfigValidation(t *testing.T) {
	ctx := context.Background()

	t.Run("EmptyPath", func(t *testing.T) {
		cfg := Config{
			Path:             "",
			VectorDimensions: 384,
		}
		_, err := New(ctx, cfg)
		if err == nil {
			t.Error("expected error for empty path")
		}
		if !errors.Is(err, ErrInvalidConfig) {
			t.Errorf("expected ErrInvalidConfig, got: %v", err)
		}
	})

	t.Run("NegativeVectorDimensions", func(t *testing.T) {
		tmpFile, err := os.CreateTemp("", "test-config-*.db")
		if err != nil {
			t.Fatalf("failed to create temp file: %v", err)
		}
		tmpPath := tmpFile.Name()
		tmpFile.Close()
		defer os.Remove(tmpPath)

		cfg := Config{
			Path:             tmpPath,
			VectorDimensions: -1,
		}
		_, err = New(ctx, cfg)
		if err == nil {
			t.Error("expected error for negative vector dimensions")
		}
		if !errors.Is(err, ErrInvalidConfig) {
			t.Errorf("expected ErrInvalidConfig, got: %v", err)
		}
	})

	t.Run("DefaultsApplied", func(t *testing.T) {
		tmpFile, err := os.CreateTemp("", "test-defaults-*.db")
		if err != nil {
			t.Fatalf("failed to create temp file: %v", err)
		}
		tmpPath := tmpFile.Name()
		tmpFile.Close()
		defer os.Remove(tmpPath)
		defer os.Remove(tmpPath + "-shm")
		defer os.Remove(tmpPath + "-wal")

		cfg := Config{
			Path: tmpPath,
			// Leave other fields at zero values
		}
		db, err := New(ctx, cfg)
		if err != nil {
			t.Fatalf("failed to create DB with defaults: %v", err)
		}
		defer db.Close()

		// Verify defaults were applied
		if db.config.MaxOpenConns == 0 {
			t.Error("expected default MaxOpenConns to be applied")
		}
		if db.config.VectorDimensions == 0 {
			t.Error("expected default VectorDimensions to be applied")
		}
	})
}

func TestClosedDatabaseErrors(t *testing.T) {
	tmpFile, err := os.CreateTemp("", "test-closed-*.db")
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

	// Close the database
	if err := db.Close(); err != nil {
		t.Fatalf("close failed: %v", err)
	}

	// Try operations on closed database
	mem := &Memory{
		ID:        "closed-test",
		Namespace: "default",
		Key:       "key",
		Value:     "value",
		Embedding: make([]float32, 384),
	}

	if err := db.Store(ctx, mem); err != ErrClosed {
		t.Errorf("expected ErrClosed for Store, got: %v", err)
	}

	if _, err := db.Retrieve(ctx, "default", "key"); err != ErrClosed {
		t.Errorf("expected ErrClosed for Retrieve, got: %v", err)
	}

	if _, err := db.RetrieveByID(ctx, "id"); err != ErrClosed {
		t.Errorf("expected ErrClosed for RetrieveByID, got: %v", err)
	}

	if err := db.Delete(ctx, "default", "key"); err != ErrClosed {
		t.Errorf("expected ErrClosed for Delete, got: %v", err)
	}

	if err := db.DeleteByID(ctx, "id"); err != ErrClosed {
		t.Errorf("expected ErrClosed for DeleteByID, got: %v", err)
	}

	if _, err := db.List(ctx, "default", 10, 0); err != ErrClosed {
		t.Errorf("expected ErrClosed for List, got: %v", err)
	}

	if _, err := db.Count(ctx, "default"); err != ErrClosed {
		t.Errorf("expected ErrClosed for Count, got: %v", err)
	}

	// Closing again should return ErrClosed
	if err := db.Close(); err != ErrClosed {
		t.Errorf("expected ErrClosed for second Close, got: %v", err)
	}
}

func TestContextCancellation(t *testing.T) {
	tmpFile, err := os.CreateTemp("", "test-context-*.db")
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

	// Create a cancelled context
	cancelledCtx, cancel := context.WithCancel(ctx)
	cancel()

	mem := &Memory{
		ID:        "context-test",
		Namespace: "default",
		Key:       "key",
		Value:     "value",
		Embedding: make([]float32, 384),
	}

	// Operations with cancelled context should fail
	if err := db.Store(cancelledCtx, mem); err == nil {
		t.Error("expected error for Store with cancelled context")
	} else if !errors.Is(err, ErrContextClosed) {
		t.Errorf("expected ErrContextClosed, got: %v", err)
	}
}

func TestInvalidVectorDimensions(t *testing.T) {
	tmpFile, err := os.CreateTemp("", "test-vec-dim-*.db")
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

	t.Run("WrongDimensionsOnStore", func(t *testing.T) {
		mem := &Memory{
			ID:        "wrong-dim",
			Namespace: "default",
			Key:       "key",
			Value:     "value",
			Embedding: make([]float32, 128), // Wrong size
		}

		err := db.Store(ctx, mem)
		if err == nil {
			t.Error("expected error for wrong vector dimensions")
		}
		if !errors.Is(err, ErrInvalidVector) {
			t.Errorf("expected ErrInvalidVector, got: %v", err)
		}
	})

	t.Run("WrongDimensionsOnSearch", func(t *testing.T) {
		queryVec := make([]float32, 128) // Wrong size
		_, err := db.Search(ctx, "default", queryVec, 5, 0.0)
		if err == nil {
			t.Error("expected error for wrong query vector dimensions")
		}
		if !errors.Is(err, ErrInvalidVector) {
			t.Errorf("expected ErrInvalidVector, got: %v", err)
		}
	})

	t.Run("WrongDimensionsOnSearchAll", func(t *testing.T) {
		queryVec := make([]float32, 128) // Wrong size
		_, err := db.SearchAll(ctx, queryVec, 5, 0.0)
		if err == nil {
			t.Error("expected error for wrong query vector dimensions")
		}
		if !errors.Is(err, ErrInvalidVector) {
			t.Errorf("expected ErrInvalidVector, got: %v", err)
		}
	})
}

func TestVectorEdgeCases(t *testing.T) {
	t.Run("NilVectorSerialization", func(t *testing.T) {
		serialized := serializeVector(nil)
		if serialized != nil {
			t.Error("expected nil for nil vector serialization")
		}

		deserialized, err := deserializeVector(nil)
		if err != nil {
			t.Errorf("deserialize nil failed: %v", err)
		}
		if deserialized != nil {
			t.Error("expected nil for nil vector deserialization")
		}
	})

	t.Run("EmptyVectorSerialization", func(t *testing.T) {
		serialized := serializeVector([]float32{})
		if len(serialized) != 0 {
			t.Errorf("expected 0 bytes for empty vector, got %d", len(serialized))
		}

		deserialized, err := deserializeVector([]byte{})
		if err != nil {
			t.Errorf("deserialize empty failed: %v", err)
		}
		if deserialized != nil && len(deserialized) != 0 {
			t.Error("expected nil or empty for empty vector deserialization")
		}
	})

	t.Run("InvalidVectorDataLength", func(t *testing.T) {
		// Not a multiple of 4
		invalidData := []byte{1, 2, 3}
		_, err := deserializeVector(invalidData)
		if err == nil {
			t.Error("expected error for invalid vector data length")
		}
	})

	t.Run("CosineSimilarityEdgeCases", func(t *testing.T) {
		// Different lengths
		sim := cosineSimilarity([]float32{1, 2}, []float32{1, 2, 3})
		if sim != 0 {
			t.Errorf("expected 0 for different length vectors, got %f", sim)
		}

		// Empty vectors
		sim = cosineSimilarity([]float32{}, []float32{})
		if sim != 0 {
			t.Errorf("expected 0 for empty vectors, got %f", sim)
		}

		// Zero vectors
		sim = cosineSimilarity([]float32{0, 0, 0}, []float32{0, 0, 0})
		if sim != 0 {
			t.Errorf("expected 0 for zero vectors, got %f", sim)
		}

		// One zero vector
		sim = cosineSimilarity([]float32{1, 2, 3}, []float32{0, 0, 0})
		if sim != 0 {
			t.Errorf("expected 0 for one zero vector, got %f", sim)
		}
	})
}

func TestDuplicateKeyHandling(t *testing.T) {
	tmpFile, err := os.CreateTemp("", "test-dup-*.db")
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

	// Store initial memory
	mem1 := &Memory{
		ID:        "id1",
		Namespace: "default",
		Key:       "duplicate-key",
		Value:     "original value",
		Metadata:  map[string]any{"version": 1},
		Embedding: make([]float32, 384),
	}

	if err := db.Store(ctx, mem1); err != nil {
		t.Fatalf("first store failed: %v", err)
	}

	// Store again with same namespace and key (should update via ON CONFLICT)
	mem2 := &Memory{
		ID:        "id2",
		Namespace: "default",
		Key:       "duplicate-key",
		Value:     "updated value",
		Metadata:  map[string]any{"version": 2},
		Embedding: make([]float32, 384),
	}

	if err := db.Store(ctx, mem2); err != nil {
		t.Fatalf("second store failed: %v", err)
	}

	// Retrieve and verify it was updated
	retrieved, err := db.Retrieve(ctx, "default", "duplicate-key")
	if err != nil {
		t.Fatalf("retrieve failed: %v", err)
	}

	if retrieved.Value != "updated value" {
		t.Errorf("expected updated value, got: %s", retrieved.Value)
	}

	// Count should be 1, not 2
	count, err := db.Count(ctx, "default")
	if err != nil {
		t.Fatalf("count failed: %v", err)
	}
	if count != 1 {
		t.Errorf("expected 1 memory after duplicate key store, got %d", count)
	}
}

func TestMemoryWithoutEmbedding(t *testing.T) {
	tmpFile, err := os.CreateTemp("", "test-no-emb-*.db")
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

	// Store memory without embedding
	mem := &Memory{
		ID:        "no-embedding",
		Namespace: "default",
		Key:       "text-only",
		Value:     "This has no vector",
		Metadata:  map[string]any{"type": "text"},
		Embedding: nil,
	}

	if err := db.Store(ctx, mem); err != nil {
		t.Fatalf("store without embedding failed: %v", err)
	}

	// Retrieve and verify
	retrieved, err := db.Retrieve(ctx, "default", "text-only")
	if err != nil {
		t.Fatalf("retrieve failed: %v", err)
	}

	if retrieved.Embedding != nil {
		t.Error("expected nil embedding")
	}

	// Vector search should not include it
	queryVec := make([]float32, 384)
	results, err := db.Search(ctx, "default", queryVec, 10, 0.0)
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	for _, result := range results {
		if result.Memory.Key == "text-only" {
			t.Error("memory without embedding should not appear in vector search results")
		}
	}
}