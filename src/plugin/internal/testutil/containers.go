//go:build integration

// Package testutil provides test utilities including Testcontainers setup for integration tests.
package testutil

import (
	"context"
	"fmt"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/testcontainers/testcontainers-go"
	"github.com/testcontainers/testcontainers-go/wait"

	"github.com/libsql-memory/plugin/internal/db"
	"github.com/libsql-memory/plugin/internal/embedding"
)

// LibSQLContainer wraps a running LibSQL container for testing.
type LibSQLContainer struct {
	Container    testcontainers.Container
	Host         string
	Port         string
	HTTPEndpoint string
}

// LibSQLContainerConfig configures the LibSQL container.
type LibSQLContainerConfig struct {
	Image   string
	Timeout time.Duration
}

// DefaultLibSQLConfig returns the default LibSQL container configuration.
func DefaultLibSQLConfig() LibSQLContainerConfig {
	return LibSQLContainerConfig{
		Image:   "ghcr.io/tursodatabase/libsql-server:latest",
		Timeout: 60 * time.Second,
	}
}

// StartLibSQLContainer starts a LibSQL container for testing.
func StartLibSQLContainer(ctx context.Context, cfg LibSQLContainerConfig) (*LibSQLContainer, error) {
	req := testcontainers.ContainerRequest{
		Image:        cfg.Image,
		ExposedPorts: []string{"8080/tcp"},
		Env: map[string]string{
			"SQLD_NODE": "primary",
		},
		WaitingFor: wait.ForHTTP("/health").
			WithPort("8080/tcp").
			WithStartupTimeout(cfg.Timeout),
	}

	container, err := testcontainers.GenericContainer(ctx, testcontainers.GenericContainerRequest{
		ContainerRequest: req,
		Started:          true,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to start LibSQL container: %w", err)
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

	return &LibSQLContainer{
		Container:    container,
		Host:         host,
		Port:         port.Port(),
		HTTPEndpoint: fmt.Sprintf("http://%s:%s", host, port.Port()),
	}, nil
}

// Terminate stops and removes the container.
func (c *LibSQLContainer) Terminate(ctx context.Context) error {
	if c.Container != nil {
		return c.Container.Terminate(ctx)
	}
	return nil
}

// TestDB wraps a test database connection with cleanup.
type TestDB struct {
	DB     *db.DB
	Path   string
	tmpDir string
}

// NewTestDB creates a new test database using a temporary file.
func NewTestDB(t *testing.T, dimensions int) *TestDB {
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
	if err != nil {
		t.Fatalf("failed to create test database: %v", err)
	}

	return &TestDB{
		DB:     database,
		Path:   dbPath,
		tmpDir: tmpDir,
	}
}

// Close cleans up the test database.
func (d *TestDB) Close() error {
	if d.DB != nil {
		return d.DB.Close()
	}
	return nil
}

// MockEmbedder implements the Embedder interface for testing.
type MockEmbedder struct {
	dimensions int
	mu         sync.Mutex
	embeddings map[string]embedding.Vector
	callCount  int
}

// NewMockEmbedder creates a new mock embedder with the specified dimensions.
func NewMockEmbedder(dimensions int) *MockEmbedder {
	return &MockEmbedder{
		dimensions: dimensions,
		embeddings: make(map[string]embedding.Vector),
	}
}

// Embed generates a deterministic embedding for the given text.
func (m *MockEmbedder) Embed(ctx context.Context, text string) (embedding.Vector, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.callCount++

	// Check cache
	if vec, ok := m.embeddings[text]; ok {
		return vec, nil
	}

	// Generate deterministic embedding
	vec := make(embedding.Vector, m.dimensions)
	hash := fnv32(text)
	for i := range vec {
		vec[i] = float32((hash+uint32(i))%1000) / 1000.0
	}

	m.embeddings[text] = vec
	return vec, nil
}

// EmbedBatch generates embeddings for multiple texts.
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

// Dimension returns the embedding dimensions.
func (m *MockEmbedder) Dimension() int {
	return m.dimensions
}

// Close is a no-op for the mock embedder.
func (m *MockEmbedder) Close() error {
	return nil
}

// CallCount returns the number of times Embed was called.
func (m *MockEmbedder) CallCount() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.callCount
}

// ResetCallCount resets the call counter.
func (m *MockEmbedder) ResetCallCount() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.callCount = 0
}

// fnv32 computes a simple FNV-1a hash of the string.
func fnv32(s string) uint32 {
	const prime = 16777619
	hash := uint32(2166136261)
	for i := 0; i < len(s); i++ {
		hash ^= uint32(s[i])
		hash *= prime
	}
	return hash
}

// SkipIfNoDocker skips the test if Docker is not available.
func SkipIfNoDocker(t *testing.T) {
	t.Helper()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// Try to connect to Docker
	req := testcontainers.ContainerRequest{
		Image: "alpine:latest",
	}

	container, err := testcontainers.GenericContainer(ctx, testcontainers.GenericContainerRequest{
		ContainerRequest: req,
		Started:          false,
	})
	if err != nil {
		t.Skipf("Docker not available: %v", err)
		return
	}
	if container != nil {
		container.Terminate(ctx)
	}
}

// RunWithTimeout runs a test function with a timeout.
func RunWithTimeout(t *testing.T, timeout time.Duration, fn func(t *testing.T)) {
	t.Helper()

	done := make(chan struct{})
	go func() {
		fn(t)
		close(done)
	}()

	select {
	case <-done:
		return
	case <-time.After(timeout):
		t.Fatalf("test timed out after %v", timeout)
	}
}

// ContainerSuite provides a test suite with shared container resources.
type ContainerSuite struct {
	T         *testing.T
	Container *LibSQLContainer
	ctx       context.Context
	cancel    context.CancelFunc
}

// NewContainerSuite creates a new container suite.
func NewContainerSuite(t *testing.T) *ContainerSuite {
	t.Helper()

	ctx, cancel := context.WithCancel(context.Background())

	suite := &ContainerSuite{
		T:      t,
		ctx:    ctx,
		cancel: cancel,
	}

	cfg := DefaultLibSQLConfig()
	container, err := StartLibSQLContainer(ctx, cfg)
	if err != nil {
		t.Skipf("Failed to start container (Docker may not be available): %v", err)
	}

	suite.Container = container

	t.Cleanup(func() {
		suite.Teardown()
	})

	return suite
}

// Context returns the suite's context.
func (s *ContainerSuite) Context() context.Context {
	return s.ctx
}

// Teardown cleans up container resources.
func (s *ContainerSuite) Teardown() {
	if s.Container != nil {
		if err := s.Container.Terminate(s.ctx); err != nil {
			s.T.Logf("Failed to terminate container: %v", err)
		}
	}
	s.cancel()
}

// TestFixture provides common test data and utilities.
type TestFixture struct {
	Namespaces []string
	Keys       []string
	Values     []string
}

// NewTestFixture creates a new test fixture with sample data.
func NewTestFixture() *TestFixture {
	return &TestFixture{
		Namespaces: []string{"default", "users", "sessions", "cache"},
		Keys: []string{
			"key-1", "key-2", "key-3", "key-4", "key-5",
			"user-prefs", "session-data", "api-cache",
		},
		Values: []string{
			"Simple text value",
			"Another text value with more content",
			"Technical documentation about Go programming",
			"User preferences and settings JSON",
			"Session data with authentication tokens",
			"Cached API response for better performance",
			"Memory about project requirements",
			"Notes from the last meeting",
		},
	}
}

// RandomNamespace returns a random namespace from the fixture.
func (f *TestFixture) RandomNamespace() string {
	return f.Namespaces[time.Now().UnixNano()%int64(len(f.Namespaces))]
}

// RandomKey returns a random key from the fixture.
func (f *TestFixture) RandomKey() string {
	return f.Keys[time.Now().UnixNano()%int64(len(f.Keys))]
}

// RandomValue returns a random value from the fixture.
func (f *TestFixture) RandomValue() string {
	return f.Values[time.Now().UnixNano()%int64(len(f.Values))]
}
