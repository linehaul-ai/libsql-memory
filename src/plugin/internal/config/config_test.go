package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestNewConfig_Defaults(t *testing.T) {
	cfg := NewConfig()

	if cfg.EmbeddingProvider != DefaultEmbeddingProvider {
		t.Errorf("expected EmbeddingProvider=%s, got %s", DefaultEmbeddingProvider, cfg.EmbeddingProvider)
	}
	if cfg.EmbeddingDimension != DefaultEmbeddingDimension {
		t.Errorf("expected EmbeddingDimension=%d, got %d", DefaultEmbeddingDimension, cfg.EmbeddingDimension)
	}
	if cfg.DefaultNamespace != DefaultNamespace {
		t.Errorf("expected DefaultNamespace=%s, got %s", DefaultNamespace, cfg.DefaultNamespace)
	}
	if cfg.MaxConnections != DefaultMaxConnections {
		t.Errorf("expected MaxConnections=%d, got %d", DefaultMaxConnections, cfg.MaxConnections)
	}
	if cfg.LogLevel != DefaultLogLevel {
		t.Errorf("expected LogLevel=%s, got %s", DefaultLogLevel, cfg.LogLevel)
	}
}

func TestNewConfig_WithOptions(t *testing.T) {
	cfg := NewConfig(
		WithDatabasePath("/path/to/db"),
		WithAuthToken("secret-token"),
		WithEmbeddingProvider("local"),
		WithOpenAIAPIKey("sk-test"),
		WithEmbeddingDimension(768),
		WithDefaultNamespace("custom"),
		WithMaxConnections(20),
		WithLogLevel("debug"),
	)

	if cfg.DatabasePath != "/path/to/db" {
		t.Errorf("expected DatabasePath=/path/to/db, got %s", cfg.DatabasePath)
	}
	if cfg.AuthToken != "secret-token" {
		t.Errorf("expected AuthToken=secret-token, got %s", cfg.AuthToken)
	}
	if cfg.EmbeddingProvider != "local" {
		t.Errorf("expected EmbeddingProvider=local, got %s", cfg.EmbeddingProvider)
	}
	if cfg.OpenAIAPIKey != "sk-test" {
		t.Errorf("expected OpenAIAPIKey=sk-test, got %s", cfg.OpenAIAPIKey)
	}
	if cfg.EmbeddingDimension != 768 {
		t.Errorf("expected EmbeddingDimension=768, got %d", cfg.EmbeddingDimension)
	}
	if cfg.DefaultNamespace != "custom" {
		t.Errorf("expected DefaultNamespace=custom, got %s", cfg.DefaultNamespace)
	}
	if cfg.MaxConnections != 20 {
		t.Errorf("expected MaxConnections=20, got %d", cfg.MaxConnections)
	}
	if cfg.LogLevel != "debug" {
		t.Errorf("expected LogLevel=debug, got %s", cfg.LogLevel)
	}
}

func TestLoadFromEnv(t *testing.T) {
	// Set environment variables
	envVars := map[string]string{
		"LIBSQL_MEMORY_DATABASE_PATH":       "/env/path/db",
		"LIBSQL_MEMORY_AUTH_TOKEN":          "env-token",
		"LIBSQL_MEMORY_EMBEDDING_PROVIDER":  "local",
		"LIBSQL_MEMORY_OPENAI_API_KEY":      "sk-env-key",
		"LIBSQL_MEMORY_EMBEDDING_DIMENSION": "512",
		"LIBSQL_MEMORY_DEFAULT_NAMESPACE":   "env-ns",
		"LIBSQL_MEMORY_MAX_CONNECTIONS":     "25",
		"LIBSQL_MEMORY_LOG_LEVEL":           "warn",
	}

	for k, v := range envVars {
		os.Setenv(k, v)
		defer os.Unsetenv(k)
	}

	cfg := LoadFromEnv()

	if cfg.DatabasePath != "/env/path/db" {
		t.Errorf("expected DatabasePath=/env/path/db, got %s", cfg.DatabasePath)
	}
	if cfg.AuthToken != "env-token" {
		t.Errorf("expected AuthToken=env-token, got %s", cfg.AuthToken)
	}
	if cfg.EmbeddingProvider != "local" {
		t.Errorf("expected EmbeddingProvider=local, got %s", cfg.EmbeddingProvider)
	}
	if cfg.OpenAIAPIKey != "sk-env-key" {
		t.Errorf("expected OpenAIAPIKey=sk-env-key, got %s", cfg.OpenAIAPIKey)
	}
	if cfg.EmbeddingDimension != 512 {
		t.Errorf("expected EmbeddingDimension=512, got %d", cfg.EmbeddingDimension)
	}
	if cfg.DefaultNamespace != "env-ns" {
		t.Errorf("expected DefaultNamespace=env-ns, got %s", cfg.DefaultNamespace)
	}
	if cfg.MaxConnections != 25 {
		t.Errorf("expected MaxConnections=25, got %d", cfg.MaxConnections)
	}
	if cfg.LogLevel != "warn" {
		t.Errorf("expected LogLevel=warn, got %s", cfg.LogLevel)
	}
}

func TestLoadFromFile_JSON(t *testing.T) {
	// Create temp JSON config file
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.json")

	jsonContent := `{
		"database_path": "/json/path/db",
		"auth_token": "json-token",
		"embedding_provider": "openai",
		"openai_api_key": "sk-json-key",
		"embedding_dimension": 1024,
		"default_namespace": "json-ns",
		"max_connections": 15,
		"log_level": "error"
	}`

	if err := os.WriteFile(configPath, []byte(jsonContent), 0644); err != nil {
		t.Fatalf("failed to write temp config: %v", err)
	}

	cfg, err := LoadFromFile(configPath)
	if err != nil {
		t.Fatalf("LoadFromFile failed: %v", err)
	}

	if cfg.DatabasePath != "/json/path/db" {
		t.Errorf("expected DatabasePath=/json/path/db, got %s", cfg.DatabasePath)
	}
	if cfg.AuthToken != "json-token" {
		t.Errorf("expected AuthToken=json-token, got %s", cfg.AuthToken)
	}
	if cfg.EmbeddingDimension != 1024 {
		t.Errorf("expected EmbeddingDimension=1024, got %d", cfg.EmbeddingDimension)
	}
}

func TestLoadFromFile_YAML(t *testing.T) {
	// Create temp YAML config file
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")

	yamlContent := `
database_path: /yaml/path/db
auth_token: yaml-token
embedding_provider: local
embedding_dimension: 768
default_namespace: yaml-ns
max_connections: 30
log_level: debug
`

	if err := os.WriteFile(configPath, []byte(yamlContent), 0644); err != nil {
		t.Fatalf("failed to write temp config: %v", err)
	}

	cfg, err := LoadFromFile(configPath)
	if err != nil {
		t.Fatalf("LoadFromFile failed: %v", err)
	}

	if cfg.DatabasePath != "/yaml/path/db" {
		t.Errorf("expected DatabasePath=/yaml/path/db, got %s", cfg.DatabasePath)
	}
	if cfg.EmbeddingProvider != "local" {
		t.Errorf("expected EmbeddingProvider=local, got %s", cfg.EmbeddingProvider)
	}
	if cfg.EmbeddingDimension != 768 {
		t.Errorf("expected EmbeddingDimension=768, got %d", cfg.EmbeddingDimension)
	}
}

func TestLoadFromFile_UnsupportedFormat(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.txt")

	if err := os.WriteFile(configPath, []byte("test"), 0644); err != nil {
		t.Fatalf("failed to write temp config: %v", err)
	}

	_, err := LoadFromFile(configPath)
	if err == nil {
		t.Error("expected error for unsupported format, got nil")
	}
}

func TestLoadFromFlags(t *testing.T) {
	args := []string{
		"-database-path", "/flag/path/db",
		"-auth-token", "flag-token",
		"-embedding-provider", "local",
		"-embedding-dimension", "256",
		"-max-connections", "50",
		"-log-level", "debug",
	}

	cfg, err := LoadFromFlags(args)
	if err != nil {
		t.Fatalf("LoadFromFlags failed: %v", err)
	}

	if cfg.DatabasePath != "/flag/path/db" {
		t.Errorf("expected DatabasePath=/flag/path/db, got %s", cfg.DatabasePath)
	}
	if cfg.AuthToken != "flag-token" {
		t.Errorf("expected AuthToken=flag-token, got %s", cfg.AuthToken)
	}
	if cfg.EmbeddingProvider != "local" {
		t.Errorf("expected EmbeddingProvider=local, got %s", cfg.EmbeddingProvider)
	}
	if cfg.EmbeddingDimension != 256 {
		t.Errorf("expected EmbeddingDimension=256, got %d", cfg.EmbeddingDimension)
	}
	if cfg.MaxConnections != 50 {
		t.Errorf("expected MaxConnections=50, got %d", cfg.MaxConnections)
	}
}

func TestValidate_ValidConfig(t *testing.T) {
	cfg := NewConfig(
		WithDatabasePath("/path/to/db"),
		WithEmbeddingProvider("local"),
	)

	err := cfg.Validate()
	if err != nil {
		t.Errorf("expected no error, got %v", err)
	}
}

func TestValidate_ValidOpenAIConfig(t *testing.T) {
	cfg := NewConfig(
		WithDatabasePath("/path/to/db"),
		WithEmbeddingProvider("openai"),
		WithOpenAIAPIKey("sk-test-key"),
	)

	err := cfg.Validate()
	if err != nil {
		t.Errorf("expected no error, got %v", err)
	}
}

func TestValidate_MissingDatabasePath(t *testing.T) {
	cfg := NewConfig(
		WithEmbeddingProvider("local"),
	)

	err := cfg.Validate()
	if err == nil {
		t.Error("expected error for missing database_path, got nil")
	}
}

func TestValidate_MissingOpenAIKey(t *testing.T) {
	cfg := NewConfig(
		WithDatabasePath("/path/to/db"),
		WithEmbeddingProvider("openai"),
	)

	err := cfg.Validate()
	if err == nil {
		t.Error("expected error for missing openai_api_key, got nil")
	}
}

func TestValidate_InvalidProvider(t *testing.T) {
	cfg := NewConfig(
		WithDatabasePath("/path/to/db"),
		WithEmbeddingProvider("invalid"),
	)

	err := cfg.Validate()
	if err == nil {
		t.Error("expected error for invalid provider, got nil")
	}
}

func TestValidate_InvalidDimension(t *testing.T) {
	cfg := NewConfig(
		WithDatabasePath("/path/to/db"),
		WithEmbeddingProvider("local"),
		WithEmbeddingDimension(0),
	)

	err := cfg.Validate()
	if err == nil {
		t.Error("expected error for invalid dimension, got nil")
	}
}

func TestValidate_InvalidLogLevel(t *testing.T) {
	cfg := NewConfig(
		WithDatabasePath("/path/to/db"),
		WithEmbeddingProvider("local"),
		WithLogLevel("invalid"),
	)

	err := cfg.Validate()
	if err == nil {
		t.Error("expected error for invalid log level, got nil")
	}
}

func TestIsTursoURL(t *testing.T) {
	tests := []struct {
		path     string
		expected bool
	}{
		{"libsql://my-db.turso.io", true},
		{"https://my-db.turso.io", true},
		{"my-db.turso.io", true},
		{"/path/to/local.db", false},
		{"./local.db", false},
		{"memory:", false},
	}

	for _, tt := range tests {
		cfg := NewConfig(WithDatabasePath(tt.path))
		if got := cfg.IsTursoURL(); got != tt.expected {
			t.Errorf("IsTursoURL(%s) = %v, want %v", tt.path, got, tt.expected)
		}
	}
}

func TestIsLocalDatabase(t *testing.T) {
	tests := []struct {
		path     string
		expected bool
	}{
		{"/path/to/local.db", true},
		{"./local.db", true},
		{"libsql://my-db.turso.io", false},
		{"https://my-db.turso.io", false},
	}

	for _, tt := range tests {
		cfg := NewConfig(WithDatabasePath(tt.path))
		if got := cfg.IsLocalDatabase(); got != tt.expected {
			t.Errorf("IsLocalDatabase(%s) = %v, want %v", tt.path, got, tt.expected)
		}
	}
}

func TestString_MasksSensitiveFields(t *testing.T) {
	cfg := NewConfig(
		WithDatabasePath("/path/to/db"),
		WithAuthToken("secret-token"),
		WithOpenAIAPIKey("sk-secret-key"),
	)

	str := cfg.String()

	if contains(str, "secret-token") {
		t.Error("String() should not contain actual auth token")
	}
	if contains(str, "sk-secret-key") {
		t.Error("String() should not contain actual API key")
	}
	if !contains(str, "***") {
		t.Error("String() should contain masked values")
	}
}

func TestClone(t *testing.T) {
	original := NewConfig(
		WithDatabasePath("/path/to/db"),
		WithAuthToken("token"),
		WithEmbeddingDimension(768),
	)

	cloned := original.Clone()

	// Verify values are equal
	if cloned.DatabasePath != original.DatabasePath {
		t.Errorf("Clone() DatabasePath mismatch")
	}
	if cloned.AuthToken != original.AuthToken {
		t.Errorf("Clone() AuthToken mismatch")
	}
	if cloned.EmbeddingDimension != original.EmbeddingDimension {
		t.Errorf("Clone() EmbeddingDimension mismatch")
	}

	// Verify it's a different instance
	cloned.DatabasePath = "/different/path"
	if original.DatabasePath == cloned.DatabasePath {
		t.Error("Clone() should create independent copy")
	}
}

func TestLoad_PriorityOrder(t *testing.T) {
	// Create temp config file
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.json")

	jsonContent := `{
		"database_path": "/file/path",
		"embedding_dimension": 512,
		"log_level": "error"
	}`

	if err := os.WriteFile(configPath, []byte(jsonContent), 0644); err != nil {
		t.Fatalf("failed to write temp config: %v", err)
	}

	// Set env var (should override file)
	os.Setenv("LIBSQL_MEMORY_EMBEDDING_DIMENSION", "768")
	defer os.Unsetenv("LIBSQL_MEMORY_EMBEDDING_DIMENSION")

	// Flags should override env
	args := []string{"-embedding-dimension", "1024"}

	cfg, err := Load(configPath, args)
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	// File value should be used for database_path (not overridden)
	if cfg.DatabasePath != "/file/path" {
		t.Errorf("expected DatabasePath=/file/path, got %s", cfg.DatabasePath)
	}

	// Flag should take precedence for embedding_dimension
	if cfg.EmbeddingDimension != 1024 {
		t.Errorf("expected EmbeddingDimension=1024 (from flag), got %d", cfg.EmbeddingDimension)
	}

	// File value should be used for log_level (not overridden)
	if cfg.LogLevel != "error" {
		t.Errorf("expected LogLevel=error, got %s", cfg.LogLevel)
	}
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > 0 && containsHelper(s, substr))
}

func containsHelper(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
