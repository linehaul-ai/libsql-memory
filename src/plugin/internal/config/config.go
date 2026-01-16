// Package config provides configuration management for the libsql-memory plugin.
// It supports loading configuration from environment variables, config files (JSON/YAML),
// and command line flags with sensible defaults and validation.
package config

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"gopkg.in/yaml.v3"
)

// Default configuration values
const (
	DefaultEmbeddingDimension     = 768 // Default to Nomic dimension
	DefaultNamespace              = "default"
	DefaultMaxConnections         = 10
	DefaultLogLevel               = "info"
	DefaultEmbeddingProvider      = "nomic" // Default to local Nomic model
	DefaultNomicEmbeddingEndpoint = "http://192.168.128.10:1234/v1/embeddings"
)

// Environment variable prefix
const envPrefix = "LIBSQL_MEMORY_"

// Config holds all configuration options for the libsql-memory plugin.
type Config struct {
	// DatabasePath is the local file path or Turso URL for the database
	DatabasePath string `json:"database_path" yaml:"database_path"`

	// AuthToken is used for Turso authentication (optional for local databases)
	AuthToken string `json:"auth_token" yaml:"auth_token"`

	// EmbeddingProvider specifies which embedding service to use ("openai", "nomic", or "local")
	EmbeddingProvider string `json:"embedding_provider" yaml:"embedding_provider"`

	// OpenAIAPIKey is the API key for OpenAI embeddings (required if provider is "openai")
	OpenAIAPIKey string `json:"openai_api_key" yaml:"openai_api_key"`

	// EmbeddingEndpoint is the API endpoint for embeddings (used for nomic provider)
	EmbeddingEndpoint string `json:"embedding_endpoint" yaml:"embedding_endpoint"`

	// EmbeddingDimension is the dimension of embedding vectors (default: 768 for Nomic)
	EmbeddingDimension int `json:"embedding_dimension" yaml:"embedding_dimension"`

	// DefaultNamespace is the default namespace for memory operations
	DefaultNamespace string `json:"default_namespace" yaml:"default_namespace"`

	// MaxConnections is the connection pool size for the database
	MaxConnections int `json:"max_connections" yaml:"max_connections"`

	// LogLevel controls logging verbosity (debug, info, warn, error)
	LogLevel string `json:"log_level" yaml:"log_level"`
}

// Option is a functional option for configuring Config
type Option func(*Config)

// WithDatabasePath sets the database path
func WithDatabasePath(path string) Option {
	return func(c *Config) {
		c.DatabasePath = path
	}
}

// WithAuthToken sets the authentication token
func WithAuthToken(token string) Option {
	return func(c *Config) {
		c.AuthToken = token
	}
}

// WithEmbeddingProvider sets the embedding provider
func WithEmbeddingProvider(provider string) Option {
	return func(c *Config) {
		c.EmbeddingProvider = provider
	}
}

// WithOpenAIAPIKey sets the OpenAI API key
func WithOpenAIAPIKey(key string) Option {
	return func(c *Config) {
		c.OpenAIAPIKey = key
	}
}

// WithEmbeddingEndpoint sets the embedding API endpoint
func WithEmbeddingEndpoint(endpoint string) Option {
	return func(c *Config) {
		c.EmbeddingEndpoint = endpoint
	}
}

// WithEmbeddingDimension sets the embedding dimension
func WithEmbeddingDimension(dim int) Option {
	return func(c *Config) {
		c.EmbeddingDimension = dim
	}
}

// WithDefaultNamespace sets the default namespace
func WithDefaultNamespace(ns string) Option {
	return func(c *Config) {
		c.DefaultNamespace = ns
	}
}

// WithMaxConnections sets the maximum number of database connections
func WithMaxConnections(max int) Option {
	return func(c *Config) {
		c.MaxConnections = max
	}
}

// WithLogLevel sets the log level
func WithLogLevel(level string) Option {
	return func(c *Config) {
		c.LogLevel = level
	}
}

// NewConfig creates a new Config with sensible defaults and applies any provided options.
// Options are applied in order, so later options override earlier ones.
func NewConfig(opts ...Option) *Config {
	cfg := &Config{
		EmbeddingProvider:  DefaultEmbeddingProvider,
		EmbeddingEndpoint:  DefaultNomicEmbeddingEndpoint,
		EmbeddingDimension: DefaultEmbeddingDimension,
		DefaultNamespace:   DefaultNamespace,
		MaxConnections:     DefaultMaxConnections,
		LogLevel:           DefaultLogLevel,
	}

	for _, opt := range opts {
		opt(cfg)
	}

	return cfg
}

// LoadFromEnv loads configuration from environment variables.
// Environment variables are prefixed with LIBSQL_MEMORY_.
// Returns a new Config with values from environment variables merged with defaults.
func LoadFromEnv() *Config {
	cfg := NewConfig()

	if v := os.Getenv(envPrefix + "DATABASE_PATH"); v != "" {
		cfg.DatabasePath = v
	}

	if v := os.Getenv(envPrefix + "AUTH_TOKEN"); v != "" {
		cfg.AuthToken = v
	}

	if v := os.Getenv(envPrefix + "EMBEDDING_PROVIDER"); v != "" {
		cfg.EmbeddingProvider = v
	}

	if v := os.Getenv(envPrefix + "OPENAI_API_KEY"); v != "" {
		cfg.OpenAIAPIKey = v
	}

	if v := os.Getenv(envPrefix + "EMBEDDING_ENDPOINT"); v != "" {
		cfg.EmbeddingEndpoint = v
	}

	if v := os.Getenv(envPrefix + "EMBEDDING_DIMENSION"); v != "" {
		if dim, err := strconv.Atoi(v); err == nil {
			cfg.EmbeddingDimension = dim
		}
	}

	if v := os.Getenv(envPrefix + "DEFAULT_NAMESPACE"); v != "" {
		cfg.DefaultNamespace = v
	}

	if v := os.Getenv(envPrefix + "MAX_CONNECTIONS"); v != "" {
		if max, err := strconv.Atoi(v); err == nil {
			cfg.MaxConnections = max
		}
	}

	if v := os.Getenv(envPrefix + "LOG_LEVEL"); v != "" {
		cfg.LogLevel = v
	}

	return cfg
}

// LoadFromFile loads configuration from a JSON or YAML file.
// The file format is determined by the file extension.
// Supported extensions: .json, .yaml, .yml
func LoadFromFile(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	cfg := NewConfig()
	ext := strings.ToLower(filepath.Ext(path))

	switch ext {
	case ".json":
		if err := json.Unmarshal(data, cfg); err != nil {
			return nil, fmt.Errorf("failed to parse JSON config: %w", err)
		}
	case ".yaml", ".yml":
		if err := yaml.Unmarshal(data, cfg); err != nil {
			return nil, fmt.Errorf("failed to parse YAML config: %w", err)
		}
	default:
		return nil, fmt.Errorf("unsupported config file format: %s", ext)
	}

	return cfg, nil
}

// LoadFromFlags creates a new FlagSet and parses command line flags.
// This function returns a Config populated from flags and any parse errors.
// The flagSet is configured but not parsed - call Parse() on the returned FlagSet.
func LoadFromFlags(args []string) (*Config, error) {
	cfg := NewConfig()
	fs := flag.NewFlagSet("libsql-memory", flag.ContinueOnError)

	fs.StringVar(&cfg.DatabasePath, "database-path", cfg.DatabasePath,
		"Database path (local file or Turso URL)")
	fs.StringVar(&cfg.AuthToken, "auth-token", cfg.AuthToken,
		"Authentication token for Turso")
	fs.StringVar(&cfg.EmbeddingProvider, "embedding-provider", cfg.EmbeddingProvider,
		"Embedding provider (openai or local)")
	fs.StringVar(&cfg.OpenAIAPIKey, "openai-api-key", cfg.OpenAIAPIKey,
		"OpenAI API key for embeddings")
	fs.StringVar(&cfg.EmbeddingEndpoint, "embedding-endpoint", cfg.EmbeddingEndpoint,
		"Embedding API endpoint (for nomic provider)")
	fs.IntVar(&cfg.EmbeddingDimension, "embedding-dimension", cfg.EmbeddingDimension,
		"Embedding vector dimension")
	fs.StringVar(&cfg.DefaultNamespace, "default-namespace", cfg.DefaultNamespace,
		"Default namespace for memory operations")
	fs.IntVar(&cfg.MaxConnections, "max-connections", cfg.MaxConnections,
		"Maximum database connections")
	fs.StringVar(&cfg.LogLevel, "log-level", cfg.LogLevel,
		"Log level (debug, info, warn, error)")

	// Add config file flag for convenience
	var configFile string
	fs.StringVar(&configFile, "config", "",
		"Path to config file (JSON or YAML)")

	if err := fs.Parse(args); err != nil {
		return nil, fmt.Errorf("failed to parse flags: %w", err)
	}

	// If config file specified, load it first then override with flags
	if configFile != "" {
		fileCfg, err := LoadFromFile(configFile)
		if err != nil {
			return nil, err
		}
		// Merge file config with flag values (flags take precedence)
		cfg = mergeConfigs(fileCfg, cfg, fs)
	}

	return cfg, nil
}

// Load loads configuration from all sources in the following priority order
// (highest priority last):
// 1. Default values
// 2. Config file (if configPath is provided)
// 3. Environment variables
// 4. Command line flags
func Load(configPath string, args []string) (*Config, error) {
	// Start with defaults
	cfg := NewConfig()

	// Load from config file if provided
	if configPath != "" {
		fileCfg, err := LoadFromFile(configPath)
		if err != nil {
			return nil, err
		}
		cfg = fileCfg
	}

	// Override with environment variables
	envCfg := LoadFromEnv()
	cfg = mergeWithEnv(cfg, envCfg)

	// Override with command line flags
	if len(args) > 0 {
		flagCfg, err := LoadFromFlags(args)
		if err != nil {
			return nil, err
		}
		cfg = mergeWithFlags(cfg, flagCfg, args)
	}

	return cfg, nil
}

// Validate checks if the configuration is valid and returns an error if not.
// Validation rules:
// - DatabasePath must be set
// - EmbeddingProvider must be "openai" or "local"
// - OpenAIAPIKey must be set if EmbeddingProvider is "openai"
// - EmbeddingDimension must be positive
// - MaxConnections must be positive
// - LogLevel must be valid
func (c *Config) Validate() error {
	var errs []string

	if c.DatabasePath == "" {
		errs = append(errs, "database_path is required")
	}

	validProviders := map[string]bool{"openai": true, "nomic": true, "local": true}
	if !validProviders[strings.ToLower(c.EmbeddingProvider)] {
		errs = append(errs, fmt.Sprintf("embedding_provider must be 'openai', 'nomic', or 'local', got '%s'", c.EmbeddingProvider))
	}

	if strings.ToLower(c.EmbeddingProvider) == "openai" && c.OpenAIAPIKey == "" {
		errs = append(errs, "openai_api_key is required when embedding_provider is 'openai'")
	}

	if strings.ToLower(c.EmbeddingProvider) == "nomic" && c.EmbeddingEndpoint == "" {
		errs = append(errs, "embedding_endpoint is required when embedding_provider is 'nomic'")
	}

	if c.EmbeddingDimension <= 0 {
		errs = append(errs, fmt.Sprintf("embedding_dimension must be positive, got %d", c.EmbeddingDimension))
	}

	if c.MaxConnections <= 0 {
		errs = append(errs, fmt.Sprintf("max_connections must be positive, got %d", c.MaxConnections))
	}

	validLogLevels := map[string]bool{"debug": true, "info": true, "warn": true, "error": true}
	if !validLogLevels[strings.ToLower(c.LogLevel)] {
		errs = append(errs, fmt.Sprintf("log_level must be one of: debug, info, warn, error; got '%s'", c.LogLevel))
	}

	if len(errs) > 0 {
		return &ValidationError{Errors: errs}
	}

	return nil
}

// ValidationError represents configuration validation errors
type ValidationError struct {
	Errors []string
}

func (e *ValidationError) Error() string {
	return fmt.Sprintf("configuration validation failed: %s", strings.Join(e.Errors, "; "))
}

// IsTursoURL checks if the database path is a Turso URL
func (c *Config) IsTursoURL() bool {
	return strings.HasPrefix(c.DatabasePath, "libsql://") ||
		strings.HasPrefix(c.DatabasePath, "https://") ||
		strings.Contains(c.DatabasePath, ".turso.io")
}

// IsLocalDatabase checks if the database path is a local file
func (c *Config) IsLocalDatabase() bool {
	return !c.IsTursoURL()
}

// String returns a string representation of the config (with sensitive fields masked)
func (c *Config) String() string {
	masked := *c
	if masked.AuthToken != "" {
		masked.AuthToken = "***"
	}
	if masked.OpenAIAPIKey != "" {
		masked.OpenAIAPIKey = "***"
	}

	data, _ := json.MarshalIndent(masked, "", "  ")
	return string(data)
}

// Clone creates a deep copy of the config
func (c *Config) Clone() *Config {
	return &Config{
		DatabasePath:       c.DatabasePath,
		AuthToken:          c.AuthToken,
		EmbeddingProvider:  c.EmbeddingProvider,
		OpenAIAPIKey:       c.OpenAIAPIKey,
		EmbeddingEndpoint:  c.EmbeddingEndpoint,
		EmbeddingDimension: c.EmbeddingDimension,
		DefaultNamespace:   c.DefaultNamespace,
		MaxConnections:     c.MaxConnections,
		LogLevel:           c.LogLevel,
	}
}

// mergeConfigs merges two configs, with values from 'override' taking precedence
// only if they were explicitly set via flags
func mergeConfigs(base, override *Config, fs *flag.FlagSet) *Config {
	result := base.Clone()

	fs.Visit(func(f *flag.Flag) {
		switch f.Name {
		case "database-path":
			result.DatabasePath = override.DatabasePath
		case "auth-token":
			result.AuthToken = override.AuthToken
		case "embedding-provider":
			result.EmbeddingProvider = override.EmbeddingProvider
		case "openai-api-key":
			result.OpenAIAPIKey = override.OpenAIAPIKey
		case "embedding-endpoint":
			result.EmbeddingEndpoint = override.EmbeddingEndpoint
		case "embedding-dimension":
			result.EmbeddingDimension = override.EmbeddingDimension
		case "default-namespace":
			result.DefaultNamespace = override.DefaultNamespace
		case "max-connections":
			result.MaxConnections = override.MaxConnections
		case "log-level":
			result.LogLevel = override.LogLevel
		}
	})

	return result
}

// mergeWithEnv merges environment config with base config
func mergeWithEnv(base, env *Config) *Config {
	result := base.Clone()

	if os.Getenv(envPrefix+"DATABASE_PATH") != "" {
		result.DatabasePath = env.DatabasePath
	}
	if os.Getenv(envPrefix+"AUTH_TOKEN") != "" {
		result.AuthToken = env.AuthToken
	}
	if os.Getenv(envPrefix+"EMBEDDING_PROVIDER") != "" {
		result.EmbeddingProvider = env.EmbeddingProvider
	}
	if os.Getenv(envPrefix+"OPENAI_API_KEY") != "" {
		result.OpenAIAPIKey = env.OpenAIAPIKey
	}
	if os.Getenv(envPrefix+"EMBEDDING_ENDPOINT") != "" {
		result.EmbeddingEndpoint = env.EmbeddingEndpoint
	}
	if os.Getenv(envPrefix+"EMBEDDING_DIMENSION") != "" {
		result.EmbeddingDimension = env.EmbeddingDimension
	}
	if os.Getenv(envPrefix+"DEFAULT_NAMESPACE") != "" {
		result.DefaultNamespace = env.DefaultNamespace
	}
	if os.Getenv(envPrefix+"MAX_CONNECTIONS") != "" {
		result.MaxConnections = env.MaxConnections
	}
	if os.Getenv(envPrefix+"LOG_LEVEL") != "" {
		result.LogLevel = env.LogLevel
	}

	return result
}

// mergeWithFlags merges flag config with base config based on what flags were actually set
func mergeWithFlags(base, flagCfg *Config, args []string) *Config {
	result := base.Clone()

	// Create a temporary flagset to detect which flags were set
	fs := flag.NewFlagSet("temp", flag.ContinueOnError)
	var tempCfg Config
	fs.StringVar(&tempCfg.DatabasePath, "database-path", "", "")
	fs.StringVar(&tempCfg.AuthToken, "auth-token", "", "")
	fs.StringVar(&tempCfg.EmbeddingProvider, "embedding-provider", "", "")
	fs.StringVar(&tempCfg.OpenAIAPIKey, "openai-api-key", "", "")
	fs.StringVar(&tempCfg.EmbeddingEndpoint, "embedding-endpoint", "", "")
	fs.IntVar(&tempCfg.EmbeddingDimension, "embedding-dimension", 0, "")
	fs.StringVar(&tempCfg.DefaultNamespace, "default-namespace", "", "")
	fs.IntVar(&tempCfg.MaxConnections, "max-connections", 0, "")
	fs.StringVar(&tempCfg.LogLevel, "log-level", "", "")
	fs.String("config", "", "") // ignore config flag

	_ = fs.Parse(args)

	fs.Visit(func(f *flag.Flag) {
		switch f.Name {
		case "database-path":
			result.DatabasePath = flagCfg.DatabasePath
		case "auth-token":
			result.AuthToken = flagCfg.AuthToken
		case "embedding-provider":
			result.EmbeddingProvider = flagCfg.EmbeddingProvider
		case "openai-api-key":
			result.OpenAIAPIKey = flagCfg.OpenAIAPIKey
		case "embedding-endpoint":
			result.EmbeddingEndpoint = flagCfg.EmbeddingEndpoint
		case "embedding-dimension":
			result.EmbeddingDimension = flagCfg.EmbeddingDimension
		case "default-namespace":
			result.DefaultNamespace = flagCfg.DefaultNamespace
		case "max-connections":
			result.MaxConnections = flagCfg.MaxConnections
		case "log-level":
			result.LogLevel = flagCfg.LogLevel
		}
	})

	return result
}
