// Package main provides the entry point for the Claude memory plugin MCP server.
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/libsql-memory/plugin/internal/config"
	"github.com/libsql-memory/plugin/internal/db"
	"github.com/libsql-memory/plugin/internal/embedding"
	"github.com/libsql-memory/plugin/internal/mcp"
	"github.com/libsql-memory/plugin/internal/memory"
)

// Version information (set at build time via ldflags)
var (
	version   = "dev"
	commit    = "unknown"
	buildDate = "unknown"
)

func main() {
	// All logging goes to stderr (stdout is reserved for MCP protocol)
	log.SetOutput(os.Stderr)
	log.SetFlags(log.Ldate | log.Ltime | log.Lmicroseconds)

	// Parse configuration from flags, environment, and config file
	cfg, err := config.Load("", os.Args[1:])
	if err != nil {
		log.Printf("[ERROR] Failed to load configuration: %v", err)
		os.Exit(1)
	}

	// Validate configuration
	if err := cfg.Validate(); err != nil {
		log.Printf("[ERROR] Invalid configuration: %v", err)
		os.Exit(1)
	}

	if cfg.LogLevel == "debug" {
		log.Printf("[DEBUG] Configuration loaded: %s", cfg.String())
	}

	log.Printf("[INFO] Starting Claude Memory Plugin v%s (commit: %s, built: %s)", version, commit, buildDate)

	// Create context with cancellation for graceful shutdown
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Set up signal handling for graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	// Channel to signal when server is ready
	serverDone := make(chan error, 1)

	// Start the server in a goroutine
	go func() {
		serverDone <- run(ctx, cfg)
	}()

	// Wait for shutdown signal or server error
	select {
	case sig := <-sigChan:
		log.Printf("[INFO] Received signal %v, initiating graceful shutdown...", sig)
	case err := <-serverDone:
		if err != nil {
			log.Printf("[ERROR] Server error: %v", err)
			os.Exit(1)
		}
		log.Printf("[INFO] Server stopped")
		return
	}

	// Cancel the main context to signal components to stop
	cancel()

	// Create shutdown context with timeout
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer shutdownCancel()

	// Wait for server to finish or timeout
	select {
	case <-shutdownCtx.Done():
		log.Printf("[WARN] Shutdown timed out, forcing exit")
	case err := <-serverDone:
		if err != nil && err != context.Canceled {
			log.Printf("[WARN] Server shutdown with error: %v", err)
		}
	}

	log.Printf("[INFO] Claude Memory Plugin shutdown complete")
}

// run initializes and starts the memory plugin server.
func run(ctx context.Context, cfg *config.Config) error {
	log.Printf("[INFO] Initializing database connection...")

	// Initialize database with context
	dbConfig := db.Config{
		Path:             cfg.DatabasePath,
		VectorDimensions: cfg.EmbeddingDimension,
	}

	database, err := db.New(ctx, dbConfig)
	if err != nil {
		return fmt.Errorf("failed to initialize database: %w", err)
	}
	defer database.Close()

	// Ping database to verify connection
	pingCtx, pingCancel := context.WithTimeout(ctx, 10*time.Second)
	defer pingCancel()
	if err := database.Ping(pingCtx); err != nil {
		return fmt.Errorf("failed to connect to database: %w", err)
	}

	log.Printf("[INFO] Database connected and schema initialized")

	// Initialize embedder based on provider
	log.Printf("[INFO] Initializing embedding provider: %s (dim=%d)", cfg.EmbeddingProvider, cfg.EmbeddingDimension)
	var embedder embedding.Embedder
	embedder, err = initEmbedder(cfg)
	if err != nil {
		return fmt.Errorf("failed to initialize embedder: %w", err)
	}
	defer embedder.Close()

	log.Printf("[INFO] Embedding provider initialized successfully")

	// Initialize memory store
	store := memory.NewStore(database, embedder)
	log.Printf("[INFO] Memory store initialized successfully")

	// Create memory service adapter for MCP server
	memoryService := &memoryServiceAdapter{
		store:            store,
		defaultNamespace: cfg.DefaultNamespace,
	}

	// Create and start MCP server
	log.Printf("[INFO] Starting MCP server on stdio...")
	server := mcp.NewServer(memoryService)

	// Run server (blocks until context cancelled or EOF)
	return server.Run()
}

// initEmbedder creates an embedder based on configuration
func initEmbedder(cfg *config.Config) (embedding.Embedder, error) {
	switch cfg.EmbeddingProvider {
	case "openai":
		if cfg.OpenAIAPIKey == "" {
			return nil, fmt.Errorf("OpenAI API key is required for openai embedding provider")
		}
		openAICfg := embedding.DefaultOpenAIConfig(cfg.OpenAIAPIKey)
		if cfg.EmbeddingDimension > 0 {
			openAICfg.Dimension = cfg.EmbeddingDimension
		}
		return embedding.NewOpenAIEmbedder(openAICfg)
	case "nomic":
		nomicCfg := embedding.DefaultNomicConfig(cfg.EmbeddingEndpoint)
		if cfg.EmbeddingDimension > 0 {
			nomicCfg.Dimension = cfg.EmbeddingDimension
		}
		log.Printf("[INFO] Using Nomic model at endpoint: %s", nomicCfg.Endpoint)
		return embedding.NewNomicEmbedder(nomicCfg)
	case "local":
		localCfg := embedding.DefaultLocalConfig()
		if cfg.EmbeddingDimension > 0 {
			localCfg.Dimension = cfg.EmbeddingDimension
		}
		return embedding.NewLocalEmbedder(localCfg), nil
	default:
		return nil, fmt.Errorf("unknown embedding provider: %s", cfg.EmbeddingProvider)
	}
}

// memoryServiceAdapter adapts memory.Store to mcp.MemoryService interface
type memoryServiceAdapter struct {
	store            *memory.Store
	defaultNamespace string
}

func (a *memoryServiceAdapter) Store(ctx context.Context, args mcp.MemoryStoreArgs) error {
	namespace := args.Namespace
	if namespace == "" {
		namespace = a.defaultNamespace
	}

	// Convert metadata from map[string]string to map[string]any
	metadata := make(map[string]any)
	for k, v := range args.Metadata {
		metadata[k] = v
	}

	opts := []memory.StoreOption{}
	if len(args.Tags) > 0 {
		opts = append(opts, memory.WithTags(args.Tags...))
	}

	return a.store.Store(ctx, namespace, args.Key, args.Value, metadata, opts...)
}

func (a *memoryServiceAdapter) Retrieve(ctx context.Context, args mcp.MemoryRetrieveArgs) (string, map[string]string, []string, error) {
	namespace := args.Namespace
	if namespace == "" {
		namespace = a.defaultNamespace
	}

	mem, err := a.store.Retrieve(ctx, namespace, args.Key)
	if err != nil {
		return "", nil, nil, err
	}

	// Convert metadata from map[string]any to map[string]string
	metadata := make(map[string]string)
	for k, v := range mem.Metadata {
		if s, ok := v.(string); ok {
			metadata[k] = s
		} else {
			metadata[k] = fmt.Sprintf("%v", v)
		}
	}

	return mem.Value, metadata, mem.Tags, nil
}

func (a *memoryServiceAdapter) Search(ctx context.Context, args mcp.MemorySearchArgs) ([]mcp.SearchResult, error) {
	namespace := args.Namespace
	// Empty namespace searches all namespaces

	searchResults, err := a.store.Search(ctx, args.Query, namespace, args.Limit, args.Threshold)
	if err != nil {
		return nil, err
	}

	results := make([]mcp.SearchResult, len(searchResults))
	for i, sr := range searchResults {
		// Convert metadata
		metadata := make(map[string]string)
		for k, v := range sr.Memory.Metadata {
			if s, ok := v.(string); ok {
				metadata[k] = s
			} else {
				metadata[k] = fmt.Sprintf("%v", v)
			}
		}

		results[i] = mcp.SearchResult{
			Key:       sr.Memory.Key,
			Value:     sr.Memory.Value,
			Namespace: sr.Memory.Namespace,
			Metadata:  metadata,
			Tags:      sr.Memory.Tags,
			Score:     sr.Score,
		}
	}

	return results, nil
}

func (a *memoryServiceAdapter) List(ctx context.Context, args mcp.MemoryListArgs) ([]mcp.MemoryEntry, int, error) {
	namespace := args.Namespace
	if namespace == "" {
		namespace = a.defaultNamespace
	}

	memories, err := a.store.List(ctx, namespace, args.Limit)
	if err != nil {
		return nil, 0, err
	}

	entries := make([]mcp.MemoryEntry, len(memories))
	for i, mem := range memories {
		// Convert metadata
		metadata := make(map[string]string)
		for k, v := range mem.Metadata {
			if s, ok := v.(string); ok {
				metadata[k] = s
			} else {
				metadata[k] = fmt.Sprintf("%v", v)
			}
		}

		entries[i] = mcp.MemoryEntry{
			Key:       mem.Key,
			Value:     mem.Value,
			Namespace: mem.Namespace,
			Metadata:  metadata,
			Tags:      mem.Tags,
			CreatedAt: mem.CreatedAt.Format(time.RFC3339),
			UpdatedAt: mem.UpdatedAt.Format(time.RFC3339),
		}
	}

	// Get total count
	total, err := a.store.Count(ctx, namespace)
	if err != nil {
		return entries, len(entries), nil // Return what we have
	}

	return entries, int(total), nil
}

func (a *memoryServiceAdapter) Delete(ctx context.Context, args mcp.MemoryDeleteArgs) error {
	namespace := args.Namespace
	if namespace == "" {
		namespace = a.defaultNamespace
	}

	return a.store.Delete(ctx, namespace, args.Key)
}

func (a *memoryServiceAdapter) Stats(ctx context.Context, args mcp.MemoryStatsArgs) (*mcp.MemoryStats, error) {
	dbStats, err := a.store.Stats(ctx)
	if err != nil {
		return nil, err
	}

	entriesByNS := make(map[string]int)
	for ns, count := range dbStats.NamespaceCounts {
		entriesByNS[ns] = int(count)
	}

	stats := &mcp.MemoryStats{
		TotalEntries:    int(dbStats.TotalMemories),
		TotalNamespaces: len(dbStats.NamespaceCounts),
		EntriesByNS:     entriesByNS,
		StorageBytes:    0,
	}

	// Filter to specific namespace if requested
	if args.Namespace != "" {
		if count, ok := dbStats.NamespaceCounts[args.Namespace]; ok {
			stats.TotalEntries = int(count)
			stats.TotalNamespaces = 1
			stats.EntriesByNS = map[string]int{args.Namespace: int(count)}
		} else {
			stats.TotalEntries = 0
			stats.TotalNamespaces = 0
			stats.EntriesByNS = map[string]int{}
		}
	}

	return stats, nil
}
