// Package main provides the entry point for the MCP server.
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/libsql-memory/plugin/internal/mcp"
)

// StubMemoryService is a placeholder implementation until the real memory service is integrated.
// Replace this with the actual LibSQL-backed implementation.
type StubMemoryService struct{}

func (s *StubMemoryService) Store(ctx context.Context, args mcp.MemoryStoreArgs) error {
	return fmt.Errorf("memory service not implemented")
}

func (s *StubMemoryService) Retrieve(ctx context.Context, args mcp.MemoryRetrieveArgs) (string, map[string]string, []string, error) {
	return "", nil, nil, fmt.Errorf("memory service not implemented")
}

func (s *StubMemoryService) Search(ctx context.Context, args mcp.MemorySearchArgs) ([]mcp.SearchResult, error) {
	return nil, fmt.Errorf("memory service not implemented")
}

func (s *StubMemoryService) List(ctx context.Context, args mcp.MemoryListArgs) ([]mcp.MemoryEntry, int, error) {
	return nil, 0, fmt.Errorf("memory service not implemented")
}

func (s *StubMemoryService) Delete(ctx context.Context, args mcp.MemoryDeleteArgs) error {
	return fmt.Errorf("memory service not implemented")
}

func (s *StubMemoryService) Stats(ctx context.Context, args mcp.MemoryStatsArgs) (*mcp.MemoryStats, error) {
	return nil, fmt.Errorf("memory service not implemented")
}

func main() {
	logger := log.New(os.Stderr, "[mcp-server] ", log.LstdFlags)
	logger.Println("Starting libsql-memory MCP server...")

	// TODO: Initialize the real memory service with LibSQL connection
	// For now, use stub that returns not implemented errors
	memoryService := &StubMemoryService{}

	server := mcp.NewServer(memoryService)

	if err := server.Run(); err != nil {
		logger.Printf("Server error: %v", err)
		os.Exit(1)
	}

	logger.Println("Server shutdown complete")
}
