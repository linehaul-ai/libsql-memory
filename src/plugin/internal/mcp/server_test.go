package mcp

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"testing"
)

// MockMemoryService implements MemoryService for testing
type MockMemoryService struct {
	mu       sync.RWMutex
	memories map[string]map[string]*mockEntry
}

type mockEntry struct {
	value    string
	metadata map[string]string
	tags     []string
}

func NewMockMemoryService() *MockMemoryService {
	return &MockMemoryService{
		memories: make(map[string]map[string]*mockEntry),
	}
}

func (m *MockMemoryService) Store(ctx context.Context, args MemoryStoreArgs) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.memories[args.Namespace] == nil {
		m.memories[args.Namespace] = make(map[string]*mockEntry)
	}
	m.memories[args.Namespace][args.Key] = &mockEntry{
		value:    args.Value,
		metadata: args.Metadata,
		tags:     args.Tags,
	}
	return nil
}

func (m *MockMemoryService) Retrieve(ctx context.Context, args MemoryRetrieveArgs) (string, map[string]string, []string, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	ns, ok := m.memories[args.Namespace]
	if !ok {
		return "", nil, nil, fmt.Errorf("namespace not found: %s", args.Namespace)
	}
	entry, ok := ns[args.Key]
	if !ok {
		return "", nil, nil, fmt.Errorf("key not found: %s", args.Key)
	}
	return entry.value, entry.metadata, entry.tags, nil
}

func (m *MockMemoryService) Search(ctx context.Context, args MemorySearchArgs) ([]SearchResult, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	var results []SearchResult

	searchNS := func(ns string, entries map[string]*mockEntry) {
		for key, entry := range entries {
			if strings.Contains(strings.ToLower(entry.value), strings.ToLower(args.Query)) {
				results = append(results, SearchResult{
					Key:       key,
					Value:     entry.value,
					Score:     0.85,
					Namespace: ns,
					Metadata:  entry.metadata,
					Tags:      entry.tags,
				})
			}
		}
	}

	if args.Namespace != "" {
		if ns, ok := m.memories[args.Namespace]; ok {
			searchNS(args.Namespace, ns)
		}
	} else {
		for nsName, ns := range m.memories {
			searchNS(nsName, ns)
		}
	}

	if args.Limit > 0 && len(results) > args.Limit {
		results = results[:args.Limit]
	}

	return results, nil
}

func (m *MockMemoryService) List(ctx context.Context, args MemoryListArgs) ([]MemoryEntry, int, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	var entries []MemoryEntry
	ns, ok := m.memories[args.Namespace]
	if !ok {
		return entries, 0, nil
	}

	for key, entry := range ns {
		entries = append(entries, MemoryEntry{
			Key:       key,
			Value:     entry.value,
			Namespace: args.Namespace,
			Metadata:  entry.metadata,
			Tags:      entry.tags,
		})
	}

	total := len(entries)

	if args.Offset > 0 && args.Offset < len(entries) {
		entries = entries[args.Offset:]
	}
	if args.Limit > 0 && len(entries) > args.Limit {
		entries = entries[:args.Limit]
	}

	return entries, total, nil
}

func (m *MockMemoryService) Delete(ctx context.Context, args MemoryDeleteArgs) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	ns, ok := m.memories[args.Namespace]
	if !ok {
		return fmt.Errorf("namespace not found: %s", args.Namespace)
	}
	if _, ok := ns[args.Key]; !ok {
		return fmt.Errorf("key not found: %s", args.Key)
	}
	delete(ns, args.Key)
	return nil
}

func (m *MockMemoryService) Stats(ctx context.Context, args MemoryStatsArgs) (*MemoryStats, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	stats := &MemoryStats{
		EntriesByNS: make(map[string]int),
	}

	for nsName, ns := range m.memories {
		if args.Namespace == "" || args.Namespace == nsName {
			count := len(ns)
			stats.EntriesByNS[nsName] = count
			stats.TotalEntries += count
			stats.TotalNamespaces++
		}
	}

	return stats, nil
}

// Helper to create a JSON-RPC request
func makeRequest(id interface{}, method string, params interface{}) string {
	req := map[string]interface{}{
		"jsonrpc": "2.0",
		"id":      id,
		"method":  method,
	}
	if params != nil {
		paramsJSON, _ := json.Marshal(params)
		req["params"] = json.RawMessage(paramsJSON)
	}
	data, _ := json.Marshal(req)
	return string(data) + "\n"
}

func TestServer_Initialize(t *testing.T) {
	mock := NewMockMemoryService()
	input := &bytes.Buffer{}
	output := &bytes.Buffer{}

	server := NewServerWithIO(mock, input, output)

	// Write initialize request
	input.WriteString(makeRequest(1, "initialize", map[string]interface{}{
		"protocolVersion": "2024-11-05",
		"capabilities":    map[string]interface{}{},
		"clientInfo": map[string]interface{}{
			"name":    "test-client",
			"version": "1.0.0",
		},
	}))

	// Process the request
	line, _ := server.reader.ReadBytes('\n')
	server.handleMessage(line)

	// Parse response
	var resp Response
	if err := json.Unmarshal(output.Bytes(), &resp); err != nil {
		t.Fatalf("Failed to parse response: %v", err)
	}

	if resp.Error != nil {
		t.Fatalf("Unexpected error: %v", resp.Error)
	}

	result, ok := resp.Result.(map[string]interface{})
	if !ok {
		t.Fatal("Result is not a map")
	}

	if result["protocolVersion"] != MCPVersion {
		t.Errorf("Expected protocol version %s, got %v", MCPVersion, result["protocolVersion"])
	}

	serverInfo := result["serverInfo"].(map[string]interface{})
	if serverInfo["name"] != "libsql-memory" {
		t.Errorf("Expected server name 'libsql-memory', got %v", serverInfo["name"])
	}
}

func TestServer_ToolsList(t *testing.T) {
	mock := NewMockMemoryService()
	input := &bytes.Buffer{}
	output := &bytes.Buffer{}

	server := NewServerWithIO(mock, input, output)

	input.WriteString(makeRequest(1, "tools/list", nil))

	line, _ := server.reader.ReadBytes('\n')
	server.handleMessage(line)

	var resp Response
	if err := json.Unmarshal(output.Bytes(), &resp); err != nil {
		t.Fatalf("Failed to parse response: %v", err)
	}

	if resp.Error != nil {
		t.Fatalf("Unexpected error: %v", resp.Error)
	}

	result, ok := resp.Result.(map[string]interface{})
	if !ok {
		t.Fatal("Result is not a map")
	}

	tools := result["tools"].([]interface{})
	if len(tools) != 6 {
		t.Errorf("Expected 6 tools, got %d", len(tools))
	}

	expectedTools := map[string]bool{
		"memory_store":    false,
		"memory_retrieve": false,
		"memory_search":   false,
		"memory_list":     false,
		"memory_delete":   false,
		"memory_stats":    false,
	}

	for _, tool := range tools {
		toolMap := tool.(map[string]interface{})
		name := toolMap["name"].(string)
		if _, ok := expectedTools[name]; ok {
			expectedTools[name] = true
		}
	}

	for name, found := range expectedTools {
		if !found {
			t.Errorf("Expected tool %s not found", name)
		}
	}
}

func TestServer_MemoryStore(t *testing.T) {
	mock := NewMockMemoryService()
	input := &bytes.Buffer{}
	output := &bytes.Buffer{}

	server := NewServerWithIO(mock, input, output)

	input.WriteString(makeRequest(1, "tools/call", map[string]interface{}{
		"name": "memory_store",
		"arguments": map[string]interface{}{
			"key":       "test-key",
			"value":     "test-value",
			"namespace": "test-ns",
			"metadata":  map[string]string{"author": "test"},
			"tags":      []string{"tag1", "tag2"},
		},
	}))

	line, _ := server.reader.ReadBytes('\n')
	server.handleMessage(line)

	var resp Response
	if err := json.Unmarshal(output.Bytes(), &resp); err != nil {
		t.Fatalf("Failed to parse response: %v", err)
	}

	if resp.Error != nil {
		t.Fatalf("Unexpected error: %v", resp.Error)
	}

	result := resp.Result.(map[string]interface{})
	content := result["content"].([]interface{})
	if len(content) == 0 {
		t.Fatal("Expected content in result")
	}

	contentBlock := content[0].(map[string]interface{})
	text := contentBlock["text"].(string)

	var storeResult map[string]interface{}
	if err := json.Unmarshal([]byte(text), &storeResult); err != nil {
		t.Fatalf("Failed to parse store result: %v", err)
	}

	if storeResult["success"] != true {
		t.Error("Expected success to be true")
	}
}

func TestServer_MemoryRetrieve(t *testing.T) {
	mock := NewMockMemoryService()
	input := &bytes.Buffer{}
	output := &bytes.Buffer{}

	server := NewServerWithIO(mock, input, output)

	// First store a memory
	mock.Store(context.Background(), MemoryStoreArgs{
		Key:       "retrieve-key",
		Value:     "retrieve-value",
		Namespace: "default",
	})

	input.WriteString(makeRequest(1, "tools/call", map[string]interface{}{
		"name": "memory_retrieve",
		"arguments": map[string]interface{}{
			"key":       "retrieve-key",
			"namespace": "default",
		},
	}))

	line, _ := server.reader.ReadBytes('\n')
	server.handleMessage(line)

	var resp Response
	if err := json.Unmarshal(output.Bytes(), &resp); err != nil {
		t.Fatalf("Failed to parse response: %v", err)
	}

	if resp.Error != nil {
		t.Fatalf("Unexpected error: %v", resp.Error)
	}

	result := resp.Result.(map[string]interface{})
	content := result["content"].([]interface{})
	contentBlock := content[0].(map[string]interface{})
	text := contentBlock["text"].(string)

	var retrieveResult map[string]interface{}
	if err := json.Unmarshal([]byte(text), &retrieveResult); err != nil {
		t.Fatalf("Failed to parse retrieve result: %v", err)
	}

	if retrieveResult["value"] != "retrieve-value" {
		t.Errorf("Expected value 'retrieve-value', got %v", retrieveResult["value"])
	}
}

func TestServer_MemorySearch(t *testing.T) {
	mock := NewMockMemoryService()
	input := &bytes.Buffer{}
	output := &bytes.Buffer{}

	server := NewServerWithIO(mock, input, output)

	// Store some memories
	mock.Store(context.Background(), MemoryStoreArgs{
		Key:       "auth-patterns",
		Value:     "JWT authentication with refresh tokens",
		Namespace: "patterns",
	})
	mock.Store(context.Background(), MemoryStoreArgs{
		Key:       "db-patterns",
		Value:     "Database connection pooling best practices",
		Namespace: "patterns",
	})

	input.WriteString(makeRequest(1, "tools/call", map[string]interface{}{
		"name": "memory_search",
		"arguments": map[string]interface{}{
			"query":     "authentication",
			"namespace": "patterns",
			"limit":     10,
		},
	}))

	line, _ := server.reader.ReadBytes('\n')
	server.handleMessage(line)

	var resp Response
	if err := json.Unmarshal(output.Bytes(), &resp); err != nil {
		t.Fatalf("Failed to parse response: %v", err)
	}

	if resp.Error != nil {
		t.Fatalf("Unexpected error: %v", resp.Error)
	}

	result := resp.Result.(map[string]interface{})
	content := result["content"].([]interface{})
	contentBlock := content[0].(map[string]interface{})
	text := contentBlock["text"].(string)

	var searchResult map[string]interface{}
	if err := json.Unmarshal([]byte(text), &searchResult); err != nil {
		t.Fatalf("Failed to parse search result: %v", err)
	}

	count := int(searchResult["count"].(float64))
	if count != 1 {
		t.Errorf("Expected 1 result, got %d", count)
	}
}

func TestServer_MethodNotFound(t *testing.T) {
	mock := NewMockMemoryService()
	input := &bytes.Buffer{}
	output := &bytes.Buffer{}

	server := NewServerWithIO(mock, input, output)

	input.WriteString(makeRequest(1, "unknown/method", nil))

	line, _ := server.reader.ReadBytes('\n')
	server.handleMessage(line)

	var resp Response
	if err := json.Unmarshal(output.Bytes(), &resp); err != nil {
		t.Fatalf("Failed to parse response: %v", err)
	}

	if resp.Error == nil {
		t.Fatal("Expected error response")
	}

	if resp.Error.Code != MethodNotFound {
		t.Errorf("Expected error code %d, got %d", MethodNotFound, resp.Error.Code)
	}
}

func TestServer_InvalidJSON(t *testing.T) {
	mock := NewMockMemoryService()
	input := &bytes.Buffer{}
	output := &bytes.Buffer{}

	server := NewServerWithIO(mock, input, output)

	input.WriteString("not valid json\n")

	line, _ := server.reader.ReadBytes('\n')
	server.handleMessage(line)

	var resp Response
	if err := json.Unmarshal(output.Bytes(), &resp); err != nil {
		t.Fatalf("Failed to parse response: %v", err)
	}

	if resp.Error == nil {
		t.Fatal("Expected error response")
	}

	if resp.Error.Code != ParseError {
		t.Errorf("Expected error code %d, got %d", ParseError, resp.Error.Code)
	}
}

func TestServer_Ping(t *testing.T) {
	mock := NewMockMemoryService()
	input := &bytes.Buffer{}
	output := &bytes.Buffer{}

	server := NewServerWithIO(mock, input, output)

	input.WriteString(makeRequest(1, "ping", nil))

	line, _ := server.reader.ReadBytes('\n')
	server.handleMessage(line)

	var resp Response
	if err := json.Unmarshal(output.Bytes(), &resp); err != nil {
		t.Fatalf("Failed to parse response: %v", err)
	}

	if resp.Error != nil {
		t.Fatalf("Unexpected error: %v", resp.Error)
	}

	if resp.Result == nil {
		t.Error("Expected empty object result")
	}
}

func TestServer_MissingRequiredParams(t *testing.T) {
	mock := NewMockMemoryService()
	input := &bytes.Buffer{}
	output := &bytes.Buffer{}

	server := NewServerWithIO(mock, input, output)

	// Try to store without key
	input.WriteString(makeRequest(1, "tools/call", map[string]interface{}{
		"name": "memory_store",
		"arguments": map[string]interface{}{
			"value": "test-value",
		},
	}))

	line, _ := server.reader.ReadBytes('\n')
	server.handleMessage(line)

	var resp Response
	if err := json.Unmarshal(output.Bytes(), &resp); err != nil {
		t.Fatalf("Failed to parse response: %v", err)
	}

	if resp.Error != nil {
		t.Fatalf("Unexpected JSON-RPC error: %v", resp.Error)
	}

	result := resp.Result.(map[string]interface{})
	if result["isError"] != true {
		t.Error("Expected isError to be true")
	}
}
