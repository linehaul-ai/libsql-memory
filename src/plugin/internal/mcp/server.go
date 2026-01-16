// Package mcp implements the Model Context Protocol (MCP) server for the Claude memory plugin.
// It provides JSON-RPC 2.0 over stdio transport for memory operations.
package mcp

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"os/signal"
	"sync"
	"syscall"
)

// JSON-RPC 2.0 constants
const (
	JSONRPCVersion = "2.0"
	MCPVersion     = "2024-11-05"
)

// JSON-RPC 2.0 error codes
const (
	ParseError     = -32700
	InvalidRequest = -32600
	MethodNotFound = -32601
	InvalidParams  = -32602
	InternalError  = -32603
)

// Request represents a JSON-RPC 2.0 request
type Request struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      interface{}     `json:"id,omitempty"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
}

// Response represents a JSON-RPC 2.0 response
type Response struct {
	JSONRPC string      `json:"jsonrpc"`
	ID      interface{} `json:"id,omitempty"`
	Result  interface{} `json:"result,omitempty"`
	Error   *Error      `json:"error,omitempty"`
}

// Error represents a JSON-RPC 2.0 error
type Error struct {
	Code    int         `json:"code"`
	Message string      `json:"message"`
	Data    interface{} `json:"data,omitempty"`
}

// Notification represents a JSON-RPC 2.0 notification (no ID)
type Notification struct {
	JSONRPC string          `json:"jsonrpc"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
}

// MCP Protocol Types

// ServerCapabilities defines the server's capabilities
type ServerCapabilities struct {
	Tools *ToolsCapability `json:"tools,omitempty"`
}

// ToolsCapability defines tools-related capabilities
type ToolsCapability struct {
	ListChanged bool `json:"listChanged,omitempty"`
}

// InitializeParams represents the initialize request parameters
type InitializeParams struct {
	ProtocolVersion string             `json:"protocolVersion"`
	Capabilities    ClientCapabilities `json:"capabilities"`
	ClientInfo      ClientInfo         `json:"clientInfo"`
}

// ClientCapabilities represents client capabilities
type ClientCapabilities struct {
	Roots    *RootsCapability    `json:"roots,omitempty"`
	Sampling *SamplingCapability `json:"sampling,omitempty"`
}

// RootsCapability represents roots-related capabilities
type RootsCapability struct {
	ListChanged bool `json:"listChanged,omitempty"`
}

// SamplingCapability represents sampling capabilities
type SamplingCapability struct{}

// ClientInfo represents information about the client
type ClientInfo struct {
	Name    string `json:"name"`
	Version string `json:"version,omitempty"`
}

// InitializeResult represents the initialize response
type InitializeResult struct {
	ProtocolVersion string             `json:"protocolVersion"`
	Capabilities    ServerCapabilities `json:"capabilities"`
	ServerInfo      ServerInfo         `json:"serverInfo"`
}

// ServerInfo represents information about the server
type ServerInfo struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

// Tool represents an MCP tool definition
type Tool struct {
	Name        string      `json:"name"`
	Description string      `json:"description"`
	InputSchema InputSchema `json:"inputSchema"`
}

// InputSchema represents the JSON schema for tool inputs
type InputSchema struct {
	Type       string              `json:"type"`
	Properties map[string]Property `json:"properties,omitempty"`
	Required   []string            `json:"required,omitempty"`
}

// Property represents a property in a JSON schema
type Property struct {
	Type        string   `json:"type"`
	Description string   `json:"description,omitempty"`
	Default     any      `json:"default,omitempty"`
	Enum        []string `json:"enum,omitempty"`
	Minimum     *float64 `json:"minimum,omitempty"`
	Maximum     *float64 `json:"maximum,omitempty"`
}

// ToolsListResult represents the tools/list response
type ToolsListResult struct {
	Tools []Tool `json:"tools"`
}

// ToolCallParams represents the tools/call request parameters
type ToolCallParams struct {
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments,omitempty"`
}

// ToolCallResult represents the tools/call response
type ToolCallResult struct {
	Content []ContentBlock `json:"content"`
	IsError bool           `json:"isError,omitempty"`
}

// ContentBlock represents content in a tool result
type ContentBlock struct {
	Type string `json:"type"`
	Text string `json:"text,omitempty"`
}

// Memory Tool Argument Types

// MemoryStoreArgs represents arguments for memory_store
type MemoryStoreArgs struct {
	Key       string            `json:"key"`
	Value     string            `json:"value"`
	Namespace string            `json:"namespace,omitempty"`
	Metadata  map[string]string `json:"metadata,omitempty"`
	Tags      []string          `json:"tags,omitempty"`
}

// MemoryRetrieveArgs represents arguments for memory_retrieve
type MemoryRetrieveArgs struct {
	Key       string `json:"key"`
	Namespace string `json:"namespace,omitempty"`
}

// MemorySearchArgs represents arguments for memory_search
type MemorySearchArgs struct {
	Query     string  `json:"query"`
	Namespace string  `json:"namespace,omitempty"`
	Limit     int     `json:"limit,omitempty"`
	Threshold float64 `json:"threshold,omitempty"`
}

// MemoryListArgs represents arguments for memory_list
type MemoryListArgs struct {
	Namespace string `json:"namespace,omitempty"`
	Limit     int    `json:"limit,omitempty"`
	Offset    int    `json:"offset,omitempty"`
}

// MemoryDeleteArgs represents arguments for memory_delete
type MemoryDeleteArgs struct {
	Key       string `json:"key"`
	Namespace string `json:"namespace,omitempty"`
}

// MemoryStatsArgs represents arguments for memory_stats
type MemoryStatsArgs struct {
	Namespace string `json:"namespace,omitempty"`
}

// MemoryService defines the interface for memory operations
type MemoryService interface {
	Store(ctx context.Context, args MemoryStoreArgs) error
	Retrieve(ctx context.Context, args MemoryRetrieveArgs) (string, map[string]string, []string, error)
	Search(ctx context.Context, args MemorySearchArgs) ([]SearchResult, error)
	List(ctx context.Context, args MemoryListArgs) ([]MemoryEntry, int, error)
	Delete(ctx context.Context, args MemoryDeleteArgs) error
	Stats(ctx context.Context, args MemoryStatsArgs) (*MemoryStats, error)
}

// SearchResult represents a search result
type SearchResult struct {
	Key       string            `json:"key"`
	Value     string            `json:"value"`
	Score     float64           `json:"score"`
	Namespace string            `json:"namespace"`
	Metadata  map[string]string `json:"metadata,omitempty"`
	Tags      []string          `json:"tags,omitempty"`
}

// MemoryEntry represents a memory entry
type MemoryEntry struct {
	Key       string            `json:"key"`
	Value     string            `json:"value"`
	Namespace string            `json:"namespace"`
	Metadata  map[string]string `json:"metadata,omitempty"`
	Tags      []string          `json:"tags,omitempty"`
	CreatedAt string            `json:"created_at"`
	UpdatedAt string            `json:"updated_at"`
}

// MemoryStats represents memory statistics
type MemoryStats struct {
	TotalEntries    int            `json:"total_entries"`
	TotalNamespaces int            `json:"total_namespaces"`
	EntriesByNS     map[string]int `json:"entries_by_namespace"`
	StorageBytes    int64          `json:"storage_bytes"`
}

// Server represents the MCP server
type Server struct {
	memory    MemoryService
	logger    *log.Logger
	reader    *bufio.Reader
	writer    io.Writer
	writeMu   sync.Mutex
	ctx       context.Context
	cancel    context.CancelFunc
	wg        sync.WaitGroup
	tools     []Tool
	isRunning bool
	runMu     sync.Mutex
}

// NewServer creates a new MCP server instance
func NewServer(memory MemoryService) *Server {
	ctx, cancel := context.WithCancel(context.Background())

	s := &Server{
		memory: memory,
		logger: log.New(os.Stderr, "[mcp] ", log.LstdFlags|log.Lmicroseconds),
		reader: bufio.NewReader(os.Stdin),
		writer: os.Stdout,
		ctx:    ctx,
		cancel: cancel,
	}

	s.initTools()
	return s
}

// NewServerWithIO creates a new MCP server with custom IO (useful for testing)
func NewServerWithIO(memory MemoryService, reader io.Reader, writer io.Writer) *Server {
	ctx, cancel := context.WithCancel(context.Background())

	s := &Server{
		memory: memory,
		logger: log.New(os.Stderr, "[mcp] ", log.LstdFlags|log.Lmicroseconds),
		reader: bufio.NewReader(reader),
		writer: writer,
		ctx:    ctx,
		cancel: cancel,
	}

	s.initTools()
	return s
}

// initTools initializes the available memory tools
func (s *Server) initTools() {
	minZero := 0.0
	maxOne := 1.0

	s.tools = []Tool{
		{
			Name:        "memory_store",
			Description: "Store a memory with a key, value, optional namespace, metadata, and tags. Use this to persist information for later retrieval.",
			InputSchema: InputSchema{
				Type: "object",
				Properties: map[string]Property{
					"key": {
						Type:        "string",
						Description: "Unique identifier for the memory within the namespace",
					},
					"value": {
						Type:        "string",
						Description: "The content to store",
					},
					"namespace": {
						Type:        "string",
						Description: "Namespace to organize memories (default: 'default')",
						Default:     "default",
					},
					"metadata": {
						Type:        "object",
						Description: "Additional key-value metadata to store with the memory",
					},
					"tags": {
						Type:        "array",
						Description: "Tags for categorizing and filtering memories",
					},
				},
				Required: []string{"key", "value"},
			},
		},
		{
			Name:        "memory_retrieve",
			Description: "Retrieve a specific memory by its key and namespace. Returns the stored value along with metadata and tags.",
			InputSchema: InputSchema{
				Type: "object",
				Properties: map[string]Property{
					"key": {
						Type:        "string",
						Description: "The key of the memory to retrieve",
					},
					"namespace": {
						Type:        "string",
						Description: "Namespace where the memory is stored (default: 'default')",
						Default:     "default",
					},
				},
				Required: []string{"key"},
			},
		},
		{
			Name:        "memory_search",
			Description: "Perform semantic search across memories using natural language. Returns relevant memories ranked by similarity score.",
			InputSchema: InputSchema{
				Type: "object",
				Properties: map[string]Property{
					"query": {
						Type:        "string",
						Description: "Natural language search query",
					},
					"namespace": {
						Type:        "string",
						Description: "Namespace to search in (searches all namespaces if not specified)",
					},
					"limit": {
						Type:        "integer",
						Description: "Maximum number of results to return (default: 10)",
						Default:     10,
					},
					"threshold": {
						Type:        "number",
						Description: "Minimum similarity score threshold (0.0 to 1.0, default: 0.0)",
						Default:     0.0,
						Minimum:     &minZero,
						Maximum:     &maxOne,
					},
				},
				Required: []string{"query"},
			},
		},
		{
			Name:        "memory_list",
			Description: "List all memories in a namespace with pagination support.",
			InputSchema: InputSchema{
				Type: "object",
				Properties: map[string]Property{
					"namespace": {
						Type:        "string",
						Description: "Namespace to list memories from (default: 'default')",
						Default:     "default",
					},
					"limit": {
						Type:        "integer",
						Description: "Maximum number of entries to return (default: 50)",
						Default:     50,
					},
					"offset": {
						Type:        "integer",
						Description: "Number of entries to skip for pagination (default: 0)",
						Default:     0,
					},
				},
				Required: []string{},
			},
		},
		{
			Name:        "memory_delete",
			Description: "Delete a memory by its key and namespace.",
			InputSchema: InputSchema{
				Type: "object",
				Properties: map[string]Property{
					"key": {
						Type:        "string",
						Description: "The key of the memory to delete",
					},
					"namespace": {
						Type:        "string",
						Description: "Namespace where the memory is stored (default: 'default')",
						Default:     "default",
					},
				},
				Required: []string{"key"},
			},
		},
		{
			Name:        "memory_stats",
			Description: "Get statistics about stored memories including counts by namespace and storage usage.",
			InputSchema: InputSchema{
				Type: "object",
				Properties: map[string]Property{
					"namespace": {
						Type:        "string",
						Description: "Filter stats to a specific namespace (shows all if not specified)",
					},
				},
				Required: []string{},
			},
		},
	}
}

// Run starts the MCP server and handles incoming requests
func (s *Server) Run() error {
	s.runMu.Lock()
	if s.isRunning {
		s.runMu.Unlock()
		return fmt.Errorf("server is already running")
	}
	s.isRunning = true
	s.runMu.Unlock()

	defer func() {
		s.runMu.Lock()
		s.isRunning = false
		s.runMu.Unlock()
	}()

	// Setup signal handling for graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		select {
		case sig := <-sigChan:
			s.logger.Printf("Received signal %v, initiating graceful shutdown", sig)
			s.Shutdown()
		case <-s.ctx.Done():
		}
	}()

	s.logger.Println("MCP server starting, listening on stdin")

	// Main request processing loop
	for {
		select {
		case <-s.ctx.Done():
			s.logger.Println("Server context cancelled, stopping request loop")
			s.wg.Wait()
			return nil
		default:
		}

		line, err := s.reader.ReadBytes('\n')
		if err != nil {
			if err == io.EOF {
				s.logger.Println("EOF received, shutting down")
				return nil
			}
			if s.ctx.Err() != nil {
				return nil
			}
			s.logger.Printf("Error reading input: %v", err)
			continue
		}

		if len(line) == 0 || string(line) == "\n" {
			continue
		}

		s.wg.Add(1)
		go func(data []byte) {
			defer s.wg.Done()
			s.handleMessage(data)
		}(line)
	}
}

// Shutdown gracefully shuts down the server
func (s *Server) Shutdown() {
	s.logger.Println("Shutting down MCP server")
	s.cancel()
}

// handleMessage processes a single JSON-RPC message
func (s *Server) handleMessage(data []byte) {
	var req Request
	if err := json.Unmarshal(data, &req); err != nil {
		s.logger.Printf("Failed to parse JSON-RPC request: %v", err)
		s.sendError(nil, ParseError, "Parse error", err.Error())
		return
	}

	if req.JSONRPC != JSONRPCVersion {
		s.sendError(req.ID, InvalidRequest, "Invalid Request", "jsonrpc must be '2.0'")
		return
	}

	// Handle notification (no response expected)
	if req.ID == nil {
		s.handleNotification(req.Method, req.Params)
		return
	}

	// Handle request (response expected)
	result, rpcErr := s.handleRequest(req.Method, req.Params)
	if rpcErr != nil {
		s.sendError(req.ID, rpcErr.Code, rpcErr.Message, rpcErr.Data)
		return
	}

	s.sendResult(req.ID, result)
}

// handleNotification processes notifications (no response)
func (s *Server) handleNotification(method string, params json.RawMessage) {
	switch method {
	case "notifications/initialized":
		s.logger.Println("Client initialized notification received")
	case "notifications/cancelled":
		s.logger.Println("Request cancelled notification received")
	default:
		s.logger.Printf("Unknown notification: %s", method)
	}
}

// handleRequest processes a request and returns result or error
func (s *Server) handleRequest(method string, params json.RawMessage) (interface{}, *Error) {
	s.logger.Printf("Handling request: %s", method)

	switch method {
	case "initialize":
		return s.handleInitialize(params)
	case "tools/list":
		return s.handleToolsList()
	case "tools/call":
		return s.handleToolsCall(params)
	case "ping":
		return map[string]interface{}{}, nil
	default:
		return nil, &Error{
			Code:    MethodNotFound,
			Message: "Method not found",
			Data:    method,
		}
	}
}

// handleInitialize processes the initialize request
func (s *Server) handleInitialize(params json.RawMessage) (*InitializeResult, *Error) {
	var initParams InitializeParams
	if len(params) > 0 {
		if err := json.Unmarshal(params, &initParams); err != nil {
			return nil, &Error{
				Code:    InvalidParams,
				Message: "Invalid params",
				Data:    err.Error(),
			}
		}
	}

	s.logger.Printf("Initialize request from client: %s %s", initParams.ClientInfo.Name, initParams.ClientInfo.Version)

	return &InitializeResult{
		ProtocolVersion: MCPVersion,
		Capabilities: ServerCapabilities{
			Tools: &ToolsCapability{
				ListChanged: false,
			},
		},
		ServerInfo: ServerInfo{
			Name:    "libsql-memory",
			Version: "1.0.0",
		},
	}, nil
}

// handleToolsList returns the list of available tools
func (s *Server) handleToolsList() (*ToolsListResult, *Error) {
	return &ToolsListResult{
		Tools: s.tools,
	}, nil
}

// handleToolsCall executes a tool call
func (s *Server) handleToolsCall(params json.RawMessage) (*ToolCallResult, *Error) {
	var callParams ToolCallParams
	if err := json.Unmarshal(params, &callParams); err != nil {
		return nil, &Error{
			Code:    InvalidParams,
			Message: "Invalid params",
			Data:    err.Error(),
		}
	}

	s.logger.Printf("Tool call: %s", callParams.Name)

	ctx, cancel := context.WithCancel(s.ctx)
	defer cancel()

	var result interface{}
	var toolErr error

	switch callParams.Name {
	case "memory_store":
		result, toolErr = s.executeMemoryStore(ctx, callParams.Arguments)
	case "memory_retrieve":
		result, toolErr = s.executeMemoryRetrieve(ctx, callParams.Arguments)
	case "memory_search":
		result, toolErr = s.executeMemorySearch(ctx, callParams.Arguments)
	case "memory_list":
		result, toolErr = s.executeMemoryList(ctx, callParams.Arguments)
	case "memory_delete":
		result, toolErr = s.executeMemoryDelete(ctx, callParams.Arguments)
	case "memory_stats":
		result, toolErr = s.executeMemoryStats(ctx, callParams.Arguments)
	default:
		return nil, &Error{
			Code:    InvalidParams,
			Message: "Unknown tool",
			Data:    callParams.Name,
		}
	}

	if toolErr != nil {
		s.logger.Printf("Tool error: %v", toolErr)
		return &ToolCallResult{
			Content: []ContentBlock{{
				Type: "text",
				Text: fmt.Sprintf("Error: %v", toolErr),
			}},
			IsError: true,
		}, nil
	}

	resultJSON, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		return &ToolCallResult{
			Content: []ContentBlock{{
				Type: "text",
				Text: fmt.Sprintf("Error serializing result: %v", err),
			}},
			IsError: true,
		}, nil
	}

	return &ToolCallResult{
		Content: []ContentBlock{{
			Type: "text",
			Text: string(resultJSON),
		}},
		IsError: false,
	}, nil
}

// executeMemoryStore executes the memory_store tool
func (s *Server) executeMemoryStore(ctx context.Context, args json.RawMessage) (interface{}, error) {
	var storeArgs MemoryStoreArgs
	if err := json.Unmarshal(args, &storeArgs); err != nil {
		return nil, fmt.Errorf("invalid arguments: %w", err)
	}

	if storeArgs.Key == "" {
		return nil, fmt.Errorf("key is required")
	}
	if storeArgs.Value == "" {
		return nil, fmt.Errorf("value is required")
	}

	// Set default namespace
	if storeArgs.Namespace == "" {
		storeArgs.Namespace = "default"
	}

	if s.memory == nil {
		return nil, fmt.Errorf("memory service not available")
	}

	if err := s.memory.Store(ctx, storeArgs); err != nil {
		return nil, fmt.Errorf("failed to store memory: %w", err)
	}

	return map[string]interface{}{
		"success":   true,
		"key":       storeArgs.Key,
		"namespace": storeArgs.Namespace,
		"message":   "Memory stored successfully",
	}, nil
}

// executeMemoryRetrieve executes the memory_retrieve tool
func (s *Server) executeMemoryRetrieve(ctx context.Context, args json.RawMessage) (interface{}, error) {
	var retrieveArgs MemoryRetrieveArgs
	if err := json.Unmarshal(args, &retrieveArgs); err != nil {
		return nil, fmt.Errorf("invalid arguments: %w", err)
	}

	if retrieveArgs.Key == "" {
		return nil, fmt.Errorf("key is required")
	}

	// Set default namespace
	if retrieveArgs.Namespace == "" {
		retrieveArgs.Namespace = "default"
	}

	if s.memory == nil {
		return nil, fmt.Errorf("memory service not available")
	}

	value, metadata, tags, err := s.memory.Retrieve(ctx, retrieveArgs)
	if err != nil {
		return nil, fmt.Errorf("failed to retrieve memory: %w", err)
	}

	return map[string]interface{}{
		"key":       retrieveArgs.Key,
		"namespace": retrieveArgs.Namespace,
		"value":     value,
		"metadata":  metadata,
		"tags":      tags,
	}, nil
}

// executeMemorySearch executes the memory_search tool
func (s *Server) executeMemorySearch(ctx context.Context, args json.RawMessage) (interface{}, error) {
	var searchArgs MemorySearchArgs
	if err := json.Unmarshal(args, &searchArgs); err != nil {
		return nil, fmt.Errorf("invalid arguments: %w", err)
	}

	if searchArgs.Query == "" {
		return nil, fmt.Errorf("query is required")
	}

	// Set defaults
	if searchArgs.Limit <= 0 {
		searchArgs.Limit = 10
	}

	if s.memory == nil {
		return nil, fmt.Errorf("memory service not available")
	}

	results, err := s.memory.Search(ctx, searchArgs)
	if err != nil {
		return nil, fmt.Errorf("failed to search memories: %w", err)
	}

	return map[string]interface{}{
		"query":   searchArgs.Query,
		"results": results,
		"count":   len(results),
	}, nil
}

// executeMemoryList executes the memory_list tool
func (s *Server) executeMemoryList(ctx context.Context, args json.RawMessage) (interface{}, error) {
	var listArgs MemoryListArgs
	if len(args) > 0 {
		if err := json.Unmarshal(args, &listArgs); err != nil {
			return nil, fmt.Errorf("invalid arguments: %w", err)
		}
	}

	// Set defaults
	if listArgs.Namespace == "" {
		listArgs.Namespace = "default"
	}
	if listArgs.Limit <= 0 {
		listArgs.Limit = 50
	}

	if s.memory == nil {
		return nil, fmt.Errorf("memory service not available")
	}

	entries, total, err := s.memory.List(ctx, listArgs)
	if err != nil {
		return nil, fmt.Errorf("failed to list memories: %w", err)
	}

	return map[string]interface{}{
		"namespace": listArgs.Namespace,
		"entries":   entries,
		"count":     len(entries),
		"total":     total,
		"offset":    listArgs.Offset,
		"limit":     listArgs.Limit,
	}, nil
}

// executeMemoryDelete executes the memory_delete tool
func (s *Server) executeMemoryDelete(ctx context.Context, args json.RawMessage) (interface{}, error) {
	var deleteArgs MemoryDeleteArgs
	if err := json.Unmarshal(args, &deleteArgs); err != nil {
		return nil, fmt.Errorf("invalid arguments: %w", err)
	}

	if deleteArgs.Key == "" {
		return nil, fmt.Errorf("key is required")
	}

	// Set default namespace
	if deleteArgs.Namespace == "" {
		deleteArgs.Namespace = "default"
	}

	if s.memory == nil {
		return nil, fmt.Errorf("memory service not available")
	}

	if err := s.memory.Delete(ctx, deleteArgs); err != nil {
		return nil, fmt.Errorf("failed to delete memory: %w", err)
	}

	return map[string]interface{}{
		"success":   true,
		"key":       deleteArgs.Key,
		"namespace": deleteArgs.Namespace,
		"message":   "Memory deleted successfully",
	}, nil
}

// executeMemoryStats executes the memory_stats tool
func (s *Server) executeMemoryStats(ctx context.Context, args json.RawMessage) (interface{}, error) {
	var statsArgs MemoryStatsArgs
	if len(args) > 0 {
		if err := json.Unmarshal(args, &statsArgs); err != nil {
			return nil, fmt.Errorf("invalid arguments: %w", err)
		}
	}

	if s.memory == nil {
		return nil, fmt.Errorf("memory service not available")
	}

	stats, err := s.memory.Stats(ctx, statsArgs)
	if err != nil {
		return nil, fmt.Errorf("failed to get memory stats: %w", err)
	}

	return stats, nil
}

// sendResult sends a successful response
func (s *Server) sendResult(id interface{}, result interface{}) {
	response := Response{
		JSONRPC: JSONRPCVersion,
		ID:      id,
		Result:  result,
	}
	s.sendResponse(response)
}

// sendError sends an error response
func (s *Server) sendError(id interface{}, code int, message string, data interface{}) {
	response := Response{
		JSONRPC: JSONRPCVersion,
		ID:      id,
		Error: &Error{
			Code:    code,
			Message: message,
			Data:    data,
		},
	}
	s.sendResponse(response)
}

// sendResponse writes a JSON-RPC response to stdout
func (s *Server) sendResponse(response Response) {
	s.writeMu.Lock()
	defer s.writeMu.Unlock()

	data, err := json.Marshal(response)
	if err != nil {
		s.logger.Printf("Failed to marshal response: %v", err)
		return
	}

	// Write response with newline delimiter
	if _, err := fmt.Fprintf(s.writer, "%s\n", data); err != nil {
		s.logger.Printf("Failed to write response: %v", err)
	}
}

// sendNotification sends a JSON-RPC notification to the client
func (s *Server) sendNotification(method string, params interface{}) {
	s.writeMu.Lock()
	defer s.writeMu.Unlock()

	notification := struct {
		JSONRPC string      `json:"jsonrpc"`
		Method  string      `json:"method"`
		Params  interface{} `json:"params,omitempty"`
	}{
		JSONRPC: JSONRPCVersion,
		Method:  method,
		Params:  params,
	}

	data, err := json.Marshal(notification)
	if err != nil {
		s.logger.Printf("Failed to marshal notification: %v", err)
		return
	}

	if _, err := fmt.Fprintf(s.writer, "%s\n", data); err != nil {
		s.logger.Printf("Failed to write notification: %v", err)
	}
}
