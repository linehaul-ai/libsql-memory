// Package types provides shared type definitions for the Claude memory plugin.
// This file contains additional API types for requests, responses, and errors.
package types

import (
	"encoding/json"
	"time"
)

// Memory represents a stored memory entry
type Memory struct {
	Key       string            `json:"key"`
	Value     string            `json:"value"`
	Namespace string            `json:"namespace"`
	Metadata  map[string]string `json:"metadata,omitempty"`
	Tags      []string          `json:"tags,omitempty"`
	Embedding []float32         `json:"-"`
	CreatedAt time.Time         `json:"created_at"`
	UpdatedAt time.Time         `json:"updated_at"`
}

// SearchResult represents a memory search result with similarity score
type SearchResult struct {
	Memory Memory  `json:"memory"`
	Score  float64 `json:"score"`
}

// Stats represents memory storage statistics
type Stats struct {
	TotalEntries    int            `json:"total_entries"`
	TotalNamespaces int            `json:"total_namespaces"`
	EntriesByNS     map[string]int `json:"entries_by_namespace"`
	StorageBytes    int64          `json:"storage_bytes"`
}

// MemoryMetadata contains additional information about a memory entry.
type MemoryMetadata struct {
	Source     string            `json:"source,omitempty"`
	Category   string            `json:"category,omitempty"`
	Importance float32           `json:"importance,omitempty"`
	SessionID  string            `json:"session_id,omitempty"`
	Extra      map[string]string `json:"extra,omitempty"`
}

// StoreRequest represents a request to store a new memory.
type StoreRequest struct {
	Key       string            `json:"key"`
	Value     string            `json:"value"`
	Namespace string            `json:"namespace,omitempty"`
	Metadata  map[string]string `json:"metadata,omitempty"`
	Tags      []string          `json:"tags,omitempty"`
}

// StoreResponse represents the response after storing a memory.
type StoreResponse struct {
	Key       string    `json:"key"`
	Namespace string    `json:"namespace"`
	Success   bool      `json:"success"`
	CreatedAt time.Time `json:"created_at"`
	Error     string    `json:"error,omitempty"`
}

// SearchRequest represents a request to search memories.
type SearchRequest struct {
	Query     string   `json:"query"`
	Namespace string   `json:"namespace,omitempty"`
	Limit     int      `json:"limit,omitempty"`
	Threshold float64  `json:"threshold,omitempty"`
	Tags      []string `json:"tags,omitempty"`
}

// SearchResponse represents the response from a memory search.
type SearchResponse struct {
	Results []SearchResult `json:"results"`
	Total   int            `json:"total"`
	Query   string         `json:"query"`
	Error   string         `json:"error,omitempty"`
}

// RetrieveRequest represents a request to retrieve a memory.
type RetrieveRequest struct {
	Key       string `json:"key"`
	Namespace string `json:"namespace,omitempty"`
}

// RetrieveResponse represents the response from retrieving a memory.
type RetrieveResponse struct {
	Memory *Memory `json:"memory,omitempty"`
	Found  bool    `json:"found"`
	Error  string  `json:"error,omitempty"`
}

// DeleteRequest represents a request to delete a memory.
type DeleteRequest struct {
	Key       string `json:"key"`
	Namespace string `json:"namespace,omitempty"`
}

// DeleteResponse represents the response after deleting a memory.
type DeleteResponse struct {
	Key       string `json:"key"`
	Namespace string `json:"namespace"`
	Success   bool   `json:"success"`
	Error     string `json:"error,omitempty"`
}

// ListRequest represents a request to list memories.
type ListRequest struct {
	Namespace string   `json:"namespace,omitempty"`
	Limit     int      `json:"limit,omitempty"`
	Offset    int      `json:"offset,omitempty"`
	Tags      []string `json:"tags,omitempty"`
}

// ListResponse represents the response from listing memories.
type ListResponse struct {
	Memories []Memory `json:"memories"`
	Total    int      `json:"total"`
	Offset   int      `json:"offset"`
	Limit    int      `json:"limit"`
	Error    string   `json:"error,omitempty"`
}

// StatsRequest represents a request to get memory statistics.
type StatsRequest struct {
	Namespace string `json:"namespace,omitempty"`
}

// StatsResponse represents the response with memory statistics.
type StatsResponse struct {
	Stats *Stats `json:"stats,omitempty"`
	Error string `json:"error,omitempty"`
}

// ErrorCode represents specific error types for the memory plugin.
type ErrorCode string

const (
	ErrCodeNotFound        ErrorCode = "NOT_FOUND"
	ErrCodeInvalidInput    ErrorCode = "INVALID_INPUT"
	ErrCodeDatabaseError   ErrorCode = "DATABASE_ERROR"
	ErrCodeEmbeddingError  ErrorCode = "EMBEDDING_ERROR"
	ErrCodeConnectionError ErrorCode = "CONNECTION_ERROR"
	ErrCodeInternal        ErrorCode = "INTERNAL_ERROR"
)

// MemoryError represents a structured error from the memory plugin.
type MemoryError struct {
	Code    ErrorCode `json:"code"`
	Message string    `json:"message"`
	Details string    `json:"details,omitempty"`
}

// Error implements the error interface.
func (e *MemoryError) Error() string {
	if e.Details != "" {
		return e.Message + ": " + e.Details
	}
	return e.Message
}

// NewMemoryError creates a new MemoryError with the given code and message.
func NewMemoryError(code ErrorCode, message string) *MemoryError {
	return &MemoryError{
		Code:    code,
		Message: message,
	}
}

// NewMemoryErrorWithDetails creates a new MemoryError with details.
func NewMemoryErrorWithDetails(code ErrorCode, message, details string) *MemoryError {
	return &MemoryError{
		Code:    code,
		Message: message,
		Details: details,
	}
}

// IsNotFound returns true if the error is a not found error.
func IsNotFound(err error) bool {
	if memErr, ok := err.(*MemoryError); ok {
		return memErr.Code == ErrCodeNotFound
	}
	return false
}

// MarshalJSON implements custom JSON marshaling for StoreResponse.
func (r *StoreResponse) MarshalJSON() ([]byte, error) {
	type Alias StoreResponse
	return json.Marshal(&struct {
		*Alias
		CreatedAt string `json:"created_at"`
	}{
		Alias:     (*Alias)(r),
		CreatedAt: r.CreatedAt.Format(time.RFC3339),
	})
}

// Validate validates the StoreRequest fields.
func (r *StoreRequest) Validate() error {
	if r.Key == "" {
		return NewMemoryError(ErrCodeInvalidInput, "key is required")
	}
	if r.Value == "" {
		return NewMemoryError(ErrCodeInvalidInput, "value is required")
	}
	return nil
}

// Validate validates the SearchRequest fields.
func (r *SearchRequest) Validate() error {
	if r.Query == "" {
		return NewMemoryError(ErrCodeInvalidInput, "query is required")
	}
	if r.Limit < 0 {
		return NewMemoryError(ErrCodeInvalidInput, "limit must be non-negative")
	}
	if r.Threshold < 0 || r.Threshold > 1 {
		return NewMemoryError(ErrCodeInvalidInput, "threshold must be between 0 and 1")
	}
	return nil
}

// SetDefaults sets default values for SearchRequest.
func (r *SearchRequest) SetDefaults() {
	if r.Limit == 0 {
		r.Limit = 10
	}
	if r.Threshold == 0 {
		r.Threshold = 0.7
	}
	if r.Namespace == "" {
		r.Namespace = "default"
	}
}

// SetDefaults sets default values for ListRequest.
func (r *ListRequest) SetDefaults() {
	if r.Limit == 0 {
		r.Limit = 50
	}
	if r.Namespace == "" {
		r.Namespace = "default"
	}
}

// Validate validates the RetrieveRequest fields.
func (r *RetrieveRequest) Validate() error {
	if r.Key == "" {
		return NewMemoryError(ErrCodeInvalidInput, "key is required")
	}
	return nil
}

// SetDefaults sets default values for RetrieveRequest.
func (r *RetrieveRequest) SetDefaults() {
	if r.Namespace == "" {
		r.Namespace = "default"
	}
}

// Validate validates the DeleteRequest fields.
func (r *DeleteRequest) Validate() error {
	if r.Key == "" {
		return NewMemoryError(ErrCodeInvalidInput, "key is required")
	}
	return nil
}

// SetDefaults sets default values for DeleteRequest.
func (r *DeleteRequest) SetDefaults() {
	if r.Namespace == "" {
		r.Namespace = "default"
	}
}

// SetDefaults sets default values for StoreRequest.
func (r *StoreRequest) SetDefaults() {
	if r.Namespace == "" {
		r.Namespace = "default"
	}
}
