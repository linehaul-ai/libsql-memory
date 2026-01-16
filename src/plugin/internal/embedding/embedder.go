// Package embedding provides vector embedding generation and similarity search
// for the Claude memory plugin. It supports multiple embedding backends including
// OpenAI's text-embedding-3-small and a local TF-IDF/hash-based fallback.
package embedding

import (
	"bytes"
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"
	"unicode"

	libsqlvector "github.com/ryanskidmore/libsql-vector-go"
)

// Common errors returned by embedding operations.
var (
	ErrEmptyText          = errors.New("embedding: text cannot be empty")
	ErrDimensionMismatch  = errors.New("embedding: vector dimension mismatch")
	ErrRateLimited        = errors.New("embedding: rate limit exceeded")
	ErrAPIKeyMissing      = errors.New("embedding: OpenAI API key is required")
	ErrAPIRequestFailed   = errors.New("embedding: API request failed")
	ErrInvalidResponse    = errors.New("embedding: invalid API response")
	ErrIndexNotFound      = errors.New("embedding: vector index not found")
	ErrDatabaseError      = errors.New("embedding: database operation failed")
)

// Default configuration values.
const (
	DefaultOpenAIDimension = 1536
	DefaultLocalDimension  = 384
	DefaultOpenAIModel     = "text-embedding-3-small"
	DefaultOpenAIEndpoint  = "https://api.openai.com/v1/embeddings"
	DefaultRequestTimeout  = 30 * time.Second
	DefaultMaxRetries      = 3
	DefaultRetryBaseDelay  = 1 * time.Second
	DefaultRateLimit       = 3000 // requests per minute for OpenAI
)

// Vector represents a float32 embedding vector.
type Vector []float32

// Dimension returns the dimensionality of the vector.
func (v Vector) Dimension() int {
	return len(v)
}

// Normalize returns an L2-normalized copy of the vector.
func (v Vector) Normalize() Vector {
	if len(v) == 0 {
		return v
	}

	var sumSquares float64
	for _, val := range v {
		sumSquares += float64(val) * float64(val)
	}

	magnitude := math.Sqrt(sumSquares)
	if magnitude == 0 {
		return v
	}

	normalized := make(Vector, len(v))
	for i, val := range v {
		normalized[i] = float32(float64(val) / magnitude)
	}
	return normalized
}

// CosineSimilarity computes the cosine similarity between two vectors.
// Both vectors should be normalized for accurate results.
func (v Vector) CosineSimilarity(other Vector) (float64, error) {
	if len(v) != len(other) {
		return 0, fmt.Errorf("%w: got %d and %d", ErrDimensionMismatch, len(v), len(other))
	}

	var dotProduct float64
	for i := range v {
		dotProduct += float64(v[i]) * float64(other[i])
	}

	return dotProduct, nil
}

// ToBytes serializes the vector to binary format for database storage.
func (v Vector) ToBytes() ([]byte, error) {
	vec := libsqlvector.NewVector([]float32(v))
	return vec.EncodeBinary(nil)
}

// VectorFromBytes deserializes a vector from binary bytes.
func VectorFromBytes(data []byte) (Vector, error) {
	var vec libsqlvector.Vector
	if err := vec.DecodeBinary(data); err != nil {
		return nil, fmt.Errorf("failed to deserialize vector: %w", err)
	}
	return Vector(vec.Slice()), nil
}

// EmbedderConfig holds configuration for embedding providers.
type EmbedderConfig struct {
	// Dimension specifies the output vector dimension.
	Dimension int

	// For OpenAI embedder
	APIKey         string
	Model          string
	Endpoint       string
	RequestTimeout time.Duration
	MaxRetries     int
	RetryBaseDelay time.Duration

	// Rate limiting
	RateLimitRPM int // Requests per minute
}

// DefaultOpenAIConfig returns the default configuration for OpenAI embedder.
func DefaultOpenAIConfig(apiKey string) EmbedderConfig {
	return EmbedderConfig{
		Dimension:      DefaultOpenAIDimension,
		APIKey:         apiKey,
		Model:          DefaultOpenAIModel,
		Endpoint:       DefaultOpenAIEndpoint,
		RequestTimeout: DefaultRequestTimeout,
		MaxRetries:     DefaultMaxRetries,
		RetryBaseDelay: DefaultRetryBaseDelay,
		RateLimitRPM:   DefaultRateLimit,
	}
}

// DefaultLocalConfig returns the default configuration for local embedder.
func DefaultLocalConfig() EmbedderConfig {
	return EmbedderConfig{
		Dimension: DefaultLocalDimension,
	}
}

// Embedder defines the interface for generating text embeddings.
type Embedder interface {
	// Embed generates an embedding vector for the given text.
	Embed(ctx context.Context, text string) (Vector, error)

	// EmbedBatch generates embeddings for multiple texts.
	EmbedBatch(ctx context.Context, texts []string) ([]Vector, error)

	// Dimension returns the output vector dimension.
	Dimension() int

	// Close releases any resources held by the embedder.
	Close() error
}

// -----------------------------------------------------------------------------
// LocalEmbedder: TF-IDF and hash-based embedding fallback
// -----------------------------------------------------------------------------

// LocalEmbedder generates embeddings using a TF-IDF and hash-based approach.
// This provides a zero-dependency fallback when external APIs are unavailable.
type LocalEmbedder struct {
	dimension int
	idf       map[string]float64
	idfMu     sync.RWMutex
	tokenizer *Tokenizer
}

// Tokenizer handles text tokenization for the local embedder.
type Tokenizer struct {
	stopWords map[string]struct{}
	wordRegex *regexp.Regexp
}

// NewTokenizer creates a new tokenizer with common English stop words.
func NewTokenizer() *Tokenizer {
	stopWords := map[string]struct{}{
		"a": {}, "an": {}, "and": {}, "are": {}, "as": {}, "at": {}, "be": {},
		"by": {}, "for": {}, "from": {}, "has": {}, "he": {}, "in": {}, "is": {},
		"it": {}, "its": {}, "of": {}, "on": {}, "or": {}, "that": {}, "the": {},
		"to": {}, "was": {}, "were": {}, "will": {}, "with": {}, "this": {},
		"but": {}, "they": {}, "have": {}, "had": {}, "what": {}, "when": {},
		"where": {}, "who": {}, "which": {}, "why": {}, "how": {}, "all": {},
		"each": {}, "every": {}, "both": {}, "few": {}, "more": {}, "most": {},
		"other": {}, "some": {}, "such": {}, "no": {}, "nor": {}, "not": {},
		"only": {}, "own": {}, "same": {}, "so": {}, "than": {}, "too": {},
		"very": {}, "can": {}, "just": {}, "should": {}, "now": {},
	}

	return &Tokenizer{
		stopWords: stopWords,
		wordRegex: regexp.MustCompile(`[a-zA-Z]+`),
	}
}

// Tokenize splits text into normalized tokens.
func (t *Tokenizer) Tokenize(text string) []string {
	// Convert to lowercase
	text = strings.ToLower(text)

	// Extract words
	words := t.wordRegex.FindAllString(text, -1)

	// Filter stop words and short tokens
	tokens := make([]string, 0, len(words))
	for _, word := range words {
		if len(word) < 2 {
			continue
		}
		if _, isStopWord := t.stopWords[word]; isStopWord {
			continue
		}
		tokens = append(tokens, word)
	}

	return tokens
}

// NewLocalEmbedder creates a new local embedder with the specified dimension.
func NewLocalEmbedder(config EmbedderConfig) *LocalEmbedder {
	dim := config.Dimension
	if dim <= 0 {
		dim = DefaultLocalDimension
	}

	return &LocalEmbedder{
		dimension: dim,
		idf:       make(map[string]float64),
		tokenizer: NewTokenizer(),
	}
}

// Dimension returns the output vector dimension.
func (e *LocalEmbedder) Dimension() int {
	return e.dimension
}

// Close releases resources (no-op for local embedder).
func (e *LocalEmbedder) Close() error {
	return nil
}

// Embed generates an embedding using TF-IDF weighted hash vectors.
func (e *LocalEmbedder) Embed(ctx context.Context, text string) (Vector, error) {
	if strings.TrimSpace(text) == "" {
		return nil, ErrEmptyText
	}

	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	tokens := e.tokenizer.Tokenize(text)
	if len(tokens) == 0 {
		// Fall back to character-level hashing for very short or unusual text
		return e.hashEmbed(text), nil
	}

	// Compute term frequencies
	tf := make(map[string]int)
	for _, token := range tokens {
		tf[token]++
	}

	// Generate weighted embedding
	embedding := make(Vector, e.dimension)

	for token, count := range tf {
		// Get or compute IDF weight
		idfWeight := e.getIDF(token)

		// TF-IDF weight
		tfIdf := float64(count) * idfWeight

		// Hash the token to get vector indices
		tokenVec := e.hashToken(token)

		// Add weighted contribution
		for i, val := range tokenVec {
			embedding[i] += float32(tfIdf * float64(val))
		}
	}

	return embedding.Normalize(), nil
}

// EmbedBatch generates embeddings for multiple texts.
func (e *LocalEmbedder) EmbedBatch(ctx context.Context, texts []string) ([]Vector, error) {
	vectors := make([]Vector, len(texts))

	for i, text := range texts {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		vec, err := e.Embed(ctx, text)
		if err != nil {
			return nil, fmt.Errorf("failed to embed text at index %d: %w", i, err)
		}
		vectors[i] = vec
	}

	return vectors, nil
}

// UpdateIDF updates IDF values based on a corpus of documents.
// This improves embedding quality when document statistics are available.
func (e *LocalEmbedder) UpdateIDF(documents []string) {
	docFreq := make(map[string]int)
	numDocs := len(documents)

	for _, doc := range documents {
		seen := make(map[string]struct{})
		tokens := e.tokenizer.Tokenize(doc)
		for _, token := range tokens {
			if _, ok := seen[token]; !ok {
				docFreq[token]++
				seen[token] = struct{}{}
			}
		}
	}

	e.idfMu.Lock()
	defer e.idfMu.Unlock()

	for token, df := range docFreq {
		// IDF with smoothing: log((N + 1) / (df + 1)) + 1
		e.idf[token] = math.Log(float64(numDocs+1)/float64(df+1)) + 1
	}
}

// getIDF returns the IDF weight for a token, using a default if unknown.
func (e *LocalEmbedder) getIDF(token string) float64 {
	e.idfMu.RLock()
	defer e.idfMu.RUnlock()

	if weight, ok := e.idf[token]; ok {
		return weight
	}
	// Default IDF for unknown tokens (assumes rare)
	return 5.0
}

// hashToken generates a sparse vector representation for a token.
func (e *LocalEmbedder) hashToken(token string) Vector {
	vec := make(Vector, e.dimension)

	// Use SHA-256 for consistent hashing
	hash := sha256.Sum256([]byte(token))

	// Use hash bytes to determine vector positions and values
	numPositions := 8 // Number of non-zero positions per token
	for i := 0; i < numPositions && i*4 < len(hash); i++ {
		// Extract position from hash
		pos := int(binary.BigEndian.Uint32(hash[i*4:i*4+4])) % e.dimension

		// Determine sign from next byte
		sign := float32(1.0)
		if i+1 < len(hash)/4 {
			if hash[(i+1)*4]&1 == 1 {
				sign = -1.0
			}
		}

		vec[pos] = sign
	}

	return vec
}

// hashEmbed generates an embedding using character-level hashing.
// Used as fallback for text that doesn't tokenize well.
func (e *LocalEmbedder) hashEmbed(text string) Vector {
	vec := make(Vector, e.dimension)

	// Normalize text
	text = strings.ToLower(text)
	text = strings.Map(func(r rune) rune {
		if unicode.IsLetter(r) || unicode.IsNumber(r) || unicode.IsSpace(r) {
			return r
		}
		return -1
	}, text)

	// Generate n-grams and hash them
	ngramSizes := []int{2, 3, 4}
	for _, n := range ngramSizes {
		for i := 0; i <= len(text)-n; i++ {
			ngram := text[i : i+n]
			hash := sha256.Sum256([]byte(ngram))
			pos := int(binary.BigEndian.Uint32(hash[:])) % e.dimension
			sign := float32(1.0)
			if hash[4]&1 == 1 {
				sign = -1.0
			}
			vec[pos] += sign * float32(1.0/float64(n))
		}
	}

	return vec.Normalize()
}

// -----------------------------------------------------------------------------
// OpenAIEmbedder: OpenAI Embeddings API client
// -----------------------------------------------------------------------------

// OpenAIEmbedder generates embeddings using OpenAI's embedding API.
type OpenAIEmbedder struct {
	config     EmbedderConfig
	httpClient *http.Client
	rateLimiter *rateLimiter
}

// rateLimiter implements a simple token bucket rate limiter.
type rateLimiter struct {
	mu           sync.Mutex
	tokens       float64
	maxTokens    float64
	refillRate   float64 // tokens per second
	lastRefill   time.Time
}

// newRateLimiter creates a rate limiter for the given requests per minute.
func newRateLimiter(rpm int) *rateLimiter {
	maxTokens := float64(rpm) / 60.0 * 10 // Allow burst of 10 seconds worth
	return &rateLimiter{
		tokens:     maxTokens,
		maxTokens:  maxTokens,
		refillRate: float64(rpm) / 60.0,
		lastRefill: time.Now(),
	}
}

// acquire attempts to acquire a token, returning true if successful.
func (r *rateLimiter) acquire() bool {
	r.mu.Lock()
	defer r.mu.Unlock()

	// Refill tokens based on elapsed time
	now := time.Now()
	elapsed := now.Sub(r.lastRefill).Seconds()
	r.tokens = math.Min(r.maxTokens, r.tokens+elapsed*r.refillRate)
	r.lastRefill = now

	if r.tokens >= 1 {
		r.tokens--
		return true
	}
	return false
}

// wait blocks until a token is available or context is cancelled.
func (r *rateLimiter) wait(ctx context.Context) error {
	for {
		if r.acquire() {
			return nil
		}

		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(100 * time.Millisecond):
			// Retry after short delay
		}
	}
}

// openAIRequest represents an OpenAI embeddings API request.
type openAIRequest struct {
	Input          interface{} `json:"input"`
	Model          string      `json:"model"`
	EncodingFormat string      `json:"encoding_format,omitempty"`
	Dimensions     int         `json:"dimensions,omitempty"`
}

// openAIResponse represents an OpenAI embeddings API response.
type openAIResponse struct {
	Object string `json:"object"`
	Data   []struct {
		Object    string    `json:"object"`
		Index     int       `json:"index"`
		Embedding []float32 `json:"embedding"`
	} `json:"data"`
	Model string `json:"model"`
	Usage struct {
		PromptTokens int `json:"prompt_tokens"`
		TotalTokens  int `json:"total_tokens"`
	} `json:"usage"`
	Error *openAIError `json:"error,omitempty"`
}

// openAIError represents an error from the OpenAI API.
type openAIError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code"`
}

// NewOpenAIEmbedder creates a new OpenAI embedder with the given configuration.
func NewOpenAIEmbedder(config EmbedderConfig) (*OpenAIEmbedder, error) {
	if config.APIKey == "" {
		return nil, ErrAPIKeyMissing
	}

	// Apply defaults
	if config.Model == "" {
		config.Model = DefaultOpenAIModel
	}
	if config.Endpoint == "" {
		config.Endpoint = DefaultOpenAIEndpoint
	}
	if config.Dimension <= 0 {
		config.Dimension = DefaultOpenAIDimension
	}
	if config.RequestTimeout <= 0 {
		config.RequestTimeout = DefaultRequestTimeout
	}
	if config.MaxRetries <= 0 {
		config.MaxRetries = DefaultMaxRetries
	}
	if config.RetryBaseDelay <= 0 {
		config.RetryBaseDelay = DefaultRetryBaseDelay
	}
	if config.RateLimitRPM <= 0 {
		config.RateLimitRPM = DefaultRateLimit
	}

	return &OpenAIEmbedder{
		config: config,
		httpClient: &http.Client{
			Timeout: config.RequestTimeout,
			Transport: &http.Transport{
				MaxIdleConns:        100,
				MaxIdleConnsPerHost: 100,
				IdleConnTimeout:     90 * time.Second,
			},
		},
		rateLimiter: newRateLimiter(config.RateLimitRPM),
	}, nil
}

// Dimension returns the output vector dimension.
func (e *OpenAIEmbedder) Dimension() int {
	return e.config.Dimension
}

// Close releases resources held by the embedder.
func (e *OpenAIEmbedder) Close() error {
	e.httpClient.CloseIdleConnections()
	return nil
}

// Embed generates an embedding for the given text using OpenAI's API.
func (e *OpenAIEmbedder) Embed(ctx context.Context, text string) (Vector, error) {
	if strings.TrimSpace(text) == "" {
		return nil, ErrEmptyText
	}

	vectors, err := e.EmbedBatch(ctx, []string{text})
	if err != nil {
		return nil, err
	}

	return vectors[0], nil
}

// EmbedBatch generates embeddings for multiple texts in a single API call.
func (e *OpenAIEmbedder) EmbedBatch(ctx context.Context, texts []string) ([]Vector, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	// Validate inputs
	for i, text := range texts {
		if strings.TrimSpace(text) == "" {
			return nil, fmt.Errorf("%w at index %d", ErrEmptyText, i)
		}
	}

	// Wait for rate limiter
	if err := e.rateLimiter.wait(ctx); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrRateLimited, err)
	}

	// Prepare request
	reqBody := openAIRequest{
		Input: texts,
		Model: e.config.Model,
	}

	// Only set dimensions if using a model that supports it
	if e.config.Model == "text-embedding-3-small" || e.config.Model == "text-embedding-3-large" {
		reqBody.Dimensions = e.config.Dimension
	}

	var lastErr error
	for attempt := 0; attempt < e.config.MaxRetries; attempt++ {
		if attempt > 0 {
			// Exponential backoff
			delay := e.config.RetryBaseDelay * time.Duration(1<<uint(attempt-1))
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(delay):
			}
		}

		vectors, err := e.doRequest(ctx, reqBody)
		if err == nil {
			return vectors, nil
		}

		lastErr = err

		// Don't retry on certain errors
		if errors.Is(err, ErrAPIKeyMissing) || errors.Is(err, ErrEmptyText) {
			return nil, err
		}

		// Check if error is retryable
		if !isRetryableError(err) {
			return nil, err
		}
	}

	return nil, fmt.Errorf("%w after %d attempts: %v", ErrAPIRequestFailed, e.config.MaxRetries, lastErr)
}

// doRequest performs the actual HTTP request to OpenAI's API.
func (e *OpenAIEmbedder) doRequest(ctx context.Context, reqBody openAIRequest) ([]Vector, error) {
	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, e.config.Endpoint, bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+e.config.APIKey)

	resp, err := e.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	// Handle HTTP errors
	if resp.StatusCode != http.StatusOK {
		return nil, e.parseErrorResponse(resp.StatusCode, body)
	}

	// Parse successful response
	var apiResp openAIResponse
	if err := json.Unmarshal(body, &apiResp); err != nil {
		return nil, fmt.Errorf("%w: failed to parse response: %v", ErrInvalidResponse, err)
	}

	if apiResp.Error != nil {
		return nil, fmt.Errorf("%w: %s", ErrAPIRequestFailed, apiResp.Error.Message)
	}

	// Extract embeddings in correct order
	vectors := make([]Vector, len(apiResp.Data))
	for _, item := range apiResp.Data {
		if item.Index >= len(vectors) {
			return nil, fmt.Errorf("%w: invalid embedding index %d", ErrInvalidResponse, item.Index)
		}
		vectors[item.Index] = Vector(item.Embedding)
	}

	// Verify all vectors were received
	for i, vec := range vectors {
		if vec == nil {
			return nil, fmt.Errorf("%w: missing embedding at index %d", ErrInvalidResponse, i)
		}
	}

	return vectors, nil
}

// parseErrorResponse converts HTTP error responses to appropriate errors.
func (e *OpenAIEmbedder) parseErrorResponse(statusCode int, body []byte) error {
	var errResp struct {
		Error openAIError `json:"error"`
	}

	if err := json.Unmarshal(body, &errResp); err == nil && errResp.Error.Message != "" {
		switch statusCode {
		case http.StatusTooManyRequests:
			return fmt.Errorf("%w: %s", ErrRateLimited, errResp.Error.Message)
		case http.StatusUnauthorized:
			return fmt.Errorf("%w: %s", ErrAPIKeyMissing, errResp.Error.Message)
		default:
			return fmt.Errorf("%w: [%d] %s", ErrAPIRequestFailed, statusCode, errResp.Error.Message)
		}
	}

	return fmt.Errorf("%w: HTTP %d", ErrAPIRequestFailed, statusCode)
}

// isRetryableError determines if an error should trigger a retry.
func isRetryableError(err error) bool {
	if errors.Is(err, ErrRateLimited) {
		return true
	}

	// Retry on transient network errors
	var netErr interface{ Temporary() bool }
	if errors.As(err, &netErr) && netErr.Temporary() {
		return true
	}

	// Check for specific HTTP status codes in error message
	errStr := err.Error()
	return strings.Contains(errStr, "429") || // Rate limit
		strings.Contains(errStr, "500") || // Internal server error
		strings.Contains(errStr, "502") || // Bad gateway
		strings.Contains(errStr, "503") || // Service unavailable
		strings.Contains(errStr, "504") // Gateway timeout
}

// -----------------------------------------------------------------------------
// VectorIndex: Similarity search using libsql vector index
// -----------------------------------------------------------------------------

// SearchResult represents a single similarity search result.
type SearchResult struct {
	ID         string
	Content    string
	Similarity float64
	Metadata   map[string]interface{}
}

// VectorIndex provides vector similarity search using libsql_vector_idx.
type VectorIndex struct {
	db        *sql.DB
	tableName string
	dimension int
	embedder  Embedder
}

// VectorIndexConfig holds configuration for the vector index.
type VectorIndexConfig struct {
	DB        *sql.DB
	TableName string
	Dimension int
	Embedder  Embedder
}

// NewVectorIndex creates a new vector index for similarity search.
func NewVectorIndex(config VectorIndexConfig) (*VectorIndex, error) {
	if config.DB == nil {
		return nil, fmt.Errorf("%w: database connection required", ErrDatabaseError)
	}
	if config.TableName == "" {
		config.TableName = "memory_embeddings"
	}
	if config.Dimension <= 0 {
		config.Dimension = DefaultOpenAIDimension
	}
	if config.Embedder == nil {
		config.Embedder = NewLocalEmbedder(DefaultLocalConfig())
	}

	return &VectorIndex{
		db:        config.DB,
		tableName: config.TableName,
		dimension: config.Dimension,
		embedder:  config.Embedder,
	}, nil
}

// InitSchema creates the table and vector index if they don't exist.
func (idx *VectorIndex) InitSchema(ctx context.Context) error {
	// Create the main table
	createTableSQL := fmt.Sprintf(`
		CREATE TABLE IF NOT EXISTS %s (
			id TEXT PRIMARY KEY,
			content TEXT NOT NULL,
			embedding BLOB NOT NULL,
			metadata TEXT,
			created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
			updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
		)
	`, idx.tableName)

	if _, err := idx.db.ExecContext(ctx, createTableSQL); err != nil {
		return fmt.Errorf("%w: failed to create table: %v", ErrDatabaseError, err)
	}

	// Create the vector index for similarity search
	// Using libsql_vector_idx with cosine distance metric
	createIndexSQL := fmt.Sprintf(`
		CREATE INDEX IF NOT EXISTS %s_vec_idx ON %s (
			libsql_vector_idx(embedding, 'metric=cosine', 'dimension=%d')
		)
	`, idx.tableName, idx.tableName, idx.dimension)

	if _, err := idx.db.ExecContext(ctx, createIndexSQL); err != nil {
		return fmt.Errorf("%w: failed to create vector index: %v", ErrDatabaseError, err)
	}

	return nil
}

// Insert adds a new entry to the vector index.
func (idx *VectorIndex) Insert(ctx context.Context, id, content string, metadata map[string]interface{}) error {
	// Generate embedding
	embedding, err := idx.embedder.Embed(ctx, content)
	if err != nil {
		return fmt.Errorf("failed to generate embedding: %w", err)
	}

	// Serialize embedding
	embeddingBytes, err := embedding.ToBytes()
	if err != nil {
		return fmt.Errorf("failed to serialize embedding: %w", err)
	}

	// Serialize metadata
	var metadataJSON []byte
	if metadata != nil {
		metadataJSON, err = json.Marshal(metadata)
		if err != nil {
			return fmt.Errorf("failed to serialize metadata: %w", err)
		}
	}

	// Insert into database
	insertSQL := fmt.Sprintf(`
		INSERT OR REPLACE INTO %s (id, content, embedding, metadata, updated_at)
		VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
	`, idx.tableName)

	if _, err := idx.db.ExecContext(ctx, insertSQL, id, content, embeddingBytes, metadataJSON); err != nil {
		return fmt.Errorf("%w: failed to insert: %v", ErrDatabaseError, err)
	}

	return nil
}

// InsertWithEmbedding adds an entry with a pre-computed embedding.
func (idx *VectorIndex) InsertWithEmbedding(ctx context.Context, id, content string, embedding Vector, metadata map[string]interface{}) error {
	if len(embedding) != idx.dimension {
		return fmt.Errorf("%w: expected %d, got %d", ErrDimensionMismatch, idx.dimension, len(embedding))
	}

	// Serialize embedding
	embeddingBytes, err := embedding.ToBytes()
	if err != nil {
		return fmt.Errorf("failed to serialize embedding: %w", err)
	}

	// Serialize metadata
	var metadataJSON []byte
	if metadata != nil {
		metadataJSON, err = json.Marshal(metadata)
		if err != nil {
			return fmt.Errorf("failed to serialize metadata: %w", err)
		}
	}

	// Insert into database
	insertSQL := fmt.Sprintf(`
		INSERT OR REPLACE INTO %s (id, content, embedding, metadata, updated_at)
		VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
	`, idx.tableName)

	if _, err := idx.db.ExecContext(ctx, insertSQL, id, content, embeddingBytes, metadataJSON); err != nil {
		return fmt.Errorf("%w: failed to insert: %v", ErrDatabaseError, err)
	}

	return nil
}

// Search performs a similarity search using the vector index.
// Returns the top-k most similar entries using cosine similarity.
func (idx *VectorIndex) Search(ctx context.Context, query string, k int) ([]SearchResult, error) {
	// Generate query embedding
	queryEmbedding, err := idx.embedder.Embed(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to generate query embedding: %w", err)
	}

	return idx.SearchByVector(ctx, queryEmbedding, k)
}

// SearchByVector performs a similarity search with a pre-computed query embedding.
func (idx *VectorIndex) SearchByVector(ctx context.Context, queryEmbedding Vector, k int) ([]SearchResult, error) {
	if len(queryEmbedding) != idx.dimension {
		return nil, fmt.Errorf("%w: expected %d, got %d", ErrDimensionMismatch, idx.dimension, len(queryEmbedding))
	}

	// Serialize query embedding
	queryBytes, err := queryEmbedding.ToBytes()
	if err != nil {
		return nil, fmt.Errorf("failed to serialize query embedding: %w", err)
	}

	// Use vector_top_k for efficient similarity search
	// The function returns (1 - cosine_distance) as similarity score
	searchSQL := fmt.Sprintf(`
		SELECT
			t.id,
			t.content,
			t.metadata,
			vector_top_k('%s_vec_idx', ?, %d) AS distance
		FROM %s t
		WHERE t.id IN (
			SELECT id FROM vector_top_k('%s_vec_idx', ?, %d)
		)
		ORDER BY distance ASC
		LIMIT ?
	`, idx.tableName, k, idx.tableName, idx.tableName, k)

	rows, err := idx.db.QueryContext(ctx, searchSQL, queryBytes, queryBytes, k)
	if err != nil {
		// Fall back to brute-force search if index query fails
		return idx.bruteForceSearch(ctx, queryEmbedding, k)
	}
	defer rows.Close()

	var results []SearchResult
	for rows.Next() {
		var (
			id          string
			content     string
			metadataStr sql.NullString
			distance    float64
		)

		if err := rows.Scan(&id, &content, &metadataStr, &distance); err != nil {
			return nil, fmt.Errorf("%w: failed to scan row: %v", ErrDatabaseError, err)
		}

		result := SearchResult{
			ID:         id,
			Content:    content,
			Similarity: 1 - distance, // Convert distance to similarity
		}

		if metadataStr.Valid {
			var metadata map[string]interface{}
			if err := json.Unmarshal([]byte(metadataStr.String), &metadata); err == nil {
				result.Metadata = metadata
			}
		}

		results = append(results, result)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("%w: row iteration error: %v", ErrDatabaseError, err)
	}

	return results, nil
}

// bruteForceSearch performs a brute-force similarity search as fallback.
func (idx *VectorIndex) bruteForceSearch(ctx context.Context, queryEmbedding Vector, k int) ([]SearchResult, error) {
	selectSQL := fmt.Sprintf(`
		SELECT id, content, embedding, metadata
		FROM %s
	`, idx.tableName)

	rows, err := idx.db.QueryContext(ctx, selectSQL)
	if err != nil {
		return nil, fmt.Errorf("%w: failed to query: %v", ErrDatabaseError, err)
	}
	defer rows.Close()

	type scoredResult struct {
		result     SearchResult
		similarity float64
	}

	var scored []scoredResult
	normalizedQuery := queryEmbedding.Normalize()

	for rows.Next() {
		var (
			id            string
			content       string
			embeddingBlob []byte
			metadataStr   sql.NullString
		)

		if err := rows.Scan(&id, &content, &embeddingBlob, &metadataStr); err != nil {
			return nil, fmt.Errorf("%w: failed to scan row: %v", ErrDatabaseError, err)
		}

		embedding, err := VectorFromBytes(embeddingBlob)
		if err != nil {
			continue // Skip invalid embeddings
		}

		normalizedEmb := embedding.Normalize()
		similarity, err := normalizedQuery.CosineSimilarity(normalizedEmb)
		if err != nil {
			continue // Skip on dimension mismatch
		}

		result := SearchResult{
			ID:         id,
			Content:    content,
			Similarity: similarity,
		}

		if metadataStr.Valid {
			var metadata map[string]interface{}
			if err := json.Unmarshal([]byte(metadataStr.String), &metadata); err == nil {
				result.Metadata = metadata
			}
		}

		scored = append(scored, scoredResult{result: result, similarity: similarity})
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("%w: row iteration error: %v", ErrDatabaseError, err)
	}

	// Sort by similarity (descending)
	sort.Slice(scored, func(i, j int) bool {
		return scored[i].similarity > scored[j].similarity
	})

	// Return top-k
	results := make([]SearchResult, 0, k)
	for i := 0; i < len(scored) && i < k; i++ {
		results = append(results, scored[i].result)
	}

	return results, nil
}

// Delete removes an entry from the vector index.
func (idx *VectorIndex) Delete(ctx context.Context, id string) error {
	deleteSQL := fmt.Sprintf(`DELETE FROM %s WHERE id = ?`, idx.tableName)

	result, err := idx.db.ExecContext(ctx, deleteSQL, id)
	if err != nil {
		return fmt.Errorf("%w: failed to delete: %v", ErrDatabaseError, err)
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("%w: failed to get rows affected: %v", ErrDatabaseError, err)
	}

	if rowsAffected == 0 {
		return fmt.Errorf("%w: no entry found with id %s", ErrIndexNotFound, id)
	}

	return nil
}

// Get retrieves a single entry by ID.
func (idx *VectorIndex) Get(ctx context.Context, id string) (*SearchResult, error) {
	selectSQL := fmt.Sprintf(`
		SELECT id, content, metadata
		FROM %s
		WHERE id = ?
	`, idx.tableName)

	var (
		content     string
		metadataStr sql.NullString
	)

	err := idx.db.QueryRowContext(ctx, selectSQL, id).Scan(&id, &content, &metadataStr)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("%w: failed to get: %v", ErrDatabaseError, err)
	}

	result := &SearchResult{
		ID:      id,
		Content: content,
	}

	if metadataStr.Valid {
		var metadata map[string]interface{}
		if err := json.Unmarshal([]byte(metadataStr.String), &metadata); err == nil {
			result.Metadata = metadata
		}
	}

	return result, nil
}

// Count returns the total number of entries in the index.
func (idx *VectorIndex) Count(ctx context.Context) (int64, error) {
	countSQL := fmt.Sprintf(`SELECT COUNT(*) FROM %s`, idx.tableName)

	var count int64
	if err := idx.db.QueryRowContext(ctx, countSQL).Scan(&count); err != nil {
		return 0, fmt.Errorf("%w: failed to count: %v", ErrDatabaseError, err)
	}

	return count, nil
}

// Close releases resources held by the vector index.
func (idx *VectorIndex) Close() error {
	return idx.embedder.Close()
}

// -----------------------------------------------------------------------------
// Factory functions for convenient embedder creation
// -----------------------------------------------------------------------------

// NewEmbedder creates an embedder based on configuration.
// If an OpenAI API key is provided, it creates an OpenAI embedder.
// Otherwise, it falls back to the local embedder.
func NewEmbedder(apiKey string, dimension int) (Embedder, error) {
	if apiKey != "" {
		config := DefaultOpenAIConfig(apiKey)
		if dimension > 0 {
			config.Dimension = dimension
		}
		return NewOpenAIEmbedder(config)
	}

	config := DefaultLocalConfig()
	if dimension > 0 {
		config.Dimension = dimension
	}
	return NewLocalEmbedder(config), nil
}

// NewEmbedderWithFallback creates an embedder with automatic fallback.
// It tries to create an OpenAI embedder first, falling back to local on failure.
func NewEmbedderWithFallback(apiKey string, dimension int) Embedder {
	embedder, err := NewEmbedder(apiKey, dimension)
	if err != nil {
		// Fall back to local embedder
		config := DefaultLocalConfig()
		if dimension > 0 {
			config.Dimension = dimension
		}
		return NewLocalEmbedder(config)
	}
	return embedder
}
