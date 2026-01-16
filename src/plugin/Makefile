# LibSQL Memory Plugin Makefile
# Claude Memory Plugin - Go implementation with LibSQL vector support

# Build variables
VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")
COMMIT ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")
BUILD_DATE ?= $(shell date -u +"%Y-%m-%dT%H:%M:%SZ")
LDFLAGS := -ldflags "-X main.version=$(VERSION) -X main.commit=$(COMMIT) -X main.buildDate=$(BUILD_DATE)"

# Go variables
GO := go
GOBIN := $(shell $(GO) env GOPATH)/bin
GOOS ?= $(shell $(GO) env GOOS)
GOARCH ?= $(shell $(GO) env GOARCH)

# Output directories
BUILD_DIR := build
DIST_DIR := dist

# Binary names
BINARY_NAME := libsql-memory
MCP_SERVER_BINARY := mcp-server

# Default target
.DEFAULT_GOAL := build

.PHONY: all build build-mcp install clean test test-verbose lint fmt vet deps tidy help

## help: Show this help message
help:
	@echo "LibSQL Memory Plugin - Claude Memory Plugin"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/## /  /'
	@echo ""
	@echo "Variables:"
	@echo "  VERSION      Build version (default: git tag or 'dev')"
	@echo "  GOOS         Target OS (default: current OS)"
	@echo "  GOARCH       Target architecture (default: current arch)"

## all: Build all binaries
all: build build-mcp

## build: Build the main plugin binary
build: $(BUILD_DIR)
	@echo "Building $(BINARY_NAME) $(VERSION)..."
	$(GO) build $(LDFLAGS) -o $(BUILD_DIR)/$(BINARY_NAME) ./cmd/

## build-mcp: Build the MCP server binary
build-mcp: $(BUILD_DIR)
	@echo "Building $(MCP_SERVER_BINARY)..."
	$(GO) build $(LDFLAGS) -o $(BUILD_DIR)/$(MCP_SERVER_BINARY) ./cmd/mcp-server/

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

## install: Install binaries to GOBIN
install:
	@echo "Installing $(BINARY_NAME) to $(GOBIN)..."
	$(GO) install $(LDFLAGS) ./cmd/
	$(GO) install $(LDFLAGS) ./cmd/mcp-server/

## clean: Remove build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(BUILD_DIR) $(DIST_DIR)
	$(GO) clean -cache -testcache

## test: Run all tests
test:
	@echo "Running tests..."
	$(GO) test -race -cover ./...

## test-verbose: Run tests with verbose output
test-verbose:
	@echo "Running tests (verbose)..."
	$(GO) test -race -cover -v ./...

## test-coverage: Run tests with coverage report
test-coverage: $(BUILD_DIR)
	@echo "Running tests with coverage..."
	$(GO) test -race -coverprofile=$(BUILD_DIR)/coverage.out ./...
	$(GO) tool cover -html=$(BUILD_DIR)/coverage.out -o $(BUILD_DIR)/coverage.html
	@echo "Coverage report: $(BUILD_DIR)/coverage.html"

## lint: Run golangci-lint
lint:
	@echo "Running linter..."
	@which golangci-lint > /dev/null || (echo "Installing golangci-lint..." && go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest)
	golangci-lint run ./...

## fmt: Format code
fmt:
	@echo "Formatting code..."
	$(GO) fmt ./...
	@which goimports > /dev/null || (echo "Installing goimports..." && go install golang.org/x/tools/cmd/goimports@latest)
	goimports -w .

## vet: Run go vet
vet:
	@echo "Running go vet..."
	$(GO) vet ./...

## deps: Download dependencies
deps:
	@echo "Downloading dependencies..."
	$(GO) mod download

## tidy: Tidy and verify dependencies
tidy:
	@echo "Tidying dependencies..."
	$(GO) mod tidy
	$(GO) mod verify

## verify: Run all checks (fmt, vet, lint, test)
verify: fmt vet lint test
	@echo "All checks passed!"

## run: Run the MCP server locally
run: build-mcp
	@echo "Running MCP server..."
	./$(BUILD_DIR)/$(MCP_SERVER_BINARY)

## dist: Build distribution binaries for all platforms
dist: $(DIST_DIR)
	@echo "Building distribution binaries..."
	GOOS=darwin GOARCH=amd64 $(GO) build $(LDFLAGS) -o $(DIST_DIR)/$(BINARY_NAME)-darwin-amd64 ./cmd/
	GOOS=darwin GOARCH=arm64 $(GO) build $(LDFLAGS) -o $(DIST_DIR)/$(BINARY_NAME)-darwin-arm64 ./cmd/
	GOOS=linux GOARCH=amd64 $(GO) build $(LDFLAGS) -o $(DIST_DIR)/$(BINARY_NAME)-linux-amd64 ./cmd/
	GOOS=linux GOARCH=arm64 $(GO) build $(LDFLAGS) -o $(DIST_DIR)/$(BINARY_NAME)-linux-arm64 ./cmd/
	GOOS=windows GOARCH=amd64 $(GO) build $(LDFLAGS) -o $(DIST_DIR)/$(BINARY_NAME)-windows-amd64.exe ./cmd/
	@echo "Distribution binaries created in $(DIST_DIR)/"

$(DIST_DIR):
	mkdir -p $(DIST_DIR)

## docker: Build Docker image
docker:
	@echo "Building Docker image..."
	docker build -t libsql-memory:$(VERSION) .

## version: Print version information
version:
	@echo "Version: $(VERSION)"
	@echo "Commit:  $(COMMIT)"
	@echo "Date:    $(BUILD_DATE)"
	@echo "Go:      $(shell $(GO) version)"
