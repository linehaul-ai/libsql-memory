//! rmcp stdio server exposing the five memory tools (spec 04).

use std::path::PathBuf;
use std::sync::Arc;

use memory_core::{MergeMode, NoteType, Retriever};
use rmcp::{
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{Implementation, ServerCapabilities, ServerInfo},
    schemars::JsonSchema,
    tool, tool_handler, tool_router, ErrorData as McpError, ServerHandler, ServiceExt,
};
use serde::{Deserialize, Serialize};
use time::Date;

use crate::service::{store_action_str, ForgetAction, MemoryService, SearchOptions, StoreRequest};

/// MCP server name (spec 04).
pub const SERVER_NAME: &str = "fff-memory";

/// rmcp-backed memory server with five tools.
#[derive(Clone)]
pub struct MemoryServer {
    service: MemoryService,
    tool_router: ToolRouter<Self>,
}

impl MemoryServer {
    /// Wire a service (store root + optional retriever) into MCP tools.
    pub fn new(service: MemoryService) -> Self {
        Self {
            service,
            tool_router: Self::tool_router(),
        }
    }

    /// Convenience: open root with optional retriever.
    pub fn open(root: impl Into<PathBuf>, retriever: Option<Arc<dyn Retriever>>) -> Self {
        Self::new(MemoryService::new(root, retriever))
    }
}

/// Args for `memory_store`.
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct MemoryStoreArgs {
    /// Human-readable title (required).
    pub title: String,
    /// Markdown body (required).
    pub body: String,
    /// Synonyms / alternate phrasings a future query might use (min 2).
    /// Write aliases in the vocabulary of future questions, not restatements of the title.
    pub aliases: Vec<String>,
    /// Namespace directory (default `default`).
    #[serde(default = "default_namespace")]
    pub namespace: String,
    /// Note kind: fact | decision | preference | lesson | reference | session-summary.
    #[serde(rename = "type")]
    pub note_type: String,
    /// Optional tags.
    #[serde(default)]
    pub tags: Vec<String>,
    /// Optional expiry `YYYY-MM-DD`.
    #[serde(default)]
    pub expires: Option<String>,
    /// How to merge if dedup finds an existing note: `append` (default) or `replace`.
    #[serde(default = "default_mode")]
    pub mode: String,
}

fn default_namespace() -> String {
    "default".into()
}

fn default_mode() -> String {
    "append".into()
}

/// Args for `memory_search`.
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct MemorySearchArgs {
    /// Lexical query. Prefer 2–3 reformulations before concluding no memory exists.
    pub query: String,
    /// Namespace subtree; omit for whole root.
    #[serde(default)]
    pub namespace: Option<String>,
    /// Max full results (default 8).
    #[serde(default = "default_limit")]
    pub limit: u32,
    /// Byte budget for full results (default 4096, max 16384).
    #[serde(default = "default_budget")]
    pub budget_bytes: u32,
    /// Include archived notes (default false).
    #[serde(default)]
    pub include_archived: bool,
}

fn default_limit() -> u32 {
    8
}

fn default_budget() -> u32 {
    4096
}

/// Args for `memory_read`.
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct MemoryReadArgs {
    /// Handle `namespace/slug` (or `slug` at root).
    pub handle: String,
}

/// Args for `memory_forget`.
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct MemoryForgetArgs {
    /// Handle `namespace/slug`.
    pub handle: String,
    /// When true, hard-delete; default archives to `.archive/`.
    #[serde(default)]
    pub hard: bool,
}

/// Args for `memory_stats`.
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct MemoryStatsArgs {
    /// Optional namespace filter.
    #[serde(default)]
    pub namespace: Option<String>,
}

#[tool_router]
impl MemoryServer {
    /// Store or update a memory. Never fails for search-related reasons.
    /// Write aliases in the vocabulary of future questions, not restatements of the title.
    #[tool(
        name = "memory_store",
        description = "Store or update a memory note. Aliases are load-bearing: write them in the vocabulary of future questions (min 2). Never fails because search is unavailable."
    )]
    async fn memory_store(
        &self,
        Parameters(args): Parameters<MemoryStoreArgs>,
    ) -> Result<rmcp::handler::server::wrapper::Json<serde_json::Value>, McpError> {
        let note_type = parse_note_type(&args.note_type)?;
        let mode = parse_mode(&args.mode)?;
        let expires = match args.expires.as_deref() {
            None | Some("") => None,
            Some(s) => Some(parse_date(s)?),
        };
        let out = self
            .service
            .store(StoreRequest {
                title: args.title,
                body: args.body,
                aliases: args.aliases,
                namespace: args.namespace,
                note_type,
                tags: args.tags,
                expires,
                mode,
            })
            .map_err(core_to_mcp)?;

        Ok(rmcp::handler::server::wrapper::Json(serde_json::json!({
            "slug": out.slug,
            "namespace": out.namespace,
            "action": store_action_str(out.action),
            "dedup_hit": out.dedup_hit,
        })))
    }

    /// Layered lexical search with ranked snippets and a why per result.
    #[tool(
        name = "memory_search",
        description = "Layered lexical search over memory notes. Returns ranked snippets with handles and a 'why' per result. Prefer 2–3 query reformulations before concluding a memory does not exist. Empty results explain stages tried."
    )]
    async fn memory_search(
        &self,
        Parameters(args): Parameters<MemorySearchArgs>,
    ) -> Result<rmcp::handler::server::wrapper::Json<serde_json::Value>, McpError> {
        let resp = self
            .service
            .search(SearchOptions {
                query: args.query,
                namespace: args.namespace,
                limit: args.limit as usize,
                budget_bytes: args.budget_bytes as usize,
                include_archived: args.include_archived,
            })
            .map_err(core_to_mcp)?;
        let value = serde_json::to_value(resp).map_err(|e| {
            McpError::internal_error(format!("serialize search response: {e}"), None)
        })?;
        Ok(rmcp::handler::server::wrapper::Json(value))
    }

    /// Fetch one full note by handle; records an access event.
    #[tool(
        name = "memory_read",
        description = "Fetch one full note by handle (namespace/slug). Records an access event for frecency ranking. Returns frontmatter, body, and one-hop linked note titles."
    )]
    async fn memory_read(
        &self,
        Parameters(args): Parameters<MemoryReadArgs>,
    ) -> Result<rmcp::handler::server::wrapper::Json<serde_json::Value>, McpError> {
        let out = self.service.read(&args.handle).map_err(core_to_mcp)?;
        let value = serde_json::to_value(out)
            .map_err(|e| McpError::internal_error(format!("serialize read response: {e}"), None))?;
        Ok(rmcp::handler::server::wrapper::Json(value))
    }

    /// Archive (default) or hard-delete a note.
    #[tool(
        name = "memory_forget",
        description = "Archive a note to .archive/ (default) or hard-delete when hard=true. Archive is reversible; hard delete is not."
    )]
    async fn memory_forget(
        &self,
        Parameters(args): Parameters<MemoryForgetArgs>,
    ) -> Result<rmcp::handler::server::wrapper::Json<serde_json::Value>, McpError> {
        let out = self
            .service
            .forget(&args.handle, args.hard)
            .map_err(core_to_mcp)?;
        let action = match out.action {
            ForgetAction::Archived => "archived",
            ForgetAction::Deleted => "deleted",
        };
        Ok(rmcp::handler::server::wrapper::Json(serde_json::json!({
            "handle": out.handle,
            "action": action,
        })))
    }

    /// Store health: counts, disk, index, decay signals.
    #[tool(
        name = "memory_stats",
        description = "Store health: counts by namespace and type, disk size, index status, access-log summary, notes nearing expiry."
    )]
    async fn memory_stats(
        &self,
        Parameters(args): Parameters<MemoryStatsArgs>,
    ) -> Result<rmcp::handler::server::wrapper::Json<serde_json::Value>, McpError> {
        let out = self
            .service
            .stats(args.namespace.as_deref())
            .map_err(core_to_mcp)?;
        let value = serde_json::to_value(out).map_err(|e| {
            McpError::internal_error(format!("serialize stats response: {e}"), None)
        })?;
        Ok(rmcp::handler::server::wrapper::Json(value))
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for MemoryServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new(
                SERVER_NAME,
                env!("CARGO_PKG_VERSION"),
            ))
            .with_instructions(
                "Persistent agent memory: store notes with rich aliases, search lexically, read by handle. Prefer memory_search with 2–3 reformulations before concluding nothing is stored.",
            )
    }
}

/// Serve MCP over stdio (stdout = protocol; log to stderr only).
pub async fn serve_stdio(server: MemoryServer) -> Result<(), rmcp::RmcpError> {
    let transport = rmcp::transport::stdio();
    let service = server.serve(transport).await?;
    service.waiting().await?;
    Ok(())
}

fn parse_note_type(s: &str) -> Result<NoteType, McpError> {
    match s.trim().to_ascii_lowercase().as_str() {
        "fact" => Ok(NoteType::Fact),
        "decision" => Ok(NoteType::Decision),
        "preference" => Ok(NoteType::Preference),
        "lesson" => Ok(NoteType::Lesson),
        "reference" => Ok(NoteType::Reference),
        "session-summary" | "session_summary" => Ok(NoteType::SessionSummary),
        other => Err(McpError::invalid_params(
            format!(
                "type {other:?} rejected; use one of: {}",
                NoteType::ALL.join(", ")
            ),
            None,
        )),
    }
}

fn parse_mode(s: &str) -> Result<MergeMode, McpError> {
    match s.trim().to_ascii_lowercase().as_str() {
        "append" | "" => Ok(MergeMode::Append),
        "replace" => Ok(MergeMode::Replace),
        other => Err(McpError::invalid_params(
            format!("mode {other:?} rejected; use append or replace"),
            None,
        )),
    }
}

fn parse_date(s: &str) -> Result<Date, McpError> {
    use time::macros::format_description;
    const FMT: &[time::format_description::FormatItem<'static>] =
        format_description!("[year]-[month]-[day]");
    Date::parse(s.trim(), &FMT).map_err(|_| {
        McpError::invalid_params(format!("expires {s:?} rejected; use YYYY-MM-DD"), None)
    })
}

fn core_to_mcp(err: memory_core::Error) -> McpError {
    use memory_core::Error::*;
    match &err {
        Validation { field, message } => McpError::invalid_params(
            format!("validation failed on `{field}`: {message}"),
            None,
        ),
        InvalidNamespace { path, reason } => McpError::invalid_params(
            format!("namespace {path:?} rejected: {reason}"),
            None,
        ),
        NotFound { handle } => McpError::invalid_params(
            format!("note not found: {handle}; check handle with memory_search or memory_stats"),
            None,
        ),
        InvalidSlug { title } => McpError::invalid_params(
            format!(
                "cannot slugify title {title:?}: produce a title with at least one ASCII letter or digit"
            ),
            None,
        ),
        other => McpError::internal_error(other.to_string(), None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rmcp::handler::server::wrapper::Parameters;
    use tempfile::tempdir;

    #[tokio::test]
    async fn tools_store_search_read_via_direct_calls() {
        let dir = tempdir().unwrap();
        let server = MemoryServer::open(dir.path(), None);

        let stored = server
            .memory_store(Parameters(MemoryStoreArgs {
                title: "Deploy Policy".into(),
                body: "Prefer blue-green.".into(),
                aliases: vec!["release process".into(), "deploy rules".into()],
                namespace: "ops".into(),
                note_type: "decision".into(),
                tags: vec!["deploy".into()],
                expires: None,
                mode: "append".into(),
            }))
            .await
            .unwrap();
        assert_eq!(stored.0["action"], "created");
        assert_eq!(stored.0["slug"], "deploy-policy");

        // Without retriever, search returns empty report (not an error)
        let search = server
            .memory_search(Parameters(MemorySearchArgs {
                query: "deploy".into(),
                namespace: Some("ops".into()),
                limit: 8,
                budget_bytes: 4096,
                include_archived: false,
            }))
            .await
            .unwrap();
        assert!(search.0.get("stages_run").is_some());
        assert!(search.0.get("empty_hint").is_some() || search.0["results"].as_array().is_some());

        let read = server
            .memory_read(Parameters(MemoryReadArgs {
                handle: "ops/deploy-policy".into(),
            }))
            .await
            .unwrap();
        assert_eq!(read.0["frontmatter"]["title"], "Deploy Policy");

        let stats = server
            .memory_stats(Parameters(MemoryStatsArgs { namespace: None }))
            .await
            .unwrap();
        assert_eq!(stats.0["total_notes"], 1);

        let forgot = server
            .memory_forget(Parameters(MemoryForgetArgs {
                handle: "ops/deploy-policy".into(),
                hard: false,
            }))
            .await
            .unwrap();
        assert_eq!(forgot.0["action"], "archived");
    }

    #[tokio::test]
    async fn invalid_type_is_mcp_invalid_params() {
        let dir = tempdir().unwrap();
        let server = MemoryServer::open(dir.path(), None);
        let err = match server
            .memory_store(Parameters(MemoryStoreArgs {
                title: "X".into(),
                body: "y".into(),
                aliases: vec!["a".into(), "b".into()],
                namespace: "default".into(),
                note_type: "nope".into(),
                tags: vec![],
                expires: None,
                mode: "append".into(),
            }))
            .await
        {
            Ok(_) => panic!("expected invalid type error"),
            Err(e) => e,
        };
        let msg = err.message.to_string();
        assert!(msg.contains("type") || msg.contains("nope"));
    }

    #[test]
    fn tool_attrs_exist() {
        let _ = MemoryServer::memory_store_tool_attr();
        let _ = MemoryServer::memory_search_tool_attr();
        let _ = MemoryServer::memory_read_tool_attr();
        let _ = MemoryServer::memory_forget_tool_attr();
        let _ = MemoryServer::memory_stats_tool_attr();
    }
}
