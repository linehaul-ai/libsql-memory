use std::{collections::BTreeSet, time::Duration};

use memory_mcp::MemoryServer;
use rmcp::ServiceExt;
use serde_json::{json, Value};
use tempfile::tempdir;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, ReadHalf, WriteHalf};

struct RawClient {
    reader: BufReader<ReadHalf<tokio::io::DuplexStream>>,
    writer: WriteHalf<tokio::io::DuplexStream>,
    next_id: u64,
}

impl RawClient {
    fn new(io: tokio::io::DuplexStream) -> Self {
        let (reader, writer) = tokio::io::split(io);
        Self {
            reader: BufReader::new(reader),
            writer,
            next_id: 1,
        }
    }

    async fn send(&mut self, message: &Value) {
        self.writer
            .write_all(serde_json::to_string(message).unwrap().as_bytes())
            .await
            .unwrap();
        self.writer.write_all(b"\n").await.unwrap();
        self.writer.flush().await.unwrap();
    }

    async fn notify(&mut self, method: &str) {
        self.send(&json!({ "jsonrpc": "2.0", "method": method }))
            .await;
    }

    async fn request(&mut self, method: &str, params: Value) -> Value {
        let id = self.next_id;
        self.next_id += 1;
        self.send(&json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        }))
        .await;

        loop {
            let mut line = String::new();
            let read =
                tokio::time::timeout(Duration::from_secs(5), self.reader.read_line(&mut line))
                    .await
                    .expect("timed out waiting for JSON-RPC response")
                    .unwrap();
            assert_ne!(read, 0, "server closed before responding to id {id}");
            let response: Value = serde_json::from_str(line.trim()).unwrap();
            if response.get("id") == Some(&json!(id)) {
                assert_eq!(response["jsonrpc"], "2.0");
                return response;
            }
        }
    }
}

fn golden(source: &str) -> Value {
    serde_json::from_str(source).unwrap()
}

fn assert_golden(actual: &Value, expected: Value, name: &str) {
    assert_eq!(
        actual,
        &expected,
        "{name} golden mismatch; actual:\n{}",
        serde_json::to_string_pretty(actual).unwrap()
    );
}

fn tools_by_name(tools: &[Value]) -> std::collections::BTreeMap<&str, &Value> {
    tools
        .iter()
        .map(|tool| (tool["name"].as_str().unwrap(), tool))
        .collect()
}

fn required(schema: &Value) -> BTreeSet<&str> {
    schema["required"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|name| name.as_str().unwrap())
        .collect()
}

fn structured(response: &Value) -> &Value {
    let result = response["result"]
        .as_object()
        .expect("successful tool response has a result object");
    assert_eq!(
        result.keys().map(String::as_str).collect::<BTreeSet<_>>(),
        BTreeSet::from(["content", "isError", "structuredContent"])
    );
    assert_eq!(result["isError"], false);
    let structured = response
        .pointer("/result/structuredContent")
        .expect("successful tool response has structuredContent");
    let content = result["content"].as_array().unwrap();
    assert_eq!(content.len(), 1);
    assert_eq!(content[0]["type"], "text");
    let text = content[0]["text"]
        .as_str()
        .expect("successful tool response has JSON text content");
    assert_eq!(serde_json::from_str::<Value>(text).unwrap(), *structured);
    structured
}

fn tool_error(response: &Value) -> &str {
    let result = response["result"]
        .as_object()
        .expect("tool argument error has a result object");
    assert_eq!(
        result.keys().map(String::as_str).collect::<BTreeSet<_>>(),
        BTreeSet::from(["content", "isError"])
    );
    assert_eq!(result["isError"], true);
    let content = result["content"].as_array().unwrap();
    assert_eq!(content.len(), 1);
    assert_eq!(content[0]["type"], "text");
    content[0]["text"].as_str().unwrap()
}

fn assert_keys(value: &Value, expected: &[&str]) {
    let actual: BTreeSet<_> = value
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(actual, expected.iter().copied().collect());
}

#[tokio::test]
async fn official_rmcp_wire_contract_matches_five_tool_goldens() {
    let dir = tempdir().unwrap();
    let (server_io, client_io) = tokio::io::duplex(64 * 1024);
    let server = MemoryServer::open(dir.path(), None);
    let server_task = tokio::spawn(async move {
        let running = server.serve(server_io).await.expect("rmcp initialize");
        running.waiting().await.expect("rmcp server loop");
    });
    let mut client = RawClient::new(client_io);

    let initialize = client
        .request(
            "initialize",
            json!({
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": { "name": "golden-test", "version": "1" }
            }),
        )
        .await;
    assert_golden(
        &initialize["result"],
        golden(include_str!("golden/initialize.json")),
        "initialize",
    );
    client.notify("notifications/initialized").await;

    let listed = client.request("tools/list", json!({})).await;
    let mut listed_result = listed["result"].clone();
    listed_result["tools"]
        .as_array_mut()
        .unwrap()
        .sort_by(|a, b| a["name"].as_str().cmp(&b["name"].as_str()));
    let tools = listed_result["tools"].as_array().unwrap();
    let names: BTreeSet<_> = tools
        .iter()
        .map(|tool| tool["name"].as_str().unwrap())
        .collect();
    assert_eq!(
        names,
        BTreeSet::from([
            "memory_forget",
            "memory_read",
            "memory_search",
            "memory_stats",
            "memory_store",
        ])
    );
    assert_eq!(tools.len(), 5, "no sixth tool is exposed");

    let tools = tools_by_name(tools);
    assert_eq!(
        required(&tools["memory_store"]["inputSchema"]),
        BTreeSet::from(["aliases", "body", "title", "type"])
    );
    assert_eq!(
        required(&tools["memory_search"]["inputSchema"]),
        BTreeSet::from(["query"])
    );
    assert_eq!(
        required(&tools["memory_read"]["inputSchema"]),
        BTreeSet::from(["handle"])
    );
    assert_eq!(
        required(&tools["memory_forget"]["inputSchema"]),
        BTreeSet::from(["handle"])
    );
    assert!(required(&tools["memory_stats"]["inputSchema"]).is_empty());

    let aliases = &tools["memory_store"]["inputSchema"]["properties"]["aliases"];
    assert_eq!(aliases["minItems"], 2);
    let limit = &tools["memory_search"]["inputSchema"]["properties"]["limit"];
    assert_eq!(limit["minimum"], 0);
    assert_eq!(limit["default"], 8);
    let budget = &tools["memory_search"]["inputSchema"]["properties"]["budget_bytes"];
    assert_eq!(budget["minimum"], 0);
    assert_eq!(budget["default"], 4096);
    assert_eq!(budget["maximum"], 16384);

    assert_golden(
        &listed_result,
        golden(include_str!("golden/tools-list.json")),
        "tools/list",
    );

    let stored = client
        .request(
            "tools/call",
            json!({
                "name": "memory_store",
                "arguments": {
                    "title": "Deploy Policy",
                    "body": "Prefer blue-green releases.",
                    "aliases": ["release process", "deployment rules"],
                    "namespace": "ops",
                    "type": "decision",
                    "tags": ["deploy"]
                }
            }),
        )
        .await;
    assert_eq!(
        structured(&stored),
        &json!({
            "action": "created",
            "dedup_hit": null,
            "namespace": "ops",
            "slug": "deploy-policy"
        })
    );

    let searched = client
        .request(
            "tools/call",
            json!({
                "name": "memory_search",
                "arguments": { "query": "deploy", "namespace": "ops" }
            }),
        )
        .await;
    let search = structured(&searched);
    assert_keys(
        search,
        &["empty_hint", "more", "results", "scope", "stages_run"],
    );
    assert_eq!(search["results"], json!([]));
    assert_eq!(search["more"], json!([]));
    assert_eq!(search["scope"], "ops");
    assert!(search["empty_hint"].as_str().unwrap().contains("broader"));

    let read = client
        .request(
            "tools/call",
            json!({
                "name": "memory_read",
                "arguments": { "handle": "ops/deploy-policy" }
            }),
        )
        .await;
    let read = structured(&read);
    assert_keys(read, &["body", "frontmatter", "linked"]);
    assert_eq!(read["frontmatter"]["title"], "Deploy Policy");
    assert_eq!(read["body"], "Prefer blue-green releases.\n");
    assert_eq!(read["linked"], json!([]));

    let stats = client
        .request(
            "tools/call",
            json!({ "name": "memory_stats", "arguments": {} }),
        )
        .await;
    let stats = structured(&stats);
    assert_keys(
        stats,
        &[
            "by_namespace",
            "by_type",
            "decay",
            "disk_bytes",
            "index",
            "total_notes",
        ],
    );
    assert_keys(
        &stats["index"],
        &[
            "access_log_events",
            "files_indexed",
            "last_scan_ms",
            "state",
        ],
    );
    assert_keys(&stats["decay"], &["expiring_soon", "never_accessed_30d"]);
    assert_eq!(stats["total_notes"], 1);
    assert_eq!(stats["by_namespace"]["ops"], 1);
    assert_eq!(stats["index"]["state"], "unavailable");

    let malformed = client
        .request(
            "tools/call",
            json!({
                "name": "memory_store",
                "arguments": {
                    "title": "Missing Aliases",
                    "body": "Required input is absent.",
                    "type": "fact"
                }
            }),
        )
        .await;
    assert_eq!(
        tool_error(&malformed),
        "failed to deserialize parameters: missing field `aliases`"
    );

    let invalid = client
        .request(
            "tools/call",
            json!({
                "name": "memory_store",
                "arguments": {
                    "title": "Invalid",
                    "body": "Too few aliases.",
                    "aliases": ["only one"],
                    "type": "fact"
                }
            }),
        )
        .await;
    assert_eq!(invalid["error"]["code"], -32602);
    let invalid_message = invalid["error"]["message"].as_str().unwrap();
    assert!(invalid_message.contains("aliases"));
    assert!(invalid_message.contains("at least 2"));

    let forgotten = client
        .request(
            "tools/call",
            json!({
                "name": "memory_forget",
                "arguments": { "handle": "ops/deploy-policy" }
            }),
        )
        .await;
    assert_eq!(
        structured(&forgotten),
        &json!({ "action": "archived", "handle": "ops/deploy-policy" })
    );

    drop(client);
    tokio::time::timeout(Duration::from_secs(5), server_task)
        .await
        .expect("server stopped after transport EOF")
        .unwrap();
}
