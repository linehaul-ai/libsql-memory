use std::io::{BufRead, BufReader, Read, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use serde_json::{json, Value};
use tempfile::tempdir;

struct McpChild {
    child: Child,
    stdin: Option<ChildStdin>,
    lines: Receiver<String>,
    stderr: Arc<Mutex<String>>,
    stdout_thread: Option<JoinHandle<()>>,
    stderr_thread: Option<JoinHandle<()>>,
    next_id: u64,
}

impl McpChild {
    fn spawn(root: &std::path::Path) -> Self {
        let mut command = Command::new(env!("CARGO_BIN_EXE_fff-memory"));
        for key in [
            "FFF_MEMORY_ROOT",
            "FFF_MEMORY_CONFIG",
            "XDG_CONFIG_HOME",
            "XDG_DATA_HOME",
            "HOME",
        ] {
            command.env_remove(key);
        }
        let mut child = command
            .args(["serve", "--root"])
            .arg(root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn fff-memory serve");
        let stdin = child.stdin.take().unwrap();
        let stdout = child.stdout.take().unwrap();
        let stderr = child.stderr.take().unwrap();
        let (tx, lines) = mpsc::channel();
        let stdout_thread = std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let line = line.unwrap_or_else(|error| format!("<stdout read error: {error}>"));
                if tx.send(line).is_err() {
                    break;
                }
            }
        });
        let stderr_text = Arc::new(Mutex::new(String::new()));
        let stderr_sink = Arc::clone(&stderr_text);
        let stderr_thread = std::thread::spawn(move || {
            let mut reader = BufReader::new(stderr);
            let mut chunk = [0; 1024];
            while let Ok(count) = reader.read(&mut chunk) {
                if count == 0 {
                    break;
                }
                stderr_sink
                    .lock()
                    .unwrap()
                    .push_str(&String::from_utf8_lossy(&chunk[..count]));
            }
        });
        Self {
            child,
            stdin: Some(stdin),
            lines,
            stderr: stderr_text,
            stdout_thread: Some(stdout_thread),
            stderr_thread: Some(stderr_thread),
            next_id: 1,
        }
    }

    fn send(&mut self, value: Value) {
        let stdin = self.stdin.as_mut().expect("stdin remains open");
        serde_json::to_writer(&mut *stdin, &value).unwrap();
        stdin.write_all(b"\n").unwrap();
        stdin.flush().unwrap();
    }

    fn notify(&mut self, method: &str) {
        self.send(json!({ "jsonrpc": "2.0", "method": method }));
    }

    fn request(&mut self, method: &str, params: Value) -> Value {
        let id = self.next_id;
        self.next_id += 1;
        self.send(json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        }));
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            let line = self.lines.recv_timeout(remaining).unwrap_or_else(|error| {
                panic!(
                    "bounded wait for JSON-RPC response: {error}; stderr={}",
                    self.stderr.lock().unwrap()
                )
            });
            let response: Value = serde_json::from_str(&line)
                .unwrap_or_else(|error| panic!("stdout protocol contamination: {line:?}: {error}"));
            assert_eq!(response["jsonrpc"], "2.0");
            if response.get("id") == Some(&json!(id)) {
                return response;
            }
        }
    }

    fn assert_clean_shutdown(&mut self) {
        self.stop();
        for line in self.lines.try_iter() {
            let value = serde_json::from_str::<Value>(&line).unwrap_or_else(|error| {
                panic!("trailing stdout protocol contamination: {line:?}: {error}")
            });
            assert_eq!(value["jsonrpc"], "2.0", "unexpected stdout JSON: {line}");
        }
    }

    fn stop(&mut self) {
        self.stdin.take();
        let deadline = Instant::now() + Duration::from_secs(2);
        while Instant::now() < deadline {
            match self.child.try_wait() {
                Ok(Some(_)) => break,
                Ok(None) => std::thread::sleep(Duration::from_millis(10)),
                Err(_) => break,
            }
        }
        if self.child.try_wait().ok().flatten().is_none() {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
        if let Some(thread) = self.stdout_thread.take() {
            let _ = thread.join();
        }
        if let Some(thread) = self.stderr_thread.take() {
            let _ = thread.join();
        }
    }
}

impl Drop for McpChild {
    fn drop(&mut self) {
        self.stop();
    }
}

fn structured(response: &Value) -> &Value {
    response
        .pointer("/result/structuredContent")
        .unwrap_or_else(|| panic!("missing structured content: {response}"))
}

#[test]
fn spawned_stdio_round_trip_store_search_read() {
    let dir = tempdir().unwrap();
    let mut client = McpChild::spawn(dir.path());

    let initialized = client.request(
        "initialize",
        json!({
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": { "name": "spawned-e2e", "version": "1" }
        }),
    );
    assert_eq!(initialized["result"]["serverInfo"]["name"], "fff-memory");
    client.notify("notifications/initialized");

    let listed = client.request("tools/list", json!({}));
    assert_eq!(listed["result"]["tools"].as_array().unwrap().len(), 5);

    let stored = client.request(
        "tools/call",
        json!({
            "name": "memory_store",
            "arguments": {
                "title": "Spawned Protocol Note",
                "body": "The real child process speaks clean JSON-RPC.",
                "aliases": ["stdio integration", "spawned server round trip"],
                "namespace": "e2e",
                "type": "fact"
            }
        }),
    );
    assert_eq!(structured(&stored)["slug"], "spawned-protocol-note");

    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        let searched = client.request(
            "tools/call",
            json!({
                "name": "memory_search",
                "arguments": { "query": "spawned server round trip" }
            }),
        );
        let results = structured(&searched)["results"].as_array().unwrap();
        if results
            .iter()
            .any(|hit| hit["handle"] == "e2e/spawned-protocol-note")
        {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "watcher did not index stored note"
        );
        std::thread::sleep(Duration::from_millis(100));
    }

    let read = client.request(
        "tools/call",
        json!({
            "name": "memory_read",
            "arguments": { "handle": "e2e/spawned-protocol-note" }
        }),
    );
    assert_eq!(
        structured(&read)["body"].as_str().unwrap().trim_end(),
        "The real child process speaks clean JSON-RPC."
    );
    client.assert_clean_shutdown();
}
