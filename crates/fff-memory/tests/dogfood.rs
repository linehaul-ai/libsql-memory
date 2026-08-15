use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use memory_core::{ContentMatch, GrepMode, MergeMode, NoteType, Retriever};
use memory_index::FffRetriever;
use memory_mcp::{MemoryService, SearchOptions, StoreRequest};
use tempfile::tempdir;

struct Seed {
    title: &'static str,
    aliases: [&'static str; 2],
    namespace: &'static str,
    note_type: NoteType,
    tags: &'static [&'static str],
    body: &'static str,
}

fn seed_corpus(service: &MemoryService) -> HashMap<&'static str, String> {
    let seeds = [
        Seed {
            title: "Production Deployment Pipeline",
            aliases: ["release process", "how do we deploy"],
            namespace: "platform",
            note_type: NoteType::Decision,
            tags: &["deployment", "ci"],
            body: "GitHub Actions builds the image, then blue-green rollout promotes it.",
        },
        Seed {
            title: "DAT Spot Rate Source",
            aliases: ["market pricing source", "where freight rates come from"],
            namespace: "freight",
            note_type: NoteType::Preference,
            tags: &["dat", "rates"],
            body: "Use the DAT rate endpoint for current lane market pricing.",
        },
        Seed {
            title: "Runtime Root Resolution",
            aliases: ["configuration precedence order", "which memory root wins"],
            namespace: "memory",
            note_type: NoteType::Fact,
            tags: &["config"],
            body: "An explicit root wins over environment, config file, and platform defaults.",
        },
        Seed {
            title: "Postgres Advisory Writer Lock",
            aliases: ["cross process writer lock", "serialize database writers"],
            namespace: "platform",
            note_type: NoteType::Lesson,
            tags: &["postgres", "concurrency"],
            body: "Use an advisory lock around the singleton repair transaction.",
        },
        Seed {
            title: "Map Attribution Control",
            aliases: ["hide maplibre attribution", "map control override"],
            namespace: "frontend",
            note_type: NoteType::Preference,
            tags: &["maplibre"],
            body: "The shared map disables attribution while callers may override options.",
        },
        Seed {
            title: "TrueNAS Bond Migration",
            aliases: ["stale apps interface", "bond ten to bond zero"],
            namespace: "infra",
            note_type: NoteType::Lesson,
            tags: &["truenas", "network"],
            body: "Verify the default route before reselecting the existing Apps pool.",
        },
        Seed {
            title: "Legacy Vector Memory",
            aliases: ["old embeddings backend", "archived semantic store"],
            namespace: "memory",
            note_type: NoteType::Reference,
            tags: &["legacy"],
            body: "The abandoned implementation depended on embeddings and a remote service.",
        },
        Seed {
            title: "Webhook Signature Verification",
            aliases: ["incoming webhook hmac", "authenticate webhook payload"],
            namespace: "security",
            note_type: NoteType::Decision,
            tags: &["hmac"],
            body: "Verify the raw request bytes before decoding the event.",
        },
        Seed {
            title: "DAT Access Token Refresh",
            aliases: ["freight api authentication", "refresh dat credentials"],
            namespace: "freight",
            note_type: NoteType::Reference,
            tags: &["dat", "auth"],
            body: "Refresh the access token through the shared authenticated client.",
        },
        Seed {
            title: "Invoice Document Upload",
            aliases: ["attach carrier invoice", "upload billing paperwork"],
            namespace: "freight",
            note_type: NoteType::Fact,
            tags: &["documents"],
            body: "Upload the document after the load has a stable external identifier.",
        },
        Seed {
            title: "Carrier Selection Policy",
            aliases: ["shared carrier ranking phrase", "which carrier wins"],
            namespace: "dispatch",
            note_type: NoteType::Decision,
            tags: &["carrier"],
            body: "Prefer qualified carriers with current insurance and reliable tracking.",
        },
        Seed {
            title: "Carrier Selection Chat",
            aliases: ["shared carrier ranking phrase", "carrier discussion recap"],
            namespace: "dispatch",
            note_type: NoteType::SessionSummary,
            tags: &["carrier"],
            body: "A temporary recap of the carrier selection conversation.",
        },
        Seed {
            title: "Blue Dispatch Preference",
            aliases: ["paired retrieval phrase", "blue dispatch choice"],
            namespace: "dispatch",
            note_type: NoteType::Fact,
            tags: &["routing"],
            body: "Blue is the remembered dispatch option for this controlled comparison.",
        },
        Seed {
            title: "Green Dispatch Fact",
            aliases: ["paired retrieval phrase", "green dispatch choice"],
            namespace: "dispatch",
            note_type: NoteType::Fact,
            tags: &["routing"],
            body: "Green is the otherwise equivalent dispatch option.",
        },
        Seed {
            title: "Alpha Retention Rule",
            aliases: ["retention boundary phrase", "alpha cleanup policy"],
            namespace: "alpha",
            note_type: NoteType::Fact,
            tags: &["retention"],
            body: "Alpha notes retain decisions for the documented lifecycle window.",
        },
        Seed {
            title: "Beta Retention Rule",
            aliases: ["retention boundary phrase", "beta cleanup policy"],
            namespace: "beta",
            note_type: NoteType::Fact,
            tags: &["retention"],
            body: "Beta notes use a different operational namespace.",
        },
        Seed {
            title: "Atomic Note Writes",
            aliases: ["safe file replacement", "temp fsync rename"],
            namespace: "memory",
            note_type: NoteType::Decision,
            tags: &["storage"],
            body: "Write a sibling temporary file, sync it, then atomically rename it.",
        },
        Seed {
            title: "Search Result Byte Budget",
            aliases: ["snippet output ceiling", "bounded retrieval response"],
            namespace: "memory",
            note_type: NoteType::Fact,
            tags: &["search"],
            body: "Measure the serialized result and spill overflow handles into more.",
        },
        Seed {
            title: "Session Summary Decay",
            aliases: ["short lived session note", "recap lifecycle policy"],
            namespace: "memory",
            note_type: NoteType::SessionSummary,
            tags: &["lifecycle"],
            body: "Session summaries decay faster than durable decisions and preferences.",
        },
        Seed {
            title: "Plugin MCP Naming",
            aliases: ["scoped plugin tool name", "claude bundled server identity"],
            namespace: "memory",
            note_type: NoteType::Reference,
            tags: &["plugin", "mcp"],
            body: "Bundled MCP tools include both the plugin and server names.",
        },
    ];

    let mut handles = HashMap::new();
    for seed in seeds {
        let outcome = service
            .store(StoreRequest {
                title: seed.title.into(),
                body: seed.body.into(),
                aliases: seed.aliases.into_iter().map(str::to_string).collect(),
                namespace: seed.namespace.into(),
                note_type: seed.note_type,
                tags: seed.tags.iter().map(|tag| (*tag).to_string()).collect(),
                expires: None,
                mode: MergeMode::Replace,
            })
            .unwrap();
        handles.insert(
            seed.title,
            format!("{}/{}", outcome.namespace, outcome.slug),
        );
    }
    assert_eq!(handles.len(), 20);
    handles
}

fn search(service: &MemoryService, query: &str) -> memory_mcp::SearchResponse {
    service
        .search(SearchOptions {
            query: query.into(),
            ..SearchOptions::default()
        })
        .unwrap()
}

#[test]
fn twenty_note_prose_corpus_proves_end_to_end_retrieval_contract() {
    let dir = tempdir().unwrap();
    let writer = MemoryService::new(dir.path(), None);
    let handles = seed_corpus(&writer);
    let archived = handles["Legacy Vector Memory"].clone();
    writer.forget(&archived, false).unwrap();

    for _ in 0..4 {
        writer.read(&handles["Green Dispatch Fact"]).unwrap();
    }

    let retriever = Arc::new(FffRetriever::open(dir.path()).unwrap());
    let alias_hits = retriever
        .grep("market pricing source", GrepMode::Plain, None, false)
        .unwrap();
    assert_eq!(
        alias_hits[0].path,
        Path::new("freight/dat-spot-rate-source.md")
    );
    assert_eq!(
        alias_hits[0].matched,
        ContentMatch::Alias("market pricing source".into())
    );
    let typo = "configuraton precedence order";
    assert!(retriever
        .grep(typo, GrepMode::Plain, None, false)
        .unwrap()
        .is_empty());
    assert_eq!(
        retriever.grep(typo, GrepMode::Fuzzy, None, false).unwrap()[0].path,
        Path::new("memory/runtime-root-resolution.md")
    );

    let service = MemoryService::new(dir.path(), Some(retriever));

    let title = search(&service, "Production Deployment Pipeline");
    assert_eq!(
        title.results[0].handle,
        handles["Production Deployment Pipeline"]
    );
    let alias = search(&service, "market pricing source");
    assert_eq!(alias.results[0].handle, handles["DAT Spot Rate Source"]);
    assert_eq!(alias.results[0].why, "alias:market pricing source");
    let typo = search(&service, typo);
    assert_eq!(typo.results[0].handle, handles["Runtime Root Resolution"]);
    assert!(typo.stages_run.iter().any(|stage| stage == "grep_fuzzy"));

    let scoped = service
        .search(SearchOptions {
            query: "retention boundary phrase".into(),
            namespace: Some("alpha".into()),
            ..SearchOptions::default()
        })
        .unwrap();
    assert_eq!(scoped.results[0].handle, handles["Alpha Retention Rule"]);
    assert!(scoped
        .results
        .iter()
        .all(|hit| hit.handle.starts_with("alpha/")));

    let hidden = search(&service, "archived semantic store");
    assert!(hidden.results.is_empty(), "{hidden:?}");
    let visible = service
        .search(SearchOptions {
            query: "archived semantic store".into(),
            include_archived: true,
            ..SearchOptions::default()
        })
        .unwrap();
    assert_eq!(visible.results[0].handle, archived);
    assert!(service
        .read(&visible.results[0].handle)
        .unwrap()
        .body
        .contains("embeddings"));

    let type_boosted = search(&service, "shared carrier ranking phrase");
    assert_eq!(
        type_boosted.results[0].handle,
        handles["Carrier Selection Policy"]
    );

    let reinforced = search(&service, "paired retrieval phrase");
    assert_eq!(reinforced.results[0].handle, handles["Green Dispatch Fact"]);
    assert!(reinforced.results[0].why.contains("frecency"));

    let empty = search(&service, "nonexistent quokka semaphore");
    assert!(empty.results.is_empty());
    let hint = empty.empty_hint.unwrap();
    assert!(hint.contains("stages tried"));
    assert!(hint.contains("try broader terms"));

    let deploy = search(&service, "how do we deploy");
    let read = service.read(&deploy.results[0].handle).unwrap();
    assert_eq!(
        deploy.results[0].handle,
        handles["Production Deployment Pipeline"]
    );
    assert!(read.body.contains("blue-green rollout"));

    let automatic = search(
        &service,
        "__fff_memory_user_prompt__:Can you remind me what our release process is before I change the production deployment workflow?",
    );
    assert_eq!(
        automatic.results[0].handle,
        handles["Production Deployment Pipeline"]
    );
}
