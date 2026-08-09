//! `fff-memory` — CLI mirrors and stdio MCP server.

mod config;

use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;

use clap::{Parser, Subcommand, ValueEnum};
use memory_core::{run_doctor, DoctorOptions, MergeMode, NoteType, Retriever, StoreAction};
use memory_index::FffRetriever;
use memory_mcp::{
    serve_stdio, ForgetAction, MemoryServer, MemoryService, SearchOptions, StoreRequest,
    BUDGET_BYTES_DEFAULT, SEARCH_LIMIT_DEFAULT,
};
use serde::Serialize;
use time::macros::format_description;
use time::Date;

#[derive(Debug, Parser)]
#[command(
    name = "fff-memory",
    version,
    about = "Persistent agent memory over lexical search (fff)",
    disable_help_subcommand = true
)]
struct Cli {
    /// Memory root directory.
    #[arg(long, global = true, value_name = "PATH", conflicts_with = "project")]
    root: Option<PathBuf>,

    /// JSON config file containing {"root":"/path"}.
    #[arg(long, global = true, value_name = "PATH")]
    config: Option<PathBuf>,

    /// Use PATH/.memory (omit PATH after the subcommand to use the current directory).
    #[arg(
        long,
        global = true,
        value_name = "PATH",
        num_args = 0..=1,
        default_missing_value = "."
    )]
    project: Option<PathBuf>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Serve the five memory tools over MCP stdio.
    Serve,
    /// Store or update a memory note.
    Store {
        #[arg(long)]
        title: String,
        #[arg(long)]
        body: String,
        #[arg(long = "alias", required = true)]
        aliases: Vec<String>,
        #[arg(long, default_value = "default")]
        namespace: String,
        #[arg(long = "type", value_enum)]
        note_type: CliNoteType,
        #[arg(long = "tag")]
        tags: Vec<String>,
        #[arg(long, value_parser = parse_date)]
        expires: Option<Date>,
        #[arg(long, value_enum, default_value_t = CliMode::Append)]
        mode: CliMode,
    },
    /// Search memory notes.
    Search {
        query: String,
        #[arg(long)]
        namespace: Option<String>,
        #[arg(long, default_value_t = SEARCH_LIMIT_DEFAULT)]
        limit: usize,
        #[arg(long, default_value_t = BUDGET_BYTES_DEFAULT)]
        budget: usize,
        #[arg(long)]
        include_archived: bool,
    },
    /// Read one note by handle.
    Read { handle: String },
    /// Archive or hard-delete one note by handle.
    Forget {
        handle: String,
        #[arg(long)]
        hard: bool,
    },
    /// Report store health as JSON.
    Stats {
        #[arg(long)]
        namespace: Option<String>,
    },
    /// Report lifecycle candidates; optionally archive and compact.
    Doctor {
        #[arg(long)]
        apply: bool,
        #[arg(long)]
        with_index: bool,
    },
    /// Drop and rebuild only the disposable fff index databases.
    Reindex,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliNoteType {
    Fact,
    Decision,
    Preference,
    Lesson,
    Reference,
    SessionSummary,
}

impl From<CliNoteType> for NoteType {
    fn from(value: CliNoteType) -> Self {
        match value {
            CliNoteType::Fact => Self::Fact,
            CliNoteType::Decision => Self::Decision,
            CliNoteType::Preference => Self::Preference,
            CliNoteType::Lesson => Self::Lesson,
            CliNoteType::Reference => Self::Reference,
            CliNoteType::SessionSummary => Self::SessionSummary,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, ValueEnum)]
enum CliMode {
    #[default]
    Append,
    Replace,
}

impl From<CliMode> for MergeMode {
    fn from(value: CliMode) -> Self {
        match value {
            CliMode::Append => Self::Append,
            CliMode::Replace => Self::Replace,
        }
    }
}

#[tokio::main]
async fn main() -> ExitCode {
    match run(Cli::parse()).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

async fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    let root = config::resolve_root(cli.root, cli.config, cli.project)?;
    std::fs::create_dir_all(&root)?;

    match cli.command {
        Command::Serve => {
            let service = service_with_optional_retriever(&root);
            serve_stdio(MemoryServer::new(service)).await?;
        }
        Command::Store {
            title,
            body,
            aliases,
            namespace,
            note_type,
            tags,
            expires,
            mode,
        } => {
            if aliases.len() < 2 {
                return Err("aliases require at least 2 values; repeat --alias VALUE".into());
            }
            let output = service_with_optional_retriever(&root).store(StoreRequest {
                title,
                body,
                aliases,
                namespace,
                note_type: note_type.into(),
                tags,
                expires,
                mode: mode.into(),
            })?;
            write_json(&serde_json::json!({
                "slug": output.slug,
                "namespace": output.namespace,
                "action": match output.action {
                    StoreAction::Created => "created",
                    StoreAction::Updated => "updated",
                },
                "dedup_hit": output.dedup_hit,
            }))?;
        }
        Command::Search {
            query,
            namespace,
            limit,
            budget,
            include_archived,
        } => write_json(
            &service_with_optional_retriever(&root).search(SearchOptions {
                query,
                namespace,
                limit,
                budget_bytes: budget,
                include_archived,
            })?,
        )?,
        Command::Read { handle } => write_json(&MemoryService::new(&root, None).read(&handle)?)?,
        Command::Forget { handle, hard } => {
            let output = MemoryService::new(&root, None).forget(&handle, hard)?;
            let action = match output.action {
                ForgetAction::Archived => "archived",
                ForgetAction::Deleted => "deleted",
            };
            write_json(&serde_json::json!({ "handle": output.handle, "action": action }))?;
        }
        Command::Stats { namespace } => {
            write_json(&service_with_optional_retriever(&root).stats(namespace.as_deref())?)?
        }
        Command::Doctor { apply, with_index } => {
            let retriever = if with_index {
                Some(FffRetriever::open(&root)?)
            } else {
                None
            };
            let report = run_doctor(
                &root,
                retriever.as_ref().map(|value| value as &dyn Retriever),
                DoctorOptions { apply },
            )?;
            print!("{report}");
        }
        Command::Reindex => {
            let _retriever = FffRetriever::rebuild(&root)?;
            println!("reindex complete for {}", root.display());
        }
    }
    Ok(())
}

fn service_with_optional_retriever(root: &Path) -> MemoryService {
    let retriever = match FffRetriever::open(root) {
        Ok(retriever) => Some(Arc::new(retriever) as Arc<dyn Retriever>),
        Err(error) => {
            eprintln!(
                "warning: retrieval unavailable for {}: {error}; continuing without index",
                root.display()
            );
            None
        }
    };
    MemoryService::new(root, retriever)
}

fn write_json(value: &impl Serialize) -> Result<(), serde_json::Error> {
    let mut stdout = io::stdout().lock();
    serde_json::to_writer(&mut stdout, value)?;
    stdout.write_all(b"\n").map_err(serde_json::Error::io)
}

fn parse_date(value: &str) -> Result<Date, String> {
    Date::parse(value, format_description!("[year]-[month]-[day]"))
        .map_err(|error| format!("invalid date {value:?}: {error}; use YYYY-MM-DD"))
}
