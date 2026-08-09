//! `fff-memory` — CLI for lifecycle and (later) tool-mirroring commands.
//!
//! Spec: `04-mcp-interface.toml` [cli], `05-lifecycle.toml` doctor/reindex.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use memory_core::{run_doctor, DoctorOptions, MemoryStore, Retriever};
use memory_index::FffRetriever;

/// Persistent agent memory over lexical search (fff).
#[derive(Debug, Parser)]
#[command(name = "fff-memory", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Report decay candidates, alias quality, unresolved links; optionally archive + compact.
    Doctor {
        /// Memory root directory (notes live here).
        #[arg(long, env = "FFF_MEMORY_ROOT", default_value = ".")]
        root: PathBuf,
        /// Archive decay candidates and compact the access log (never hard-deletes).
        #[arg(long, default_value_t = false)]
        apply: bool,
        /// Open the index to report cold/ready state (optional; slower).
        #[arg(long, default_value_t = false)]
        with_index: bool,
    },
    /// Drop/rescan the fff index from disk (files remain the source of truth).
    Reindex {
        /// Memory root directory.
        #[arg(long, env = "FFF_MEMORY_ROOT", default_value = ".")]
        root: PathBuf,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    match cli.command {
        Command::Doctor {
            root,
            apply,
            with_index,
        } => {
            let retriever = if with_index {
                Some(FffRetriever::open(&root).map_err(|e| e.to_string())?)
            } else {
                None
            };
            let report = run_doctor(
                &root,
                retriever.as_ref().map(|r| r as &dyn memory_core::Retriever),
                DoctorOptions { apply },
            )?;
            print!("{report}");
            Ok(())
        }
        Command::Reindex { root } => {
            // Ensure root exists so the walker has a path
            let store = MemoryStore::new(&root);
            std::fs::create_dir_all(store.root())?;
            let r = FffRetriever::open(&root).map_err(|e| e.to_string())?;
            r.reindex().map_err(|e| e.to_string())?;
            println!("reindex complete for {}", root.display());
            Ok(())
        }
    }
}
