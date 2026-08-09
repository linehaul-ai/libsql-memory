use std::env;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

const CONFIG_FIX: &str = r#"use JSON like {"root":"/path"}"#;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    root: PathBuf,
}

#[derive(Debug)]
pub struct ResolveError(String);

impl fmt::Display for ResolveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ResolveError {}

pub fn resolve_root(
    explicit_root: Option<PathBuf>,
    explicit_config: Option<PathBuf>,
    project: Option<PathBuf>,
) -> Result<PathBuf, ResolveError> {
    let selected_config = explicit_config
        .or_else(|| env_path("FFF_MEMORY_CONFIG"))
        .map(|path| read_selected_config(&path).map(|root| (path, root)))
        .transpose()?;

    if let Some(project) = project {
        return Ok(project.join(".memory"));
    }
    if let Some(root) = explicit_root.or_else(|| env_path("FFF_MEMORY_ROOT")) {
        return Ok(root);
    }

    if let Some((_path, root)) = selected_config {
        return Ok(root);
    }

    let config_path = default_config_path()?;
    match fs::read_to_string(&config_path) {
        Ok(contents) => parse_config(&config_path, &contents),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => default_root(),
        Err(error) => Err(ResolveError(format!(
            "cannot read config {}: {error}; check permissions or choose another with --config PATH",
            config_path.display()
        ))),
    }
}

fn read_selected_config(path: &Path) -> Result<PathBuf, ResolveError> {
    let contents = fs::read_to_string(path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            ResolveError(format!(
                "selected config {} does not exist; create the file or choose another with --config PATH",
                path.display()
            ))
        } else {
            ResolveError(format!(
                "cannot read config {}: {error}; check permissions or choose another with --config PATH",
                path.display()
            ))
        }
    })?;
    parse_config(path, &contents)
}

fn parse_config(path: &Path, contents: &str) -> Result<PathBuf, ResolveError> {
    serde_json::from_str::<Config>(contents)
        .map(|config| config.root)
        .map_err(|error| {
            ResolveError(format!(
                "invalid config {}: {error}; {CONFIG_FIX}",
                path.display()
            ))
        })
}

fn default_config_path() -> Result<PathBuf, ResolveError> {
    if let Some(root) = env_path("XDG_CONFIG_HOME") {
        return Ok(root.join("fff-memory/config.json"));
    }
    home().map(|home| home.join(".config/fff-memory/config.json"))
}

fn default_root() -> Result<PathBuf, ResolveError> {
    if let Some(root) = env_path("XDG_DATA_HOME") {
        return Ok(root.join("fff-memory"));
    }
    home().map(|home| home.join(".local/share/fff-memory"))
}

fn home() -> Result<PathBuf, ResolveError> {
    env_path("HOME").ok_or_else(|| {
        ResolveError(
            "cannot resolve fff-memory paths: HOME is unset; set HOME, XDG_CONFIG_HOME, and XDG_DATA_HOME, or pass --root PATH".into(),
        )
    })
}

fn env_path(key: &str) -> Option<PathBuf> {
    env::var_os(key)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}
