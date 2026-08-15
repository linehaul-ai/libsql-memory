//! Filesystem containment checks for paths below a memory root.

use std::fs;
use std::path::{Component, Path, PathBuf};

use crate::{Error, Result};

/// Join a relative path to `root`, rejecting traversal and existing symlink components.
pub fn safe_join(root: &Path, relative: &Path) -> Result<PathBuf> {
    // ponytail: static checks assume a locally trusted root; use openat/openat2
    // directory handles if hostile concurrent path replacement becomes a requirement.
    let mut path = root.to_path_buf();
    for component in relative.components() {
        let Component::Normal(segment) = component else {
            return Err(Error::validation(
                "path",
                format!(
                    "{} must be relative to the memory root without traversal",
                    relative.display()
                ),
            ));
        };
        path.push(segment);
        match fs::symlink_metadata(&path) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                return Err(Error::validation(
                    "path",
                    format!(
                        "symlink component {} is not allowed below the memory root; replace it with a real directory or file",
                        path.display()
                    ),
                ));
            }
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(Error::io(&path, error)),
        }
    }
    Ok(path)
}
