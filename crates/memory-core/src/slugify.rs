//! Slug and namespace identity helpers (specs 01 + 02).

use crate::error::{Error, Result};

/// Maximum length of a note slug (filename stem), per write-path rules.
pub const SLUG_MAX_LEN: usize = 60;

/// Derive a kebab-case ASCII slug from a title.
///
/// - Lowercases ASCII letters
/// - Non-alphanumeric runs become a single `-`
/// - Non-ASCII codepoints are dropped
/// - Truncated to [`SLUG_MAX_LEN`], preferring a cut at a `-` boundary
///
/// Returns [`Error::InvalidSlug`] if nothing usable remains.
pub fn slugify(title: &str) -> Result<String> {
    let mut out = String::with_capacity(title.len().min(SLUG_MAX_LEN));
    let mut prev_dash = true; // treat start as if after a dash (trim leading)

    for ch in title.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            prev_dash = false;
        } else if ch.is_ascii() && !prev_dash {
            // punctuation / whitespace → dash
            out.push('-');
            prev_dash = true;
        }
        // non-ascii: drop silently
        if out.len() >= SLUG_MAX_LEN {
            break;
        }
    }

    // Trim trailing dash from collapse or truncation
    while out.ends_with('-') {
        out.pop();
    }

    if out.is_empty() {
        return Err(Error::InvalidSlug {
            title: title.to_string(),
        });
    }

    // If we hit max mid-token, try to cut back to last full segment
    if out.len() == SLUG_MAX_LEN {
        if let Some(cut) = out.rfind('-') {
            // Only cut if we keep a meaningful prefix
            if cut >= SLUG_MAX_LEN / 2 {
                out.truncate(cut);
            }
        }
    }

    while out.ends_with('-') {
        out.pop();
    }

    if out.is_empty() {
        return Err(Error::InvalidSlug {
            title: title.to_string(),
        });
    }

    Ok(out)
}

/// Validate a namespace path relative to the memory root.
///
/// Empty string is the root namespace (allowed). Nested namespaces use `/`
/// separators. Rejects absolute paths, `..`, `.`, empty segments, and
/// backslashes (normalize to `/` on the store side before calling).
pub fn validate_namespace(namespace: &str) -> Result<()> {
    if namespace.is_empty() {
        return Ok(());
    }
    if namespace.trim() != namespace {
        return Err(Error::InvalidNamespace {
            path: namespace.to_string(),
            reason: "must not have leading or trailing whitespace".into(),
        });
    }

    if namespace.starts_with('/') || namespace.starts_with('\\') {
        return Err(Error::InvalidNamespace {
            path: namespace.to_string(),
            reason: "must be relative to the memory root (no leading slash)".into(),
        });
    }

    if namespace.contains('\\') {
        return Err(Error::InvalidNamespace {
            path: namespace.to_string(),
            reason: "use `/` separators, not backslashes".into(),
        });
    }

    if namespace.contains('\0') {
        return Err(Error::InvalidNamespace {
            path: namespace.to_string(),
            reason: "must not contain NUL bytes".into(),
        });
    }

    for segment in namespace.split('/') {
        if segment.is_empty() {
            return Err(Error::InvalidNamespace {
                path: namespace.to_string(),
                reason: "empty path segment (no leading/trailing or double `/`)".into(),
            });
        }
        if segment == ".." || segment == "." {
            return Err(Error::InvalidNamespace {
                path: namespace.to_string(),
                reason: "path traversal components (`.` / `..`) are not allowed".into(),
            });
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn slugify_preserves_identity_invariants(title in any::<String>()) {
            let has_ascii_alnum = title.chars().any(|ch| ch.is_ascii_alphanumeric());
            match slugify(&title) {
                Ok(slug) => {
                    prop_assert!(has_ascii_alnum);
                    prop_assert!(!slug.is_empty());
                    prop_assert!(slug.len() <= SLUG_MAX_LEN);
                    prop_assert!(slug.is_ascii());
                    prop_assert!(!slug.starts_with('-'));
                    prop_assert!(!slug.ends_with('-'));
                    prop_assert!(!slug.contains("--"));
                    prop_assert!(slug.bytes().all(|byte| byte.is_ascii_lowercase()
                        || byte.is_ascii_digit()
                        || byte == b'-'));
                    prop_assert_eq!(slugify(&slug).unwrap(), slug);
                }
                Err(Error::InvalidSlug { .. }) => prop_assert!(!has_ascii_alnum),
                Err(error) => prop_assert!(false, "unexpected error: {error}"),
            }
        }
    }

    #[test]
    fn slugify_basic() {
        assert_eq!(
            slugify("Deploy pipeline uses GitHub Actions + Docker").unwrap(),
            "deploy-pipeline-uses-github-actions-docker"
        );
        assert_eq!(slugify("Hello World").unwrap(), "hello-world");
        assert_eq!(slugify("  spaced  out  ").unwrap(), "spaced-out");
    }

    #[test]
    fn slugify_max_len() {
        let long = "a".repeat(100);
        let s = slugify(&long).unwrap();
        assert!(s.len() <= SLUG_MAX_LEN);
        assert_eq!(s, "a".repeat(SLUG_MAX_LEN));
    }

    #[test]
    fn slugify_truncates_at_dash_boundary() {
        // Build a title whose slug would exceed max with a dash near the end
        let title = format!("{}-tail", "word-".repeat(20));
        let s = slugify(&title).unwrap();
        assert!(s.len() <= SLUG_MAX_LEN);
        assert!(!s.ends_with('-'));
    }

    #[test]
    fn slugify_strips_unicode() {
        assert_eq!(slugify("café résumé").unwrap(), "caf-rsum");
        // pure non-ascii fails
        assert!(slugify("日本語").is_err());
    }

    #[test]
    fn slugify_empty_title() {
        assert!(slugify("").is_err());
        assert!(slugify("!!!").is_err());
        assert!(slugify("---").is_err());
    }

    #[test]
    fn namespace_ok() {
        validate_namespace("").unwrap();
        validate_namespace("linehaul").unwrap();
        validate_namespace("linehaul/tms").unwrap();
        validate_namespace("a/b/c").unwrap();
    }

    #[test]
    fn namespace_rejects_traversal_and_absolute() {
        assert!(validate_namespace("../x").is_err());
        assert!(validate_namespace("a/../b").is_err());
        assert!(validate_namespace("/abs").is_err());
        assert!(validate_namespace("a//b").is_err());
        assert!(validate_namespace("a/").is_err());
        assert!(validate_namespace("./x").is_err());
        assert!(validate_namespace(r"a\b").is_err());
        assert!(validate_namespace(" proj").is_err());
        assert!(validate_namespace("proj ").is_err());
    }
}
