//! Wikilink extraction from note bodies (spec 01 `[links]`).

/// A `[[…]]` link found in note body markdown.
///
/// Unresolved targets are legal — they mark a memory worth writing, not an error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Wikilink {
    /// Full matched text including brackets, e.g. `[[ns/slug]]`.
    pub raw: String,
    /// Interior target as written, e.g. `ns/slug` or `slug`.
    pub target: String,
    /// Namespace prefix when target contains `/`; `None` means same-namespace.
    pub namespace: Option<String>,
    /// Final path segment (the note slug).
    pub slug: String,
}

/// Find all `[[slug]]` / `[[ns/slug]]` wikilinks in `body`, in document order.
///
/// Nested namespaces use `/` (e.g. `[[linehaul/tms/foo]]` → namespace
/// `linehaul/tms`, slug `foo`). Empty `[[]]` targets are skipped.
pub fn extract_wikilinks(body: &str) -> Vec<Wikilink> {
    let mut links = Vec::new();
    let bytes = body.as_bytes();
    let mut i = 0;

    while i + 3 < bytes.len() {
        if bytes[i] == b'[' && bytes[i + 1] == b'[' {
            if let Some(end) = find_close(bytes, i + 2) {
                let interior = &body[i + 2..end];
                // Reject nested brackets inside the target
                if !interior.contains('[') && !interior.contains(']') {
                    let target = interior.trim();
                    if !target.is_empty() {
                        let (namespace, slug) = split_target(target);
                        if !slug.is_empty() {
                            links.push(Wikilink {
                                raw: body[i..end + 2].to_string(),
                                target: target.to_string(),
                                namespace,
                                slug,
                            });
                        }
                    }
                }
                i = end + 2;
                continue;
            }
        }
        i += 1;
    }

    links
}

fn find_close(bytes: &[u8], from: usize) -> Option<usize> {
    let mut j = from;
    while j + 1 < bytes.len() {
        if bytes[j] == b']' && bytes[j + 1] == b']' {
            return Some(j);
        }
        j += 1;
    }
    None
}

fn split_target(target: &str) -> (Option<String>, String) {
    match target.rfind('/') {
        Some(i) => {
            let ns = &target[..i];
            let slug = &target[i + 1..];
            if ns.is_empty() {
                (None, slug.to_string())
            } else {
                (Some(ns.to_string()), slug.to_string())
            }
        }
        None => (None, target.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_same_namespace() {
        let body = "Related: [[docker-registry-auth]]\n";
        let links = extract_wikilinks(body);
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].slug, "docker-registry-auth");
        assert_eq!(links[0].namespace, None);
        assert_eq!(links[0].target, "docker-registry-auth");
        assert_eq!(links[0].raw, "[[docker-registry-auth]]");
    }

    #[test]
    fn extract_qualified() {
        let body = "See [[linehaul/tms/foo]] and [[a/b/c]].";
        let links = extract_wikilinks(body);
        assert_eq!(links.len(), 2);
        assert_eq!(links[0].namespace.as_deref(), Some("linehaul/tms"));
        assert_eq!(links[0].slug, "foo");
        assert_eq!(links[1].namespace.as_deref(), Some("a/b"));
        assert_eq!(links[1].slug, "c");
    }

    #[test]
    fn extract_multiple_and_empty() {
        assert!(extract_wikilinks("no links here").is_empty());
        assert!(extract_wikilinks("[[]]").is_empty());
        let links = extract_wikilinks("[[one]] [[two]]");
        assert_eq!(links.len(), 2);
    }
}
