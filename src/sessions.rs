//! Server-side session list helpers.
//!
//! Conversations live on the DeepSeek backend and are listed with
//! `chat_session/fetch_page`. This module only holds the data shape returned
//! by that endpoint plus a few pure helpers (resolving a user-supplied
//! reference, formatting timestamps). No local state is kept.

use serde::Deserialize;

#[derive(Deserialize, Clone, Debug, Default)]
pub struct SessionEntry {
    pub id: String,
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub updated_at: f64,
}

impl SessionEntry {
    /// Human-readable title, falling back to a placeholder.
    pub fn display_title(&self) -> &str {
        if self.title.trim().is_empty() {
            "<untitled>"
        } else {
            &self.title
        }
    }

    /// `YYYY-MM-DD HH:MM` in local time, or `-` when unknown.
    pub fn when(&self) -> String {
        if self.updated_at <= 0.0 {
            return "-".to_string();
        }
        chrono::DateTime::from_timestamp(self.updated_at as i64, 0)
            .map(|d| {
                d.with_timezone(&chrono::Local)
                    .format("%Y-%m-%d %H:%M")
                    .to_string()
            })
            .unwrap_or_else(|| self.updated_at.to_string())
    }

    /// Short id prefix for display.
    pub fn short_id(&self) -> &str {
        &self.id[..self.id.len().min(8)]
    }
}

/// Resolve a user-supplied reference to an entry in `list`.
///
/// A pure number is a 1-based index. Anything else matches the full id, then a
/// unique id prefix. Returns `None` for empty, out-of-range, or ambiguous
/// references.
pub fn resolve<'a>(list: &'a [SessionEntry], arg: &str) -> Option<&'a SessionEntry> {
    let arg = arg.trim();
    if arg.is_empty() {
        return None;
    }
    if let Ok(n) = arg.parse::<usize>() {
        if n >= 1 {
            return list.get(n - 1);
        }
    }
    if let Some(e) = list.iter().find(|e| e.id == arg) {
        return Some(e);
    }
    let matches: Vec<&SessionEntry> = list.iter().filter(|e| e.id.starts_with(arg)).collect();
    if matches.len() == 1 {
        Some(matches[0])
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Vec<SessionEntry> {
        vec![
            SessionEntry {
                id: "abcdef1234567890".into(),
                title: "first".into(),
                updated_at: 1_700_000_000.0,
            },
            SessionEntry {
                id: "ffff000011112222".into(),
                title: String::new(),
                updated_at: 0.0,
            },
        ]
    }

    #[test]
    fn resolves_by_index() {
        let l = sample();
        assert_eq!(resolve(&l, "1").unwrap().id, "abcdef1234567890");
        assert_eq!(resolve(&l, "2").unwrap().id, "ffff000011112222");
        assert!(resolve(&l, "0").is_none());
        assert!(resolve(&l, "3").is_none());
    }

    #[test]
    fn resolves_by_full_id_and_prefix() {
        let l = sample();
        assert_eq!(resolve(&l, "abcdef1234567890").unwrap().title, "first");
        assert_eq!(resolve(&l, "abcd").unwrap().id, "abcdef1234567890");
        assert!(resolve(&l, "ffff0000").is_some());
        assert!(resolve(&l, "zzz").is_none());
        assert!(resolve(&l, "   ").is_none());
    }

    #[test]
    fn ambiguous_prefix_is_rejected() {
        let l = vec![
            SessionEntry {
                id: "aaa1".into(),
                ..Default::default()
            },
            SessionEntry {
                id: "aaa2".into(),
                ..Default::default()
            },
        ];
        assert!(resolve(&l, "aaa").is_none());
        assert!(resolve(&l, "aaa1").is_some());
    }

    #[test]
    fn display_falls_back_for_blank_title() {
        let e = SessionEntry {
            title: "  ".into(),
            ..Default::default()
        };
        assert_eq!(e.display_title(), "<untitled>");
    }
}
