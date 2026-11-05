//! Per-conversation goal persistence.
//!
//! Conversations live on the DeepSeek backend, which has nowhere to keep the
//! user's active goal.  A small local map from chat-session id to goal text
//! bridges restarts: `/goal <text>` records the goal for the current
//! conversation, and `/resume` reads it back so work continues where it left
//! off.
//!
//! Storage is best effort.  A missing or unreadable file means "no saved
//! goals"; failing to write one must never abort the session, it only costs
//! persistence.

use anyhow::{Context, Result};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Path to the goal store.
///
/// Lookup order:
///   1. `$DS_GOALS_FILE` (explicit override)
///   2. `$XDG_DATA_HOME/deepseek-web-cli/goals.json`
///   3. `~/.local/share/deepseek-web-cli/goals.json`
fn store_path() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("DS_GOALS_FILE") {
        if !p.is_empty() {
            return Some(PathBuf::from(p));
        }
    }
    if let Some(dir) = std::env::var_os("XDG_DATA_HOME") {
        if !dir.is_empty() {
            return Some(
                PathBuf::from(dir)
                    .join("deepseek-web-cli")
                    .join("goals.json"),
            );
        }
    }
    std::env::var_os("HOME").map(|h| {
        PathBuf::from(h)
            .join(".local")
            .join("share")
            .join("deepseek-web-cli")
            .join("goals.json")
    })
}

/// Read the map at `path`, treating anything unreadable as empty.
///
/// The file is user data, not config: a corrupt or half-written file must not
/// stop the CLI from starting, so parse failures fall back to an empty map.
/// Blank goals are dropped on load, which also cleans up older files.
fn load_from(path: &Path) -> BTreeMap<String, String> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return BTreeMap::new();
    };
    let parsed: BTreeMap<String, String> = serde_json::from_str(&text).unwrap_or_default();
    parsed
        .into_iter()
        .filter(|(id, goal)| !id.is_empty() && !goal.trim().is_empty())
        .collect()
}

/// Write `map` to `path` atomically (temp file + rename) so an interrupted
/// write cannot truncate the existing store.
fn save_to(path: &Path, map: &BTreeMap<String, String>) -> Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    let text = serde_json::to_string_pretty(map)?;
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, text).with_context(|| format!("writing {}", tmp.display()))?;
    std::fs::rename(&tmp, path).with_context(|| format!("renaming {}", tmp.display()))?;
    Ok(())
}

/// Apply `f` to the stored map and persist the result.
fn update<F: FnOnce(&mut BTreeMap<String, String>)>(f: F) {
    let Some(path) = store_path() else { return };
    let mut map = load_from(&path);
    f(&mut map);
    // Persistence is a convenience; a read-only home directory should not
    // break the session.
    let _ = save_to(&path, &map);
}

/// The goal saved for `session_id`, if any.
pub fn get(session_id: &str) -> Option<String> {
    if session_id.is_empty() {
        return None;
    }
    let path = store_path()?;
    load_from(&path).remove(session_id)
}

/// Remember `goal` for `session_id`.
pub fn set(session_id: &str, goal: &str) {
    let goal = goal.trim();
    if session_id.is_empty() || goal.is_empty() {
        return;
    }
    let id = session_id.to_string();
    let goal = goal.to_string();
    update(|map| {
        map.insert(id, goal);
    });
}

/// Forget any goal saved for `session_id`.
pub fn clear(session_id: &str) {
    if session_id.is_empty() {
        return;
    }
    let id = session_id.to_string();
    update(|map| {
        map.remove(&id);
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        let unique = format!(
            "ds-goals-{}-{}-{}",
            std::process::id(),
            name,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        p.push(unique);
        p.push("goals.json");
        p
    }

    #[test]
    fn missing_file_is_empty() {
        let path = temp_path("missing");
        assert!(load_from(&path).is_empty());
    }

    #[test]
    fn roundtrip_persists_map() {
        let path = temp_path("roundtrip");
        let mut map = BTreeMap::new();
        map.insert("s1".to_string(), "fix the bug".to_string());
        save_to(&path, &map).unwrap();
        assert_eq!(load_from(&path).get("s1"), Some(&"fix the bug".to_string()));
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn save_overwrites_and_supports_removal() {
        let path = temp_path("overwrite");
        let mut map = BTreeMap::new();
        map.insert("s1".to_string(), "one".to_string());
        map.insert("s2".to_string(), "two".to_string());
        save_to(&path, &map).unwrap();
        map.remove("s1");
        map.insert("s2".to_string(), "two-updated".to_string());
        save_to(&path, &map).unwrap();
        let back = load_from(&path);
        assert_eq!(back.len(), 1);
        assert_eq!(back.get("s2"), Some(&"two-updated".to_string()));
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn corrupt_file_is_treated_as_empty() {
        let path = temp_path("corrupt");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, "{not json").unwrap();
        assert!(load_from(&path).is_empty());
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    /// Serializes the tests that touch process-wide env vars.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Point `DS_GOALS_FILE` at a fresh temp file for the duration of a test.
    struct TempStore {
        _guard: std::sync::MutexGuard<'static, ()>,
        prev: Option<std::ffi::OsString>,
        path: PathBuf,
    }

    impl TempStore {
        fn new(name: &str) -> Self {
            let guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            let path = temp_path(name);
            let prev = std::env::var_os("DS_GOALS_FILE");
            std::env::set_var("DS_GOALS_FILE", &path);
            Self {
                _guard: guard,
                prev,
                path,
            }
        }
    }

    impl Drop for TempStore {
        fn drop(&mut self) {
            match &self.prev {
                Some(v) => std::env::set_var("DS_GOALS_FILE", v),
                None => std::env::remove_var("DS_GOALS_FILE"),
            }
            std::fs::remove_dir_all(self.path.parent().unwrap()).ok();
        }
    }

    /// The end-to-end contract: a goal written for a conversation is still
    /// there when a later process asks for it, and clearing removes it.
    #[test]
    fn survives_a_simulated_restart() {
        let store = TempStore::new("restart");

        // First "process": set the goal for a conversation.
        set("chat-1", "persist goal after session resume");
        set("chat-2", "unrelated work");
        assert_eq!(
            get("chat-1").as_deref(),
            Some("persist goal after session resume")
        );

        // Second "process" (nothing in memory, reading from disk): the goal
        // for the resumed conversation is available.
        assert_eq!(
            load_from(&store.path).get("chat-1").map(String::as_str),
            Some("persist goal after session resume")
        );
        assert_eq!(
            get("chat-1").as_deref(),
            Some("persist goal after session resume")
        );
        assert_eq!(get("chat-2").as_deref(), Some("unrelated work"));

        // Completion clears only the finished conversation's goal.
        clear("chat-1");
        assert_eq!(get("chat-1"), None);
        assert_eq!(get("chat-2").as_deref(), Some("unrelated work"));
    }

    #[test]
    fn set_ignores_blank_input() {
        let _store = TempStore::new("blank-set");
        set("chat-1", "   ");
        set("", "goal with no conversation");
        assert_eq!(get("chat-1"), None);
        assert_eq!(get(""), None);
    }

    #[test]
    fn unknown_conversation_has_no_goal() {
        let _store = TempStore::new("unknown");
        set("chat-1", "a goal");
        assert_eq!(get("chat-missing"), None);
    }

    #[test]
    fn blank_goals_are_dropped_on_load() {
        let path = temp_path("blank");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, r#"{"a":"  ","b":"keep me"}"#).unwrap();
        let back = load_from(&path);
        assert_eq!(back.len(), 1);
        assert_eq!(back.get("b"), Some(&"keep me".to_string()));
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }
}
