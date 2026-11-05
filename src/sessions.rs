//! Local session store so conversations can be resumed later.
//!
//! Each entry remembers the DeepSeek chat session id, a human label, when it
//! was last touched, and the last assistant message id (the conversation
//! "tip") so a resumed conversation can keep threading correctly.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct SessionEntry {
    pub id: String,
    #[serde(default)]
    pub label: String,
    #[serde(default)]
    pub updated_at: u64,
    #[serde(default)]
    pub last_message_id: Option<serde_json::Value>,
}

pub fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn store_path() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("DS_SESSIONS_FILE") {
        if !p.is_empty() {
            return Some(PathBuf::from(p));
        }
    }
    if let Some(dir) = std::env::var_os("XDG_DATA_HOME") {
        if !dir.is_empty() {
            return Some(
                PathBuf::from(dir)
                    .join("deepseek-web-cli")
                    .join("sessions.json"),
            );
        }
    }
    std::env::var_os("HOME").map(|h| {
        PathBuf::from(h)
            .join(".local")
            .join("share")
            .join("deepseek-web-cli")
            .join("sessions.json")
    })
}

fn load() -> Result<Vec<SessionEntry>> {
    let Some(path) = store_path() else {
        return Ok(Vec::new());
    };
    let contents = match std::fs::read_to_string(&path) {
        Ok(c) => c,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
    };
    if contents.trim().is_empty() {
        return Ok(Vec::new());
    }
    serde_json::from_str(&contents).with_context(|| format!("parsing {}", path.display()))
}

fn save(entries: &[SessionEntry]) -> Result<()> {
    let Some(path) = store_path() else {
        return Ok(());
    };
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    let text = serde_json::to_string_pretty(entries)?;
    std::fs::write(&path, text).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

pub fn list() -> Result<Vec<SessionEntry>> {
    let mut v = load()?;
    v.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
    Ok(v)
}

pub fn upsert(entry: SessionEntry) -> Result<()> {
    let mut v = load()?;
    match v.iter_mut().find(|e| e.id == entry.id) {
        Some(existing) => {
            if !entry.label.is_empty() {
                existing.label = entry.label;
            }
            existing.updated_at = entry.updated_at;
            if entry.last_message_id.is_some() {
                existing.last_message_id = entry.last_message_id;
            }
        }
        None => v.push(entry),
    }
    save(&v)
}

pub fn remove(id: &str) -> Result<bool> {
    let mut v = load()?;
    let before = v.len();
    v.retain(|e| e.id != id);
    let removed = v.len() != before;
    if removed {
        save(&v)?;
    }
    Ok(removed)
}

pub fn resolve(arg: &str) -> Result<Option<SessionEntry>> {
    let arg = arg.trim();
    if arg.is_empty() {
        return Ok(None);
    }
    let all = list()?;
    if let Ok(n) = arg.parse::<usize>() {
        if n >= 1 {
            return Ok(all.get(n - 1).cloned());
        }
    }
    if let Some(e) = all.iter().find(|e| e.id == arg) {
        return Ok(Some(e.clone()));
    }
    let matches: Vec<&SessionEntry> = all.iter().filter(|e| e.id.starts_with(arg)).collect();
    if matches.len() == 1 {
        return Ok(Some(matches[0].clone()));
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_json() {
        let entries = vec![SessionEntry {
            id: "abc".into(),
            label: "hello".into(),
            updated_at: 42,
            last_message_id: Some(serde_json::json!({"id": "m1"})),
        }];
        let text = serde_json::to_string(&entries).unwrap();
        let back: Vec<SessionEntry> = serde_json::from_str(&text).unwrap();
        assert_eq!(back.len(), 1);
        assert_eq!(back[0].id, "abc");
        assert_eq!(back[0].updated_at, 42);
    }

    #[test]
    fn missing_fields_default() {
        let back: SessionEntry = serde_json::from_str(r#"{"id":"x"}"#).unwrap();
        assert_eq!(back.id, "x");
        assert_eq!(back.label, "");
        assert_eq!(back.updated_at, 0);
        assert!(back.last_message_id.is_none());
    }
}
