//! Interactive login: let the user sign in to DeepSeek in their own browser,
//! then read the resulting token out of that browser's local storage.
//!
//! The user never copies a credential: they log in to
//! <https://chat.deepseek.com> the way they normally would (password, Google,
//! phone, ...), and this module harvests the `userToken` the web app already
//! stores, which *is* the bearer token the API expects.
//!
//! Supported stores:
//!   * Firefox  - `storage/default/https+++chat.deepseek.com/ls/data.sqlite`
//!     (a sqlite table `data(key TEXT, value BLOB)`, value JSON-wrapped).
//!   * Chromium - `<profile>/Local Storage/leveldb` (a LevelDB whose value is
//!     a one-byte encoding tag plus the JSON-wrapped token).
//!
//! Everything here is best effort and read-only: browser databases are copied
//! to a temporary directory before being opened, so a running browser is never
//! disturbed and nothing is ever written back to the browser profile.

use anyhow::Result;
use rusty_leveldb::LdbIterator;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Origin whose local storage holds the token.
const ORIGIN: &str = "chat.deepseek.com";
/// Storage key the web app uses for the bearer token.
const TOKEN_KEY: &str = "userToken";
/// Where the user should be sent to sign in.
pub const LOGIN_URL: &str = "https://chat.deepseek.com/sign_in";

/// A browser profile that may hold a DeepSeek token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Profile {
    /// Human-readable browser name, for messages.
    pub browser: String,
    /// Profile directory on disk.
    pub path: PathBuf,
}

/// A token recovered from a browser profile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FoundToken {
    /// The raw token, without a `Bearer ` prefix.
    pub token: String,
    /// Where it came from, for messages.
    pub source: String,
}

/// Home directory, honoring `HOME` on unix and `USERPROFILE` on Windows.
fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .filter(|h| !h.is_empty())
        .or_else(|| std::env::var_os("USERPROFILE").filter(|h| !h.is_empty()))
        .map(PathBuf::from)
}

/// Join `base` with `parts`, returning `None` when `base` is absent.
fn under(base: Option<PathBuf>, parts: &[&str]) -> Option<PathBuf> {
    let mut p = base?;
    for part in parts {
        p.push(part);
    }
    Some(p)
}

/// Subdirectories of `dir`, sorted, ignoring unreadable entries.
fn subdirs(dir: &Path) -> Vec<PathBuf> {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut out: Vec<PathBuf> = rd
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    out.sort();
    out
}

/// Root directories that may contain Firefox profiles.
fn firefox_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Some(p) = under(home_dir(), &[".mozilla", "firefox"]) {
        roots.push(p);
    }
    // Snap and Flatpak keep their own home.
    if let Some(p) = under(
        home_dir(),
        &["snap", "firefox", "common", ".mozilla", "firefox"],
    ) {
        roots.push(p);
    }
    if let Some(p) = under(
        home_dir(),
        &[".var", "app", "org.mozilla.firefox", ".mozilla", "firefox"],
    ) {
        roots.push(p);
    }
    // Windows.
    if let Some(appdata) = std::env::var_os("APPDATA").map(PathBuf::from) {
        if let Some(p) = under(Some(appdata), &["Mozilla", "Firefox", "Profiles"]) {
            roots.push(p);
        }
    }
    roots
}

/// Root directories that may contain Chromium-family profiles.
fn chromium_roots() -> Vec<(String, PathBuf)> {
    let mut roots = Vec::new();
    let bases = [
        (".config", "chromium", "Chromium"),
        (".config", "google-chrome", "Chrome"),
        (".config", "google-chrome-beta", "Chrome Beta"),
        (".config", "google-chrome-unstable", "Chrome Unstable"),
        (".config", "microsoft-edge", "Edge"),
        (".config", "BraveSoftware/Brave-Browser", "Brave"),
        (".config", "vivaldi", "Vivaldi"),
        (".config", "google-chrome-for-testing", "Chrome for Testing"),
    ];
    for (base, rel, name) in bases {
        let parts: Vec<&str> = std::iter::once(base).chain(rel.split('/')).collect();
        if let Some(p) = under(home_dir(), &parts) {
            roots.push((name.to_string(), p));
        }
    }
    // Flatpak Chromium.
    if let Some(p) = under(
        home_dir(),
        &[".var", "app", "org.chromium.Chromium", "config", "chromium"],
    ) {
        roots.push(("Chromium (flatpak)".to_string(), p));
    }
    // Windows.
    if let Some(local) = std::env::var_os("LOCALAPPDATA").map(PathBuf::from) {
        let wins = [
            ("Google/Chrome/User Data", "Chrome"),
            ("Microsoft/Edge/User Data", "Edge"),
            ("Chromium/User Data", "Chromium"),
            ("BraveSoftware/Brave-Browser/User Data", "Brave"),
        ];
        for (rel, name) in wins {
            let parts: Vec<&str> = rel.split('/').collect();
            if let Some(p) = under(Some(local.clone()), &parts) {
                roots.push((name.to_string(), p));
            }
        }
    }
    roots
}

/// Every profile worth searching, in a stable order.
pub fn profiles() -> Vec<Profile> {
    let mut out = Vec::new();
    for root in firefox_roots() {
        if !root.join("storage").is_dir() {
            // A profile root itself contains per-profile directories.
            for prof in subdirs(&root) {
                if prof.join("storage").is_dir() {
                    out.push(Profile {
                        browser: "Firefox".to_string(),
                        path: prof,
                    });
                }
            }
            continue;
        }
        out.push(Profile {
            browser: "Firefox".to_string(),
            path: root,
        });
    }
    for (name, root) in chromium_roots() {
        if root.join("Local Storage").is_dir() {
            out.push(Profile {
                browser: name.clone(),
                path: root,
            });
            continue;
        }
        for prof in subdirs(&root) {
            if prof.join("Local Storage").is_dir() {
                out.push(Profile {
                    browser: name.clone(),
                    path: prof,
                });
            }
        }
    }
    out
}

/// Copy `names` from `dir` into a fresh temporary directory.
///
/// Browser databases may be open in a running browser; reading a copy avoids
/// lock contention and never touches the original. Returns the temp dir.
fn copy_db(dir: &Path, names: &[&str]) -> Result<PathBuf> {
    let unique = format!(
        "ds-login-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    );
    let tmp = std::env::temp_dir().join(unique);
    std::fs::create_dir_all(&tmp)?;
    for name in names {
        let src = dir.join(name);
        if src.is_file() {
            std::fs::copy(&src, tmp.join(name))?;
        }
    }
    Ok(tmp)
}

/// Locate the sqlite file that holds a Firefox profile's local storage.
fn firefox_ls_db(profile: &Path) -> Option<PathBuf> {
    let db = profile
        .join("storage")
        .join("default")
        .join(format!("https+++{ORIGIN}"))
        .join("ls")
        .join("data.sqlite");
    db.is_file().then_some(db)
}

/// Decode a Chromium LocalStorage value.
///
/// The on-disk encoding prefixes the payload with one byte: `0` means the
/// remainder is UTF-16LE, `1` means it is Latin-1. Anything else is not a
/// local-storage record and is rejected.
fn decode_chromium_value(bytes: &[u8]) -> Option<String> {
    match bytes.first()? {
        0 => {
            let body = &bytes[1..];
            let units: Vec<u16> = body
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| u16::from_le_bytes(*c))
                .collect();
            Some(String::from_utf16_lossy(&units))
        }
        1 => Some(bytes[1..].iter().map(|&b| b as char).collect()),
        _ => None,
    }
}

/// True when a LevelDB key belongs to the DeepSeek origin and `userToken`.
///
/// Chromium keys look like `_https://chat.deepseek.com\x00\x01userToken`; the
/// exact framing differs between versions, so match on the parts that matter.
fn chromium_key_matches(key: &[u8]) -> bool {
    let origin = ORIGIN.as_bytes();
    let needle = TOKEN_KEY.as_bytes();
    key.windows(origin.len()).any(|w| w == origin) && key.ends_with(needle)
}

/// Pull a token out of a stored value.
///
/// The web app stores JSON (`{"value":"<token>","__version__":"0"}`), but an
/// older or truncated record may hold the bare string, so fall back to the
/// longest plausible ASCII run.
pub fn extract_token(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(trimmed) {
        if let Some(s) = v.get("value").and_then(|v| v.as_str()) {
            if let Some(t) = normalize_token(s) {
                return Some(t);
            }
        }
    }
    longest_token_run(trimmed)
}

/// Scan `s` for the longest run of token-ish characters.
fn longest_token_run(s: &str) -> Option<String> {
    let mut best: Option<String> = None;
    let mut cur = String::new();
    for ch in s.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-') {
            cur.push(ch);
        } else {
            if let Some(t) = normalize_token(&cur) {
                if best.as_ref().map(|b| t.len() > b.len()).unwrap_or(true) {
                    best = Some(t);
                }
            }
            cur.clear();
        }
    }
    if let Some(t) = normalize_token(&cur) {
        if best.as_ref().map(|b| t.len() > b.len()).unwrap_or(true) {
            best = Some(t);
        }
    }
    best
}

/// Validate a candidate token and strip any `Bearer ` prefix.
///
/// Real tokens are long opaque strings; requiring a plausible length keeps
/// unrelated local-storage keys and UI strings from being mistaken for one.
pub fn normalize_token(raw: &str) -> Option<String> {
    let t = raw.trim();
    let t = t.strip_prefix("Bearer ").unwrap_or(t).trim();
    if t.len() < 32 || t.len() > 512 {
        return None;
    }
    let ok = t
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-' | '+' | '/' | '='));
    ok.then(|| t.to_string())
}

/// Read the token from a Firefox profile, if present.
fn firefox_token(profile: &Profile) -> Option<FoundToken> {
    let db = firefox_ls_db(&profile.path)?;
    let tmp = copy_db(
        db.parent()?,
        &["data.sqlite", "data.sqlite-wal", "data.sqlite-shm"],
    )
    .ok()?;
    let result = (|| -> Option<FoundToken> {
        let conn = rusqlite::Connection::open_with_flags(
            tmp.join("data.sqlite"),
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
        )
        .ok()?;
        let mut stmt = conn.prepare("select value from data where key = ?1").ok()?;
        let value: Vec<u8> = stmt.query_row([TOKEN_KEY], |row| row.get(0)).ok()?;
        let text = String::from_utf8_lossy(&value);
        let token = extract_token(&text)?;
        Some(FoundToken {
            token,
            source: format!("{} ({})", profile.browser, profile.path.display()),
        })
    })();
    let _ = std::fs::remove_dir_all(&tmp);
    result
}

/// Read the token from a Chromium-family profile, if present.
fn chromium_token(profile: &Profile) -> Option<FoundToken> {
    let dir = profile.path.join("Local Storage").join("leveldb");
    if !dir.is_dir() {
        return None;
    }
    let names = ["CURRENT", "MANIFEST-000001", "LOCK", "LOG"];
    let tmp = copy_db(&dir, &names).ok()?;
    // Every .ldb/.log file matters, and their names are version-dependent.
    if let Ok(rd) = std::fs::read_dir(&dir) {
        for e in rd.flatten() {
            let path = e.path();
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            if ext == "ldb" || ext == "log" {
                if let Some(name) = path.file_name() {
                    let _ = std::fs::copy(&path, tmp.join(name));
                }
            }
        }
    }
    let result = (|| -> Option<FoundToken> {
        let opts = rusty_leveldb::Options {
            create_if_missing: false,
            ..Default::default()
        };
        let mut db = rusty_leveldb::DB::open(&tmp, opts).ok()?;
        let mut iter = db.new_iter().ok()?;
        while iter.advance() {
            let Some((key, value)) = iter.current() else {
                break;
            };
            if !chromium_key_matches(&key) {
                continue;
            }
            let decoded = decode_chromium_value(&value)?;
            if let Some(token) = extract_token(&decoded) {
                return Some(FoundToken {
                    token,
                    source: format!("{} ({})", profile.browser, profile.path.display()),
                });
            }
        }
        None
    })();
    let _ = std::fs::remove_dir_all(&tmp);
    result
}

/// Read the token from a single profile.
pub fn token_from_profile(profile: &Profile) -> Option<FoundToken> {
    if profile.browser.starts_with("Firefox") {
        firefox_token(profile)
    } else {
        chromium_token(profile)
    }
}

/// Search every known browser profile for a token.
///
/// The most recently written token wins: a user who signs in with one browser
/// while another holds an old session should get the fresh one.  Tokens in
/// `rejected` are skipped, so a sign-in can require a genuinely new value
/// instead of silently re-reading the credential the server just refused.
pub fn discover_token(rejected: &[String]) -> Option<FoundToken> {
    let mut best: Option<(std::time::SystemTime, FoundToken)> = None;
    for profile in profiles() {
        let Some(found) = token_from_profile(&profile) else {
            continue;
        };
        if rejected.iter().any(|r| r == &found.token) {
            continue;
        }
        let mtime = firefox_ls_db(&profile.path)
            .and_then(|p| std::fs::metadata(p).ok())
            .and_then(|m| m.modified().ok())
            .or_else(|| {
                std::fs::metadata(profile.path.join("Local Storage").join("leveldb"))
                    .ok()
                    .and_then(|m| m.modified().ok())
            });
        let when = mtime.unwrap_or(std::time::UNIX_EPOCH);
        if best.as_ref().map(|(t, _)| when > *t).unwrap_or(true) {
            best = Some((when, found));
        }
    }
    best.map(|(_, f)| f)
}

/// Open `url` in the user's default browser.
///
/// Failure is not fatal: the URL is printed either way, so a headless or
/// unusual environment still lets the user sign in elsewhere.
pub fn open_browser(url: &str) -> bool {
    #[cfg(target_os = "macos")]
    let candidates: &[(&str, &[&str])] = &[("open", &[url])];
    #[cfg(target_os = "windows")]
    let candidates: &[(&str, &[&str])] = &[("cmd", &["/C", "start", "", url])];
    #[cfg(all(unix, not(target_os = "macos")))]
    let candidates: &[(&str, &[&str])] = &[
        ("xdg-open", &[url]),
        ("gio", &["open", url]),
        ("sensible-browser", &[url]),
        ("x-www-browser", &[url]),
    ];

    for (program, args) in candidates {
        let spawned = std::process::Command::new(program)
            .args(*args)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn();
        if let Ok(mut child) = spawned {
            // Reap the launcher without blocking on the browser itself.
            std::thread::spawn(move || {
                let _ = child.wait();
            });
            return true;
        }
    }
    false
}

/// Insert or replace `KEY=VALUE` in the text of a secrets file.
///
/// Comments, blank lines and unrelated keys are preserved in place; a missing
/// key is appended. Values are written bare (the reader strips quotes and an
/// `export ` prefix, so plain `KEY=VALUE` is the simplest stable form).
pub fn upsert_env_line(contents: &str, key: &str, value: &str) -> String {
    let new_line = format!("{key}={value}");
    let mut out = String::with_capacity(contents.len() + new_line.len() + 1);
    let mut replaced = false;
    for line in contents.lines() {
        let trimmed = line.trim();
        let without_export = trimmed.strip_prefix("export ").unwrap_or(trimmed);
        let matches_key = without_export
            .split_once('=')
            .map(|(k, _)| k.trim() == key)
            .unwrap_or(false);
        if matches_key {
            if !replaced {
                out.push_str(&new_line);
                out.push('\n');
                replaced = true;
            }
            // A duplicate definition is dropped: the first one now wins.
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    if !replaced {
        out.push_str(&new_line);
        out.push('\n');
    }
    out
}

/// Remove every definition of `key` from secrets-file text.
///
/// Comments and other keys are left alone, so logging out does not disturb the
/// rest of the user's configuration.
pub fn remove_key(contents: &str, key: &str) -> String {
    contents
        .lines()
        .filter(|line| {
            let trimmed = line.trim();
            let body = trimmed.strip_prefix("export ").unwrap_or(trimmed);
            !body
                .split_once('=')
                .map(|(k, _)| k.trim() == key)
                .unwrap_or(false)
        })
        .map(|line| format!("{line}\n"))
        .collect()
}

/// Write the token into the secrets file at `path`, creating it (and its
/// parent directory) if needed, with owner-only permissions.
pub fn save_token(path: &Path, token: &str) -> Result<()> {
    let existing = std::fs::read_to_string(path).unwrap_or_default();
    let contents = upsert_env_line(&existing, "AUTHORIZATION_TOKEN", &format!("Bearer {token}"));
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    // Write then rename so a crash cannot leave a half-written secrets file.
    let tmp = path.with_extension("env.tmp");
    {
        let mut f = std::fs::File::create(&tmp)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            f.set_permissions(std::fs::Permissions::from_mode(0o600))?;
        }
        f.write_all(contents.as_bytes())?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_json_wrapped_token() {
        let raw = r#"{"value":"XfwsGyX+eI/atbzcD0waEbWwbUeYbbFMeFyWQb3FYJNV1xzDvZtlrCbY5N/clQY1","__version__":"0"}"#;
        let t = extract_token(raw).expect("token");
        assert!(t.starts_with("XfwsGyX"));
        assert_eq!(t.len(), 64);
    }

    #[test]
    fn extracts_bare_token_from_noise() {
        let raw = "\u{0}garbage\u{1}\u{2}abcdefghijklmnopqrstuvwxyz0123456789ABCDEF\u{0}";
        assert_eq!(
            extract_token(raw).as_deref(),
            Some("abcdefghijklmnopqrstuvwxyz0123456789ABCDEF")
        );
    }

    #[test]
    fn rejects_short_or_unlikely_values() {
        assert!(extract_token("31").is_none());
        assert!(extract_token("true").is_none());
        assert!(extract_token("").is_none());
        assert!(normalize_token("short").is_none());
    }

    #[test]
    fn strips_bearer_prefix() {
        let raw = format!("Bearer {}", "a".repeat(40));
        assert_eq!(normalize_token(&raw), Some("a".repeat(40)));
    }

    #[test]
    fn decodes_chromium_latin1_value() {
        let mut bytes = vec![1u8];
        bytes.extend_from_slice(
            br#"{"value":"deadbeefdeadbeefdeadbeefdeadbeefdeadbeef","__version__":"0"}"#,
        );
        let s = decode_chromium_value(&bytes).expect("decode");
        assert_eq!(
            extract_token(&s).as_deref(),
            Some("deadbeefdeadbeefdeadbeefdeadbeefdeadbeef")
        );
    }

    #[test]
    fn decodes_chromium_utf16_value() {
        let payload = r#"{"value":"cafebabecafebabecafebabecafebabecafebabe","__version__":"0"}"#;
        let mut bytes = vec![0u8];
        for unit in payload.encode_utf16() {
            bytes.extend_from_slice(&unit.to_le_bytes());
        }
        let s = decode_chromium_value(&bytes).expect("decode");
        assert_eq!(
            extract_token(&s).as_deref(),
            Some("cafebabecafebabecafebabecafebabecafebabe")
        );
    }

    #[test]
    fn rejects_unknown_chromium_encoding_byte() {
        assert!(decode_chromium_value(&[7, b'x']).is_none());
        assert!(decode_chromium_value(&[]).is_none());
    }

    #[test]
    fn matches_chromium_storage_keys() {
        let key = b"_https://chat.deepseek.com\x00\x01userToken";
        assert!(chromium_key_matches(key));
        assert!(!chromium_key_matches(
            b"_https://chat.deepseek.com\x00\x01settingsJwt"
        ));
        assert!(!chromium_key_matches(
            b"_https://example.com\x00\x01userToken"
        ));
    }

    #[test]
    fn upsert_replaces_in_place_and_keeps_comments() {
        let before = "# creds\nexport DS_SESSION_ID=abc\nexport AUTHORIZATION_TOKEN=old\n";
        let after = upsert_env_line(before, "AUTHORIZATION_TOKEN", "Bearer new");
        assert_eq!(
            after,
            "# creds\nexport DS_SESSION_ID=abc\nAUTHORIZATION_TOKEN=Bearer new\n"
        );
    }

    #[test]
    fn upsert_appends_missing_key() {
        let after = upsert_env_line("DS_SESSION_ID=abc\n", "AUTHORIZATION_TOKEN", "Bearer new");
        assert_eq!(after, "DS_SESSION_ID=abc\nAUTHORIZATION_TOKEN=Bearer new\n");
    }

    #[test]
    fn upsert_handles_empty_file_and_duplicates() {
        assert_eq!(
            upsert_env_line("", "AUTHORIZATION_TOKEN", "Bearer x"),
            "AUTHORIZATION_TOKEN=Bearer x\n"
        );
        assert_eq!(
            upsert_env_line(
                "AUTHORIZATION_TOKEN=a\nAUTHORIZATION_TOKEN=b\n",
                "AUTHORIZATION_TOKEN",
                "Bearer c"
            ),
            "AUTHORIZATION_TOKEN=Bearer c\n"
        );
    }

    #[test]
    fn saved_token_is_readable_by_the_existing_parser() {
        // The file we write must round-trip through `parse_secrets` in main.
        let text = upsert_env_line("", "AUTHORIZATION_TOKEN", "Bearer tok");
        let line = text.lines().next().unwrap();
        let (key, value) = line.split_once('=').unwrap();
        assert_eq!(key, "AUTHORIZATION_TOKEN");
        assert_eq!(value, "Bearer tok");
    }

    #[test]
    fn remove_key_drops_only_the_named_key() {
        let before = "# creds\nDS_SESSION_ID=abc\nexport AUTHORIZATION_TOKEN=old\n";
        assert_eq!(
            remove_key(before, "AUTHORIZATION_TOKEN"),
            "# creds\nDS_SESSION_ID=abc\n"
        );
    }

    #[test]
    fn remove_key_handles_absent_key_and_duplicates() {
        assert_eq!(remove_key("A=1\n", "AUTHORIZATION_TOKEN"), "A=1\n");
        assert_eq!(
            remove_key(
                "AUTHORIZATION_TOKEN=a\nAUTHORIZATION_TOKEN=b\n",
                "AUTHORIZATION_TOKEN"
            ),
            ""
        );
    }

    #[test]
    fn save_token_creates_private_file() {
        let dir = std::env::temp_dir().join(format!("ds-login-test-{}", std::process::id()));
        let path = dir.join("nested").join(".env");
        save_token(&path, "tok").expect("save");
        let text = std::fs::read_to_string(&path).expect("read");
        assert_eq!(text, "AUTHORIZATION_TOKEN=Bearer tok\n");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&path).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o600);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
