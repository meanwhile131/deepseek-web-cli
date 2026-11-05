use anyhow::Result;
use std::process::Stdio;
use std::time::Duration;
use tokio::process::Command;

/// Default wall-clock limit for a single shell command.
///
/// A command that runs longer is killed and reported back to the model as a
/// timeout rather than hanging the session. The model may override this per
/// command with a `timeout` attribute on the opening shell tag.
pub const DEFAULT_TIMEOUT_SECS: u64 = 120;

/// Default timeout as a `Duration`.
pub fn default_timeout() -> Duration {
    std::env::var("DS_SHELL_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|&s| s > 0)
        .map(Duration::from_secs)
        .unwrap_or(Duration::from_secs(DEFAULT_TIMEOUT_SECS))
}

/// A shell command extracted from model output, plus its per-use timeout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShellCommand {
    pub cmd: String,
    pub timeout: Duration,
}

/// Run `cmd` under `sh -c`, killing it if it exceeds `timeout`.
///
/// On timeout the child is killed (via `kill_on_drop`) and a descriptive
/// message is returned as normal output so the model can react to it.
pub async fn run_shell(cmd: &str, timeout: Duration) -> Result<String> {
    let child = Command::new("sh")
        .arg("-c")
        .arg(cmd)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()?;

    let output = match tokio::time::timeout(timeout, child.wait_with_output()).await {
        Ok(res) => res?,
        Err(_) => {
            // `wait_with_output` was dropped, which (with kill_on_drop)
            // terminates the child process.
            return Ok(format!(
                "[command timed out after {}s and was killed]",
                timeout.as_secs_f64()
            ));
        }
    };

    let mut result = String::new();
    if !output.stdout.is_empty() {
        result.push_str(&String::from_utf8_lossy(&output.stdout));
    }
    if !output.stderr.is_empty() {
        result.push_str("\n[stderr]\n");
        result.push_str(&String::from_utf8_lossy(&output.stderr));
    }
    Ok(result)
}

// Tag delimiters, spelled with escapes so this source file does not itself
// contain a literal shell-tag pair.
const OPEN: &str = "\x3cshell";
const CLOSE: &str = "\x3c/shell\x3e";

/// Parse a duration attribute value such as `30`, `30s`, `2m`, `1h`, `500ms`.
fn parse_duration(raw: &str) -> Option<Duration> {
    let raw = raw.trim().trim_matches('"').trim_matches('\'').trim();
    if raw.is_empty() {
        return None;
    }
    let (num, mult) = if let Some(n) = raw.strip_suffix("ms") {
        (n, 0.001)
    } else if let Some(n) = raw.strip_suffix('s') {
        (n, 1.0)
    } else if let Some(n) = raw.strip_suffix('m') {
        (n, 60.0)
    } else if let Some(n) = raw.strip_suffix('h') {
        (n, 3600.0)
    } else {
        (raw, 1.0)
    };
    let value: f64 = num.trim().parse().ok()?;
    if value <= 0.0 {
        return None;
    }
    Some(Duration::from_secs_f64(value * mult))
}

/// Parse a `timeout=...` attribute out of an opening tag's attribute text.
fn parse_timeout_attr(attrs: &str) -> Option<Duration> {
    for part in attrs.split_whitespace() {
        if let Some(v) = part.strip_prefix("timeout=") {
            return parse_duration(v);
        }
    }
    None
}

/// Extract shell command blocks from model output.
///
/// Supported forms:
///   plain block                        -> default timeout
///   block with `timeout="30"` attr     -> 30 seconds
///   block with `timeout=2m` attr       -> 2 minutes
pub fn extract_shell_commands(text: &str) -> Vec<ShellCommand> {
    let mut cmds = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find(OPEN) {
        let after_name = &rest[start + OPEN.len()..];
        // The tag name must be followed by whitespace or `>` so we do not
        // match something like `<shellfish>`.
        let ok = match after_name.chars().next() {
            Some('>') => true,
            Some(c) if c.is_whitespace() => true,
            _ => false,
        };
        if !ok {
            rest = after_name;
            continue;
        }
        // Find the `>` that closes the opening tag.
        let Some(gt) = after_name.find('>') else {
            break;
        };
        let attrs = &after_name[..gt];
        let body_start = start + OPEN.len() + gt + 1;
        let Some(end) = rest[body_start..].find(CLOSE) else {
            break;
        };
        let cmd = rest[body_start..body_start + end].trim().to_string();
        let timeout = parse_timeout_attr(attrs).unwrap_or_else(default_timeout);
        cmds.push(ShellCommand { cmd, timeout });
        rest = &rest[body_start + end + CLOSE.len()..];
    }
    cmds
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a shell block without writing the literal tag pair.
    fn block(attrs: &str, body: &str) -> String {
        format!("{OPEN}{attrs}\x3e{body}{CLOSE}")
    }

    #[test]
    fn extracts_plain_command_with_default_timeout() {
        let text = format!("run {} now", block("", "ls -la"));
        let cmds = extract_shell_commands(&text);
        assert_eq!(cmds.len(), 1);
        assert_eq!(cmds[0].cmd, "ls -la");
        assert_eq!(cmds[0].timeout, default_timeout());
    }

    #[test]
    fn extracts_custom_timeout_attribute() {
        let cmds = extract_shell_commands(&block(r#" timeout="30""#, "sleep 5"));
        assert_eq!(cmds.len(), 1);
        assert_eq!(cmds[0].timeout, Duration::from_secs(30));

        let cmds = extract_shell_commands(&block(" timeout=2m", "long job"));
        assert_eq!(cmds[0].timeout, Duration::from_secs(120));

        let cmds = extract_shell_commands(&block(" timeout='1h'", "job"));
        assert_eq!(cmds[0].timeout, Duration::from_secs(3600));
    }

    #[test]
    fn multiple_commands_each_keep_their_timeout() {
        let text = format!("{}\n{}", block("", "echo a"), block(r#" timeout="5""#, "echo b"));
        let cmds = extract_shell_commands(&text);
        assert_eq!(cmds.len(), 2);
        assert_eq!(cmds[0].timeout, default_timeout());
        assert_eq!(cmds[1].timeout, Duration::from_secs(5));
        assert_eq!(cmds[1].cmd, "echo b");
    }

    #[test]
    fn invalid_timeout_falls_back_to_default() {
        let cmds = extract_shell_commands(&block(r#" timeout="abc""#, "x"));
        assert_eq!(cmds[0].timeout, default_timeout());
    }

    #[test]
    fn ignores_unclosed_or_similar_tags() {
        assert!(extract_shell_commands("\x3cshell>no end").is_empty());
        assert!(extract_shell_commands("\x3cshellfish>oops\x3c/shellfish\x3e").is_empty());
    }

    #[tokio::test]
    async fn run_shell_kills_on_timeout() {
        let start = std::time::Instant::now();
        let out = run_shell("sleep 10", Duration::from_millis(200))
            .await
            .expect("run_shell should not error on timeout");
        assert!(out.contains("timed out"), "unexpected output: {out}");
        // Must return promptly rather than waiting the full 10s.
        assert!(start.elapsed() < Duration::from_secs(5));
    }

    #[tokio::test]
    async fn run_shell_returns_output_when_fast() {
        let out = run_shell("echo hello", Duration::from_secs(5))
            .await
            .expect("run_shell ok");
        assert_eq!(out.trim(), "hello");
    }
}
