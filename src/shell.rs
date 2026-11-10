//! Shell command extraction and execution.
//!
//! Commands are extracted from model output, run under `sh -c` with a
//! wall-clock limit, and their output is bounded before it is echoed back to
//! the model.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::OnceLock;
use std::time::Duration;
use tokio::process::Command;

/// Default wall-clock limit for a single shell command.
///
/// A command that runs longer is killed and reported back to the model as a
/// timeout rather than hanging the session. The model may override this per
/// command with a `timeout` attribute on the opening shell tag.
pub const DEFAULT_TIMEOUT_SECS: u64 = 120;

/// Default cap on the bytes of output kept per stream (stdout, stderr).
pub const DEFAULT_MAX_OUTPUT_BYTES: usize = 16 * 1024;

/// Smallest accepted output cap; below this the truncation marker dominates.
const MIN_OUTPUT_BYTES: usize = 256;

/// Default timeout, overridable with `DS_SHELL_TIMEOUT_SECS`.
pub fn default_timeout() -> Duration {
    static TIMEOUT: OnceLock<Duration> = OnceLock::new();
    *TIMEOUT.get_or_init(|| {
        std::env::var("DS_SHELL_TIMEOUT_SECS")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .filter(|&s| s > 0)
            .map(Duration::from_secs)
            .unwrap_or(Duration::from_secs(DEFAULT_TIMEOUT_SECS))
    })
}

/// Per-stream output cap in bytes, overridable with `DS_SHELL_MAX_OUTPUT_BYTES`.
pub fn max_output_bytes() -> usize {
    static MAX: OnceLock<usize> = OnceLock::new();
    *MAX.get_or_init(|| {
        std::env::var("DS_SHELL_MAX_OUTPUT_BYTES")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|&n| n >= MIN_OUTPUT_BYTES)
            .unwrap_or(DEFAULT_MAX_OUTPUT_BYTES)
    })
}

/// A shell command extracted from model output, plus its per-use timeout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShellCommand {
    pub cmd: String,
    pub timeout: Duration,
}

/// How strongly a command is confined to the sandbox root.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Confinement {
    /// Not sandboxed at all (`--unsandboxed`).
    None,
    /// Confined by an OS sandbox (bubblewrap on Linux, `sandbox-exec` on macOS):
    /// the rest of the filesystem is read-only and writes outside the root fail.
    Os,
    /// Best effort only: the working directory and `TMPDIR` are redirected into
    /// the root, but nothing stops an absolute path from being read or written.
    /// Used when no OS sandbox helper is installed.
    WorkingDir,
}

impl Confinement {
    /// Short human-readable label for the notice printed before each command.
    pub fn label(&self) -> &'static str {
        match self {
            Confinement::None => "unsandboxed",
            Confinement::Os => "sandboxed",
            Confinement::WorkingDir => "sandboxed (working dir only)",
        }
    }
}

/// Decide how `cmd` can be confined to `root` on this machine.
fn confinement_for(root: &Path) -> Confinement {
    let _ = root;
    if os_sandbox_available() {
        Confinement::Os
    } else {
        Confinement::WorkingDir
    }
}

/// Label describing how a command will be confined: `"sandboxed"` when an OS
/// sandbox is in force, `"sandboxed (working dir only)"` when only the working
/// directory is redirected, and `"unsandboxed"` when `root` is `None`.
pub fn sandbox_label(root: Option<&Path>) -> &'static str {
    match root {
        None => Confinement::None.label(),
        Some(r) => confinement_for(r).label(),
    }
}

/// Whether `root` will be confined by an OS sandbox rather than merely by
/// redirecting the working directory. Callers use this to warn the user.
pub fn os_sandbox_available() -> bool {
    (cfg!(target_os = "linux") && which("bwrap").is_some())
        || (cfg!(target_os = "macos") && which("sandbox-exec").is_some())
}

/// Look `name` up on `PATH`, returning its path when it is an executable file.
fn which(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

/// Build the process that will run `cmd`, confined to `sandbox_root` when set.
fn build_command(cmd: &str, sandbox_root: Option<&Path>) -> (Command, Confinement) {
    let Some(root) = sandbox_root else {
        return (plain_shell(cmd), Confinement::None);
    };
    let level = confinement_for(root);
    let mut c = match level {
        Confinement::Os if cfg!(target_os = "linux") => bwrap_shell(cmd, root),
        Confinement::Os if cfg!(target_os = "macos") => sandbox_exec_shell(cmd, root),
        // WorkingDir, or an OS branch that was compiled out.
        _ => plain_shell(cmd),
    };
    // In every sandboxed mode the command starts in the root, so relative
    // paths land inside the sandbox. `HOME` is deliberately left alone: the
    // user's real home is visible (read-only under the OS sandbox) and tools
    // that key off `HOME` keep working as usual. `TMPDIR` and `PWD` are
    // redirected so scratch files and relative paths stay inside the root.
    c.current_dir(root).env("TMPDIR", root).env("PWD", root);
    (c, level)
}

/// A bare `sh -c` (or `cmd /C` on Windows) with no confinement.
fn plain_shell(cmd: &str) -> Command {
    #[cfg(windows)]
    {
        let mut c = Command::new("cmd");
        c.arg("/C").arg(cmd);
        c
    }
    #[cfg(not(windows))]
    {
        let mut c = Command::new("sh");
        c.arg("-c").arg(cmd);
        c
    }
}

/// bubblewrap: everything read-only, the sandbox root writable, `/dev` and
/// `/proc` provided so ordinary tools keep working.
#[cfg(target_os = "linux")]
fn bwrap_shell(cmd: &str, root: &Path) -> Command {
    let mut c = Command::new("bwrap");
    c.arg("--die-with-parent")
        .arg("--ro-bind")
        .arg("/")
        .arg("/")
        .arg("--dev")
        .arg("/dev")
        .arg("--proc")
        .arg("/proc")
        .arg("--bind")
        .arg(root)
        .arg(root)
        .arg("--chdir")
        .arg(root)
        .arg("sh")
        .arg("-c")
        .arg(cmd);
    c
}

#[cfg(not(target_os = "linux"))]
fn bwrap_shell(cmd: &str, _root: &Path) -> Command {
    plain_shell(cmd)
}

/// macOS `sandbox-exec`: deny all writes except under the root and the usual
/// scratch directories.
#[cfg(target_os = "macos")]
fn sandbox_exec_shell(cmd: &str, root: &Path) -> Command {
    let profile = format!(
        "(version 1)\n(allow default)\n(deny file-write*)\n\
         (allow file-write* (subpath \"{root}\") (subpath \"/tmp\") \
         (subpath \"/private/tmp\") (subpath \"/dev\") (literal \"/dev/null\"))\n",
        root = root.display()
    );
    let mut c = Command::new("sandbox-exec");
    c.arg("-p").arg(profile).arg("sh").arg("-c").arg(cmd);
    c
}

#[cfg(not(target_os = "macos"))]
fn sandbox_exec_shell(cmd: &str, _root: &Path) -> Command {
    plain_shell(cmd)
}

/// Run `cmd` under `sh -c`, killing it if it exceeds `timeout`.
///
/// stdin is `/dev/null` so an interactive command cannot fight the line editor
/// for the terminal. On timeout the child *and its process group* are killed,
/// so backgrounded grandchildren do not outlive the turn; the model is told
/// about it as ordinary output. Output is truncated to `max_output_bytes()`
/// per stream.
///
/// When `sandbox_root` is `Some`, the command runs with that directory as its
/// working directory and - where an OS sandbox helper is available - with the
/// rest of the filesystem read-only. `HOME` is inherited unchanged. Passing
/// `None` runs the command unconfined.
pub async fn run_shell(
    cmd: &str,
    timeout: Duration,
    sandbox_root: Option<&Path>,
) -> Result<String> {
    let (mut command, _level) = build_command(cmd, sandbox_root);
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    #[cfg(unix)]
    // Give the command its own process group so the whole tree can be
    // signalled at once on timeout.
    command.process_group(0);

    let child = command.spawn().context("failed to spawn shell")?;
    #[cfg(unix)]
    let pid = child.id();

    let output = match tokio::time::timeout(timeout, child.wait_with_output()).await {
        Ok(res) => res.context("failed to run shell command")?,
        Err(_) => {
            // Dropping the wait future kills the direct child; the group
            // signal below takes care of anything it forked.
            #[cfg(unix)]
            if let Some(pid) = pid {
                // SAFETY: kill(2) with a negative pid signals the process
                // group led by `pid`, which we created above. ESRCH (already
                // gone) is fine and ignored.
                unsafe {
                    libc::kill(-(pid as i32), libc::SIGKILL);
                }
            }
            return Ok(format!(
                "[command timed out after {}s and was killed. Default timeout is {}s; \
                 to allow more time, add a timeout attribute to the opening tag, e.g. \
                 <shell timeout=\"300\"> or <shell timeout=5m> (s/m/h suffixes accepted).]",
                timeout.as_secs_f64(),
                default_timeout().as_secs(),
            ));
        }
    };

    let cap = max_output_bytes();
    let mut result = String::new();
    if !output.stdout.is_empty() {
        result.push_str(&bounded_output(&output.stdout, cap));
    }
    if !output.stderr.is_empty() {
        result.push_str("\n[stderr]\n");
        result.push_str(&bounded_output(&output.stderr, cap));
    }
    match output.status.code() {
        Some(0) => {}
        Some(code) => result.push_str(&format!("\n[exit status: {code}]")),
        None => result.push_str("\n[terminated by signal]"),
    }
    Ok(result)
}

/// Decode and cap one stream of command output.
fn bounded_output(bytes: &[u8], cap: usize) -> String {
    truncate_middle(&String::from_utf8_lossy(bytes), cap)
}

/// Keep the head and tail of `s`, replacing the middle when it is longer than
/// `cap` bytes. Splits only on character boundaries.
fn truncate_middle(s: &str, cap: usize) -> String {
    if s.len() <= cap {
        return s.to_string();
    }
    let head_budget = cap / 2;
    let tail_budget = cap - head_budget;
    let head_end = floor_boundary(s, head_budget);
    let tail_start = ceil_boundary(s, s.len() - tail_budget);
    format!(
        "{}\n[\u{2026} {} bytes omitted \u{2026}]\n{}",
        &s[..head_end],
        tail_start - head_end,
        &s[tail_start..]
    )
}

fn floor_boundary(s: &str, mut i: usize) -> usize {
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

fn ceil_boundary(s: &str, mut i: usize) -> usize {
    while i < s.len() && !s.is_char_boundary(i) {
        i += 1;
    }
    i
}

// Tag delimiters, spelled with escapes so this source file does not itself
// contain a literal shell-tag pair.

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

/// Find an attribute of the form `name = "value"` (either quote style,
/// whitespace allowed around `=`) and return the byte offset of `name`.
///
/// Used to read the native tool-call shape some models emit instead of the
/// tag form: an `invoke name="shell"` header wrapping a
/// `parameter name="command"` element.
fn find_name_attr(haystack: &str, value: &str) -> Option<usize> {
    let mut search = 0usize;
    while let Some(i) = haystack[search..].find("name") {
        let abs = search + i;
        let rest = haystack[abs + "name".len()..].trim_start();
        if let Some(rest) = rest.strip_prefix('=') {
            let rest = rest.trim_start();
            for q in ['"', '\''] {
                if let Some(rest) = rest.strip_prefix(q) {
                    if let Some(rest) = rest.strip_prefix(value) {
                        if rest.starts_with(q) {
                            return Some(abs);
                        }
                    }
                }
            }
        }
        search = abs + "name".len();
    }
    None
}

/// Recover commands from a native tool-call syntax, e.g.
///
/// ```text
/// <|...|> invoke name="shell">
/// <|...|> parameter name="command">ls -la</|...|> parameter>
/// </|...|> invoke>
/// ```
///
/// The delimiters around those elements vary between models, so only the
/// `name=...` attributes and the body of the `command` parameter are matched.
/// A reviewer that reaches for this shape must still be able to run commands:
/// silently running nothing would leave it without evidence.
fn extract_tool_call_commands(text: &str) -> Vec<ShellCommand> {
    let mut cmds = Vec::new();
    let mut rest = text;
    while let Some(invoke) = find_name_attr(rest, "shell") {
        let after_invoke = &rest[invoke..];
        let Some(param) = find_name_attr(after_invoke, "command") else {
            break;
        };
        let after_param = &after_invoke[param..];
        let Some(gt) = after_param.find('>') else {
            break;
        };
        let body = &after_param[gt + 1..];
        let end = body.find("</").unwrap_or(body.len());
        let cmd = body[..end].trim();
        if !cmd.is_empty() {
            cmds.push(ShellCommand {
                cmd: cmd.to_string(),
                timeout: default_timeout(),
            });
        }
        if end >= body.len() {
            break;
        }
        rest = &body[end..];
    }
    cmds
}

/// Extract commands from JSON tool-call blocks such as
///
/// ```text
/// <tool_call>
/// {"name":"shell","arguments":{"command":"ls -la"}}
/// </tool_call>
/// ```
///
/// Models drift between plain function calls, Anthropic-style `arguments`
/// objects and OpenAI-style `parameters`. Rather than depend on a JSON parser
/// (the surrounding text is often not valid JSON), look for the `"command"`
/// key and read its string value.
fn extract_json_tool_commands(text: &str) -> Vec<ShellCommand> {
    let mut cmds = Vec::new();
    let mut rest = text;
    while let Some(i) = rest.find("\"command\"") {
        let after = &rest[i + "\"command\"".len()..];
        let after = after.trim_start();
        let Some(after) = after.strip_prefix(':') else {
            rest = after;
            continue;
        };
        let after = after.trim_start();
        let Some(quote) = after.chars().next() else {
            break;
        };
        if quote != '"' && quote != '\'' {
            rest = after;
            continue;
        }
        // Read the quoted value, honouring backslash escapes.
        let mut value = String::new();
        let mut escaped = false;
        let mut consumed = 1;
        let mut closed = false;
        for ch in after[1..].chars() {
            consumed += ch.len_utf8();
            if escaped {
                value.push(match ch {
                    'n' => '\n',
                    't' => '\t',
                    other => other,
                });
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == quote {
                closed = true;
                break;
            } else {
                value.push(ch);
            }
        }
        if closed && !value.trim().is_empty() {
            cmds.push(ShellCommand {
                cmd: value.trim().to_string(),
                timeout: default_timeout(),
            });
        }
        rest = &after[consumed..];
    }
    cmds
}

/// Opening tag names accepted for a command block.
///
/// `<shell>` is the documented form. The aliases exist because models drift
/// into other tag names - `<cmd>`, `<command>`, `<bash>` - and a reviewer that
/// used one would otherwise run nothing and be left without evidence.
const OPEN_TAGS: [&str; 4] = ["\x3cshell", "\x3ccmd", "\x3ccommand", "\x3cbash"];

/// Extract one command block starting at `start` (which points at `<`).
///
/// Returns the command, the index just past the block, and the attributes of
/// the opening tag. `None` means the text at `start` is not a usable block.
fn take_tag_block(text: &str, start: usize, open: &str) -> Option<(ShellCommand, usize, usize)> {
    // A short tail (`...<cm`) must simply not match.
    let after_name = text.get(start + open.len()..)?;
    // The tag name must be followed by whitespace or `>` so we do not match
    // something like `<shellfish>`.
    match after_name.chars().next() {
        Some('>') => {}
        Some(c) if c.is_whitespace() => {}
        _ => return None,
    }
    let gt = after_name.find('>')?;
    let attrs = &after_name[..gt];
    let body_start = start + open.len() + gt + 1;
    let name = &open[1..];
    let close = format!("\x3c/{name}\x3e");
    let end = text[body_start..].find(&close)?;
    let cmd = text[body_start..body_start + end].trim().to_string();
    if cmd.is_empty() {
        return None;
    }
    let timeout = parse_timeout_attr(attrs).unwrap_or_else(default_timeout);
    Some((
        ShellCommand { cmd, timeout },
        attrs.len(),
        body_start + end + close.len(),
    ))
}

/// Extract shell command blocks from model output.
///
/// Supported forms:
///   plain block                        -> default timeout
///   block with `timeout="30"` attr     -> 30 seconds
///   block with `timeout=2m` attr       -> 2 minutes
///   `<cmd>`, `<command>`, `<bash>`     -> aliases for the block tag
pub fn extract_shell_commands(text: &str) -> Vec<ShellCommand> {
    let mut cmds = Vec::new();
    let mut i = 0usize;
    while i < text.len() {
        // Step over whole characters: indexing at a non-boundary would panic.
        let ch = text[i..].chars().next().unwrap_or_default();
        if ch != '\x3c' {
            i += ch.len_utf8().max(1);
            continue;
        }
        let mut advanced = false;
        for open in OPEN_TAGS {
            if let Some((cmd, _, next)) = take_tag_block(text, i, open) {
                cmds.push(cmd);
                i = next;
                advanced = true;
                break;
            }
        }
        if !advanced {
            i += 1;
        }
    }
    // Tolerate the native tool-call shape as well, so a model that reaches
    // for it instead of the tag form still runs its commands.
    for cmd in extract_tool_call_commands(text)
        .into_iter()
        .chain(extract_json_tool_commands(text))
    {
        if !cmds.iter().any(|c| c.cmd == cmd.cmd) {
            cmds.push(cmd);
        }
    }
    cmds
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Spelled with escapes so this source file does not itself contain a
    /// literal shell-tag pair.
    const OPEN: &str = "\x3cshell";
    const CLOSE: &str = "\x3c/shell\x3e";

    /// Build a shell block without writing the literal tag pair.
    fn block(attrs: &str, body: &str) -> String {
        format!("{OPEN}{attrs}\x3e{body}{CLOSE}")
    }

    #[test]
    fn extracts_dsml_with_extra_parameters() {
        // The invoke can carry more than one parameter (e.g. a workdir); the
        // command must still be read, and the extra one ignored.
        let text = concat!(
            "<｜｜DSML｜｜ calls>\n",
            "<｜｜DSML｜｜ invoke name=\"shell\">\n",
            "<｜｜DSML｜｜ parameter name=\"command\" string=\"true\">pwd && ls</｜｜DSML｜｜ parameter>\n",
            "<｜｜DSML｜｜ parameter name=\"workdir\" string=\"true\">.</｜｜DSML｜｜ parameter>\n",
            "</｜｜DSML｜｜ invoke>\n",
            "</｜｜DSML｜｜ calls>"
        );
        let cmds = extract_shell_commands(text);
        assert_eq!(cmds.len(), 1, "{cmds:?}");
        assert_eq!(cmds[0].cmd, "pwd && ls");
    }

    #[test]
    fn extracts_bare_dsml_invoke_without_calls_wrapper() {
        // The invoke element can arrive without the surrounding "calls" line.
        let text = concat!(
            "<｜｜DSML｜｜ invoke name=\"shell\">\n",
            "<｜｜DSML｜｜ parameter name=\"command\" string=\"true\">pwd && ls -la</｜｜DSML｜｜ parameter>\n",
            "</｜｜DSML｜｜ invoke>"
        );
        let cmds = extract_shell_commands(text);
        assert_eq!(cmds.len(), 1, "{cmds:?}");
        assert_eq!(cmds[0].cmd, "pwd && ls -la");
    }

    #[test]
    fn extracts_json_tool_call() {
        // The exact shape a reviewer emitted: a JSON tool-call block.
        let text = concat!(
            "<tool_call>\n",
            "{\"name\":\"shell\",\"arguments\":{\"command\":\"pwd && ls -la\"}}\n",
            "</tool_call>"
        );
        let cmds = extract_shell_commands(text);
        assert_eq!(cmds.len(), 1, "{cmds:?}");
        assert_eq!(cmds[0].cmd, "pwd && ls -la");

        // Escaped quotes and newlines survive.
        let text = r#"{"parameters":{"command":"printf \"a\"\nls"}}"#;
        let cmds = extract_shell_commands(text);
        assert_eq!(cmds.len(), 1, "{cmds:?}");
        assert_eq!(cmds[0].cmd, "printf \"a\"\nls");
    }

    #[test]
    fn extracts_dsml_style_tool_call() {
        // The shape an agent session emits verbatim, fullwidth bars and all.
        let text = concat!(
            "<｜｜DSML｜｜ calls>\n",
            "<｜｜DSML｜｜ invoke name=\"shell\">\n",
            "<｜｜DSML｜｜ parameter name=\"command\" string=\"true\">pwd && ls</｜｜DSML｜｜ parameter>\n",
            "</｜｜DSML｜｜ invoke>"
        );
        let cmds = extract_shell_commands(text);
        assert_eq!(cmds.len(), 1, "{cmds:?}");
        assert_eq!(cmds[0].cmd, "pwd && ls");
    }

    #[test]
    fn extracts_multiline_bash_blocks() {
        // Multi-line bodies and two blocks in one reply must both work.
        let text = "<bash>\npwd && ls -la\n</bash>\n\n<bash>\nfind . -type f | head -200\n</bash>";
        let cmds = extract_shell_commands(text);
        assert_eq!(cmds.len(), 2, "{cmds:?}");
        assert_eq!(cmds[0].cmd, "pwd && ls -la");
        assert_eq!(cmds[1].cmd, "find . -type f | head -200");
    }

    #[test]
    fn accepts_common_tag_aliases() {
        // Models drift into other tag names; a reviewer using one must still
        // be able to run its commands.
        for name in ["cmd", "command", "bash"] {
            let text = format!("\x3c{name}\x3epwd\x3c/{name}\x3e");
            let cmds = extract_shell_commands(&text);
            assert_eq!(cmds.len(), 1, "{name}: {cmds:?}");
            assert_eq!(cmds[0].cmd, "pwd");
        }
        // A near-miss must not be treated as a block.
        let text = "<cmdx>pwd</cmdx>";
        assert!(extract_shell_commands(text).is_empty());
    }

    #[test]
    fn extracts_native_tool_call_syntax() {
        // Some models answer with their native tool-call shape instead of the
        // documented tag form. The command must still run: a reviewer that
        // reaches for this shape would otherwise get no evidence at all.
        let text = concat!(
            "<|delim|> calls\n",
            "<|delim|> invoke name=\"shell\">\n",
            "<|delim|> parameter name=\"command\" string=\"true\">pwd && ls</|delim|> parameter>\n",
            "</|delim|> invoke>\n"
        );
        let cmds = extract_shell_commands(text);
        assert_eq!(cmds.len(), 1, "{cmds:?}");
        assert_eq!(cmds[0].cmd, "pwd && ls");
    }

    #[test]
    fn a_non_json_command_key_is_not_a_false_positive() {
        // Prose mentioning the key must not become a command.
        assert!(extract_shell_commands("\"command\": not quoted here").is_empty());
        assert!(extract_shell_commands("the \"command\" key is undocumented").is_empty());
    }

    #[test]
    fn tag_form_and_tool_call_form_are_not_duplicated() {
        let text = format!(
            "{}\n<|d|> invoke name=\"shell\"><|d|> parameter name=\"command\">ls</|d|> parameter></|d|> invoke>",
            block("", "ls")
        );
        let cmds = extract_shell_commands(&text);
        assert_eq!(cmds.len(), 1, "{cmds:?}");
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
        let text = format!(
            "{}\n{}",
            block("", "echo a"),
            block(r#" timeout="5""#, "echo b")
        );
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

    #[test]
    fn short_output_is_not_truncated() {
        assert_eq!(truncate_middle("hello", 256), "hello");
    }

    #[test]
    fn long_output_keeps_head_and_tail() {
        let s = "a".repeat(500) + &"b".repeat(500);
        let out = truncate_middle(&s, 200);
        assert!(out.len() < 400, "not truncated: {} bytes", out.len());
        assert!(out.contains("bytes omitted"));
        assert!(out.starts_with("aaa"));
        assert!(out.ends_with("bbb"));
    }

    #[test]
    fn truncation_respects_char_boundaries() {
        // Multi-byte characters must never be split.
        let s = "\u{e9}".repeat(400);
        let out = truncate_middle(&s, 101);
        assert!(out.contains("bytes omitted"));
    }

    #[tokio::test]
    async fn run_shell_kills_on_timeout() {
        let start = std::time::Instant::now();
        let out = run_shell("sleep 10", Duration::from_millis(200), None)
            .await
            .expect("run_shell should not error on timeout");
        assert!(out.contains("timed out"), "unexpected output: {out}");
        // The hint about raising the limit must travel with the error.
        assert!(
            out.contains("timeout attribute"),
            "missing timeout hint: {out}"
        );
        // Must return promptly rather than waiting the full 10s.
        assert!(start.elapsed() < Duration::from_secs(5));
    }

    #[tokio::test]
    async fn run_shell_returns_output_when_fast() {
        let out = run_shell("echo hello", Duration::from_secs(5), None)
            .await
            .expect("run_shell ok");
        assert_eq!(out.trim(), "hello");
    }

    #[tokio::test]
    async fn run_shell_does_not_read_the_terminal() {
        // With stdin at /dev/null, `read` hits EOF immediately instead of
        // waiting for input.
        let out = run_shell("read x; echo got:[$x]", Duration::from_secs(5), None)
            .await
            .expect("run_shell ok");
        assert_eq!(out.trim(), "got:[]");
    }

    #[tokio::test]
    async fn run_shell_reports_nonzero_exit() {
        let out = run_shell("exit 3", Duration::from_secs(5), None)
            .await
            .expect("run_shell ok");
        assert!(out.contains("exit status: 3"), "unexpected output: {out}");
    }

    #[tokio::test]
    async fn sandboxed_command_runs_in_the_root() {
        let root = std::env::temp_dir().join(format!("ds-sandbox-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        // `pwd` must report the sandbox root, and a relative write must land
        // there rather than in the test runner's cwd.
        let out = run_shell(
            "pwd; echo hi > rel.txt",
            Duration::from_secs(10),
            Some(&root),
        )
        .await
        .expect("run_shell ok");
        let canon_root = root.canonicalize().unwrap();
        assert!(
            out.contains(&canon_root.display().to_string())
                || out.contains(&root.display().to_string()),
            "pwd did not report the sandbox root: {out}"
        );
        assert!(
            root.join("rel.txt").exists(),
            "relative write escaped the root"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn sandbox_leaves_home_alone() {
        let root = std::env::temp_dir().join(format!("ds-home-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let expected = std::env::var("HOME").unwrap_or_default();
        let out = run_shell("echo $HOME", Duration::from_secs(10), Some(&root))
            .await
            .expect("run_shell ok");
        assert_eq!(out.trim(), expected, "HOME was overridden in the sandbox");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn confinement_label_is_set() {
        assert_eq!(Confinement::None.label(), "unsandboxed");
        assert!(Confinement::Os.label().starts_with("sandboxed"));
    }
}
