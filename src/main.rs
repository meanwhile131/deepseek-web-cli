mod api;
mod color;
mod goals;
mod pow;
mod sessions;
mod shell;

use anyhow::Result;
use color::paint;
use futures_util::StreamExt;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use std::io::Write;
use std::path::PathBuf;
use tokio::sync::watch;

const SYSTEM_PROMPT: &str = r#"You are a helpful assistant with access to a local shell.
When you need to run a command, emit it exactly as:
<shell>command goes here</shell>
The user will execute it and reply with [SHELL OUTPUT] followed by the result.
Use one <shell> block per command.  Do not wrap commands in markdown fences.
Always batch multiple commands into one response.
Command blocks must appear in your visible reply, never inside private reasoning.
Each command has a default timeout of 120 seconds; a command that runs longer is killed. To change the timeout for a single command, add a timeout attribute in seconds (or with an s/m/h suffix), for example <shell timeout="300">long command</shell>. Omit the attribute to use the default."#;

/// Token the model must include in its visible reply once the goal is done.
const GOAL_DONE: &str = "[GOAL COMPLETE]";

const THINK_OPEN: &str = "\u{27EA}THINK\u{27EB}";
const THINK_CLOSE: &str = "\u{27EA}/THINK\u{27EB}";

fn out(s: &str) {
    print!("{s}");
    std::io::stdout().flush().ok();
}

/// Cooperative Ctrl+C handling.
///
/// A background task listens for `SIGINT` and bumps a generation counter on
/// every press.  Long-running work (streaming replies, shell commands) awaits
/// this via `tokio::select!` so a Ctrl+C cancels just the current turn and
/// returns control to the prompt, instead of terminating the whole process.
///
/// Generation counters (rather than a sticky boolean) make stale interrupts
/// impossible: each turn subscribes fresh, so a signal delivered while sitting
/// at the prompt cannot leak into the next turn.
#[derive(Clone)]
struct Interrupt {
    tx: watch::Sender<u64>,
}

impl Interrupt {
    fn new() -> Self {
        let (tx, _rx) = watch::channel(0u64);
        Self { tx }
    }
    fn fire(&self) {
        self.tx.send_modify(|g| *g = g.wrapping_add(1));
    }
    fn subscribe(&self) -> watch::Receiver<u64> {
        self.tx.subscribe()
    }
}

/// Install the SIGINT handler that feeds `interrupt`.
fn spawn_sigint_handler(interrupt: &Interrupt) {
    let intr = interrupt.clone();
    tokio::spawn(async move {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt()) {
            Ok(mut sig) => {
                while sig.recv().await.is_some() {
                    intr.fire();
                }
            }
            Err(e) => eprintln!(
                "{} could not install Ctrl+C handler: {e}",
                paint("1;33", "[warn]")
            ),
        }
    });
}

/// How a turn ended, so callers can tell a completed turn from an aborted one.
enum TurnResult {
    Done { completed: bool },
    Interrupted,
}

/// Parse a boolean-ish env var. Empty/unset falls back to `default`.
fn env_flag(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(v) => {
            let v = v.trim();
            if v.is_empty() {
                default
            } else {
                v != "0" && !v.eq_ignore_ascii_case("false") && !v.eq_ignore_ascii_case("no")
            }
        }
        Err(_) => default,
    }
}

/// Parse a numeric env var, falling back to `default` when unset/invalid.
fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(default)
}

/// Command-line options.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct Args {
    /// Run model-proposed shell commands without asking for confirmation.
    auto_approve: bool,
    /// Print usage and exit.
    help: bool,
}

/// Parse process arguments (excluding the program name).
///
/// Kept deliberately small and dependency-free: the CLI only has a couple of
/// flags.  Unknown arguments are collected so the caller can report them
/// rather than silently ignoring a typo such as `--auto-aprove`.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct ParsedArgs {
    args: Args,
    unknown: Vec<String>,
}

fn parse_args<I: IntoIterator<Item = String>>(argv: I) -> ParsedArgs {
    let mut parsed = ParsedArgs::default();
    for a in argv {
        match a.as_str() {
            "-y" | "--auto-approve" => parsed.args.auto_approve = true,
            "-h" | "--help" => parsed.args.help = true,
            other => parsed.unknown.push(other.to_string()),
        }
    }
    parsed
}

/// Help text shown for `--help` and on argument errors.
fn usage() -> &'static str {
    "Usage: deepseek [OPTIONS]\n\
     \n\
     Options:\n\
     \x20 -y, --auto-approve   run model-proposed shell commands without asking\n\
     \x20 -h, --help           show this help and exit\n\
     \n\
     Credentials and tuning are read from the environment or the user-wide\n\
     secrets file (see README)."
}

/// Path to the user-wide secrets file.
///
/// Lookup order:
///   1. `$DS_SECRETS_FILE` (explicit override)
///   2. `$XDG_CONFIG_HOME/deepseek-web-cli/.env`
///   3. `~/.config/deepseek-web-cli/.env`
fn user_secrets_path() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("DS_SECRETS_FILE") {
        if !p.is_empty() {
            return Some(PathBuf::from(p));
        }
    }
    if let Some(dir) = std::env::var_os("XDG_CONFIG_HOME") {
        if !dir.is_empty() {
            return Some(PathBuf::from(dir).join("deepseek-web-cli").join(".env"));
        }
    }
    std::env::var_os("HOME").map(|h| {
        PathBuf::from(h)
            .join(".config")
            .join("deepseek-web-cli")
            .join(".env")
    })
}

/// Parse the contents of a secrets file: one `KEY=VALUE` per line.  Blank
/// lines and `#` comments are ignored, a leading `export ` is accepted (so a
/// shell-style `.env` keeps working), and matching single or double quotes
/// around the value are stripped.
fn parse_secrets(contents: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for raw in contents.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let line = line.strip_prefix("export ").unwrap_or(line).trim_start();
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let key = key.trim();
        if key.is_empty() {
            continue;
        }
        let mut value = value.trim();
        for q in ['"', '\''] {
            if value.len() >= 2 && value.starts_with(q) && value.ends_with(q) {
                value = &value[1..value.len() - 1];
                break;
            }
        }
        out.push((key.to_string(), value.to_string()));
    }
    out
}

/// Load the user-wide secrets file (if any) into the process environment.
/// Variables already present in the environment are left untouched, so real
/// environment variables always take precedence over the file.  Returns the
/// path that was read, for diagnostics.
fn load_user_secrets() -> Option<PathBuf> {
    let path = user_secrets_path()?;
    let contents = std::fs::read_to_string(&path).ok()?;
    for (key, value) in parse_secrets(&contents) {
        if std::env::var_os(&key).is_none() {
            std::env::set_var(&key, &value);
        }
    }
    Some(path)
}

/// Fetch a required secret, failing with a message that points at both the
/// environment and the user-wide secrets file.
fn require_secret(name: &str, hint: &str) -> Result<String> {
    std::env::var(name).map_err(|_| {
        let where_ = user_secrets_path()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "~/.config/deepseek-web-cli/.env".to_string());
        anyhow::anyhow!("set {name} ({hint}) in the environment or in {where_}")
    })
}

/// Attach the active-goal reminder to an outgoing message so the model keeps
/// the user's stated objective in view on every turn (including shell
/// follow-ups), and tell it how to signal completion.  No goal set means the
/// message is passed through untouched.
fn with_goal(goal: &Option<String>, msg: &str) -> String {
    match goal {
        Some(g) => format!(
            "{msg}\n\n[ACTIVE GOAL] {g}\nWhen the goal is fully accomplished, include the exact token {GOAL_DONE} in your visible reply."
        ),
        None => msg.to_string(),
    }
}

/// True when the model has declared the active goal finished.
fn goal_completed(text: &str) -> bool {
    text.contains(GOAL_DONE)
}

/// Mirror the in-memory goal into the per-conversation store.
///
/// A conversation id only exists once the first message has been sent, so this
/// runs after a turn: before that there is nothing to key the goal on, and the
/// goal lives only in memory (it is picked up by the first `persist_goal`
/// call).  Persisting the absence of a goal clears any stale saved entry.
fn persist_goal(client: &api::DeepSeekClient, goal: &Option<String>) {
    let Some(id) = client.chat_session_id() else {
        return;
    };
    match goal {
        Some(g) => goals::set(id, g),
        None => goals::clear(id),
    }
}

/// Print the active goal (or a hint when none is set).
///
/// Shared by bare `/goal` and explicit `/goal show`/`/goal status` so both
/// spellings report the same thing and neither can be mistaken for a new
/// goal whose text happens to be "show".
fn print_goal(goal: &Option<String>) {
    match goal {
        Some(g) => println!("{} {g}", paint("36", "[goal]")),
        None => println!(
            "{} {}",
            paint("36", "[goal]"),
            paint(
                "2",
                "<none> \u{2014} usage: /goal <text> | /goal show | /goal clear"
            )
        ),
    }
}

/// What a `/goal ...` invocation should do.
#[derive(Debug, PartialEq, Eq)]
enum GoalCommand {
    /// No argument, `show`, or `status`: report the active goal.
    Show,
    /// `clear`, `none`, or `off`: drop the active goal.
    Clear,
    /// Anything else: start working toward this goal.
    Set(String),
}

/// Classify the text after `/goal`.
///
/// Kept separate from the event loop so the read-only subcommands can be
/// tested directly: `show`/`status` must be reported, never turned into a goal
/// whose text happens to be the word "show".
fn parse_goal_command(arg: &str) -> GoalCommand {
    let arg = arg.trim();
    if arg.is_empty() || arg.eq_ignore_ascii_case("show") || arg.eq_ignore_ascii_case("status") {
        GoalCommand::Show
    } else if arg.eq_ignore_ascii_case("clear")
        || arg.eq_ignore_ascii_case("none")
        || arg.eq_ignore_ascii_case("off")
    {
        GoalCommand::Clear
    } else {
        GoalCommand::Set(arg.to_string())
    }
}

/// Streaming renderer: dims the reasoning region delimited by the think
/// markers, survives markers split across chunks, and records visible
/// (non-thinking) text and reasoning text separately.  Command extraction
/// prefers the visible text but falls back to reasoning, since the model
/// sometimes emits its tool call inside the reasoning region.
struct Renderer {
    in_think: bool,
    buf: String,
}

impl Renderer {
    fn new() -> Self {
        Self {
            in_think: false,
            buf: String::new(),
        }
    }

    /// Feed a stream chunk; visible text is appended to `visible`,
    /// reasoning text to `reasoning`.
    fn feed(&mut self, chunk: &str, visible: &mut String, reasoning: &mut String) {
        self.buf.push_str(chunk);
        self.drain(false, visible, reasoning);
    }

    /// Flush leftovers at end of stream and restore terminal attributes.
    fn finish(&mut self, visible: &mut String, reasoning: &mut String) {
        self.drain(true, visible, reasoning);
        if self.in_think {
            out(&format!("{}\n", color::reset()));
            self.in_think = false;
        }
    }

    fn drain(&mut self, at_end: bool, visible: &mut String, reasoning: &mut String) {
        loop {
            let Some(i) = self.buf.find('\u{27EA}') else {
                if !self.buf.is_empty() {
                    let s = std::mem::take(&mut self.buf);
                    self.emit(&s, visible, reasoning);
                }
                return;
            };
            if i > 0 {
                let pre: String = self.buf.drain(..i).collect();
                self.emit(&pre, visible, reasoning);
                continue;
            }
            if self.buf.starts_with(THINK_OPEN) {
                self.buf.drain(..THINK_OPEN.len());
                self.in_think = true;
                out(&format!("{}[thinking] ", color::dim()));
                continue;
            }
            if self.buf.starts_with(THINK_CLOSE) {
                self.buf.drain(..THINK_CLOSE.len());
                self.in_think = false;
                out(&format!("{}\n", color::reset()));
                continue;
            }
            let partial = THINK_OPEN.starts_with(self.buf.as_str())
                || THINK_CLOSE.starts_with(self.buf.as_str());
            if partial && !at_end {
                // Wait for the rest of the marker in a later chunk.
                return;
            }
            // A stray marker character that is not the start of a marker.
            self.buf.drain(..'\u{27EA}'.len_utf8());
            self.emit("\u{27EA}", visible, reasoning);
        }
    }

    fn emit(&self, s: &str, visible: &mut String, reasoning: &mut String) {
        if s.is_empty() {
            return;
        }
        out(s);
        if self.in_think {
            reasoning.push_str(s);
        } else {
            visible.push_str(s);
        }
    }
}

impl Drop for Renderer {
    /// Safety net: if the stream is dropped mid-thinking (error, early
    /// return, panic) without `finish()` being called, restore the default
    /// terminal attributes so the dim `\x1b[2m` does not leak into the rest
    /// of the session.
    fn drop(&mut self) {
        if self.in_think {
            out(&format!("{}\n", color::reset()));
            self.in_think = false;
        }
    }
}

/// Send one message, render the stream, and return both the visible
/// (non-thinking) text and the reasoning text.  The caller normally extracts
/// shell commands from the visible text, but falls back to the reasoning text
/// when the model mistakenly emits the tool call inside its thinking region.
async fn stream_reply(
    client: &mut api::DeepSeekClient,
    msg: &str,
    thinking: bool,
    search: bool,
) -> Result<(String, String)> {
    let mut s = client.send_message(msg, thinking, search).await?;
    let mut visible = String::new();
    let mut reasoning = String::new();
    let mut renderer = Renderer::new();
    let mut stream_err: Option<anyhow::Error> = None;
    while let Some(c) = s.next().await {
        match c {
            Ok(chunk) => renderer.feed(&chunk, &mut visible, &mut reasoning),
            Err(e) => {
                stream_err = Some(e);
                break;
            }
        }
    }
    // Always close out the renderer (restores color) even after an error.
    renderer.finish(&mut visible, &mut reasoning);
    out("\n");
    drop(s);
    if let Some(e) = stream_err {
        return Err(e);
    }
    Ok((visible, reasoning))
}

/// Send `message`, then keep extracting and running shell commands until the
/// model stops asking for them.  Returns `true` if the model signalled the
/// active goal was completed at any point during the turn.
/// Turn-wide settings, bundled so `run_turn` does not take a long argument
/// list of booleans (which are easy to mix up at the call site).
struct TurnConfig<'a> {
    thinking: bool,
    search: bool,
    auto_approve: bool,
    debug_pending: bool,
    goal: &'a Option<String>,
}

async fn run_turn(
    client: &mut api::DeepSeekClient,
    rl: &mut DefaultEditor,
    message: &str,
    cfg: &TurnConfig<'_>,
    interrupt: &Interrupt,
) -> Result<TurnResult> {
    let mut rx = interrupt.subscribe();
    let (thinking, search) = (cfg.thinking, cfg.search);
    let (auto_approve, debug_pending) = (cfg.auto_approve, cfg.debug_pending);
    let goal = cfg.goal;

    let (mut pending, mut reasoning) = tokio::select! {
        r = stream_reply(client, message, thinking, search) => r?,
        _ = rx.changed() => {
            println!("\n{}", paint("1;33", "[interrupted]"));
            return Ok(TurnResult::Interrupted);
        }
    };
    let mut completed = goal_completed(&pending) || goal_completed(&reasoning);
    let mut iter = 0usize;

    // Shell loop: keep extracting commands from each reply
    // (initial + every follow-up) until the model stops.
    loop {
        iter += 1;

        if debug_pending {
            eprintln!(
                "\n[debug] iter {iter}: pending ({} bytes):\n---\n{pending}\n---",
                pending.len()
            );
            if !reasoning.is_empty() {
                eprintln!(
                    "[debug] reasoning ({} bytes):\n---\n{reasoning}\n---",
                    reasoning.len()
                );
            }
        }

        let mut cmds = shell::extract_shell_commands(&pending);
        if cmds.is_empty() {
            // Fallback: the model sometimes emits the tool call
            // inside its reasoning region and leaves the visible
            // output empty.  Recover the command from reasoning.
            let from_reasoning = shell::extract_shell_commands(&reasoning);
            if !from_reasoning.is_empty() {
                println!(
                    "{} tool call found in reasoning instead of visible output \u{2014} recovering it",
                    paint("1;33", "[warn]")
                );
                cmds = from_reasoning;
            }
        }
        if cmds.is_empty() {
            if debug_pending && pending.contains("<shell>") {
                eprintln!("[warn] pending contains a command tag but extraction returned nothing");
            }
            break;
        }
        if debug_pending {
            eprintln!("[debug] extracted {} command(s): {:?}", cmds.len(), cmds);
        }

        // Run every command in this batch first, then send ONE
        // combined follow-up so the conversation thread is kept.
        let mut combined = String::new();
        for cmd in cmds {
            let cmd_text = cmd.cmd.clone();
            let cmd_timeout = cmd.timeout;
            println!("\n{} {}", paint("1;35", "[shell]"), paint("35", &cmd_text));
            if auto_approve {
                println!("{}", paint("2", "execute? [auto-approved]"));
            } else {
                // Use the rustyline editor so we never read raw
                // stdin behind its back.
                let approve_prompt = format!("{} ", paint("33", "execute? [y/N]"));
                let approved = match rl.readline(&approve_prompt) {
                    Ok(a) => a.trim().eq_ignore_ascii_case("y"),
                    Err(ReadlineError::Interrupted) => {
                        println!("\n{}", paint("1;33", "[interrupted]"));
                        return Ok(TurnResult::Interrupted);
                    }
                    Err(_) => false,
                };
                if !approved {
                    println!("{}", paint("2", "skipped"));
                    continue;
                }
            }
            let output = match tokio::select! {
                r = shell::run_shell(&cmd_text, cmd_timeout) => r,
                _ = rx.changed() => {
                    println!("\n{}", paint("1;33", "[interrupted]"));
                    return Ok(TurnResult::Interrupted);
                }
            } {
                Ok(o) => o,
                Err(e) => format!("[error running command: {e}]"),
            };
            println!(
                "{}\n{output}\n{}",
                paint("2", "--- output ---"),
                paint("2", "--------------")
            );
            combined.push_str(&format!("\n=== CMD: {cmd_text}\n{output}\n"));
        }

        if combined.is_empty() {
            println!("{}", paint("2", "[all commands skipped]"));
            break;
        }

        let follow_up = with_goal(goal, &format!("[SHELL OUTPUT]{combined}"));
        let (v, r) = tokio::select! {
            r = stream_reply(client, &follow_up, thinking, search) => r?,
            _ = rx.changed() => {
                println!("\n{}", paint("1;33", "[interrupted]"));
                return Ok(TurnResult::Interrupted);
            }
        };
        if goal_completed(&v) || goal_completed(&r) {
            completed = true;
        }
        pending = v;
        reasoning = r;
    }
    Ok(TurnResult::Done { completed })
}

/// Largest page the `fetch_page` endpoint accepts; anything above is rejected
/// with `illegal count`.
const SESSION_PAGE_SIZE: usize = 100;

/// Hard cap on cursor walks, so a misbehaving `has_more` cannot spin forever.
const SESSION_MAX_PAGES: usize = 100;

/// Conversations loaded so far, plus enough cursor state to fetch the next
/// page.
///
/// `fetch_page`'s `lte_cursor` is inclusive, so the entry the cursor points at
/// comes back with the next page; `fetch_next` drops such repeats.
#[derive(Default)]
struct SessionCache {
    entries: Vec<sessions::SessionEntry>,
    /// The server says older conversations exist beyond what we hold.
    has_more: bool,
    /// Whether a fetch has run at all (distinguishes "empty" from "unloaded").
    started: bool,
}

impl SessionCache {
    fn reset(&mut self) {
        *self = Self::default();
    }

    /// Fetch the next page and append it. Returns how many new entries were
    /// added.
    async fn fetch_next(&mut self, client: &api::DeepSeekClient) -> Result<usize> {
        // Own the cursor so the borrow on `entries` ends before we mutate it.
        let cursor_owned = self
            .entries
            .last()
            .map(|last| (last.updated_at, last.id.clone()));
        let cursor = cursor_owned.as_ref().map(|(u, id)| (*u, id.as_str()));

        let (page, has_more) = client.list_sessions(SESSION_PAGE_SIZE, cursor).await?;
        let added = self.merge_page(page, has_more);
        self.started = true;
        Ok(added)
    }

    /// Append `page`, skipping ids already held, and record `has_more`.
    ///
    /// Returns how many new entries were added. Because `lte_cursor` is
    /// inclusive the page repeats its cursor row, so a page that adds nothing
    /// means paging has stalled; `has_more` is cleared in that case.
    fn merge_page(&mut self, page: Vec<sessions::SessionEntry>, has_more: bool) -> usize {
        let before = self.entries.len();
        for e in page {
            if !self.entries.iter().any(|x| x.id == e.id) {
                self.entries.push(e);
            }
        }
        let added = self.entries.len() - before;
        self.has_more = has_more && added > 0;
        added
    }

    /// Load pages until `n` entries are available, the server runs out, or the
    /// page cap is hit.
    async fn ensure_len(&mut self, client: &api::DeepSeekClient, n: usize) -> Result<()> {
        let mut pages = 0;
        while self.entries.len() < n && self.has_more && pages < SESSION_MAX_PAGES {
            self.fetch_next(client).await?;
            pages += 1;
        }
        Ok(())
    }

    /// Load every page the server will give us (bounded by the page cap).
    async fn ensure_all(&mut self, client: &api::DeepSeekClient) -> Result<()> {
        let mut pages = 0;
        while self.has_more && pages < SESSION_MAX_PAGES {
            self.fetch_next(client).await?;
            pages += 1;
        }
        Ok(())
    }
}

/// Print part of the loaded conversation list, numbering from `first`.
fn print_session_rows(list: &[sessions::SessionEntry], first: usize) {
    println!("{}", paint("1;36", "[sessions]"));
    for (i, e) in list.iter().enumerate() {
        println!(
            "  {}  {}  {}  {}",
            paint("33", &format!("{:>4}", first + i + 1)),
            paint("2", &e.when()),
            e.display_title(),
            paint("2", &format!("(id {})", e.short_id())),
        );
    }
}

/// Print the loaded conversations, saying whether more can be loaded.
fn print_loaded_sessions(cache: &SessionCache, from: usize) {
    if cache.entries.is_empty() {
        println!(
            "{} {}",
            paint("36", "[sessions]"),
            paint("2", "<none on the server>")
        );
        return;
    }
    let from = from.min(cache.entries.len());
    print_session_rows(&cache.entries[from..], from);
    if cache.has_more {
        println!(
            "{} {}",
            paint("36", "[sessions]"),
            paint(
                "2",
                &format!(
                    "showing {} so far \u{2014} run /sessions more to load up to {} more",
                    cache.entries.len(),
                    SESSION_PAGE_SIZE
                )
            )
        );
    }
    println!("{}", paint("2", "resume with: /resume <number|id-prefix>"));
}

/// Handle `/sessions` and `/sessions more`.
async fn show_sessions(client: &api::DeepSeekClient, cache: &mut SessionCache, more: bool) {
    if more && cache.started {
        let from = cache.entries.len();
        if !cache.has_more {
            println!(
                "{} {}",
                paint("36", "[sessions]"),
                paint("2", "no more conversations to load")
            );
            return;
        }
        match cache.fetch_next(client).await {
            Ok(0) => println!(
                "{} {}",
                paint("36", "[sessions]"),
                paint("2", "no more conversations to load")
            ),
            Ok(_) => print_loaded_sessions(cache, from),
            Err(e) => eprintln!(
                "{} could not list sessions: {e:#}",
                paint("1;31", "[error]")
            ),
        }
        return;
    }

    // A plain `/sessions` restarts from the newest conversation.
    cache.reset();
    match cache.fetch_next(client).await {
        Ok(_) => print_loaded_sessions(cache, 0),
        Err(e) => eprintln!(
            "{} could not list sessions: {e:#}",
            paint("1;31", "[error]")
        ),
    }
}

/// Resolve a user reference against the conversation list, loading more pages
/// from the server as needed.
///
/// A numeric reference loads only as far as that index; a name or id prefix
/// loads every page so the match can be found anywhere in the history.
async fn resolve_session(
    client: &api::DeepSeekClient,
    cache: &mut SessionCache,
    arg: &str,
) -> Result<Option<sessions::SessionEntry>> {
    let trimmed = arg.trim();
    if !cache.started {
        cache.fetch_next(client).await?;
    }
    match trimmed.parse::<usize>() {
        Ok(n) if n >= 1 => cache.ensure_len(client, n).await?,
        _ => cache.ensure_all(client).await?,
    }
    Ok(sessions::resolve(&cache.entries, trimmed).cloned())
}

/// Apply a title the user set before any conversation existed.
///
/// New conversations get an auto-generated server title; a `/label` issued
/// before the first message is remembered in `pending` and pushed once the
/// session exists.
async fn apply_pending_title(
    client: &api::DeepSeekClient,
    label: &mut Option<String>,
    pending: &mut Option<String>,
) {
    let Some(title) = pending.clone() else { return };
    let Some(id) = client.chat_session_id() else {
        return;
    };
    match client.rename_session(id, &title).await {
        Ok(()) => {
            *label = Some(title);
            *pending = None;
        }
        Err(e) => eprintln!("{} could not apply title: {e:#}", paint("1;33", "[warn]")),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let parsed = parse_args(std::env::args().skip(1));
    if parsed.args.help {
        println!("{}", usage());
        return Ok(());
    }
    if !parsed.unknown.is_empty() {
        eprintln!(
            "{} unknown argument(s): {}",
            paint("1;33", "[warn]"),
            parsed.unknown.join(" ")
        );
        eprintln!("{}", usage());
        std::process::exit(2);
    }
    let args = parsed.args;

    // Pick up secrets from the user-wide file before anything reads them.
    let secrets_path = load_user_secrets();
    if env_flag("DS_DEBUG_SECRETS", false) {
        match &secrets_path {
            Some(p) => eprintln!("[secrets] loaded {}", p.display()),
            None => eprintln!("[secrets] no user-wide secrets file found"),
        }
    }

    let session_id = require_secret("DS_SESSION_ID", "ds_session_id cookie")?;
    let bearer = require_secret("AUTHORIZATION_TOKEN", "full 'Bearer ...' value")?;

    let debug_pending = env_flag("DS_DEBUG_PENDING", false);
    // Default OFF: executing model-authored commands without confirmation is
    // opt-in via the `--auto-approve` / `-y` command-line flag.
    let auto_approve = args.auto_approve;
    let thinking = env_flag("DS_THINKING", true);
    let search = env_flag("DS_SEARCH", true);

    let mut client = api::DeepSeekClient::new(session_id, bearer)?;

    // Rate-limit policy, tunable from the environment.
    let policy = api::RetryPolicy {
        base: std::time::Duration::from_millis(env_u64("DS_RETRY_BASE_MS", 500)),
        cap: std::time::Duration::from_millis(env_u64("DS_RETRY_CAP_MS", 60_000)),
        min_gap: std::time::Duration::from_millis(env_u64("DS_MIN_GAP_MS", 0)),
    };
    if !policy.min_gap.is_zero() {
        eprintln!(
            "[rate-limit] retries=unlimited base={:?} cap={:?} min_gap={:?}",
            policy.base, policy.cap, policy.min_gap
        );
    }
    client.set_policy(policy);
    client.set_http_timeout(std::time::Duration::from_millis(env_u64(
        "DS_HTTP_TIMEOUT_MS",
        180_000,
    )));

    // The system prompt is prepended to the first user message of each
    // conversation instead of being sent as a separate startup message.
    let mut primed = false;
    let mut goal: Option<String> = None;
    // Human-readable name for the current conversation, persisted in the store.
    let mut label: Option<String> = None;
    // A title set before the first message, applied once a session exists.
    let mut pending_title: Option<String> = None;
    // Conversations loaded from the server, reusable across /sessions calls.
    let mut session_cache = SessionCache::default();

    let mut rl = DefaultEditor::new()?;

    // Route SIGINT to cooperative cancellation so Ctrl+C aborts the current
    // turn instead of killing the process while the model streams or a shell
    // command runs.
    let interrupt = Interrupt::new();
    spawn_sigint_handler(&interrupt);

    println!("{}", paint("1;36", "DeepSeek Web CLI \u{2014} multi-turn"));
    println!("  {}   new conversation", paint("32", "/new"));
    println!("  {}   show conversation tip", paint("33", "/id"));
    println!("  {}   set/start/show/clear goal", paint("35", "/goal"));
    println!(
        "  {}   list conversations ({} more to page)",
        paint("34", "/sessions"),
        paint("2", "/sessions more")
    );
    println!("  {}   resume a saved conversation", paint("34", "/resume"));
    println!(
        "  {}   rename the current conversation",
        paint("34", "/label")
    );
    println!("  {}   forget a saved conversation", paint("34", "/forget"));
    println!("  {}   quit (Ctrl+D also quits)", paint("31", "exit"));
    println!(
        "  {} Ctrl+C cancels the current reply/command (or clears the prompt)",
        paint("2", "tip:")
    );
    println!();

    loop {
        // Give the client a fresh view of the interrupt channel so retry
        // backoffs and session calls can be cancelled with Ctrl+C too.
        client.set_cancel(interrupt.subscribe());
        let prompt = paint("1;35", ">> ");
        let line = rl.readline(&prompt);
        match line {
            Ok(input) if input.trim() == "exit" => break,
            Ok(input) if input.trim() == "/new" => {
                // The old conversation stays in the store so it can be resumed.
                client.reset();
                primed = false;
                label = None;
                pending_title = None;
                println!("{}", paint("32", "[new conversation started]"));
                if let Some(g) = &goal {
                    println!("{} {g}", paint("36", "[goal retained]"));
                }
                continue;
            }
            Ok(input) if input.trim() == "/id" => {
                match client.last_message_id() {
                    Some(v) => println!("{} {v}", paint("36", "[tip]")),
                    None => println!("{} {}", paint("36", "[tip]"), paint("2", "<none yet>")),
                }
                continue;
            }
            Ok(input) if input.trim().is_empty() => continue,
            Ok(input)
                if input.trim() == "/sessions"
                    || input.trim().eq_ignore_ascii_case("/sessions more") =>
            {
                let more = input.trim().eq_ignore_ascii_case("/sessions more");
                show_sessions(&client, &mut session_cache, more).await;
                continue;
            }
            Ok(input) if input.trim() == "/resume" || input.trim().starts_with("/resume ") => {
                let arg = input.trim()["/resume".len()..].trim().to_string();
                if arg.is_empty() {
                    show_sessions(&client, &mut session_cache, false).await;
                    println!("{}", paint("2", "usage: /resume <number|id-prefix>"));
                    continue;
                }
                match resolve_session(&client, &mut session_cache, &arg).await {
                    Ok(Some(entry)) => match client.latest_message_id(&entry.id).await {
                        Ok(tip) => {
                            client.resume(entry.id.clone(), tip);
                            // The remote conversation already contains the
                            // system prompt, so we must not prepend it again.
                            primed = true;
                            label = Some(entry.display_title().to_string());
                            // Restore the goal that was active when this
                            // conversation was last used, so a restart does
                            // not lose the work in progress.
                            goal = goals::get(&entry.id);
                            println!("{} {}", paint("32", "[resumed]"), entry.display_title());
                            if let Some(g) = &goal {
                                println!("{} {g}", paint("36", "[goal restored]"));
                            }
                        }
                        Err(e) => eprintln!(
                            "{} could not load conversation: {e:#}",
                            paint("1;31", "[error]")
                        ),
                    },
                    Ok(None) => println!(
                        "{} no unique session matches {:?}",
                        paint("1;33", "[warn]"),
                        arg
                    ),
                    Err(e) => eprintln!("{} {e:#}", paint("1;31", "[error]")),
                }
                continue;
            }
            Ok(input) if input.trim() == "/forget" || input.trim().starts_with("/forget ") => {
                let arg = input.trim()["/forget".len()..].trim().to_string();
                if arg.is_empty() {
                    show_sessions(&client, &mut session_cache, false).await;
                    println!("{}", paint("2", "usage: /forget <number|id-prefix>"));
                    continue;
                }
                match resolve_session(&client, &mut session_cache, &arg).await {
                    Ok(Some(entry)) => match client.delete_session(&entry.id).await {
                        Ok(()) => {
                            session_cache.reset();
                            // The conversation is gone, so its saved goal is
                            // meaningless; drop it either way.
                            goals::clear(&entry.id);
                            if client.chat_session_id() == Some(entry.id.as_str()) {
                                client.reset();
                                primed = false;
                                label = None;
                                pending_title = None;
                            }
                            println!("{} {}", paint("32", "[deleted]"), entry.display_title());
                        }
                        Err(e) => eprintln!(
                            "{} could not delete session: {e:#}",
                            paint("1;31", "[error]")
                        ),
                    },
                    Ok(None) => println!(
                        "{} no unique session matches {:?}",
                        paint("1;33", "[warn]"),
                        arg
                    ),
                    Err(e) => eprintln!("{} {e:#}", paint("1;31", "[error]")),
                }
                continue;
            }
            Ok(input) if input.trim() == "/label" || input.trim().starts_with("/label ") => {
                let arg = input.trim()["/label".len()..].trim().to_string();
                if arg.is_empty() {
                    match &label {
                        Some(l) => println!("{} {l}", paint("36", "[label]")),
                        None => println!("{} {}", paint("36", "[label]"), paint("2", "<none>")),
                    }
                    continue;
                }
                match client.chat_session_id() {
                    Some(id) => match client.rename_session(id, &arg).await {
                        Ok(()) => {
                            label = Some(arg.clone());
                            println!("{} {arg}", paint("32", "[label set]"));
                        }
                        Err(e) => eprintln!(
                            "{} could not rename session: {e:#}",
                            paint("1;31", "[error]")
                        ),
                    },
                    None => {
                        pending_title = Some(arg.clone());
                        println!(
                            "{} {arg} {}",
                            paint("32", "[label set]"),
                            paint("2", "(applied once a conversation starts)")
                        );
                    }
                }
                continue;
            }
            Ok(input) if input.trim() == "/goal" || input.trim().starts_with("/goal ") => {
                let arg = input.trim()["/goal".len()..].trim().to_string();
                match parse_goal_command(&arg) {
                    // Read-only: report the goal without starting a turn.
                    GoalCommand::Show => print_goal(&goal),
                    GoalCommand::Clear => {
                        goal = None;
                        persist_goal(&client, &goal);
                        println!("{}", paint("32", "[goal cleared]"));
                    }
                    GoalCommand::Set(text) => {
                        goal = Some(text.clone());
                        persist_goal(&client, &goal);
                        println!("{} {text}", paint("32", "[goal set]"));
                        // Automatically start the model and keep reprompting it
                        // until it marks the goal complete.
                        let mut turn = 0usize;
                        loop {
                            turn += 1;
                            let base = if turn == 1 {
                                format!(
                                "Work toward the following goal, using the shell as needed. \
                                 When it is fully accomplished, include the token {GOAL_DONE} in your visible reply.\n\nGoal: {text}"
                            )
                            } else {
                                format!(
                                "Continue working toward the active goal. \
                                 When it is fully accomplished, include the token {GOAL_DONE} in your visible reply."
                            )
                            };
                            let base = if primed {
                                base
                            } else {
                                primed = true;
                                format!("{SYSTEM_PROMPT}\n\n{base}")
                            };
                            let message = with_goal(&goal, &base);
                            let cfg = TurnConfig {
                                thinking,
                                search,
                                auto_approve,
                                debug_pending,
                                goal: &goal,
                            };
                            let completed = match run_turn(
                                &mut client,
                                &mut rl,
                                &message,
                                &cfg,
                                &interrupt,
                            )
                            .await
                            {
                                Ok(TurnResult::Done { completed }) => completed,
                                Ok(TurnResult::Interrupted) => {
                                    // Stop reprompting; keep the goal so the
                                    // user can resume it with /goal.
                                    println!(
                                    "{} goal paused \u{2014} view with /goal show, restart with /goal <text> or drop with /goal clear",
                                    paint("1;33", "[interrupted]")
                                );
                                    break;
                                }
                                Err(e) => {
                                    eprintln!("{} {e:#}", paint("1;31", "[error]"));
                                    break;
                                }
                            };
                            apply_pending_title(&client, &mut label, &mut pending_title).await;
                            // The first turn creates the conversation id; record
                            // the goal against it now that one exists.
                            persist_goal(&client, &goal);
                            if completed {
                                println!("{}", paint("32", "[goal completed]"));
                                goal = None;
                                persist_goal(&client, &goal);
                                break;
                            }
                            println!("{} reprompting model (turn {turn})", paint("36", "[goal]"));
                        }
                    }
                }
                continue;
            }
            Ok(input) => {
                let message = if primed {
                    input.clone()
                } else {
                    primed = true;
                    format!("{SYSTEM_PROMPT}\n\n{input}")
                };
                let message = with_goal(&goal, &message);
                let cfg = TurnConfig {
                    thinking,
                    search,
                    auto_approve,
                    debug_pending,
                    goal: &goal,
                };
                match run_turn(&mut client, &mut rl, &message, &cfg, &interrupt).await {
                    Ok(TurnResult::Done { completed }) => {
                        apply_pending_title(&client, &mut label, &mut pending_title).await;
                        if completed && goal.is_some() {
                            println!("{}", paint("32", "[goal completed]"));
                            goal = None;
                        }
                        // Sync the store: a new conversation gets its goal
                        // recorded, and completion clears the saved entry.
                        persist_goal(&client, &goal);
                    }
                    // Turn cancelled by Ctrl+C: just return to the prompt.
                    Ok(TurnResult::Interrupted) => {}
                    Err(e) => eprintln!("{} {e:#}", paint("1;31", "[error]")),
                }
            }
            // Ctrl+C at the prompt: print a freshline and show the prompt
            // again instead of quitting.
            Err(ReadlineError::Interrupted) => {
                println!();
                continue;
            }
            // Ctrl+D exits.
            Err(ReadlineError::Eof) => break,
            Err(e) => eprintln!("{} readline error: {e}", paint("1;31", "[error]")),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        goal_completed, parse_args, parse_goal_command, parse_secrets, with_goal, GoalCommand,
        ParsedArgs, SessionCache,
    };
    use crate::sessions;

    #[test]
    fn no_goal_is_passthrough() {
        assert_eq!(with_goal(&None, "hi"), "hi");
    }

    #[test]
    fn goal_is_appended() {
        let g = Some("fix the bug".to_string());
        assert_eq!(
            with_goal(&g, "hi"),
            "hi\n\n[ACTIVE GOAL] fix the bug\nWhen the goal is fully accomplished, include the exact token [GOAL COMPLETE] in your visible reply."
        );
    }

    #[test]
    fn goal_wraps_shell_output() {
        let g = Some("fix the bug".to_string());
        let msg = format!("[SHELL OUTPUT]{}", "\n=== CMD: ls\nok\n");
        assert!(with_goal(&g, &msg).contains("[ACTIVE GOAL] fix the bug"));
    }

    #[test]
    fn goal_show_is_read_only() {
        // The help text promises `/goal` can "show"; these must report the
        // goal rather than starting a new one whose text is the word.
        assert_eq!(parse_goal_command(""), GoalCommand::Show);
        assert_eq!(parse_goal_command("   "), GoalCommand::Show);
        assert_eq!(parse_goal_command("show"), GoalCommand::Show);
        assert_eq!(parse_goal_command("SHOW"), GoalCommand::Show);
        assert_eq!(parse_goal_command("status"), GoalCommand::Show);
        assert_eq!(parse_goal_command(" Status "), GoalCommand::Show);
    }

    #[test]
    fn goal_clear_variants() {
        assert_eq!(parse_goal_command("clear"), GoalCommand::Clear);
        assert_eq!(parse_goal_command("NONE"), GoalCommand::Clear);
        assert_eq!(parse_goal_command("off"), GoalCommand::Clear);
    }

    #[test]
    fn goal_text_is_set() {
        assert_eq!(
            parse_goal_command("fix the bug"),
            GoalCommand::Set("fix the bug".to_string())
        );
        // A goal that merely contains "show" is still a goal to set.
        assert_eq!(
            parse_goal_command("make /goal show functional"),
            GoalCommand::Set("make /goal show functional".to_string())
        );
        assert_eq!(
            parse_goal_command("  trim me  "),
            GoalCommand::Set("trim me".to_string())
        );
    }

    #[test]
    fn detects_completion_marker() {
        assert!(goal_completed("all done [GOAL COMPLETE]"));
        assert!(!goal_completed("still working"));
    }

    fn entry(id: &str, updated_at: f64) -> sessions::SessionEntry {
        sessions::SessionEntry {
            id: id.into(),
            title: id.into(),
            updated_at,
        }
    }

    #[test]
    fn merge_dedups_inclusive_cursor_row() {
        let mut c = SessionCache::default();
        assert_eq!(
            c.merge_page(vec![entry("a", 3.0), entry("b", 2.0)], true),
            2
        );
        assert!(c.has_more);
        // The next page repeats "b" (the cursor row) before "c".
        assert_eq!(
            c.merge_page(vec![entry("b", 2.0), entry("c", 1.0)], false),
            1
        );
        assert_eq!(
            c.entries.iter().map(|e| e.id.as_str()).collect::<Vec<_>>(),
            vec!["a", "b", "c"]
        );
        assert!(!c.has_more);
    }

    #[test]
    fn merge_stops_paging_when_nothing_new() {
        let mut c = SessionCache::default();
        c.merge_page(vec![entry("a", 3.0)], true);
        assert!(c.has_more);
        // A page that only echoes the cursor row means paging stalled.
        assert_eq!(c.merge_page(vec![entry("a", 3.0)], true), 0);
        assert!(!c.has_more);
    }

    fn parse(v: &[&str]) -> ParsedArgs {
        parse_args(v.iter().map(|s| s.to_string()))
    }

    #[test]
    fn auto_approve_is_off_by_default() {
        let p = parse(&[]);
        assert!(!p.args.auto_approve);
        assert!(!p.args.help);
        assert!(p.unknown.is_empty());
    }

    #[test]
    fn accepts_long_and_short_auto_approve() {
        assert!(parse(&["--auto-approve"]).args.auto_approve);
        assert!(parse(&["-y"]).args.auto_approve);
    }

    #[test]
    fn accepts_help_flag() {
        assert!(parse(&["--help"]).args.help);
        assert!(parse(&["-h"]).args.help);
    }

    #[test]
    fn collects_unknown_arguments_instead_of_ignoring_them() {
        let p = parse(&["--auto-aprove", "extra"]);
        assert!(!p.args.auto_approve);
        assert_eq!(p.unknown, vec!["--auto-aprove", "extra"]);
    }

    #[test]
    fn flags_can_be_combined() {
        let p = parse(&["-y", "--help"]);
        assert!(p.args.auto_approve);
        assert!(p.args.help);
        assert!(p.unknown.is_empty());
    }

    #[test]
    fn parses_secrets_file() {
        let contents = "# a comment\n\nexport DS_SESSION_ID=abc\nAUTHORIZATION_TOKEN=\"Bearer xyz\"\nQ='v'\nnovalue\n";
        assert_eq!(
            parse_secrets(contents),
            vec![
                ("DS_SESSION_ID".to_string(), "abc".to_string()),
                ("AUTHORIZATION_TOKEN".to_string(), "Bearer xyz".to_string()),
                ("Q".to_string(), "v".to_string()),
            ]
        );
    }
}
