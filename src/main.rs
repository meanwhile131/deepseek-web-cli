mod api;
mod color;
mod goals;
mod login;
mod markdown;
mod pow;
mod reviewer;
mod sessions;
mod shell;

use anyhow::{Context, Result};
use color::paint;
use futures_util::StreamExt;
use rustyline::error::ReadlineError;
use rustyline::{Config, DefaultEditor};
use std::io::{IsTerminal, Write};
use std::path::PathBuf;
use tokio::sync::watch;

/// How to call the shell tool. Shared by the worker's system prompt and the
/// reviewer's prompt: the reviewer inspects the workspace with the same shell,
/// so it must be told the same command format the worker is.
///
/// The "no other syntax" clause is load-bearing: the documented form below is
/// the one that reliably runs. [`shell::extract_shell_commands`] also recovers
/// a few drifted forms (`<cmd>`, `<bash>`, native `invoke name="shell"` tool
/// calls, JSON `"command"` arguments) as a safety net, but those are
/// best-effort - a model that invents its own syntax runs nothing and silently
/// loses its evidence.
pub(crate) const SHELL_INSTRUCTIONS: &str = r#"The ONLY way to run a command is a block of exactly this form (never in reasoning, never in markdown fences):
<shell>command goes here</shell>
Do not use any other tool-call syntax. In particular do NOT emit DeepSeek's native DSML
tool calls (invoke/parameter elements) or JSON tool calls: this harness does not execute
them, so nothing runs.
One command per block; batch several per reply. Blocks run automatically; output returns as [SHELL OUTPUT].
For example, to inspect the repo:
<shell>git log --oneline -3 && git status --short</shell>
Commands run in the current directory. The rest of the filesystem is read-only (bubblewrap/sandbox-exec), so do not write outside it; HOME is your real home directory and is read-only there."#;

/// The worker's system prompt: persona plus the shared shell instructions.
///
/// Built with `format!` rather than `concat!` because the shell instructions
/// live in one `const` shared with the reviewer, and `concat!` only accepts
/// literals.
fn system_prompt() -> String {
    format!("You are a helpful assistant with a local shell.\n{SHELL_INSTRUCTIONS}")
}

/// Token the model must include in its visible reply once the goal is done.
const GOAL_DONE: &str = "[GOAL COMPLETE]";

/// Default cap on consecutive goal turns; `0` means unlimited.
///
/// By default `/goal` keeps reprompting the model until it emits the completion
/// token and the reviewer verifies it, so a goal has no fixed turn budget.
/// Ctrl+C always interrupts the loop and keeps the goal for later. Set
/// `DS_MAX_GOAL_TURNS` to a positive number to impose a request budget (useful
/// for unattended runs or accounts sensitive to request volume).
const DEFAULT_MAX_GOAL_TURNS: usize = 0;

/// Optional hard cap on shell round-trips inside a single turn.
///
/// `0` means unlimited, which is the default: the loop keeps running shell
/// commands for as long as the model emits them, and Ctrl+C always interrupts
/// the turn. Set `DS_MAX_SHELL_ITERS` to a positive number to impose a request
/// budget.
const DEFAULT_MAX_SHELL_ITERS: usize = 0;

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

/// Install the Ctrl+C (SIGINT) handler that feeds `interrupt`.
fn spawn_sigint_handler(interrupt: &Interrupt) {
    let intr = interrupt.clone();
    tokio::spawn(async move {
        #[cfg(unix)]
        {
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
        }
        #[cfg(not(unix))]
        {
            loop {
                match tokio::signal::ctrl_c().await {
                    Ok(()) => intr.fire(),
                    Err(e) => {
                        eprintln!(
                            "{} could not install Ctrl+C handler: {e}",
                            paint("1;33", "[warn]")
                        );
                        break;
                    }
                }
            }
        }
    });
}

/// How a turn ended, so callers can tell a finished turn from an aborted one.
enum TurnResult {
    /// `claim` is the model's *unverified* claim that the active goal is done.
    /// `/goal` never acts on it directly: [`review_goal_claim`] checks it
    /// first.  A turn with no active goal always claims nothing.
    ///
    /// `report` is the visible text of the turn's last reply, handed to the
    /// reviewer as the claim under examination.
    Done {
        claim: bool,
        report: String,
    },
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
    /// Disable the current-directory sandbox and give commands full,
    /// unsandboxed filesystem access.
    unsandboxed: bool,
    /// Sign in through the browser even if a saved token exists.
    login: bool,
    /// Do not launch a browser during login; just print the URL.
    no_browser: bool,
    /// Print usage and exit.
    help: bool,
    /// Print the version and exit.
    version: bool,
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
            "-u" | "--unsandboxed" | "--no-sandbox" => parsed.args.unsandboxed = true,
            "--login" => parsed.args.login = true,
            "--no-browser" => parsed.args.no_browser = true,
            "-h" | "--help" => parsed.args.help = true,
            "-V" | "--version" => parsed.args.version = true,
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
     \x20 -u, --unsandboxed    give shell commands full, unsandboxed filesystem\n\
     \x20                      access (default: sandboxed to the current dir)\n\
     \x20     --login          sign in through the browser even if a token exists\n\
     \x20     --no-browser     during login, print the URL instead of opening it\n\
     \x20 -h, --help           show this help and exit\n\
     \x20 -V, --version        show the version and exit\n\
     \n\
     Model-proposed shell commands run without asking, in a sandbox rooted at\n\
     the directory the CLI was started from. Pass --unsandboxed to lift that.\n\
     With no saved credentials the tool opens chat.deepseek.com in your browser\n\
     and picks up the token once you have signed in - nothing to copy or paste.\n\
     Other settings are read from the environment or the user-wide secrets file\n\
     (see README)."
}

/// Home directory, honoring `HOME` on unix and `USERPROFILE` on Windows.
fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .filter(|h| !h.is_empty())
        .or_else(|| std::env::var_os("USERPROFILE").filter(|h| !h.is_empty()))
        .map(PathBuf::from)
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
    home_dir().map(|h| h.join(".config").join("deepseek-web-cli").join(".env"))
}

/// Path to the persistent prompt-history file.
///
/// Up/down at the prompt cycle through the lines typed in earlier sessions,
/// stored here. Lookup order:
///   1. `$DS_HISTORY_FILE` (explicit override)
///   2. `$XDG_CONFIG_HOME/deepseek-web-cli/history`
///   3. `~/.config/deepseek-web-cli/history`
///
/// Returns `None` when no home/config directory is known; history then simply
/// lives in memory for the session.
fn history_path() -> Option<PathBuf> {
    resolve_history_path(
        std::env::var_os("DS_HISTORY_FILE"),
        std::env::var_os("XDG_CONFIG_HOME"),
        home_dir(),
    )
}

/// Pure lookup behind [`history_path`], split out so it can be tested without
/// touching the process environment.
fn resolve_history_path(
    override_file: Option<std::ffi::OsString>,
    xdg_config_home: Option<std::ffi::OsString>,
    home: Option<PathBuf>,
) -> Option<PathBuf> {
    if let Some(p) = override_file {
        if !p.is_empty() {
            return Some(PathBuf::from(p));
        }
    }
    if let Some(dir) = xdg_config_home {
        if !dir.is_empty() {
            return Some(PathBuf::from(dir).join("deepseek-web-cli").join("history"));
        }
    }
    home.map(|h| h.join(".config").join("deepseek-web-cli").join("history"))
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
        // An empty variable counts as unset: a blank `AUTHORIZATION_TOKEN=`
        // exported by a wrapper script must not shadow the saved token.
        let already_set = std::env::var_os(&key).is_some_and(|v| !v.is_empty());
        if !already_set {
            std::env::set_var(&key, &value);
        }
    }
    Some(path)
}

/// Where a freshly harvested token is written when no override is set.
fn secrets_path_or_default() -> PathBuf {
    user_secrets_path().unwrap_or_else(|| PathBuf::from("~/.config/deepseek-web-cli/.env"))
}

/// How long to wait for the user to finish signing in.
const LOGIN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(300);

/// Sign in without asking the user to copy anything.
///
/// The user signs in to chat.deepseek.com in their own browser; the web app
/// stores the bearer token in local storage, and this reads it from there.
/// The token is then saved to the user-wide secrets file so later runs need no
/// browser at all.
fn interactive_login(no_browser: bool, rejected: &[String]) -> Result<String> {
    let path = secrets_path_or_default();
    let mut found = login::discover_token(rejected);

    if found.is_some() {
        // A previous browser sign-in is still on disk. Reuse it, but say so:
        // this is the browser's own session credential, not a separate API
        // key, and the user should know the CLI is riding on that session.
        println!(
            "{} reusing the DeepSeek session already stored in your browser \
             (this is the same session, not a separate key)",
            paint("2", "[login]")
        );
    } else {
        println!();
        println!("{}", paint("1;36", "Sign in to DeepSeek"));
        println!(
            "  {}",
            paint(
                "2",
                "A browser window will open. Sign in however you normally do"
            )
        );
        println!(
            "  {}",
            paint(
                "2",
                "(password, Google, phone) - there is nothing to copy or paste."
            )
        );
        if !no_browser && !login::open_browser(login::LOGIN_URL) {
            println!(
                "{} could not launch a browser; open this URL manually:",
                paint("1;33", "[warn]")
            );
        } else {
            println!("  {}", paint("2", "If it did not open, visit:"));
        }
        println!("    {}", paint("34", login::LOGIN_URL));
        println!();

        let interactive = std::io::stdin().is_terminal();
        let deadline = std::time::Instant::now() + LOGIN_TIMEOUT;
        let mut last_note = std::time::Instant::now();
        loop {
            std::thread::sleep(std::time::Duration::from_millis(750));
            if let Some(t) = login::discover_token(rejected) {
                found = Some(t);
                break;
            }
            if !interactive {
                anyhow::bail!(
                    "no DeepSeek token found in any browser profile (stdin is not a terminal, so\
                     \x20not waiting); set AUTHORIZATION_TOKEN or rerun `deepseek --login` in a terminal"
                );
            }
            if std::time::Instant::now() >= deadline {
                anyhow::bail!(
                    "timed out after {}s waiting for a sign-in; set AUTHORIZATION_TOKEN or try again",
                    LOGIN_TIMEOUT.as_secs()
                );
            }
            if last_note.elapsed() >= std::time::Duration::from_secs(20) {
                last_note = std::time::Instant::now();
                println!(
                    "{}",
                    paint("2", "  still waiting for you to finish signing in...")
                );
            }
        }
    }

    let found = found.ok_or_else(|| {
        anyhow::anyhow!("could not read a DeepSeek token from any browser profile")
    })?;
    login::save_token(&path, &found.token).with_context(|| {
        format!(
            "signed in via {}, but could not save the token to {}",
            found.source,
            path.display()
        )
    })?;
    println!(
        "{} signed in via {}; token saved to {}",
        paint("32", "[login]"),
        found.source,
        path.display()
    );
    Ok(found.token)
}

/// Build a client with the given credentials and settings.
///
/// Factored out so `/login` can swap in freshly harvested credentials without
/// restarting the process.
fn build_client(
    token: &str,
    session_id: &Option<String>,
    policy: api::RetryPolicy,
    http_timeout: std::time::Duration,
) -> Result<api::DeepSeekClient> {
    // Accept a bare token as well as a full `Bearer ...` value, so a token
    // harvested from the browser can be stored without a scheme prefix.
    let token = token.trim();
    let bearer = if token.to_ascii_lowercase().starts_with("bearer ") {
        token.to_string()
    } else {
        format!("Bearer {token}")
    };
    let mut client = api::DeepSeekClient::new(session_id.clone(), bearer)?;
    client.set_policy(policy);
    client.set_http_timeout(http_timeout);
    Ok(client)
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

/// The opening prompt for a goal that is being started or restarted.
///
/// Shared by `/goal <text>` and by `/resume` when the conversation being
/// resumed had an active goal: resuming the goal means putting the model back
/// to work on it, not just printing the saved text.
fn goal_start_prompt(goal: &str) -> String {
    format!(
        "Work toward the following goal, using the shell as needed. \
         When it is fully accomplished, include the token {GOAL_DONE} in your visible reply.\n\nGoal: {goal}"
    )
}

/// The prompt for every turn after the first while a goal is still active.
fn goal_continue_prompt() -> String {
    format!(
        "Continue working toward the active goal. \
         When it is fully accomplished, include the token {GOAL_DONE} in your visible reply."
    )
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

/// The separate conversation that checks `/goal` completion claims.
///
/// The worker's `[GOAL COMPLETE]` is only a *claim*: a model that is tired of
/// the task, or that misjudged its own work, can emit it anyway.  Every claim
/// is therefore handed to this reviewer, which
///
///   * runs in a brand-new conversation, so it is not biased by having
///     written the work (and cannot see the worker's reasoning),
///   * has the same shell access, so it can inspect the files, run the tests
///     and look at the actual artifacts instead of trusting the report,
///   * must end with an explicit verdict, which is parsed by
///     [`reviewer::parse_verdict`].
///
/// A missing or unreadable verdict counts as "not achieved": an unverified
/// claim never closes a goal.
struct Reviewer {
    bearer: String,
    session_id: Option<String>,
    policy: api::RetryPolicy,
    http_timeout: std::time::Duration,
}

impl Reviewer {
    fn new(
        bearer: &str,
        session_id: &Option<String>,
        policy: api::RetryPolicy,
        http_timeout: std::time::Duration,
    ) -> Self {
        Self {
            bearer: bearer.to_string(),
            session_id: session_id.clone(),
            policy,
            http_timeout,
        }
    }

    /// Have the reviewer examine `report` (the worker's completion claim) for
    /// `goal` and return its full reply, verdict line included.
    ///
    /// The review runs as an ordinary turn in a fresh conversation: it may
    /// issue shell commands and reads back their output, exactly like the
    /// worker, and `run_turn` returns the visible text of its last reply.
    async fn review(
        &self,
        goal: &str,
        report: &str,
        thinking: bool,
        search: bool,
        sandbox_root: &Option<std::path::PathBuf>,
        interrupt: &Interrupt,
    ) -> Result<String> {
        let mut client = build_client(
            &self.bearer,
            &self.session_id,
            self.policy,
            self.http_timeout,
        )?;
        client.set_cancel(interrupt.subscribe());
        // The review runs in a brand-new conversation, so it needs the same
        // opening system prompt the worker gets: that is where the shell tool
        // contract lives. Without it the reviewer only has whatever the review
        // prompt repeats and tends to invent its own command syntax.
        let prompt = format!(
            "{}\n\n{}",
            system_prompt(),
            reviewer::review_prompt(goal, report)
        );
        // The reviewer must not carry the active-goal reminder: it is not
        // working on the goal, it is judging it.
        let no_goal = None;
        let cfg = TurnConfig {
            thinking,
            search,
            sandbox_root: sandbox_root.clone(),
            debug_pending: false,
            goal: &no_goal,
            follow_up_suffix: Some(reviewer::VERDICT_REMINDER),
        };
        match run_turn(&mut client, ReplySource::Message(&prompt), &cfg, interrupt).await? {
            TurnResult::Done { report, .. } => Ok(report),
            TurnResult::Interrupted => anyhow::bail!("review interrupted"),
        }
    }
}

/// Whether a goal's opt-in turn budget has run out after `turn` turns.
///
/// `max_turns == 0` means unlimited (the default), so this is only ever true
/// when the user explicitly set `DS_MAX_GOAL_TURNS` to a positive value.
fn goal_budget_exhausted(max_turns: usize, turn: usize) -> bool {
    max_turns != 0 && turn > max_turns
}

/// Drive the active goal to a *verified* completion, with no turn limit unless
/// `DS_MAX_GOAL_TURNS` imposes one (see [`DEFAULT_MAX_GOAL_TURNS`]).
///
/// Shared by `/goal <text>` (which starts a new goal) and `/resume` (which
/// restarts the goal saved with the conversation it just loaded).  Because the
/// same loop runs in both cases, resuming a conversation with an active goal
/// puts the model back to work instead of merely printing the saved text.
///
/// The worker's `[GOAL COMPLETE]` is a claim, not a result: every claim is
/// checked by [`Reviewer`] in a separate conversation.  Only an explicit
/// `VERDICT: ACHIEVED` closes the goal.  A rejection (or a review that ends
/// without a readable verdict) is fed back to the worker as its next
/// instruction, so it keeps working on whatever the reviewer found missing.
///
/// `primed` is updated the same way as a normal turn: the system prompt is
/// prepended only when the conversation is brand new.
#[allow(clippy::too_many_arguments)]
async fn run_goal_turns(
    client: &mut api::DeepSeekClient,
    reviewer: &Reviewer,
    goal: &mut Option<String>,
    primed: &mut bool,
    label: &mut Option<String>,
    pending_title: &mut Option<String>,
    thinking: bool,
    search: bool,
    sandbox_root: Option<std::path::PathBuf>,
    debug_pending: bool,
    interrupt: &Interrupt,
) {
    let Some(text) = goal.clone() else { return };
    let max_turns = env_u64("DS_MAX_GOAL_TURNS", DEFAULT_MAX_GOAL_TURNS as u64) as usize;
    // Feedback from a rejected review, sent to the worker instead of the
    // generic "keep going" prompt.
    let mut feedback: Option<String> = None;
    let mut reviews = 0usize;
    let mut turn = 0usize;
    loop {
        turn += 1;
        if goal_budget_exhausted(max_turns, turn) {
            println!(
                "{} stopped after {max_turns} turns without a verified {GOAL_DONE}; \
                 goal kept - continue with /goal <text> or clear with /goal clear",
                paint("1;33", "[goal paused]")
            );
            break;
        }
        let base = match feedback.take() {
            Some(f) => f,
            None if turn == 1 => goal_start_prompt(&text),
            None => goal_continue_prompt(),
        };
        let base = if *primed {
            base
        } else {
            *primed = true;
            format!("{}\n\n{base}", system_prompt())
        };
        let message = with_goal(goal, &base);
        let cfg = TurnConfig {
            thinking,
            search,
            sandbox_root: sandbox_root.clone(),
            debug_pending,
            goal,
            follow_up_suffix: None,
        };
        let (claim, report) = match run_turn(
            client,
            ReplySource::Message(&message),
            &cfg,
            interrupt,
        )
        .await
        {
            Ok(TurnResult::Done { claim, report }) => (claim, report),
            Ok(TurnResult::Interrupted) => {
                // Stop reprompting; keep the goal so the user can resume it
                // with /goal.
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
        apply_pending_title(client, label, pending_title).await;
        // The first turn creates the conversation id; record the goal against
        // it now that one exists.
        persist_goal(client, goal);
        if !claim {
            println!("{} reprompting model (turn {turn})", paint("36", "[goal]"));
            continue;
        }

        // The worker says it is done. Verify before believing it.
        reviews += 1;
        println!("{} checking the completion claim", paint("36", "[review]"));
        let review = reviewer
            .review(&text, &report, thinking, search, &sandbox_root, interrupt)
            .await;
        match review {
            Ok(review) => match reviewer::parse_verdict(&review) {
                Some(true) => {
                    println!("{}", paint("32", "[goal completed]"));
                    println!(
                        "{} {} confirmed the goal",
                        paint("32", "[review]"),
                        paint("2", "reviewer")
                    );
                    *goal = None;
                    persist_goal(client, goal);
                    break;
                }
                Some(false) => {
                    println!(
                        "{} rejected the completion claim; sending the findings back",
                        paint("1;33", "[review]")
                    );
                    feedback = Some(reviewer::feedback_prompt(&text, &review));
                }
                None => {
                    println!(
                        "{} no readable verdict; treating the claim as unverified",
                        paint("1;33", "[review]")
                    );
                    feedback = Some(reviewer::feedback_prompt(&text, &review));
                }
            },
            Err(e) => {
                // A broken review must not silently accept the claim, but it
                // also must not discard the user's goal: pause and let them
                // retry.
                eprintln!("{} review failed: {e:#}", paint("1;31", "[error]"));
                println!(
                    "{} goal kept unverified - continue with /goal <text> or clear with /goal clear",
                    paint("1;33", "[goal paused]")
                );
                break;
            }
        }
        println!(
            "{} reprompting model (turn {turn}, {reviews} review{})",
            paint("36", "[goal]"),
            if reviews == 1 { "" } else { "s" }
        );
    }
}

/// Verify a completion claim that arrived from an ordinary turn.
///
/// `/goal` runs its own verify-then-reprompt loop ([`run_goal_turns`]);
/// plain messages, `/continue` and resumed replies can also carry a
/// `[GOAL COMPLETE]`, and those claims must not close a goal unchecked.  Here
/// the reviewer gets one look: on `VERDICT: ACHIEVED` the goal is cleared,
/// otherwise it stays active and the rejection is printed so the user (or the
/// next `/goal`) knows why.
#[allow(clippy::too_many_arguments)]
async fn handle_goal_claim(
    client: &mut api::DeepSeekClient,
    reviewer: &Reviewer,
    goal: &mut Option<String>,
    report: &str,
    thinking: bool,
    search: bool,
    sandbox_root: &Option<std::path::PathBuf>,
    interrupt: &Interrupt,
) {
    let Some(text) = goal.clone() else { return };
    println!("{} checking the completion claim", paint("36", "[review]"));
    match reviewer
        .review(&text, report, thinking, search, sandbox_root, interrupt)
        .await
    {
        Ok(review) => match reviewer::parse_verdict(&review) {
            Some(true) => {
                println!("{}", paint("32", "[goal completed]"));
                println!(
                    "{} {} confirmed the goal",
                    paint("32", "[review]"),
                    paint("2", "reviewer")
                );
                *goal = None;
                persist_goal(client, goal);
            }
            Some(false) => {
                println!(
                    "{} rejected the completion claim - goal kept; \
                     run /goal to keep working on it",
                    paint("1;33", "[review]")
                );
            }
            None => {
                println!(
                    "{} no readable verdict - goal kept unverified",
                    paint("1;33", "[review]")
                );
            }
        },
        Err(e) => eprintln!("{} review failed: {e:#}", paint("1;31", "[error]")),
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
    /// Renders the visible (non-reasoning) text as Markdown. Reasoning stays
    /// plain so its dimmed styling is not rewritten by the Markdown pass.
    md: markdown::MarkdownStream,
}

impl Renderer {
    fn new() -> Self {
        Self {
            in_think: false,
            buf: String::new(),
            md: markdown::MarkdownStream::new(),
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
        self.md.finish();
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

    fn emit(&mut self, s: &str, visible: &mut String, reasoning: &mut String) {
        if s.is_empty() {
            return;
        }
        if self.in_think {
            out(s);
            reasoning.push_str(s);
        } else {
            self.md.push(s);
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
        self.md.finish();
        if self.in_think {
            out(&format!("{}\n", color::reset()));
            self.in_think = false;
        }
    }
}

/// What the first reply of a turn should ask the server for.
#[derive(Clone, Copy)]
enum ReplySource<'a> {
    /// A fresh `chat/completion` for this message.
    Message(&'a str),
    /// Resume the current assistant message via `chat/continue`.
    Continue,
}

/// Send one message, render the stream, and return both the visible
/// (non-thinking) text and the reasoning text.  The caller normally extracts
/// shell commands from the visible text, but falls back to the reasoning text
/// when the model mistakenly emits the tool call inside its thinking region.
///
/// With [`ReplySource::Continue`] the reply is resumed through
/// `chat/continue` instead of a fresh `chat/completion`.
///
/// A reply the server cuts short mid-generation is resumed automatically: the
/// stream reports a [`api::Truncated`] error after the partial text has been
/// rendered and the conversation tip recorded, so the very endpoint `/continue`
/// drives is re-opened without the user having to type it. There is no cap on
/// these resumes: a truncated reply keeps being resumed until it completes or
/// the user interrupts with Ctrl+C.
async fn stream_reply(
    client: &mut api::DeepSeekClient,
    source: ReplySource<'_>,
    thinking: bool,
    search: bool,
) -> Result<(String, String)> {
    let mut resumes = 0usize;
    let mut source = source;
    // Text from earlier, cut-short attempts is kept so command extraction sees
    // the whole reply rather than only the piece that arrived after resuming.
    let mut visible = String::new();
    let mut reasoning = String::new();
    loop {
        let (v, r, err) = stream_once(client, source, thinking, search).await;
        visible.push_str(&v);
        reasoning.push_str(&r);
        let Some(err) = err else {
            return Ok((visible, reasoning));
        };
        // Only a mid-generation cut is resumable; anything else (auth, bad
        // request, exhausted retries) belongs to the caller.
        if !api::is_truncated(&err) {
            return Err(err);
        }
        resumes += 1;
        println!(
            "{} {} - resuming automatically (attempt {resumes})",
            paint("1;33", "[cut short]"),
            format_args!("{err}"),
        );
        source = ReplySource::Continue;
    }
}

/// One streaming attempt. Returns the text rendered so far plus the error, if
/// any, so a cut-short reply can keep its partial output. The renderer is
/// always finished (restoring terminal color) before returning.
async fn stream_once(
    client: &mut api::DeepSeekClient,
    source: ReplySource<'_>,
    thinking: bool,
    search: bool,
) -> (String, String, Option<anyhow::Error>) {
    let mut s = match match source {
        ReplySource::Message(message) => client.send_message(message, thinking, search).await,
        ReplySource::Continue => client.continue_message().await,
    } {
        Ok(s) => s,
        Err(e) => return (String::new(), String::new(), Some(e)),
    };
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
    (visible, reasoning, stream_err)
}

/// Send `message`, then keep extracting and running shell commands until the
/// model stops asking for them.  Returns `true` if the model signalled the
/// active goal was completed at any point during the turn.
/// Turn-wide settings, bundled so `run_turn` does not take a long argument
/// list of booleans (which are easy to mix up at the call site).
struct TurnConfig<'a> {
    thinking: bool,
    search: bool,
    /// `Some(root)` runs commands sandboxed to `root`; `None` is unsandboxed.
    sandbox_root: Option<std::path::PathBuf>,
    debug_pending: bool,
    goal: &'a Option<String>,
    /// Appended to every shell-output follow-up in this turn.
    ///
    /// The reviewer needs this: its conversation has no system prompt and its
    /// follow-up turns carry only the command output, so without a reminder it
    /// can finish without the verdict line and the claim goes unverified.
    follow_up_suffix: Option<&'a str>,
}

async fn run_turn(
    client: &mut api::DeepSeekClient,
    first: ReplySource<'_>,
    cfg: &TurnConfig<'_>,
    interrupt: &Interrupt,
) -> Result<TurnResult> {
    let mut rx = interrupt.subscribe();
    let (thinking, search) = (cfg.thinking, cfg.search);
    let sandbox_root = cfg.sandbox_root.as_deref();
    let debug_pending = cfg.debug_pending;
    let goal = cfg.goal;

    let (mut pending, mut reasoning) = tokio::select! {
        r = stream_reply(client, first, thinking, search) => r?,
        _ = rx.changed() => {
            println!("\n{}", paint("1;33", "[interrupted]"));
            return Ok(TurnResult::Interrupted);
        }
    };
    // A completion token anywhere in the turn is a *claim*; it is verified by
    // the reviewer before the goal is actually closed.
    let mut claim = goal_completed(&pending) || goal_completed(&reasoning);
    let mut iter = 0usize;

    // Shell loop: keep extracting commands from each reply
    // (initial + every follow-up) until the model stops. The loop is unlimited
    // by default; `DS_MAX_SHELL_ITERS` can impose an optional request budget.
    let max_iters = env_u64("DS_MAX_SHELL_ITERS", DEFAULT_MAX_SHELL_ITERS as u64) as usize;
    loop {
        iter += 1;
        if max_iters != 0 && iter > max_iters {
            println!(
                "{} stopped after {max_iters} shell round-trips in one turn",
                paint("1;33", "[shell limit]")
            );
            break;
        }

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
            // Commands are auto-approved by default; the sandbox, not a
            // confirmation prompt, is what bounds what they may touch.
            let label = shell::sandbox_label(sandbox_root);
            match sandbox_root {
                Some(root) => println!("{}", paint("2", &format!("[{label}: {}]", root.display()))),
                None => println!("{}", paint("1;33", &format!("[{label}]"))),
            }
            let output = match tokio::select! {
                r = shell::run_shell(&cmd_text, cmd_timeout, sandbox_root) => r,
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

        let mut follow_up = with_goal(goal, &format!("[SHELL OUTPUT]{combined}"));
        if let Some(suffix) = cfg.follow_up_suffix {
            follow_up.push_str("\n\n");
            follow_up.push_str(suffix);
        }
        let (v, r) = tokio::select! {
            r = stream_reply(client, ReplySource::Message(&follow_up), thinking, search) => r?,
            _ = rx.changed() => {
                println!("\n{}", paint("1;33", "[interrupted]"));
                return Ok(TurnResult::Interrupted);
            }
        };
        if goal_completed(&v) || goal_completed(&r) {
            claim = true;
        }
        pending = v;
        reasoning = r;
    }
    let report = if pending.trim().is_empty() {
        reasoning.trim().to_string()
    } else {
        pending.trim().to_string()
    };
    Ok(TurnResult::Done { claim, report })
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
        // Entries are kept oldest-first, so the first row is the oldest loaded
        // and paging continues backwards in time from there.
        let cursor_owned = self
            .entries
            .first()
            .map(|first| (first.updated_at, first.id.clone()));
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
    ///
    /// After merging, the list is sorted oldest-first by `updated_at`, so a
    /// server that returns pages (or rows inside a page) in a different order
    /// still yields a chronological listing. Sorting ascending also keeps the
    /// cursor sane: `fetch_next` pages from `entries.first()`, which is the
    /// oldest row loaded so far.
    fn merge_page(&mut self, page: Vec<sessions::SessionEntry>, has_more: bool) -> usize {
        let before = self.entries.len();
        for e in page {
            if !self.entries.iter().any(|x| x.id == e.id) {
                self.entries.push(e);
            }
        }
        let added = self.entries.len() - before;
        self.entries.sort_by(|a, b| {
            a.updated_at
                .partial_cmp(&b.updated_at)
                .unwrap_or(std::cmp::Ordering::Equal)
                // Stable tie-break so paging and numbering are deterministic.
                .then_with(|| a.id.cmp(&b.id))
        });
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
            // Entries are kept oldest-first, so an older page is prepended and
            // every row's number shifts; reprint the whole list.
            Ok(_) => print_loaded_sessions(cache, 0),
            Err(e) => eprintln!(
                "{} could not list sessions: {e:#}",
                paint("1;31", "[error]")
            ),
        }
        return;
    }

    // A plain `/sessions` restarts from the most recent page of conversations,
    // listed oldest-to-newest.
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
    if parsed.args.version {
        println!("deepseek {}", env!("CARGO_PKG_VERSION"));
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

    // Credentials: an explicit environment/file token wins, otherwise sign in
    // through the browser.  `--login` forces a fresh browser sign-in.
    let bearer = if args.login {
        interactive_login(args.no_browser, &[])?
    } else {
        match std::env::var("AUTHORIZATION_TOKEN")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
        {
            Some(t) => t,
            None => interactive_login(args.no_browser, &[])?,
        }
    };
    // Optional: only the bearer token is checked by the API (see
    // `DeepSeekClient::new`).  Treat an empty value as absent so a `.env` line
    // like `DS_SESSION_ID=` does not send a blank cookie.
    let session_id = std::env::var("DS_SESSION_ID")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty());

    let debug_pending = env_flag("DS_DEBUG_PENDING", false);
    // Commands are auto-approved. Unless `--unsandboxed` is given they run
    // sandboxed to the directory the CLI was started from.
    let sandbox_root = if args.unsandboxed {
        None
    } else {
        let root = std::env::current_dir().context("could not determine the current directory")?;
        if !shell::os_sandbox_available() {
            eprintln!(
                "{} no OS sandbox helper (bubblewrap on Linux, sandbox-exec on macOS) found; \
                 commands run with the current directory as their working directory, \
                 but absolute paths are not confined. Use --unsandboxed to silence this.",
                paint("1;33", "[warn]")
            );
        }
        Some(root)
    };
    let thinking = env_flag("DS_THINKING", true);
    let search = env_flag("DS_SEARCH", true);

    // Rate-limit policy, tunable from the environment.
    //
    // Defaults are deliberately generous: retries start after a real pause,
    // grow fast, and wait a long time at the ceiling. The spacing (not a fixed
    // attempt count) is what keeps the traffic polite, so there is no retry
    // cap by default; `DS_MAX_RETRIES` can still set one explicitly.
    let policy = api::RetryPolicy {
        base: std::time::Duration::from_millis(env_u64("DS_RETRY_BASE_MS", 2_000)),
        cap: std::time::Duration::from_millis(env_u64("DS_RETRY_CAP_MS", 300_000)),
        min_gap: std::time::Duration::from_millis(env_u64("DS_MIN_GAP_MS", 1_500)),
        max_attempts: env_u64("DS_MAX_RETRIES", 0) as u32,
    };
    eprintln!(
        "[rate-limit] max_retries={} base={:?} cap={:?} min_gap={:?}",
        if policy.max_attempts == 0 {
            "unlimited".to_string()
        } else {
            policy.max_attempts.to_string()
        },
        policy.base,
        policy.cap,
        policy.min_gap
    );
    let http_timeout = std::time::Duration::from_millis(env_u64("DS_HTTP_TIMEOUT_MS", 180_000));
    // Route SIGINT to cooperative cancellation so Ctrl+C aborts the current
    // turn instead of killing the process while the model streams or a shell
    // command runs.
    let interrupt = Interrupt::new();
    spawn_sigint_handler(&interrupt);

    let mut client = build_client(&bearer, &session_id, policy, http_timeout)?;
    client.set_cancel(interrupt.subscribe());
    // The independent agent that verifies `/goal` completion claims.
    let mut reviewer = Reviewer::new(&bearer, &session_id, policy, http_timeout);

    // Prove the token still works before starting the prompt loop.
    //
    // A rejected token is NOT silently retried. Re-harvesting the browser
    // token on every failure was the root cause of the "accounts get logged
    // out" reports: each rejected request causes the server to rotate the
    // session, which in turn signs the user out of the browser they logged in
    // with. Worse, retrying three times meant three rotations per start.
    //
    // Now a rejected token fails fast with an explicit instruction. Signing in
    // again is something the user asks for, never something the tool does
    // behind their back.
    match client.check_auth().await {
        Ok(email) => {
            if env_flag("DS_DEBUG_SECRETS", false) {
                eprintln!("[auth] signed in as {email}");
            }
        }
        Err(e) if api::is_auth_error(&format!("{e:#}")) => {
            eprintln!(
                "{} DeepSeek rejected the saved token: {e:#}",
                paint("1;31", "[error]")
            );
            eprintln!(
                "{} the CLI will not keep re-reading it from your browser: doing so rotates the\
                 \n  session and signs you out there. Remove the stale token from {} and run\
                 \n  `deepseek --login` when you are ready to sign in again.",
                paint("2", "note:"),
                secrets_path_or_default().display()
            );
            std::process::exit(1);
        }
        Err(e) => {
            // Anything else (DNS hiccup, captive portal, ...) is not proof the
            // token is bad; warn and let the turn-level retry logic deal with
            // it.
            eprintln!(
                "{} could not verify credentials yet: {e:#}",
                paint("1;33", "[warn]")
            );
        }
    }

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

    // Prompt history: remember what was typed (including slash commands) and
    // make it reachable with the up/down arrow keys, across runs too.
    let history_file = history_path();
    let rl_config = Config::builder()
        .max_history_size(1000)?
        .history_ignore_space(true)
        .history_ignore_dups(true)?
        .auto_add_history(false)
        .build();
    let mut rl = DefaultEditor::with_config(rl_config)?;
    if let Some(path) = &history_file {
        match rl.load_history(path) {
            Ok(()) => {}
            // No history yet is the normal first-run case, not an error.
            Err(ReadlineError::Io(ref e)) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => eprintln!(
                "{} could not read history from {}: {e}",
                paint("1;33", "[warn]"),
                path.display()
            ),
        }
    }

    println!("{}", paint("1;36", "DeepSeek Web CLI \u{2014} multi-turn"));
    println!("  {}   new conversation", paint("32", "/new"));
    println!("  {}   show conversation tip", paint("33", "/id"));
    println!("  {}   set/start/show/clear goal", paint("35", "/goal"));
    println!(
        "  {}   resume the interrupted reply",
        paint("34", "/continue")
    );
    println!(
        "  {}   list conversations ({} to page)",
        paint("34", "/sessions"),
        paint("2", "/sessions more")
    );
    println!("  {}   resume a saved conversation", paint("34", "/resume"));
    println!(
        "  {}   rename the current conversation",
        paint("34", "/label")
    );
    println!("  {}   forget a saved conversation", paint("34", "/forget"));
    println!(
        "  {}   sign in again through the browser",
        paint("34", "/login")
    );
    println!("  {}  drop the saved token", paint("34", "/logout"));
    println!("  {}   quit (Ctrl+D also quits)", paint("31", "exit"));
    println!(
        "  {} Ctrl+C cancels the current reply/command (or clears the prompt)",
        paint("2", "tip:")
    );
    println!(
        "  {} up/down recall earlier prompts and commands",
        paint("2", "tip:")
    );
    println!();

    loop {
        // Give the client a fresh view of the interrupt channel so retry
        // backoffs and session calls can be cancelled with Ctrl+C too.
        client.set_cancel(interrupt.subscribe());
        let prompt = paint("1;35", ">> ");
        let line = rl.readline(&prompt);
        // Remember the line so up/down can recall it, but keep blank input and
        // whitespace-led secrets (already excluded by `history_ignore_space`)
        // out of the history file.
        if let Ok(input) = &line {
            if !input.trim().is_empty() {
                if let Err(e) = rl.add_history_entry(input.as_str()) {
                    eprintln!("{} could not record history: {e}", paint("1;33", "[warn]"));
                }
            }
        }
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
                                // Resuming a conversation with an active goal
                                // means putting the model back to work on it,
                                // not just printing the saved text.
                                run_goal_turns(
                                    &mut client,
                                    &reviewer,
                                    &mut goal,
                                    &mut primed,
                                    &mut label,
                                    &mut pending_title,
                                    thinking,
                                    search,
                                    sandbox_root.clone(),
                                    debug_pending,
                                    &interrupt,
                                )
                                .await;
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
            Ok(input) if input.trim() == "/login" || input.trim() == "/logout" => {
                if input.trim() == "/logout" {
                    // Forget the token: remove it from the environment and the
                    // secrets file, so the next run signs in again.
                    std::env::remove_var("AUTHORIZATION_TOKEN");
                    let path = secrets_path_or_default();
                    match std::fs::read_to_string(&path) {
                        Ok(text) => {
                            let kept = login::remove_key(&text, "AUTHORIZATION_TOKEN");
                            match std::fs::write(&path, kept) {
                                Ok(()) => {
                                    println!(
                                        "{} token removed from {}",
                                        paint("32", "[logout]"),
                                        path.display()
                                    );
                                    println!(
                                        "{} the browser session is untouched: sign out there too, \
                                         otherwise the next run signs in with it again",
                                        paint("2", "note:")
                                    );
                                }
                                Err(e) => eprintln!(
                                    "{} could not rewrite {}: {e:#}",
                                    paint("1;31", "[error]"),
                                    path.display()
                                ),
                            }
                        }
                        Err(e) => eprintln!(
                            "{} could not read {}: {e:#}",
                            paint("1;31", "[error]"),
                            path.display()
                        ),
                    }
                    continue;
                }
                // A fresh browser sign-in, replacing the current credentials.
                match interactive_login(args.no_browser, &[]) {
                    Ok(token) => match build_client(&token, &session_id, policy, http_timeout) {
                        Ok(mut fresh) => match fresh.check_auth().await {
                            Ok(email) => {
                                fresh.set_cancel(interrupt.subscribe());
                                client = fresh;
                                // The reviewer talks to the same account with
                                // the same (possibly rotated) credential.
                                reviewer.bearer = token.clone();
                                println!("{} signed in as {email}", paint("32", "[login]"));
                            }
                            Err(e) => eprintln!(
                                "{} signed in, but the token was rejected: {e:#}",
                                paint("1;31", "[error]")
                            ),
                        },
                        Err(e) => eprintln!("{} {e:#}", paint("1;31", "[error]")),
                    },
                    Err(e) => eprintln!("{} sign-in failed: {e:#}", paint("1;31", "[error]")),
                }
                continue;
            }
            Ok(input) if input.trim() == "/continue" => {
                // Resume the current (interrupted) assistant reply instead of
                // paying for a fresh completion. chat/continue threads from the
                // message id the stream last reported, which `last_message_id`
                // already tracks.
                if client.chat_session_id().is_none() || client.last_message_id().is_none() {
                    println!(
                        "{} nothing to continue \u{2014} send a message first",
                        paint("1;33", "[warn]")
                    );
                    continue;
                }
                println!(
                    "{} resuming message {}",
                    paint("36", "[continue]"),
                    client
                        .last_message_id()
                        .map(|v| v.to_string())
                        .unwrap_or_default()
                );
                let cfg = TurnConfig {
                    thinking,
                    search,
                    sandbox_root: sandbox_root.clone(),
                    debug_pending,
                    goal: &goal,
                    follow_up_suffix: None,
                };
                match run_turn(&mut client, ReplySource::Continue, &cfg, &interrupt).await {
                    Ok(TurnResult::Done { claim, report }) => {
                        apply_pending_title(&client, &mut label, &mut pending_title).await;
                        if claim && goal.is_some() {
                            // A resumed reply can carry a completion claim
                            // too; verify it like any other.
                            handle_goal_claim(
                                &mut client,
                                &reviewer,
                                &mut goal,
                                &report,
                                thinking,
                                search,
                                &sandbox_root,
                                &interrupt,
                            )
                            .await;
                        }
                        persist_goal(&client, &goal);
                    }
                    Ok(TurnResult::Interrupted) => {}
                    Err(e) => eprintln!("{} {e:#}", paint("1;31", "[error]")),
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
                        // Start the model and reprompt it until it marks the
                        // goal complete (no turn limit by default; see
                        // DEFAULT_MAX_GOAL_TURNS).
                        run_goal_turns(
                            &mut client,
                            &reviewer,
                            &mut goal,
                            &mut primed,
                            &mut label,
                            &mut pending_title,
                            thinking,
                            search,
                            sandbox_root.clone(),
                            debug_pending,
                            &interrupt,
                        )
                        .await;
                    }
                }
                continue;
            }
            Ok(input) => {
                let message = if primed {
                    input.clone()
                } else {
                    primed = true;
                    format!("{}\n\n{input}", system_prompt())
                };
                let message = with_goal(&goal, &message);
                let cfg = TurnConfig {
                    thinking,
                    search,
                    sandbox_root: sandbox_root.clone(),
                    debug_pending,
                    goal: &goal,
                    follow_up_suffix: None,
                };
                match run_turn(
                    &mut client,
                    ReplySource::Message(&message),
                    &cfg,
                    &interrupt,
                )
                .await
                {
                    Ok(TurnResult::Done { claim, report }) => {
                        apply_pending_title(&client, &mut label, &mut pending_title).await;
                        if claim && goal.is_some() {
                            // Verify the completion claim before closing the
                            // goal, exactly like `/goal` does.
                            handle_goal_claim(
                                &mut client,
                                &reviewer,
                                &mut goal,
                                &report,
                                thinking,
                                search,
                                &sandbox_root,
                                &interrupt,
                            )
                            .await;
                        }
                        // Sync the store: a new conversation gets its goal
                        // recorded, and a verified completion clears it.
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

    // Persist the session's history so the next run can recall it.
    if let Some(path) = &history_file {
        if let Some(dir) = path.parent() {
            if let Err(e) = std::fs::create_dir_all(dir) {
                eprintln!(
                    "{} could not create history directory {}: {e}",
                    paint("1;33", "[warn]"),
                    dir.display()
                );
            }
        }
        if let Err(e) = rl.save_history(path) {
            eprintln!(
                "{} could not save history to {}: {e}",
                paint("1;33", "[warn]"),
                path.display()
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        goal_budget_exhausted, goal_completed, goal_continue_prompt, goal_start_prompt, parse_args,
        parse_goal_command, parse_secrets, usage, with_goal, GoalCommand, ParsedArgs, SessionCache,
        DEFAULT_MAX_GOAL_TURNS,
    };
    use crate::sessions;

    #[test]
    fn goal_turns_are_unlimited_by_default() {
        // The shipped default is "no budget": even a very long goal keeps
        // reprompting instead of pausing at some fixed turn count.
        assert_eq!(DEFAULT_MAX_GOAL_TURNS, 0, "default must mean unlimited");
        for turn in [1, 25, 26, 1_000, usize::MAX / 2] {
            assert!(
                !goal_budget_exhausted(DEFAULT_MAX_GOAL_TURNS, turn),
                "goal must not pause at turn {turn} with the default budget"
            );
        }
    }

    #[test]
    fn an_opt_in_budget_still_pauses() {
        assert!(!goal_budget_exhausted(3, 1));
        assert!(!goal_budget_exhausted(3, 3));
        assert!(goal_budget_exhausted(3, 4), "turn 4 exceeds a budget of 3");
        assert!(goal_budget_exhausted(1, 2));
    }

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
    fn goal_start_prompt_names_the_goal_and_the_done_token() {
        // `/resume` reuses this prompt, so a restored goal actually sends the
        // model back to work instead of only printing the saved text.
        let p = goal_start_prompt("fix the bug");
        assert!(p.contains("Goal: fix the bug"), "{p}");
        assert!(p.contains("[GOAL COMPLETE]"), "{p}");
        assert!(p.contains("Work toward the following goal"), "{p}");
    }

    #[test]
    fn goal_continue_prompt_asks_to_continue() {
        let p = goal_continue_prompt();
        assert!(p.contains("Continue working toward the active goal"), "{p}");
        assert!(p.contains("[GOAL COMPLETE]"), "{p}");
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
            vec!["c", "b", "a"]
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

    #[test]
    fn merge_orders_oldest_first() {
        let mut c = SessionCache::default();
        // Server returns rows out of order; the cache must normalise them.
        c.merge_page(
            vec![entry("old", 1.0), entry("new", 3.0), entry("mid", 2.0)],
            false,
        );
        assert_eq!(
            c.entries.iter().map(|e| e.id.as_str()).collect::<Vec<_>>(),
            vec!["old", "mid", "new"]
        );
    }

    #[test]
    fn merge_keeps_oldest_first_across_pages() {
        let mut c = SessionCache::default();
        c.merge_page(vec![entry("a", 3.0), entry("b", 2.0)], true);
        // An older page arrives late and is not in order either.
        c.merge_page(vec![entry("d", 1.0), entry("c", 1.5)], false);
        assert_eq!(
            c.entries.iter().map(|e| e.id.as_str()).collect::<Vec<_>>(),
            vec!["d", "c", "b", "a"]
        );
        // The cursor for the next page is the oldest row loaded.
        assert_eq!(c.entries.first().unwrap().id, "d");
    }

    #[test]
    fn latest_session_is_last_in_the_list() {
        // The server hands back the most recently updated conversations first;
        // the cache must keep them so the latest one prints at the end of the
        // list, not the beginning.
        let mut c = SessionCache::default();
        c.merge_page(
            vec![
                entry("newest", 30.0),
                entry("middle", 20.0),
                entry("oldest", 10.0),
            ],
            true,
        );
        c.merge_page(vec![entry("older", 5.0)], false);
        let ids: Vec<&str> = c.entries.iter().map(|e| e.id.as_str()).collect();
        assert_eq!(ids.last().copied(), Some("newest"));
        assert_eq!(ids.first().copied(), Some("older"));
        assert_eq!(ids, vec!["older", "oldest", "middle", "newest"]);
    }

    #[test]
    fn history_round_trips_through_a_file() {
        // The same builder/config the prompt loop uses: entries added during a
        // session survive a save/reload cycle, so up/down works across runs.
        use rustyline::{Config, DefaultEditor};

        let dir = std::env::temp_dir().join(format!("ds-hist-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("history");

        let config = Config::builder()
            .max_history_size(1000)
            .unwrap()
            .history_ignore_space(true)
            .history_ignore_dups(true)
            .unwrap()
            .auto_add_history(false)
            .build();
        let mut rl = DefaultEditor::with_config(config).unwrap();
        // A missing file on first run is not an error.
        assert!(rl.load_history(&path).is_err());
        rl.add_history_entry("hello").unwrap();
        rl.add_history_entry("hello").unwrap(); // consecutive dup dropped
        rl.add_history_entry("/goal ship it").unwrap();
        rl.add_history_entry(" secret").unwrap(); // leading space dropped
        rl.save_history(&path).unwrap();

        let mut rl2 = DefaultEditor::with_config(
            Config::builder()
                .max_history_size(1000)
                .unwrap()
                .history_ignore_space(true)
                .history_ignore_dups(true)
                .unwrap()
                .auto_add_history(false)
                .build(),
        )
        .unwrap();
        rl2.load_history(&path).unwrap();
        let entries: Vec<String> = rl2.history().iter().map(|e| e.to_string()).collect();
        assert_eq!(
            entries,
            vec!["hello".to_string(), "/goal ship it".to_string()]
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn history_path_prefers_explicit_override() {
        use super::resolve_history_path;
        use std::ffi::OsString;

        // $DS_HISTORY_FILE wins over everything else.
        assert_eq!(
            resolve_history_path(
                Some(OsString::from("/tmp/h")),
                Some(OsString::from("/xdg")),
                Some("/home/u".into()),
            ),
            Some("/tmp/h".into())
        );
        // An empty override falls through to the XDG config dir.
        assert_eq!(
            resolve_history_path(
                Some(OsString::new()),
                Some(OsString::from("/xdg")),
                Some("/home/u".into()),
            ),
            Some("/xdg/deepseek-web-cli/history".into())
        );
        // Without either, the home config dir is used.
        assert_eq!(
            resolve_history_path(None, None, Some("/home/u".into())),
            Some("/home/u/.config/deepseek-web-cli/history".into())
        );
        assert_eq!(resolve_history_path(None, None, None), None);
    }

    fn parse(v: &[&str]) -> ParsedArgs {
        parse_args(v.iter().map(|s| s.to_string()))
    }

    #[test]
    fn sandbox_is_on_by_default() {
        let p = parse(&[]);
        assert!(!p.args.unsandboxed);
        assert!(!p.args.help);
        assert!(p.unknown.is_empty());
    }

    #[test]
    fn accepts_unsandboxed_flags() {
        assert!(parse(&["--unsandboxed"]).args.unsandboxed);
        assert!(parse(&["--no-sandbox"]).args.unsandboxed);
        assert!(parse(&["-u"]).args.unsandboxed);
    }

    #[test]
    fn accepts_help_flag() {
        assert!(parse(&["--help"]).args.help);
        assert!(parse(&["-h"]).args.help);
    }

    #[test]
    fn accepts_version_flag() {
        assert!(parse(&["--version"]).args.version);
        assert!(parse(&["-V"]).args.version);
        // `-v` (lowercase) is not a documented alias and stays an error.
        assert_eq!(parse(&["-v"]).unknown, vec!["-v"]);
    }

    #[test]
    fn help_text_documents_version_flag() {
        assert!(usage().contains("--version"));
    }

    #[test]
    fn collects_unknown_arguments_instead_of_ignoring_them() {
        let p = parse(&["--unsandbaxed", "extra"]);
        assert!(!p.args.unsandboxed);
        assert_eq!(p.unknown, vec!["--unsandbaxed", "extra"]);
    }

    #[test]
    fn auto_approve_flag_is_gone() {
        // Auto-approval is the default now; the old opt-in flag is unknown.
        let p = parse(&["--auto-approve"]);
        assert_eq!(p.unknown, vec!["--auto-approve"]);
    }

    #[test]
    fn flags_can_be_combined() {
        let p = parse(&["-u", "--help"]);
        assert!(p.args.unsandboxed);
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
