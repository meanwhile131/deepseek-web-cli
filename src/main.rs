mod api;
mod color;
mod pow;
mod shell;

use anyhow::Result;
use futures_util::StreamExt;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use std::io::Write;
use color::paint;

const SYSTEM_PROMPT: &str = r#"You are a helpful assistant with access to a local shell.
When you need to run a command, emit it exactly as:
<shell>command goes here</shell>
The user will execute it and reply with [SHELL OUTPUT] followed by the result.
Use one <shell> block per command.  Do not wrap commands in markdown fences.
Use multiple commands per response if possible.
Command blocks must appear in your visible reply, never inside private reasoning."#;

const MAX_SHELL_ITERS: usize = 32;
/// Safety cap on how many times we auto-reprompt the model to finish a goal.
const MAX_GOAL_REPROMPTS: usize = 24;
/// Token the model must include in its visible reply once the goal is done.
const GOAL_DONE: &str = "[GOAL COMPLETE]";

const THINK_OPEN: &str = "\u{27EA}THINK\u{27EB}";
const THINK_CLOSE: &str = "\u{27EA}/THINK\u{27EB}";

fn out(s: &str) {
    print!("{s}");
    std::io::stdout().flush().ok();
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

fn env_u32(name: &str, default: u32) -> u32 {
    env_u64(name, default as u64).min(u32::MAX as u64) as u32
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
async fn run_turn(
    client: &mut api::DeepSeekClient,
    rl: &mut DefaultEditor,
    message: &str,
    thinking: bool,
    search: bool,
    auto_approve: bool,
    debug_pending: bool,
    goal: &Option<String>,
) -> Result<bool> {
    let (mut pending, mut reasoning) =
        stream_reply(client, message, thinking, search).await?;
    let mut completed = goal_completed(&pending) || goal_completed(&reasoning);
    let mut iter = 0usize;

    // Shell loop: keep extracting commands from each reply
    // (initial + every follow-up) until the model stops.
    loop {
        iter += 1;
        if iter > MAX_SHELL_ITERS {
            println!("{} shell loop safety limit reached", paint("1;33", "[warn]"));
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
                eprintln!(
                    "[warn] pending contains a command tag but extraction returned nothing"
                );
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
            println!("\n{} {}", paint("1;35", "[shell]"), paint("35", &cmd));
            if auto_approve {
                println!("{}", paint("2", "execute? [auto-approved]"));
            } else {
                // Use the rustyline editor so we never read raw
                // stdin behind its back.
                let approve_prompt = format!("{} ", paint("33", "execute? [y/N]"));
                let approved = match rl.readline(&approve_prompt) {
                    Ok(a) => a.trim().eq_ignore_ascii_case("y"),
                    Err(_) => false,
                };
                if !approved {
                    println!("{}", paint("2", "skipped"));
                    continue;
                }
            }
            // A failing command must not kill the session.
            let output = match shell::run_shell(&cmd).await {
                Ok(o) => o,
                Err(e) => format!("[error running command: {e}]"),
            };
            println!("{}\n{output}\n{}", paint("2", "--- output ---"), paint("2", "--------------"));
            combined.push_str(&format!("\n=== CMD: {cmd}\n{output}\n"));
        }

        if combined.is_empty() {
            println!("{}", paint("2", "[all commands skipped]"));
            break;
        }

        let follow_up =
            with_goal(goal, &format!("[SHELL OUTPUT]{combined}"));
        match stream_reply(client, &follow_up, thinking, search).await {
            Ok((v, r)) => {
                if goal_completed(&v) || goal_completed(&r) {
                    completed = true;
                }
                pending = v;
                reasoning = r;
            }
            Err(e) => return Err(e),
        }
    }
    Ok(completed)
}

#[tokio::main]
async fn main() -> Result<()> {
    let session_id =
        std::env::var("DS_SESSION_ID").expect("set DS_SESSION_ID (ds_session_id cookie)");
    let bearer = std::env::var("AUTHORIZATION_TOKEN")
        .expect("set AUTHORIZATION_TOKEN (full 'Bearer ...' value)");

    let debug_pending = env_flag("DS_DEBUG_PENDING", false);
    // Default OFF: executing model-authored commands without confirmation is opt-in.
    let auto_approve = env_flag("DS_AUTO_APPROVE", true);
    let thinking = env_flag("DS_THINKING", true);
    let search = env_flag("DS_SEARCH", true);

    let mut client = api::DeepSeekClient::new(session_id, bearer)?;

    // Rate-limit policy, tunable from the environment.
    let policy = api::RetryPolicy {
        max_retries: env_u32("DS_MAX_RETRIES", 6),
        base: std::time::Duration::from_millis(env_u64("DS_RETRY_BASE_MS", 500)),
        cap: std::time::Duration::from_millis(env_u64("DS_RETRY_CAP_MS", 30_000)),
        min_gap: std::time::Duration::from_millis(env_u64("DS_MIN_GAP_MS", 0)),
    };
    if policy.max_retries > 0 || !policy.min_gap.is_zero() {
        eprintln!(
            "[rate-limit] retries={} base={:?} cap={:?} min_gap={:?}",
            policy.max_retries, policy.base, policy.cap, policy.min_gap
        );
    }
    client.set_policy(policy);

    // The system prompt is prepended to the first user message of each
    // conversation instead of being sent as a separate startup message.
    let mut primed = false;
    let mut goal: Option<String> = None;

    let mut rl = DefaultEditor::new()?;
    println!(
        "{}",
        paint("1;36", "DeepSeek Web CLI \u{2014} multi-turn")
    );
    println!("  {}   new conversation", paint("32", "/new"));
    println!("  {}   show conversation tip", paint("33", "/id"));
    println!("  {}   set/start/show/clear goal", paint("35", "/goal"));
    println!("  {}   quit", paint("31", "exit"));
    println!();

    loop {
        let prompt = paint("1;35", ">> ");
        let line = rl.readline(&prompt);
        match line {
            Ok(input) if input.trim() == "exit" => break,
            Ok(input) if input.trim() == "/new" => {
                client.reset();
                primed = false;
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
                if input.trim() == "/goal" || input.trim().starts_with("/goal ") =>
            {
                let arg = input.trim()["/goal".len()..].trim().to_string();
                if arg.is_empty() {
                    match &goal {
                        Some(g) => println!("{} {g}", paint("36", "[goal]")),
                        None => println!(
                            "{} {}",
                            paint("36", "[goal]"),
                            paint("2", "<none> \u{2014} usage: /goal <text> | /goal clear")
                        ),
                    }
                } else if arg.eq_ignore_ascii_case("clear")
                    || arg.eq_ignore_ascii_case("none")
                    || arg.eq_ignore_ascii_case("off")
                {
                    goal = None;
                    println!("{}", paint("32", "[goal cleared]"));
                } else {
                    goal = Some(arg.clone());
                    println!("{} {arg}", paint("32", "[goal set]"));
                    // Automatically start the model and keep reprompting it
                    // until it marks the goal complete (or we hit the cap).
                    let mut turn = 0usize;
                    loop {
                        turn += 1;
                        let base = if turn == 1 {
                            format!(
                                "Work toward the following goal, using the shell as needed. \
                                 When it is fully accomplished, include the token {GOAL_DONE} in your visible reply.\n\nGoal: {arg}"
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
                        let completed = match run_turn(
                            &mut client,
                            &mut rl,
                            &message,
                            thinking,
                            search,
                            auto_approve,
                            debug_pending,
                            &goal,
                        )
                        .await
                        {
                            Ok(c) => c,
                            Err(e) => {
                                eprintln!("{} {e:#}", paint("1;31", "[error]"));
                                break;
                            }
                        };
                        if completed {
                            println!("{}", paint("32", "[goal completed]"));
                            goal = None;
                            break;
                        }
                        if turn >= MAX_GOAL_REPROMPTS {
                            println!(
                                "{} reprompt limit reached; goal stays active",
                                paint("1;33", "[warn]")
                            );
                            break;
                        }
                        println!(
                            "{} reprompting model (turn {turn})",
                            paint("36", "[goal]")
                        );
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
                match run_turn(
                    &mut client,
                    &mut rl,
                    &message,
                    thinking,
                    search,
                    auto_approve,
                    debug_pending,
                    &goal,
                )
                .await
                {
                    Ok(completed) => {
                        if completed && goal.is_some() {
                            println!("{}", paint("32", "[goal completed]"));
                            goal = None;
                        }
                    }
                    Err(e) => eprintln!("{} {e:#}", paint("1;31", "[error]")),
                }
            }
            Err(ReadlineError::Interrupted) | Err(ReadlineError::Eof) => break,
            Err(e) => eprintln!("{} readline error: {e}", paint("1;31", "[error]")),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{goal_completed, with_goal};

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
    fn detects_completion_marker() {
        assert!(goal_completed("all done [GOAL COMPLETE]"));
        assert!(!goal_completed("still working"));
    }
}
