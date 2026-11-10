//! End-to-end tests: drive the real `deepseek` binary as a subprocess against
//! a local mock of the DeepSeek web API.
//!
//! Nothing here talks to the network. The binary is pointed at an in-process
//! HTTP server through `DS_API_BASE`, fed prompts on stdin (rustyline falls
//! back to line-by-line reads when stdin is not a terminal), and its stdout,
//! stderr and exit status are asserted on.
//!
//! The mock implements just enough of the API for a full turn:
//!   GET  /api/v0/users/current              - credential check at startup
//!   POST /api/v0/chat/create_pow_challenge  - proof-of-work challenge
//!   POST /api/v0/chat_session/create        - conversation creation
//!   POST /api/v0/chat/completion            - the streamed reply (SSE)
//!
//! Proof of work is solved for real: the challenge's target hash is computed
//! with the same `pow` implementation the binary uses (included by path), so
//! the PoW header the CLI sends is genuinely valid for the challenge served.

#![allow(dead_code)]

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[path = "../src/pow.rs"]
mod pow;

// ---------------------------------------------------------------------------
// A tiny HTTP mock
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Req {
    method: String,
    path: String,
    headers: Vec<(String, String)>,
    body: String,
}

impl Req {
    fn header(&self, name: &str) -> Option<&str> {
        let name = name.to_ascii_lowercase();
        self.headers
            .iter()
            .find(|(k, _)| *k == name)
            .map(|(_, v)| v.as_str())
    }

    /// The request body parsed as JSON (empty value when it is not JSON).
    fn json(&self) -> serde_json::Value {
        serde_json::from_str(&self.body).unwrap_or(serde_json::Value::Null)
    }
}

struct Resp {
    status: u16,
    content_type: &'static str,
    body: String,
}

impl Resp {
    fn json(body: impl Into<String>) -> Self {
        Self {
            status: 200,
            content_type: "application/json",
            body: body.into(),
        }
    }

    fn sse(body: impl Into<String>) -> Self {
        Self {
            status: 200,
            content_type: "text/event-stream",
            body: body.into(),
        }
    }

    fn status(code: u16, body: impl Into<String>) -> Self {
        Self {
            status: code,
            content_type: "application/json",
            body: body.into(),
        }
    }
}

type Handler = dyn Fn(&Req) -> Resp + Send + Sync;

/// A mock API server on an ephemeral localhost port.
struct Mock {
    base: String,
    requests: Arc<Mutex<Vec<Req>>>,
}

impl Mock {
    fn start(handler: impl Fn(&Req) -> Resp + Send + Sync + 'static) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock server");
        let port = listener.local_addr().unwrap().port();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let log = requests.clone();
        let handler: Arc<Handler> = Arc::new(handler);

        std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let handler = handler.clone();
                let log = log.clone();
                std::thread::spawn(move || serve(stream, &*handler, &log));
            }
        });

        Self {
            base: format!("http://127.0.0.1:{port}"),
            requests,
        }
    }

    fn requests(&self) -> Vec<Req> {
        self.requests.lock().unwrap().clone()
    }

    fn requests_to(&self, path: &str) -> Vec<Req> {
        self.requests()
            .into_iter()
            .filter(|r| r.path == path)
            .collect()
    }
}

fn serve(mut stream: TcpStream, handler: &Handler, log: &Mutex<Vec<Req>>) {
    let Some(req) = read_request(&mut stream) else {
        return;
    };
    let resp = handler(&req);
    log.lock().unwrap().push(req);
    let reason = match resp.status {
        200 => "OK",
        401 => "Unauthorized",
        403 => "Forbidden",
        500 => "Internal Server Error",
        _ => "Status",
    };
    let head = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        resp.status,
        reason,
        resp.content_type,
        resp.body.len()
    );
    let _ = stream.write_all(head.as_bytes());
    let _ = stream.write_all(resp.body.as_bytes());
    let _ = stream.flush();
    let _ = stream.shutdown(std::net::Shutdown::Both);
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

fn read_request(stream: &mut TcpStream) -> Option<Req> {
    stream
        .set_read_timeout(Some(Duration::from_secs(10)))
        .ok()?;
    let mut buf = Vec::new();
    let mut tmp = [0u8; 2048];
    let head_end = loop {
        if let Some(pos) = find(&buf, b"\r\n\r\n") {
            break pos;
        }
        let n = stream.read(&mut tmp).ok()?;
        if n == 0 {
            return None;
        }
        buf.extend_from_slice(&tmp[..n]);
    };

    let head = String::from_utf8_lossy(&buf[..head_end]).to_string();
    let mut lines = head.lines();
    let mut first = lines.next().unwrap_or("").split_whitespace();
    let method = first.next()?.to_string();
    let path = first.next()?.to_string();
    let mut headers = Vec::new();
    for line in lines {
        if let Some((k, v)) = line.split_once(':') {
            headers.push((k.trim().to_ascii_lowercase(), v.trim().to_string()));
        }
    }
    let len: usize = headers
        .iter()
        .find(|(k, _)| k == "content-length")
        .and_then(|(_, v)| v.parse().ok())
        .unwrap_or(0);
    let mut body = buf[head_end + 4..].to_vec();
    while body.len() < len {
        let n = stream.read(&mut tmp).ok()?;
        if n == 0 {
            break;
        }
        body.extend_from_slice(&tmp[..n]);
    }
    body.truncate(len);
    Some(Req {
        method,
        path,
        headers,
        body: String::from_utf8_lossy(&body).to_string(),
    })
}

// ---------------------------------------------------------------------------
// Canned API payloads
// ---------------------------------------------------------------------------

fn user_ok() -> Resp {
    Resp::json(
        r#"{"code":0,"msg":"","data":{"biz_code":0,"biz_msg":"","biz_data":{"email":"e2e@example.com"}}}"#,
    )
}

fn user_rejected() -> Resp {
    Resp::json(r#"{"code":40003,"msg":"INVALID_TOKEN","data":null}"#)
}

fn session_created() -> Resp {
    Resp::json(r#"{"code":0,"data":{"biz_code":0,"biz_data":{"chat_session":{"id":"sess-e2e"}}}}"#)
}

/// `true` when a completion prompt is the reviewer's, not the worker's.
fn is_review(prompt: &str) -> bool {
    prompt.contains("strict, independent reviewer")
}

/// A PoW challenge whose target hash is reachable within `difficulty` guesses.
///
/// `expire_at` is 0, which the solver treats as "never expires" - a real
/// challenge carries a timestamp, but the mock is not a real server and a
/// fixed timestamp would make the fixture go stale.
fn pow_challenge() -> Resp {
    let salt = "e2e-salt";
    let expire_at: u64 = 0;
    let answer: u64 = 4242;
    let target = hex::encode(pow::deepseek_hash_v1(
        format!("{salt}_{expire_at}_{answer}").as_bytes(),
    ));
    let body = serde_json::json!({
        "code": 0,
        "data": {"biz_code": 0, "biz_data": {"challenge": {
            "algorithm": "DeepSeekHashV1",
            "challenge": target,
            "salt": salt,
            "signature": "e2e-signature",
            "difficulty": answer + 1000,
            "expire_at": expire_at,
            "expire_after": 0,
            "target_path": "/api/v0/chat/completion",
        }}}
    });
    Resp::json(body.to_string())
}

/// Build an SSE reply body: a message id, some visible text, then FINISHED.
fn sse_reply(text: &str) -> Resp {
    let mut body = String::new();
    for evt in [
        serde_json::json!({"p": "response/message_id", "v": "msg-e2e"}),
        serde_json::json!({"p": "response/content", "v": text}),
        serde_json::json!({"p": "response/status", "v": "FINISHED"}),
    ] {
        body.push_str(&format!("data: {evt}\n\n"));
    }
    Resp::sse(body)
}

/// The handler every chat test starts from: auth, PoW and session creation are
/// answered; `completion` is delegated to the caller.
fn mock_api<F>(completion: F) -> Mock
where
    F: Fn(&Req, usize) -> Resp + Send + Sync + 'static,
{
    let calls = AtomicUsize::new(0);
    Mock::start(move |req| match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/api/v0/users/current") => user_ok(),
        ("POST", "/api/v0/chat/create_pow_challenge") => pow_challenge(),
        ("POST", "/api/v0/chat_session/create") => session_created(),
        ("POST", "/api/v0/chat/completion") => {
            let n = calls.fetch_add(1, Ordering::SeqCst);
            completion(req, n)
        }
        (m, p) => Resp::status(404, format!(r#"{{"error":"unexpected {m} {p}"}}"#)),
    })
}

// ---------------------------------------------------------------------------
// Running the binary
// ---------------------------------------------------------------------------

/// A throwaway config/data directory so tests never read or write the user's
/// real secrets, history or goal store.
struct Sandbox {
    dir: PathBuf,
}

impl Sandbox {
    fn new(name: &str) -> Self {
        static N: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "deepseek-e2e-{}-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::SeqCst),
            name
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create sandbox dir");
        Self { dir }
    }

    fn path(&self) -> &Path {
        &self.dir
    }

    fn file(&self, name: &str) -> PathBuf {
        self.dir.join(name)
    }

    /// `PATH` for the child: a directory of no-op stand-ins for `xdg-open`
    /// and friends, followed by the real `PATH` so `sh` still works.
    fn bin_dir(&self) -> std::ffi::OsString {
        let bin = self.file("bin");
        std::fs::create_dir_all(&bin).expect("create bin dir");
        for name in ["xdg-open", "gio", "sensible-browser", "x-www-browser"] {
            let path = bin.join(name);
            std::fs::write(
                &path,
                "#!/bin/sh\necho \"$0 $*\" >> \"$DS_BROWSER_LOG\"\nexit 0\n",
            )
            .expect("write browser shim");
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755))
                    .expect("chmod browser shim");
            }
        }
        let real = std::env::var_os("PATH").unwrap_or_default();
        let mut paths = vec![bin];
        paths.extend(std::env::split_paths(&real));
        std::env::join_paths(paths).expect("join PATH")
    }
}

impl Drop for Sandbox {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// Run the CLI with a clean, hermetic environment plus `extra` overrides.
fn run(sandbox: &Sandbox, args: &[&str], extra: &[(&str, &str)], stdin: &str) -> Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_deepseek"));
    cmd.args(args)
        .current_dir(sandbox.path())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        // Start from a clean slate: no ambient token, proxy or config.
        .env_remove("AUTHORIZATION_TOKEN")
        .env_remove("DS_SESSION_ID")
        .env_remove("DS_API_BASE")
        .env_remove("DS_SHUMEI_DEVICE_ID")
        .env_remove("CLICOLOR_FORCE")
        .env("NO_COLOR", "1")
        .env("HOME", sandbox.path())
        .env("XDG_CONFIG_HOME", sandbox.file("config"))
        .env("XDG_DATA_HOME", sandbox.file("data"))
        .env("DS_SECRETS_FILE", sandbox.file("secrets.env"))
        .env("DS_HISTORY_FILE", sandbox.file("history"))
        .env("DS_GOALS_FILE", sandbox.file("goals.json"))
        // No pacing or long retry sleeps: the mock answers immediately.
        .env("DS_MIN_GAP_MS", "0")
        .env("DS_RETRY_BASE_MS", "1")
        .env("DS_RETRY_CAP_MS", "20")
        .env("DS_MAX_RETRIES", "3")
        .env("DS_HTTP_TIMEOUT_MS", "10000")
        .env("AUTHORIZATION_TOKEN", "e2e-token")
        // Never let a test open a real browser: a shim on PATH records the
        // attempt instead and exits immediately.
        .env("PATH", sandbox.bin_dir())
        .env("DS_BROWSER_LOG", sandbox.file("browser.log"))
        // Keep the mock's own requests out of any user proxy.
        .env_remove("HTTP_PROXY")
        .env_remove("HTTPS_PROXY")
        .env_remove("ALL_PROXY")
        .env_remove("http_proxy")
        .env_remove("https_proxy")
        .env_remove("all_proxy");
    for (k, v) in extra {
        cmd.env(k, v);
    }

    let mut child = cmd.spawn().expect("spawn deepseek");
    child
        .stdin
        .take()
        .expect("stdin")
        .write_all(stdin.as_bytes())
        .expect("write stdin");
    // Dropping the pipe's handle closes stdin, so the prompt loop sees EOF and
    // exits even if the input did not end with `exit`.

    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        match child.try_wait().expect("wait for deepseek") {
            Some(_) => break,
            None if Instant::now() > deadline => {
                let _ = child.kill();
                panic!("deepseek did not exit within 60s");
            }
            None => std::thread::sleep(Duration::from_millis(10)),
        }
    }
    child.wait_with_output().expect("collect output")
}

fn stdout(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).to_string()
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).to_string()
}

/// Assert helper that shows both streams on failure.
fn assert_contains(haystack: &str, needle: &str, what: &str) {
    assert!(
        haystack.contains(needle),
        "{what}: expected to find {needle:?} in output\n--- output ---\n{haystack}\n--- end ---"
    );
}

// ---------------------------------------------------------------------------
// Tests: argument handling (no server needed)
// ---------------------------------------------------------------------------

#[test]
fn version_flag_prints_version_and_exits_zero() {
    let sb = Sandbox::new("version");
    let out = run(&sb, &["--version"], &[], "");
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    let text = stdout(&out);
    assert_contains(&text, "deepseek", "version output");
    assert_contains(&text, env!("CARGO_PKG_VERSION"), "version output");
    // No server should be contacted for a pure flag.
    assert!(
        !text.contains("DeepSeek Web CLI"),
        "should not start a session"
    );
}

#[test]
fn help_flag_documents_options_and_exits_zero() {
    let sb = Sandbox::new("help");
    let out = run(&sb, &["--help"], &[], "");
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    let text = stdout(&out);
    for needle in ["Usage: deepseek", "--unsandboxed", "--login", "--version"] {
        assert_contains(&text, needle, "--help output");
    }
}

#[test]
fn unknown_argument_is_rejected_with_exit_code_two() {
    let sb = Sandbox::new("unknown");
    let out = run(&sb, &["--auto-aprove"], &[], "");
    assert_eq!(out.status.code(), Some(2), "stderr: {}", stderr(&out));
    let err = stderr(&out);
    assert_contains(&err, "unknown argument", "error message");
    assert_contains(&err, "--auto-aprove", "error message");
}

// ---------------------------------------------------------------------------
// Tests: full turns against the mock API
// ---------------------------------------------------------------------------

#[test]
fn one_shot_turn_streams_the_reply_and_authenticates() {
    let mock = mock_api(|_req, _n| sse_reply("hello from the mock model"));
    let sb = Sandbox::new("turn");

    let out = run(
        &sb,
        &[],
        &[("DS_API_BASE", &mock.base)],
        "what is up\nexit\n",
    );

    let text = stdout(&out);
    assert_contains(&text, "DeepSeek Web CLI", "banner");
    assert_contains(&text, "hello from the mock model", "streamed reply");

    let completions = mock.requests_to("/api/v0/chat/completion");
    assert_eq!(completions.len(), 1, "exactly one completion request");
    let req = &completions[0];

    // The token from AUTHORIZATION_TOKEN is sent as a bearer credential.
    assert_eq!(req.header("authorization"), Some("Bearer e2e-token"));
    // A conversation was created and threaded into the completion.
    assert!(
        !mock.requests_to("/api/v0/chat_session/create").is_empty(),
        "the CLI must create a conversation before completing"
    );
    let body = req.json();
    assert_eq!(body["chat_session_id"], "sess-e2e");
    // The first prompt of a conversation is prefixed with the system prompt;
    // the user's text comes last, unmodified.
    let prompt = body["prompt"].as_str().expect("prompt is a string");
    assert!(
        prompt.ends_with("what is up"),
        "prompt should end with the user's message: {prompt}"
    );
    assert!(
        prompt.contains("You are a helpful assistant"),
        "first prompt should carry the system prompt: {prompt}"
    );
    // ...and a solved proof of work was attached.
    assert!(
        req.header("x-ds-pow-response").is_some(),
        "completion must carry a proof-of-work header"
    );
}

#[test]
fn shell_command_runs_and_its_output_is_sent_back_to_the_model() {
    let mock = mock_api(|req, n| {
        if n == 0 {
            // First reply asks for a command.
            sse_reply("Let me check.\n<shell>echo e2e-shell-ran</shell>")
        } else {
            // The follow-up must carry the command's output.
            let prompt = req.json()["prompt"].as_str().unwrap_or("").to_string();
            assert!(
                prompt.contains("[SHELL OUTPUT]"),
                "follow-up should quote the shell output: {prompt}"
            );
            assert!(
                prompt.contains("e2e-shell-ran"),
                "follow-up should contain the command output: {prompt}"
            );
            sse_reply("the command worked")
        }
    });
    let sb = Sandbox::new("shell");

    let out = run(&sb, &[], &[("DS_API_BASE", &mock.base)], "run it\nexit\n");
    let text = stdout(&out);

    // The CLI announces and runs the command, echoing its output locally...
    assert_contains(&text, "echo e2e-shell-ran", "shell announcement");
    assert_contains(&text, "e2e-shell-ran", "command output");
    // ...and the model's second reply is shown.
    assert_contains(&text, "the command worked", "follow-up reply");

    let completions = mock.requests_to("/api/v0/chat/completion");
    assert_eq!(completions.len(), 2, "one completion per shell round-trip");
    // The second request threads from the first message, keeping one conversation.
    assert_eq!(completions[1].json()["chat_session_id"], "sess-e2e");
}

#[test]
fn goal_is_reprompted_until_the_reviewer_confirms_completion() {
    // Three voices in the mock: the worker (which claims completion twice),
    // the reviewer (which first rejects, then accepts), and the shell.
    let mock = mock_api(|req, _n| {
        let prompt = req.json()["prompt"].as_str().unwrap_or("").to_string();
        if is_review(&prompt) {
            assert!(
                prompt.contains("Goal: tidy the repo"),
                "reviewer must be told the goal: {prompt}"
            );
            // The reviewer inspects the workspace with the same shell the
            // worker uses, so it must be told the command format.
            assert!(
                prompt.contains("<shell>") && prompt.contains("[SHELL OUTPUT]"),
                "reviewer must be told how to run commands: {prompt}"
            );
            if prompt.contains("removed the cruft") {
                // Second review: the worker's report fixed the finding.
                sse_reply("Checked the tree; clean.\nVERDICT: ACHIEVED")
            } else {
                assert!(
                    prompt.contains("all done [GOAL COMPLETE]"),
                    "reviewer must see the claim under examination: {prompt}"
                );
                sse_reply("README still has cruft.\nVERDICT: NOT ACHIEVED - tests fail")
            }
        } else if prompt.contains("Reviewer's verdict and findings") {
            // The rejection was fed back to the worker.
            assert!(
                prompt.contains("README still has cruft"),
                "worker must receive the reviewer's findings: {prompt}"
            );
            sse_reply("removed the cruft [GOAL COMPLETE]")
        } else if prompt.contains("[ACTIVE GOAL] tidy the repo") {
            sse_reply("all done [GOAL COMPLETE]")
        } else {
            sse_reply("still working on it")
        }
    });
    let sb = Sandbox::new("goal");

    let out = run(
        &sb,
        &[],
        &[("DS_API_BASE", &mock.base)],
        "/goal tidy the repo\nexit\n",
    );
    let text = stdout(&out);

    assert_contains(&text, "[goal set] tidy the repo", "goal command");
    assert_contains(
        &text,
        "[review] checking the completion claim",
        "review runs",
    );
    assert_contains(
        &text,
        "[review] rejected the completion claim",
        "a rejected claim is reported",
    );
    assert_contains(&text, "[goal completed]", "goal completion");
    assert!(
        !text.contains("[goal paused]"),
        "goal should finish without pausing: {text}"
    );

    let completions = mock.requests_to("/api/v0/chat/completion");
    // 2 worker turns + 2 reviews: the reviewer really is consulted twice.
    assert_eq!(completions.len(), 4, "two worker turns and two reviews");
    let reviews = completions
        .iter()
        .filter(|r| is_review(r.json()["prompt"].as_str().unwrap_or("")))
        .count();
    assert_eq!(reviews, 2, "one review per completion claim");

    // The goal is persisted against the conversation and cleared on completion.
    let store = std::fs::read_to_string(sb.file("goals.json")).unwrap_or_default();
    assert!(
        !store.contains("tidy the repo"),
        "completed goal must not stay in the store: {store}"
    );
}

#[test]
fn a_reviewer_using_dsml_tool_calls_still_runs_commands() {
    // Exactly the shape a reviewer emitted when it ignored the instructions:
    // a DSML invoke/parameter tool call with fullwidth bars.
    let bar = "｜｜";
    let mock = mock_api(move |req, _n| {
        let prompt = req.json()["prompt"].as_str().unwrap_or("").to_string();
        if prompt.contains("=== CMD:") && !prompt.contains("[ACTIVE GOAL]") {
            // The reviewer's follow-up has no system prompt, so it must be
            // reminded how to finish or it may answer without a verdict.
            assert!(
                prompt.contains("Reminder: you are the reviewer"),
                "follow-up must carry the verdict reminder: {prompt}"
            );
            sse_reply("Checked.\nVERDICT: ACHIEVED")
        } else if is_review(&prompt) {
            sse_reply(&format!(
                "<{bar}DSML{bar} calls>\n<{bar}DSML{bar} invoke name=\"shell\">\n<{bar}DSML{bar} parameter name=\"command\" string=\"true\">echo e2e-dsml-ran</{bar}DSML{bar} parameter>\n</{bar}DSML{bar} invoke>"
            ))
        } else {
            sse_reply("done [GOAL COMPLETE]")
        }
    });
    let sb = Sandbox::new("goal-dsml");

    let out = run(
        &sb,
        &[],
        &[("DS_API_BASE", &mock.base)],
        "/goal tidy the repo\nexit\n",
    );
    let text = stdout(&out);

    assert_contains(&text, "e2e-dsml-ran", "DSML command output");
    assert_contains(
        &text,
        "[goal completed]",
        "goal completes after real evidence",
    );
}

#[test]
fn a_reviewer_using_an_alternate_tag_still_runs_commands() {
    // The documented tag is the block form, but a model can reach for a short
    // alias. It must still run, or the reviewer has no evidence at all.
    let mock = mock_api(|req, _n| {
        let prompt = req.json()["prompt"].as_str().unwrap_or("").to_string();
        if prompt.contains("=== CMD:") && !prompt.contains("[ACTIVE GOAL]") {
            sse_reply("Checked.\nVERDICT: ACHIEVED")
        } else if is_review(&prompt) {
            sse_reply("\x3ccmd\x3eecho e2e-cmd-alias-ran\x3c/cmd\x3e")
        } else {
            sse_reply("done [GOAL COMPLETE]")
        }
    });
    let sb = Sandbox::new("goal-cmd-alias");

    let out = run(
        &sb,
        &[],
        &[("DS_API_BASE", &mock.base)],
        "/goal tidy the repo\nexit\n",
    );
    let text = stdout(&out);

    assert_contains(&text, "e2e-cmd-alias-ran", "alias command output");
    assert_contains(
        &text,
        "[goal completed]",
        "goal completes after real evidence",
    );
}

#[test]
fn a_reviewer_using_a_native_tool_call_still_runs_commands() {
    // The reviewer is told the tag form, but a model can still answer with its
    // native tool-call syntax. Those commands must actually run: otherwise the
    // reviewer gets no shell output and can only judge the report.
    let mock = mock_api(|req, _n| {
        let prompt = req.json()["prompt"].as_str().unwrap_or("").to_string();
        // The reviewer's *follow-up* turn is the shell output it asked for;
        // it does not carry the review instructions, only the executed
        // command. The worker's follow-ups carry an active goal, so the two
        // are told apart by that marker.
        if prompt.contains("=== CMD:") && !prompt.contains("[ACTIVE GOAL]") {
            sse_reply("Checked the output.\nVERDICT: ACHIEVED")
        } else if is_review(&prompt) {
            sse_reply(concat!(
                "<|d|> invoke name=\"shell\">\n",
                "<|d|> parameter name=\"command\" string=\"true\">echo e2e-reviewer-ran</|d|> parameter>\n",
                "</|d|> invoke>"
            ))
        } else {
            sse_reply("done [GOAL COMPLETE]")
        }
    });
    let sb = Sandbox::new("goal-native-tool-call");

    let out = run(
        &sb,
        &[],
        &[("DS_API_BASE", &mock.base)],
        "/goal tidy the repo\nexit\n",
    );
    let text = stdout(&out);

    assert_contains(&text, "echo e2e-reviewer-ran", "reviewer command announced");
    assert_contains(&text, "e2e-reviewer-ran", "reviewer command output");
    assert_contains(
        &text,
        "[goal completed]",
        "goal completes after real evidence",
    );

    // The review's first turn emitted a command, and the output came back as
    // a follow-up turn in the same (reviewer) conversation.
    assert_contains(&text, "--- output ---", "reviewer shell output is shown");
    assert_contains(&text, "e2e-reviewer-ran\n", "reviewer sees what it ran");
    let requests = mock.requests_to("/api/v0/chat/completion");
    let worker_turns = requests
        .iter()
        .filter(|r| {
            let p = r.json()["prompt"].as_str().unwrap_or("").to_string();
            p.contains("[ACTIVE GOAL]")
        })
        .count();
    assert_eq!(
        requests.len(),
        worker_turns + 2,
        "one review turn plus its shell follow-up"
    );
}

#[test]
fn an_unverified_claim_does_not_complete_the_goal() {
    // The worker always claims completion; the reviewer never gives a
    // readable verdict.  The goal must stay open (and persisted).
    let mock = mock_api(|req, _n| {
        let prompt = req.json()["prompt"].as_str().unwrap_or("").to_string();
        if is_review(&prompt) {
            sse_reply("Looks fine to me, I guess.")
        } else {
            sse_reply("done [GOAL COMPLETE]")
        }
    });
    let sb = Sandbox::new("goal-unverified");

    let out = run(
        &sb,
        &[],
        &[("DS_API_BASE", &mock.base), ("DS_MAX_GOAL_TURNS", "3")],
        "/goal tidy the repo\nexit\n",
    );
    let text = stdout(&out);

    assert_contains(
        &text,
        "no readable verdict",
        "a missing verdict must not be accepted",
    );
    assert!(
        !text.contains("[goal completed]"),
        "an unverified claim must not close the goal: {text}"
    );
    assert_contains(
        &text,
        "[goal paused]",
        "goal stays open after the opt-in turn budget",
    );

    let store = std::fs::read_to_string(sb.file("goals.json")).unwrap_or_default();
    assert!(
        store.contains("tidy the repo"),
        "the unfinished goal must stay in the store: {store}"
    );
}

#[test]
fn goal_turns_are_unlimited_by_default() {
    // 40 worker turns is far beyond the old 25-turn cap.  With no
    // DS_MAX_GOAL_TURNS set the loop must keep going until the reviewer
    // confirms completion.
    let mock = mock_api(|req, n| {
        let prompt = req.json()["prompt"].as_str().unwrap_or("").to_string();
        if is_review(&prompt) {
            sse_reply("Verified.\nVERDICT: ACHIEVED")
        } else if n >= 39 {
            sse_reply("finally done [GOAL COMPLETE]")
        } else {
            sse_reply("still working on it")
        }
    });
    let sb = Sandbox::new("goal-unlimited");

    let out = run(
        &sb,
        &[],
        &[
            ("DS_API_BASE", &mock.base),
            // 40 sequential requests: don't spend a minute on request spacing.
            ("DS_MIN_GAP_MS", "1"),
        ],
        "/goal tidy the repo\nexit\n",
    );
    let text = stdout(&out);

    assert!(
        !text.contains("[goal paused]"),
        "goal must not pause: {text}"
    );
    assert_contains(
        &text,
        "[goal] reprompting model (turn 39)",
        "39+ turns ran (well past the old 25-turn cap)",
    );
    assert_contains(&text, "[goal completed]", "goal eventually completes");

    let store = std::fs::read_to_string(sb.file("goals.json")).unwrap_or_default();
    assert!(
        !store.contains("tidy the repo"),
        "completed goal must not stay in the store: {store}"
    );
}

#[test]
fn rejected_token_fails_fast_with_an_actionable_message() {
    let mock = Mock::start(|req| match req.path.as_str() {
        "/api/v0/users/current" => user_rejected(),
        other => Resp::status(404, format!(r#"{{"error":"unexpected {other}"}}"#)),
    });
    let sb = Sandbox::new("rejected");

    let out = run(&sb, &[], &[("DS_API_BASE", &mock.base)], "hi\nexit\n");

    assert_eq!(out.status.code(), Some(1), "stderr: {}", stderr(&out));
    let err = stderr(&out);
    assert_contains(&err, "rejected the saved token", "error message");
    assert_contains(&err, "--login", "error message should say how to recover");
    // A rejected token must not be retried or silently re-harvested.
    assert_eq!(
        mock.requests_to("/api/v0/users/current").len(),
        1,
        "the credential is checked exactly once"
    );
    assert!(
        mock.requests_to("/api/v0/chat/completion").is_empty(),
        "no completions are attempted with a rejected token"
    );
}

#[test]
fn token_can_be_loaded_from_the_secrets_file() {
    let mock = mock_api(|_req, _n| sse_reply("from the secrets file"));
    let sb = Sandbox::new("secrets");
    std::fs::write(
        sb.file("secrets.env"),
        "# stored by a previous sign-in\nexport AUTHORIZATION_TOKEN=\"file-token\"\n",
    )
    .expect("write secrets file");

    let out = run(
        &sb,
        &[],
        &[("DS_API_BASE", &mock.base), ("AUTHORIZATION_TOKEN", "")],
        "hi\nexit\n",
    );
    assert_contains(&stdout(&out), "from the secrets file", "reply");

    let completions = mock.requests_to("/api/v0/chat/completion");
    assert_eq!(completions.len(), 1);
    assert_eq!(
        completions[0].header("authorization"),
        Some("Bearer file-token"),
        "the token quoted in the secrets file is used"
    );
}

#[test]
fn missing_token_without_a_terminal_explains_itself_instead_of_hanging() {
    // With no token anywhere and stdin not a terminal there is nobody to
    // complete a browser sign-in, so the CLI must say so and exit non-zero
    // rather than wait out the login timeout.
    let sb = Sandbox::new("no-token");
    let started = Instant::now();
    let out = run(&sb, &[], &[("AUTHORIZATION_TOKEN", "")], "hi\nexit\n");
    let elapsed = started.elapsed();

    assert!(!out.status.success(), "should fail without credentials");
    let err = stderr(&out);
    assert_contains(&err, "stdin is not a terminal", "error message");
    assert_contains(
        &err,
        "AUTHORIZATION_TOKEN",
        "error message should say what to set",
    );
    assert!(
        elapsed < Duration::from_secs(20),
        "must not sit through the login timeout (took {elapsed:?})"
    );
    // The shim on PATH caught the browser attempt; nothing real was opened.
    let log = std::fs::read_to_string(sb.file("browser.log")).unwrap_or_default();
    assert_contains(&log, "https://chat.deepseek.com/sign_in", "browser launch");
}

#[test]
fn no_browser_flag_still_prints_the_sign_in_url() {
    let sb = Sandbox::new("no-browser");
    let out = run(
        &sb,
        &["--no-browser"],
        &[("AUTHORIZATION_TOKEN", "")],
        "hi\nexit\n",
    );
    let text = stdout(&out) + &stderr(&out);
    assert_contains(&text, "https://chat.deepseek.com/sign_in", "printed URL");
    let log = std::fs::read_to_string(sb.file("browser.log")).unwrap_or_default();
    assert!(
        log.is_empty(),
        "--no-browser must not launch anything: {log}"
    );
}
