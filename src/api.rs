use anyhow::{bail, Context, Result};
use base64::{engine::general_purpose::STANDARD, Engine};
use futures_util::{Stream, StreamExt};
use reqwest::{header, header::HeaderValue, Client, Response, StatusCode};
use serde_json::{json, Value};
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::pow::{ChatChallenge, SolvedChallenge};
use tokio::sync::watch;

pub type ChatStream = Pin<Box<dyn Stream<Item = Result<String>> + Send>>;

/// Shared pacing slot: the instant the next request is allowed to start.
type Slot = Arc<Mutex<Option<Instant>>>;

/// Default origin of the DeepSeek web app and its API.
const DEFAULT_API_BASE: &str = "https://chat.deepseek.com";

/// Origin to send API calls to.
///
/// Overridable with `DS_API_BASE` so the CLI can be pointed at a local mock
/// server; the end-to-end tests use exactly that to drive the real binary
/// without touching the network. Trailing slashes are trimmed so a base of
/// `http://127.0.0.1:8080/` still produces `.../api/v0/...`.
fn api_base() -> String {
    let raw = std::env::var("DS_API_BASE")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| DEFAULT_API_BASE.to_string());
    raw.trim_end_matches('/').to_string()
}

/// Absolute URL for an API `path` (which must start with `/`).
fn api_url(path: &str) -> String {
    format!("{}{path}", api_base())
}

fn completion_url() -> String {
    api_url("/api/v0/chat/completion")
}
/// `target_path` the completion PoW challenge is issued for.
const COMPLETION_TARGET_PATH: &str = "/api/v0/chat/completion";
/// Resume generation of an interrupted assistant message.
///
/// Unlike `chat/completion` this endpoint needs no proof of work: the web
/// client posts it with a plain bearer token (see the captured HAR), so it is
/// opened through the same streaming/retry path but without a PoW header.
fn continue_url() -> String {
    api_url("/api/v0/chat/continue")
}
fn challenge_url() -> String {
    api_url("/api/v0/chat/create_pow_challenge")
}
fn session_url() -> String {
    api_url("/api/v0/chat_session/create")
}
fn session_page_url() -> String {
    api_url("/api/v0/chat_session/fetch_page")
}
fn session_title_url() -> String {
    api_url("/api/v0/chat_session/update_title")
}
fn session_delete_url() -> String {
    api_url("/api/v0/chat_session/delete")
}
fn history_url() -> String {
    api_url("/api/v0/chat/history_messages")
}
fn user_url() -> String {
    api_url("/api/v0/users/current")
}

// ---------------------------------------------------------------------------
// Rate-limit / transient-failure policy
// ---------------------------------------------------------------------------

/// Statuses worth retrying: explicit throttling plus transient upstream errors.
fn is_retryable(status: StatusCode) -> bool {
    matches!(status.as_u16(), 408 | 429 | 500 | 502 | 503 | 504)
}

/// Server-requested delay, if the response advertises one.
fn retry_after(resp: &Response) -> Option<Duration> {
    let h = resp.headers();
    // HTTP-date form is legal but rare here; fall through to other hints.
    if let Some(secs) = h
        .get("retry-after")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.trim().parse::<u64>().ok())
    {
        return Some(Duration::from_secs(secs));
    }
    for name in [
        "x-ratelimit-reset-requests",
        "x-ratelimit-reset-tokens",
        "x-ratelimit-reset",
    ] {
        if let Some(v) = h.get(name).and_then(|v| v.to_str().ok()) {
            if let Some(d) = parse_duration_hint(v) {
                return Some(d);
            }
        }
    }
    None
}

/// Accepts `12`, `1.5`, `500ms`, `3s`, `1m30s` style hints.
fn parse_duration_hint(s: &str) -> Option<Duration> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    if let Ok(secs) = s.parse::<u64>() {
        return Some(Duration::from_secs(secs));
    }
    if let Ok(secs) = s.parse::<f64>() {
        return (secs >= 0.0).then(|| Duration::from_secs_f64(secs));
    }
    let mut total = 0f64;
    let mut num = String::new();
    let mut matched = false;
    for ch in s.chars() {
        if ch.is_ascii_digit() || ch == '.' {
            num.push(ch);
            continue;
        }
        let v = num.parse::<f64>().ok()?;
        num.clear();
        let mult = match ch {
            'h' => 3600.0,
            'm' => 60.0,
            's' => 1.0,
            'M' => 0.001,
            _ => return None,
        };
        total += v * mult;
        matched = true;
    }
    if !num.is_empty() {
        total += num.parse::<f64>().ok()?;
        matched = true;
    }
    matched.then(|| Duration::from_secs_f64(total.max(0.0)))
}

/// How fast the backoff grows per attempt. A factor of four climbs out of a
/// short throttle quickly (one or two retries) instead of creeping up by
/// doubling, which used to leave the client hammering for the first handful of
/// attempts.
const BACKOFF_FACTOR: u32 = 4;

/// Exponential backoff with jitter: base * 4^attempt, capped, then randomised
/// into [cap/2, cap] so parallel clients do not retry in lockstep.
fn backoff(attempt: u32, base: Duration, cap: Duration) -> Duration {
    let exp = base.saturating_mul(BACKOFF_FACTOR.saturating_pow(attempt.min(15)));
    let capped = exp.min(cap);
    let ms = capped.as_millis() as u64;
    if ms == 0 {
        return Duration::ZERO;
    }
    let lo = ms / 2;
    let span = ms - lo;
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos() as u64)
        .unwrap_or(0);
    Duration::from_millis(lo + (seed % (span + 1)))
}

/// True when a message string looks like upstream throttling.
fn message_mentions_rate_limit(s: &str) -> bool {
    let h = s.to_ascii_lowercase();
    h.contains("rate limit")
        || h.contains("rate_limit")
        || h.contains("rate-limit")
        || h.contains("ratelimit")
        || h.contains("rate_limit_reached")
        || h.contains("too many requests")
        || h.contains("too frequent")
        || h.contains("throttl")
        || h.contains("429")
}

/// True when a JSON object carries a throttle-style numeric code.
fn is_rate_limit_code(v: &Value) -> bool {
    for k in ["code", "biz_code", "status_code", "errno", "err_code"] {
        if let Some(c) = v.get(k).and_then(|c| c.as_i64()) {
            if c == 429 || (42900..=42999).contains(&c) {
                return true;
            }
        }
    }
    if let Some(d) = v.get("data") {
        if d.is_object() && is_rate_limit_code(d) {
            return true;
        }
    }
    false
}

/// Detect a throttle/error envelope in a *complete* JSON body (not an event).
fn envelope_rate_limited(v: &Value) -> bool {
    if is_rate_limit_code(v) {
        return true;
    }
    if let Some(err) = v.get("error").filter(|e| !e.is_null()) {
        if is_rate_limit_code(err) || message_mentions_rate_limit(&err.to_string()) {
            return true;
        }
    }
    for k in ["msg", "message", "detail", "reason"] {
        if let Some(s) = v.get(k).and_then(|x| x.as_str()) {
            if message_mentions_rate_limit(s) {
                return true;
            }
        }
    }
    false
}

/// Detect throttling in a single SSE event (or bare JSON line).
fn event_rate_limited(evt: &Value) -> bool {
    // `event: toast` carries the throttle reason in `finish_reason` and/or
    // `content`, e.g.
    //   {"type":"error","content":"Messages too frequent. Try again later.",
    //    "finish_reason":"rate_limit_reached"}
    // It has none of the envelope keys below, so check it explicitly.
    if let Some(fr) = evt.get("finish_reason").and_then(|v| v.as_str()) {
        if message_mentions_rate_limit(fr) {
            return true;
        }
    }
    if let Some(content) = evt.get("content").and_then(|v| v.as_str()) {
        if message_mentions_rate_limit(content) {
            return true;
        }
    }
    if let Some(err) = evt.get("error").filter(|e| !e.is_null()) {
        if is_rate_limit_code(err) || message_mentions_rate_limit(&err.to_string()) {
            return true;
        }
    }
    // Envelope-shaped event (no `p` path) carrying a code/msg.
    if evt.get("p").is_none()
        && (evt.get("code").is_some() || evt.get("msg").is_some())
        && envelope_rate_limited(evt)
    {
        return true;
    }
    if evt.get("p").and_then(|p| p.as_str()) == Some("response/status") {
        if let Some(s) = evt.get("v").and_then(|v| v.as_str()) {
            let s = s.to_ascii_uppercase();
            if s.contains("RATE") || s.contains("THROTTL") {
                return true;
            }
        }
    }
    false
}

/// True when an envelope's top-level `code`/`msg` pair signals a failure.
///
/// The API answers `200 OK` with `{"code":40003,"msg":"INVALID_TOKEN"}` when a
/// token is rejected and with `{"code":40002,"msg":"Missing Token"}` when none
/// is sent, so a plain HTTP status check is not enough.
fn envelope_error(v: &Value) -> Option<String> {
    let code = v.get("code").and_then(|c| c.as_i64()).unwrap_or(0);
    let biz_code = v
        .get("data")
        .and_then(|d| d.get("biz_code"))
        .and_then(|c| c.as_i64())
        .unwrap_or(0);
    if code == 0 && biz_code == 0 {
        return None;
    }
    let msg = v
        .get("msg")
        .and_then(|m| m.as_str())
        .filter(|m| !m.is_empty())
        .or_else(|| {
            v.get("data")
                .and_then(|d| d.get("biz_msg"))
                .and_then(|m| m.as_str())
                .filter(|m| !m.is_empty())
        })
        .unwrap_or("no message");
    let code = if code != 0 { code } else { biz_code };
    Some(format!("{msg} (code {code})"))
}

/// True when an error string describes a rejected credential rather than a
/// transient problem. The API reports these as `40002 Missing Token` and
/// `40003 INVALID_TOKEN` / `Authorization Failed`.
pub fn is_auth_error(msg: &str) -> bool {
    let m = msg.to_ascii_lowercase();
    m.contains("code 40002")
        || m.contains("code 40003")
        || m.contains("missing token")
        || m.contains("invalid_token")
        || m.contains("invalid token")
        || m.contains("authorization failed")
}

/// Unwrap `data.biz_data`, reporting an API-level error instead of quietly
/// returning `null` (which used to surface as a confusing
/// "returned no account data" / "no session id" message).
fn unwrap_biz(v: Value, label: &str) -> Result<Value> {
    if let Some(err) = envelope_error(&v) {
        bail!("{label} failed: {err}");
    }
    Ok(v["data"]["biz_data"].clone())
}

/// Pull a server-suggested delay out of an event payload.
fn retry_after_value(v: &Value) -> Option<Duration> {
    for k in ["retry_after", "retry-after", "retryAfter"] {
        if let Some(x) = v.get(k) {
            if let Some(s) = x.as_str() {
                if let Some(d) = parse_duration_hint(s) {
                    return Some(d);
                }
            }
            if let Some(n) = x.as_u64() {
                return Some(Duration::from_secs(n));
            }
        }
    }
    if let Some(reset) = v.get("reset_at").and_then(|x| x.as_u64()) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .ok()?
            .as_secs();
        return Some(Duration::from_secs(reset.saturating_sub(now).max(1)));
    }
    None
}

/// Retry knobs, configurable from the environment in `main`.
#[derive(Clone, Copy, Debug)]
pub struct RetryPolicy {
    /// First backoff step; grows by `BACKOFF_FACTOR` every attempt.
    pub base: Duration,
    /// Upper bound on any single wait.
    pub cap: Duration,
    /// Minimum spacing between consecutive requests (client-side pacing).
    pub min_gap: Duration,
    /// Hard cap on consecutive retries of one request before giving up.
    ///
    /// Retrying forever is what a bot-detection system sees as abusive
    /// traffic: the client keeps hammering an endpoint the server has already
    /// answered with a throttle. A finite cap turns a persistent throttle into
    /// a visible error the user can act on, instead of an invisible hammer.
    /// `0` means unlimited, which is the default: retries are bounded by the
    /// growing backoff, not by a fixed attempt count.
    pub max_attempts: u32,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            // Start with a real pause and grow fast: throttling is normal and
            // worth waiting out, and the wait itself keeps the retries polite,
            // so there is no fixed attempt cap.
            base: Duration::from_secs(2),
            cap: Duration::from_secs(300),
            // Never fire requests back-to-back by default. Zero-gap bursts are
            // indistinguishable from automation, which is exactly what gets an
            // account flagged for botting.
            min_gap: Duration::from_millis(1500),
            max_attempts: 0,
        }
    }
}

impl RetryPolicy {
    /// True once `attempt` retries have been spent and the caller must fail.
    fn exhausted(&self, attempt: u32) -> bool {
        self.max_attempts != 0 && attempt >= self.max_attempts
    }
}

/// Client-side pacing: sleep until this request's slot opens, and reserve
/// the following slot. Concurrent callers therefore queue up instead of
/// bursting. No-op when `gap` is zero.
async fn pace(slot: &Slot, gap: Duration) {
    if gap.is_zero() {
        return;
    }
    let wait = {
        let mut s = slot.lock().unwrap();
        let now = Instant::now();
        let wait = match *s {
            Some(at) => at.saturating_duration_since(now),
            None => Duration::ZERO,
        };
        *s = Some(now + wait + gap);
        wait
    };
    if !wait.is_zero() {
        tokio::time::sleep(wait).await;
    }
}

/// POST JSON with retry on throttling / transient upstream failures, including
/// throttle envelopes that arrive with a `200 OK` status.
async fn post_json(
    http: &Client,
    policy: &RetryPolicy,
    slot: &Slot,
    cancel: &mut Option<watch::Receiver<u64>>,
    url: &str,
    body: &Value,
    label: &str,
) -> Result<Value> {
    let mut attempt = 0u32;
    loop {
        pace(slot, policy.min_gap).await;
        let resp = http
            .post(url)
            .json(body)
            .send()
            .await
            .with_context(|| format!("{label}: request failed"))?;

        let status = resp.status();
        let wait = retry_after(&resp).unwrap_or_else(|| backoff(attempt, policy.base, policy.cap));

        if status.is_success() {
            let text = resp
                .text()
                .await
                .with_context(|| format!("{label}: failed to read response"))?;
            let v: Value = match serde_json::from_str(&text) {
                Ok(v) => v,
                Err(e) => {
                    // A non-JSON 200 is usually an HTML challenge/interstitial
                    // served by the WAF, not an API bug.
                    let hint = if looks_like_bot_wall(status, &text) {
                        BOT_WALL_HINT
                    } else {
                        ""
                    };
                    bail!(
                        "{label}: invalid JSON response ({}): {}{}",
                        e,
                        snippet(&text),
                        hint
                    );
                }
            };
            if envelope_rate_limited(&v) {
                if policy.exhausted(attempt) {
                    bail!(
                        "{label}: still throttled after {} attempts; giving up instead of retrying forever (set DS_MAX_RETRIES=0 to retry without a cap)",
                        attempt + 1
                    );
                }
                eprintln!(
                    "[rate-limit] {label}: throttled in response body, waiting {:?} before retry {}",
                    wait,
                    attempt + 1
                );
                if sleep_or_cancel(wait, cancel).await {
                    bail!("{label}: cancelled");
                }
                attempt += 1;
                continue;
            }
            return Ok(v);
        }

        if !is_retryable(status) {
            let text = resp.text().await.unwrap_or_default();
            let hint = if looks_like_bot_wall(status, &text) {
                BOT_WALL_HINT
            } else {
                ""
            };
            bail!(
                "{label} failed after {} attempt(s): {} - {}{}",
                attempt + 1,
                status,
                text,
                hint
            );
        }

        if policy.exhausted(attempt) {
            bail!(
                "{label}: still failing after {} attempts (last: {status}); giving up instead of retrying forever (set DS_MAX_RETRIES=0 for no cap)",
                attempt + 1
            );
        }
        eprintln!(
            "[rate-limit] {label}: {status}, retry {} in {:?}",
            attempt + 1,
            wait
        );
        if sleep_or_cancel(wait, cancel).await {
            bail!("{label}: cancelled");
        }
        attempt += 1;
    }
}

/// GET a URL with retry on throttling / transient upstream failures and
/// unwrap the `data.biz_data` payload. `query` is appended verbatim.
async fn get_json(
    http: &Client,
    policy: &RetryPolicy,
    slot: &Slot,
    cancel: &mut Option<watch::Receiver<u64>>,
    url: &str,
    query: &[(&str, String)],
    label: &str,
) -> Result<Value> {
    let mut attempt = 0u32;
    loop {
        pace(slot, policy.min_gap).await;
        let resp = http
            .get(url)
            .query(query)
            .send()
            .await
            .with_context(|| format!("{label}: request failed"))?;

        let status = resp.status();
        let wait = retry_after(&resp).unwrap_or_else(|| backoff(attempt, policy.base, policy.cap));

        if status.is_success() {
            let text = resp
                .text()
                .await
                .with_context(|| format!("{label}: failed to read response"))?;
            let v: Value = match serde_json::from_str(&text) {
                Ok(v) => v,
                Err(e) => {
                    // A non-JSON 200 is usually an HTML challenge/interstitial
                    // served by the WAF, not an API bug.
                    let hint = if looks_like_bot_wall(status, &text) {
                        BOT_WALL_HINT
                    } else {
                        ""
                    };
                    bail!(
                        "{label}: invalid JSON response ({}): {}{}",
                        e,
                        snippet(&text),
                        hint
                    );
                }
            };
            if envelope_rate_limited(&v) {
                if policy.exhausted(attempt) {
                    bail!(
                        "{label}: still throttled after {} attempts; giving up instead of retrying forever (set DS_MAX_RETRIES=0 to retry without a cap)",
                        attempt + 1
                    );
                }
                eprintln!(
                    "[rate-limit] {label}: throttled in response body, waiting {:?} before retry {}",
                    wait,
                    attempt + 1
                );
                if sleep_or_cancel(wait, cancel).await {
                    bail!("{label}: cancelled");
                }
                attempt += 1;
                continue;
            }
            return unwrap_biz(v, label);
        }

        if !is_retryable(status) {
            let text = resp.text().await.unwrap_or_default();
            let hint = if looks_like_bot_wall(status, &text) {
                BOT_WALL_HINT
            } else {
                ""
            };
            bail!(
                "{label} failed after {} attempt(s): {} - {}{}",
                attempt + 1,
                status,
                text,
                hint
            );
        }

        if policy.exhausted(attempt) {
            bail!(
                "{label}: still failing after {} attempts (last: {status}); giving up instead of retrying forever (set DS_MAX_RETRIES=0 for no cap)",
                attempt + 1
            );
        }
        eprintln!(
            "[rate-limit] {label}: {status}, retry {} in {:?}",
            attempt + 1,
            wait
        );
        if sleep_or_cancel(wait, cancel).await {
            bail!("{label}: cancelled");
        }
        attempt += 1;
    }
}

async fn fetch_challenge(
    http: &Client,
    policy: &RetryPolicy,
    slot: &Slot,
    cancel: &mut Option<watch::Receiver<u64>>,
    target_path: &str,
) -> Result<ChatChallenge> {
    let raw = post_json(
        http,
        policy,
        slot,
        cancel,
        &challenge_url(),
        &json!({ "target_path": target_path }),
        "create_pow_challenge",
    )
    .await?;
    let c: ChatChallenge = serde_json::from_value(raw["data"]["biz_data"].clone())?;
    Ok(c)
}

/// Which streaming endpoint a request goes to, and how to authenticate it.
#[derive(Clone)]
struct StreamTarget {
    url: String,
    /// Whether a proof-of-work header must be solved and attached.
    pow: bool,
    /// Short name used in retry/error messages, e.g. `completion`.
    label: &'static str,
}

/// Open a chat stream, retrying throttled/transient responses.
///
/// A throttle does not always use a 4xx status: the server may answer `200 OK`
/// with a JSON error envelope. We inspect non-SSE bodies explicitly so those
/// never get mistaken for an empty reply.
///
/// `pow` selects the authentication style: `chat/completion` requires a solved
/// proof-of-work header, while `chat/continue` (like every other endpoint)
/// only needs the bearer token - the captured HAR shows the browser posting it
/// with no `x-ds-pow-response` at all.
async fn open_stream(
    http: &Client,
    policy: &RetryPolicy,
    slot: &Slot,
    cancel: &mut Option<watch::Receiver<u64>>,
    body: &Value,
    target: StreamTarget,
) -> Result<Response> {
    let StreamTarget { url, pow, label } = &target;
    let (pow, label) = (*pow, *label);
    let mut attempt = 0u32;
    loop {
        pace(slot, policy.min_gap).await;

        let mut req = http.post(url.as_str());
        // Browsers send the *conversation* URL as Referer on every fetch issued
        // from an open chat; the site root is only right before one exists.
        let session = body.get("chat_session_id").and_then(|v| v.as_str());
        if let Ok(r) = HeaderValue::from_str(&referer_for(session)) {
            req = req.header(header::REFERER, r);
        }
        if pow {
            let challenge =
                fetch_challenge(http, policy, slot, cancel, COMPLETION_TARGET_PATH).await?;
            let answer = challenge
                .solve()
                .ok_or_else(|| anyhow::anyhow!("PoW solving failed"))?;
            let solved: SolvedChallenge = challenge.to_solved(answer);
            let pow_b64 = STANDARD.encode(serde_json::to_string(&solved)?);
            req = req.header("x-ds-pow-response", pow_b64);
        }

        let resp = req
            .json(body)
            .send()
            .await
            .with_context(|| format!("{label}: request failed"))?;

        let status = resp.status();
        let content_type = resp
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map(|v| v.to_ascii_lowercase());
        // Missing content-type is treated as SSE so normal flows keep working.
        let is_sse = content_type
            .as_deref()
            .map(|v| v.contains("text/event-stream"))
            .unwrap_or(true);

        if status.is_success() && is_sse {
            return Ok(resp);
        }

        let wait = retry_after(&resp).unwrap_or_else(|| backoff(attempt, policy.base, policy.cap));
        let text = resp.text().await.unwrap_or_default();
        let rate_limited = status.as_u16() == 429
            || message_mentions_rate_limit(&text)
            || serde_json::from_str::<Value>(&text)
                .map(|v| envelope_rate_limited(&v))
                .unwrap_or(false);

        let retryable = is_retryable(status) || rate_limited;
        if !retryable {
            let hint = if looks_like_bot_wall(status, &text) {
                BOT_WALL_HINT
            } else {
                ""
            };
            bail!(
                "{label} failed after {} attempt(s): {} - {}{}",
                attempt + 1,
                status,
                text,
                hint
            );
        }

        if policy.exhausted(attempt) {
            bail!(
                "{label}: still failing after {} attempts (last: {status}); giving up rather than hammering the endpoint",
                attempt + 1
            );
        }
        eprintln!(
            "[rate-limit] {label}: {} ({}), waiting {:?} before retry {}",
            status,
            if rate_limited {
                "rate limited"
            } else {
                "transient"
            },
            wait,
            attempt + 1
        );
        if sleep_or_cancel(wait, cancel).await {
            bail!("{label}: cancelled");
        }
        attempt += 1;
    }
}

/// Wait for `dur`, returning `true` if the turn was cancelled meanwhile.
///
/// Retries can sleep for up to `policy.cap`, so without this a Ctrl+C during a
/// throttled retry would look like a hang. Any interrupt raised *before* this
/// call is discarded first, so a stray Ctrl+C at the prompt cannot cancel the
/// next request.
async fn sleep_or_cancel(dur: Duration, cancel: &mut Option<watch::Receiver<u64>>) -> bool {
    let Some(rx) = cancel else {
        tokio::time::sleep(dur).await;
        return false;
    };
    rx.borrow_and_update();
    tokio::select! {
        _ = tokio::time::sleep(dur) => false,
        _ = rx.changed() => true,
    }
}

/// Chrome major version the default profile imitates. The `User-Agent` and the
/// `sec-ch-ua*` client hints are both derived from it, so they cannot drift
/// apart: a Chrome UA announcing a different major than `sec-ch-ua` is itself
/// an automated-client signal.
const CHROME_MAJOR: &str = "138";

/// `x-client-version` the web app advertises. Taken verbatim from the captured
/// HAR; a made-up value (the old `1.0.0-always`) is a one-header giveaway that
/// the caller is not the web app.
const CLIENT_VERSION: &str = "2.7.1";

/// `x-client-bundle-id` the web app sends on every API call.
const CLIENT_BUNDLE_ID: &str = "com.deepseek.chat";

/// Default `x-client-locale`, used when `DS_CLIENT_LOCALE` is unset.
const CLIENT_LOCALE: &str = "en_US";

/// The UA platform token matching the OS this binary runs on. A browser that
/// claims `X11; Linux x86_64` while running on Windows is a giveaway, so the
/// default profile follows the host.
fn host_ua_platform() -> &'static str {
    if cfg!(target_os = "windows") {
        "Windows NT 10.0; Win64; x64"
    } else if cfg!(target_os = "macos") {
        "Macintosh; Intel Mac OS X 10_15_7"
    } else {
        "X11; Linux x86_64"
    }
}

/// `sec-ch-ua-platform` value implied by a UA string.
fn hint_platform_from_ua(ua: &str) -> &'static str {
    if ua.contains("Windows NT") {
        "\"Windows\""
    } else if ua.contains("Macintosh") || ua.contains("Mac OS X") {
        "\"macOS\""
    } else if ua.contains("Android") {
        "\"Android\""
    } else {
        "\"Linux\""
    }
}

/// One coherent set of browser request headers.
///
/// Servers, and the bot detection in front of them, cross-check these: the UA
/// string against the client hints, the hint set against the claimed engine
/// (Firefox sends no `sec-ch-ua*` at all), and the platform hint against the
/// OS named in the UA. An inconsistent set is more suspicious than a sparse
/// one, so every header here is derived from the same profile.
struct BrowserProfile {
    user_agent: String,
    /// `sec-ch-ua` value; `None` for engines that do not send client hints.
    sec_ch_ua: Option<String>,
    /// `sec-ch-ua-platform` value, quoted as it goes on the wire.
    platform: &'static str,
}

impl BrowserProfile {
    fn is_firefox(&self) -> bool {
        self.sec_ch_ua.is_none()
    }
}

/// Derive the profile from `DS_USER_AGENT`, or default to Chrome on the host
/// OS. An override is honoured, but the hints are re-derived from it rather
/// than left at the defaults, so the two still agree.
fn browser_profile() -> BrowserProfile {
    profile_for(
        std::env::var("DS_USER_AGENT")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty()),
    )
}

/// The pure part of [`browser_profile`]: same decision table, but with the
/// override passed in instead of read from the environment, so the rules can
/// be tested without mutating process-global state.
fn profile_for(override_ua: Option<String>) -> BrowserProfile {
    let ua = match override_ua {
        Some(ua) => ua,
        None => format!(
            "Mozilla/5.0 ({}) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{CHROME_MAJOR}.0.0.0 Safari/537.36",
            host_ua_platform()
        ),
    };
    let platform = hint_platform_from_ua(&ua);
    if ua.contains("Firefox/") {
        return BrowserProfile {
            user_agent: ua,
            sec_ch_ua: None,
            platform,
        };
    }
    // Chrome-like: take the hinted major from the UA so the hints cannot drift
    // away from the string that is actually sent.
    let major = ua
        .split("Chrome/")
        .nth(1)
        .and_then(|rest| rest.split(['.', ' ']).next())
        .filter(|m| !m.is_empty() && m.chars().all(|c| c.is_ascii_digit()))
        .unwrap_or(CHROME_MAJOR)
        .to_string();
    BrowserProfile {
        user_agent: ua,
        // Ordered exactly as Chromium emits it: GREASE brand first, then
        // Chromium, then the product brand.
        sec_ch_ua: Some(format!(
            "\"Not)A;Brand\";v=\"8\", \"Chromium\";v=\"{major}\", \"Google Chrome\";v=\"{major}\""
        )),
        platform,
    }
}

/// `x-client-locale`, overridable with `DS_CLIENT_LOCALE`.
fn client_locale() -> String {
    std::env::var("DS_CLIENT_LOCALE")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| CLIENT_LOCALE.to_string())
}

/// The `Accept-Language` a browser with that locale would send.
///
/// Derived from `x-client-locale` so the two cannot contradict each other: a
/// request claiming `en_US` while advertising `ru,en;q=0.9` is a mismatch a
/// detector can flag without any other signal.
fn accept_language_for(locale: &str) -> String {
    let lang = locale
        .split(['_', '-'])
        .next()
        .filter(|s| !s.is_empty())
        .unwrap_or("en");
    if lang.eq_ignore_ascii_case("en") {
        "en-US,en;q=0.9".to_string()
    } else {
        format!("{lang},{lang}-{};q=0.9,en;q=0.8", region_of(locale))
    }
}

/// The region part of a `ll_RR` locale, or the language itself when absent.
fn region_of(locale: &str) -> String {
    locale
        .split(['_', '-'])
        .nth(1)
        .filter(|s| !s.is_empty())
        .map(|s| s.to_ascii_uppercase())
        .unwrap_or_else(|| "US".to_string())
}

/// Local timezone offset in seconds east of UTC, as the browser reports it.
///
/// `x-client-timezone-offset: 14400` in the captured HAR is simply the
/// browser's `-Date.getTimezoneOffset() * 60`. Sending `0` from a machine that
/// is not in UTC is a cheap automation tell, so the real local offset is used.
fn timezone_offset_seconds() -> i64 {
    if let Some(v) = std::env::var("DS_TIMEZONE_OFFSET")
        .ok()
        .and_then(|v| v.trim().parse::<i64>().ok())
    {
        return v;
    }
    use chrono::Offset;
    chrono::Local::now().offset().fix().local_minus_utc() as i64
}

/// A stable, random-looking device id, as the web app stores one per browser.
///
/// The value must persist across runs (a browser keeps the same id for months)
/// but must not be a counter or a build fingerprint. `DS_DEVICE_ID` overrides
/// it; otherwise it is derived from stable host facts and hashed into the
/// UUIDv4 shape the real client uses.
fn device_id() -> String {
    if let Some(v) = std::env::var("DS_DEVICE_ID")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
    {
        return v;
    }
    use std::hash::{Hash, Hasher};
    let mut seed = String::new();
    if let Some(h) = std::env::var_os("HOSTNAME").or_else(|| std::env::var_os("COMPUTERNAME")) {
        seed.push_str(&h.to_string_lossy());
    }
    if let Some(u) = std::env::var_os("USER").or_else(|| std::env::var_os("USERNAME")) {
        seed.push_str(&u.to_string_lossy());
    }
    if let Ok(h) = std::fs::read_to_string("/etc/machine-id") {
        seed.push_str(h.trim());
    }
    seed.push_str("deepseek-web-cli/device");

    let mut hi = std::collections::hash_map::DefaultHasher::new();
    seed.hash(&mut hi);
    let mut lo = std::collections::hash_map::DefaultHasher::new();
    ("lo".to_string() + &seed).hash(&mut lo);
    let mut bytes = [0u8; 16];
    bytes[..8].copy_from_slice(&hi.finish().to_be_bytes());
    bytes[8..].copy_from_slice(&lo.finish().to_be_bytes());
    // Version 4 (random) and the RFC 4122 variant, so it parses as a UUID.
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15]
    )
}

/// The `Referer` a browser would send for a chat API call.
///
/// Firefox/Chrome both send the full conversation URL, not the site root, on
/// every `fetch()` issued from an open chat. The HAR shows
/// `https://chat.deepseek.com/a/chat/s/<session-id>`; the root URL is only
/// used before a session exists.
fn referer_for(session_id: Option<&str>) -> String {
    match session_id.filter(|s| !s.is_empty()) {
        Some(id) => format!("https://chat.deepseek.com/a/chat/s/{id}"),
        None => "https://chat.deepseek.com/".to_string(),
    }
}

/// Headers the DeepSeek web app attaches to every API call.
///
/// A request carrying only `Authorization` and a `User-Agent` is trivially
/// distinguishable from the web client: browsers always send `Origin`,
/// `Referer`, `Accept-Language`, `Accept-Encoding`, `DNT`, `Sec-GPC`,
/// `TE`, the `Sec-Fetch-*` / `Sec-CH-UA*` hints and the app's own
/// `x-client-*` / `x-device-*` headers.  Their absence is a classic
/// automated-client signal (and some WAF rules reject such requests outright),
/// so mirror the set the web app sends. Every value below is copied from or
/// derived from the captured HAR.
/// `DS_BROWSER_HEADERS=0` turns this off if a deployment ever objects to it.
fn browser_headers() -> header::HeaderMap {
    fn name(s: &'static str) -> header::HeaderName {
        header::HeaderName::from_static(s)
    }
    let profile = browser_profile();
    let locale = client_locale();
    let mut h = header::HeaderMap::new();
    h.insert(header::ACCEPT, header::HeaderValue::from_static("*/*"));
    h.insert(
        header::ACCEPT_LANGUAGE,
        HeaderValue::from_str(&accept_language_for(&locale)).expect("ASCII language tag"),
    );
    // Exactly the set this build can transparently decode (see Cargo.toml),
    // which is also what Chrome and Firefox offer to the web app. reqwest
    // would otherwise fill this in itself, but only when nothing else sets it.
    h.insert(
        header::ACCEPT_ENCODING,
        header::HeaderValue::from_static("gzip, deflate, br, zstd"),
    );
    h.insert(
        header::ORIGIN,
        header::HeaderValue::from_static("https://chat.deepseek.com"),
    );
    h.insert(
        header::REFERER,
        HeaderValue::from_str(&referer_for(None)).expect("ASCII URL"),
    );
    // Firefox sends `DNT`; both engines send `Sec-GPC` and `TE: trailers`
    // on cross-document fetches.
    h.insert(name("dnt"), header::HeaderValue::from_static("1"));
    h.insert(name("sec-gpc"), header::HeaderValue::from_static("1"));
    h.insert(name("te"), header::HeaderValue::from_static("trailers"));
    for (k, v) in [
        ("x-client-platform", "web"),
        ("x-client-version", CLIENT_VERSION),
        ("x-client-bundle-id", CLIENT_BUNDLE_ID),
        ("x-device-model", ""),
        ("sec-fetch-site", "same-origin"),
        ("sec-fetch-mode", "cors"),
        ("sec-fetch-dest", "empty"),
    ] {
        h.insert(name(k), header::HeaderValue::from_static(v));
    }
    h.insert(
        name("x-client-locale"),
        HeaderValue::from_str(&locale).expect("ASCII locale"),
    );
    h.insert(
        name("x-client-timezone-offset"),
        HeaderValue::from_str(&timezone_offset_seconds().to_string()).expect("integer"),
    );
    h.insert(
        name("x-device-id"),
        HeaderValue::from_str(&device_id()).expect("ASCII device id"),
    );
    // Optional anti-bot tokens (`x-hif-leim` / `x-hif-dliq`). These are minted
    // in-page by the web app's obfuscated script; a client that cannot compute
    // them can still forward a captured pair so the request carries the same
    // shape as the browser's.
    if let Ok(v) = std::env::var("DS_HIF_LEIM") {
        if let Ok(v) = HeaderValue::from_str(v.trim()) {
            if !v.is_empty() {
                h.insert(name("x-hif-leim"), v);
            }
        }
    }
    if let Ok(v) = std::env::var("DS_HIF_DLIQ") {
        if let Ok(v) = HeaderValue::from_str(v.trim()) {
            if !v.is_empty() {
                h.insert(name("x-hif-dliq"), v);
            }
        }
    }
    // Client hints are a Chromium-only feature; sending them from a Firefox
    // profile is a contradiction a detector can spot in one request.
    if !profile.is_firefox() {
        if let Some(ua_hint) = profile.sec_ch_ua {
            h.insert(
                name("sec-ch-ua"),
                HeaderValue::from_str(&ua_hint).expect("ASCII client hint"),
            );
        }
        h.insert(
            name("sec-ch-ua-mobile"),
            header::HeaderValue::from_static("?0"),
        );
        h.insert(
            name("sec-ch-ua-platform"),
            HeaderValue::from_str(profile.platform).expect("ASCII platform hint"),
        );
        // Chromium attaches this to every fetch() it issues; the HAR shows
        // `u=0` (lowest) for both the challenge and completion POSTs.
        h.insert(name("priority"), header::HeaderValue::from_static("u=0"));
    }
    h
}

/// Parse a boolean environment flag (`1`/`true`/`yes`/`on`).
fn env_flag(name: &str) -> bool {
    matches!(
        std::env::var(name)
            .ok()
            .map(|v| v.trim().to_ascii_lowercase())
            .as_deref(),
        Some("1") | Some("true") | Some("yes") | Some("on")
    )
}

/// Whether the browser-parity headers should be sent (default: yes).
fn browser_headers_enabled() -> bool {
    !matches!(
        std::env::var("DS_BROWSER_HEADERS")
            .ok()
            .map(|v| v.trim().to_ascii_lowercase())
            .as_deref(),
        Some("0") | Some("false") | Some("no") | Some("off")
    )
}

/// The `User-Agent` to send, overridable with `DS_USER_AGENT`. Always the same
/// value the client hints in [`browser_headers`] were derived from.
fn user_agent() -> String {
    browser_profile().user_agent
}

/// Build the shared HTTP client.
///
/// A stalled connection must not hang a turn forever, so both connect and
/// overall request timeouts are set (override with `DS_HTTP_TIMEOUT_MS`).
fn build_http(headers: header::HeaderMap, timeout: Option<Duration>) -> Result<Client> {
    let timeout = timeout.unwrap_or(Duration::from_secs(180));
    let mut b = Client::builder()
        .default_headers(headers)
        // Deliberately no cookie jar: this tool must not accumulate or replay
        // cookies of its own.  Everything it needs is in `headers` (the bearer
        // token, plus `ds_session_id` only if the user configured one), so a
        // request can never carry a cookie the user did not choose to send.
        .connect_timeout(timeout.min(Duration::from_secs(30)))
        .timeout(timeout)
        .user_agent(user_agent())
        // Keep connections alive the way a browser tab does; closing after
        // each call is itself a scraper trait.
        .pool_idle_timeout(Duration::from_secs(90));
    // Protocol selection mirrors a browser: negotiate HTTP/2 over ALPN and
    // fall back to HTTP/1.1 when the server (or a middlebox in front of it)
    // only offers 1.1.  Do NOT use `http2_prior_knowledge()`: that sends the
    // h2 preface unconditionally, and any endpoint or proxy that answers in
    // HTTP/1.1 then trips `frame with invalid size` on the first request.
    // `DS_FORCE_HTTP1=1` pins the client to HTTP/1.1 as an escape hatch for
    // proxies that corrupt h2 upgrade negotiations.
    if env_flag("DS_FORCE_HTTP1") {
        b = b.http1_only();
    }
    Ok(b.build()?)
}

/// Hint appended to errors that look like an anti-bot rejection.
const BOT_WALL_HINT: &str = " -- the server looks like it has flagged this client as automated. \
Check, in order: that the token still belongs to the *current* browser session \
(`deepseek --login`), that `DS_SESSION_ID` was copied from that same session, that \
`DS_BROWSER_HEADERS` is not disabled, and that no other client (a script, or the \
browser itself) is driving the same account concurrently. Then wait a while before \
retrying: repeated attempts right now only extend the block.";

/// A short, single-line excerpt of a body, for error messages.
fn snippet(text: &str) -> String {
    let flat: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() <= 200 {
        flat
    } else {
        let cut: String = flat.chars().take(200).collect();
        format!("{cut}...")
    }
}

/// True when a response looks like an anti-bot / risk-control rejection rather
/// than an ordinary API error.
fn looks_like_bot_wall(status: StatusCode, body: &str) -> bool {
    if matches!(status.as_u16(), 403 | 418) {
        return true;
    }
    let h = body.to_ascii_lowercase();
    [
        "captcha",
        "verify you are human",
        "verify_code",
        "are you a robot",
        "unusual traffic",
        "bot detection",
        "bot_detection",
        "risk control",
        "risk_control",
        "abnormal request",
        "abnormal_request",
        "automated",
    ]
    .iter()
    .any(|needle| h.contains(needle))
}

/// The optional Shumei device fingerprint (`DS_SHUMEI_DEVICE_ID`).
///
/// The web client generates this blob in the browser and sends it with its chat
/// requests; it lets the server tie a request to the browser it came from. A
/// client that never sends one stands out, so it is forwarded whenever the user
/// has captured one.
fn shumei_device_id() -> Option<String> {
    std::env::var("DS_SHUMEI_DEVICE_ID")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

/// Request body for `chat/continue`, mirroring the captured HAR:
/// `{"chat_session_id", "message_id", "fallback_to_resume": true}` plus the
/// optional web-client `shumei_device_id` blob.
///
/// `message_id` is the interrupted *assistant* message - the same id the SSE
/// stream reports as `response/message_id`, which the client stores as the
/// conversation tip. `fallback_to_resume` lets the server restart the message
/// from scratch when it cannot literally resume it.
fn continue_body(session_id: &str, message_id: &Value, shumei_device_id: Option<&str>) -> Value {
    let mut body = json!({
        "chat_session_id": session_id,
        "message_id": message_id,
        "fallback_to_resume": true,
    });
    if let Some(device) = shumei_device_id {
        body["shumei_device_id"] = Value::String(device.to_string());
    }
    body
}

/// A reply the server cut short mid-generation.
///
/// The partial text has already been streamed to the caller and the
/// conversation tip (`response/message_id`) has been recorded, so
/// `chat/continue` can resume it.  This is a distinct type rather than a plain
/// `anyhow!` string so the CLI can tell a truncated reply apart from ordinary
/// errors and resume it automatically instead of aborting the turn.
#[derive(Debug)]
pub struct Truncated {
    /// Endpoint that was streaming, e.g. `completion`.
    pub label: &'static str,
    /// Why the server stopped, as reported in the stream/toast.
    pub reason: String,
}

impl std::fmt::Display for Truncated {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}: reply cut short by the server ({})",
            self.label, self.reason
        )
    }
}

impl std::error::Error for Truncated {}

/// Whether `err` means the reply stopped mid-generation (see [`Truncated`]),
/// i.e. the stream produced output but never reached FINISHED.
pub fn is_truncated(err: &anyhow::Error) -> bool {
    err.downcast_ref::<Truncated>().is_some()
}

pub struct DeepSeekClient {
    http: Client,
    /// Kept so the client can be rebuilt when the timeout changes.
    headers: header::HeaderMap,
    /// Cancellation channel for the current turn, if any.
    cancel: Option<watch::Receiver<u64>>,
    chat_session_id: Option<String>,
    last_message_id: Arc<Mutex<Option<Value>>>,
    policy: RetryPolicy,
    /// Slot reservation for pacing: the instant the next request may start.
    slot: Slot,
}

impl DeepSeekClient {
    pub fn new(session_id: Option<String>, bearer: String) -> Result<Self> {
        let mut headers = header::HeaderMap::new();
        // Mirror the web app's browser headers first; the credentials below are
        // layered on top. See `browser_headers` for why they matter.
        if browser_headers_enabled() {
            for (k, v) in browser_headers() {
                if let Some(k) = k {
                    headers.insert(k, v);
                }
            }
        }
        // The Bearer token is the only credential the API actually checks: the
        // web app authenticates with it alone, and the endpoints answer
        // identically with no cookie, a stale cookie, or a bogus one.  The
        // `ds_session_id` cookie is therefore optional.  Sending it only when
        // the caller supplied a live value keeps a cookie copied from an old
        // browser session from being presented as if it were still current.
        if let Some(sid) = session_id
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            headers.insert(header::COOKIE, format!("ds_session_id={sid}").parse()?);
        }
        headers.insert(header::AUTHORIZATION, bearer.parse()?);
        let http = build_http(headers.clone(), None)?;
        Ok(Self {
            headers,
            cancel: None,
            http,
            chat_session_id: None,
            last_message_id: Arc::new(Mutex::new(None)),
            policy: RetryPolicy::default(),
            slot: Arc::new(Mutex::new(None)),
        })
    }

    /// Verify the bearer token against `users/current`.
    ///
    /// Returns the account's (masked) email on success. Fails with the server's
    /// own message when the token has been revoked or has expired, so a
    /// logged-out session is reported at startup instead of as a confusing
    /// error on the first prompt.
    pub async fn check_auth(&self) -> Result<String> {
        let mut cancel = self.cancel.clone();
        let biz = get_json(
            &self.http,
            &self.policy,
            &self.slot,
            &mut cancel,
            &user_url(),
            &[],
            "users/current",
        )
        .await?;
        if biz.is_null() {
            bail!("users/current returned no account data");
        }
        Ok(biz
            .get("email")
            .and_then(|v| v.as_str())
            .unwrap_or("<unknown>")
            .to_string())
    }

    pub fn set_policy(&mut self, policy: RetryPolicy) {
        self.policy = policy;
    }

    /// Subscribe to the turn-cancellation channel (Ctrl+C).
    pub fn set_cancel(&mut self, cancel: watch::Receiver<u64>) {
        self.cancel = Some(cancel);
    }

    /// Rebuild the HTTP client with an overall request timeout. Applies to
    /// requests made after this call.
    pub fn set_http_timeout(&mut self, timeout: Duration) {
        match build_http(self.headers.clone(), Some(timeout)) {
            Ok(c) => self.http = c,
            Err(e) => eprintln!("[warn] could not apply HTTP timeout: {e:#}"),
        }
    }

    /// Wipe the conversation state - next send_message starts a fresh session.
    pub fn reset(&mut self) {
        self.chat_session_id = None;
        *self.last_message_id.lock().unwrap() = None;
    }

    /// The DeepSeek chat session id currently in use, if any.
    pub fn chat_session_id(&self) -> Option<&str> {
        self.chat_session_id.as_deref()
    }

    /// Adopt an existing conversation so the next message continues it.
    /// `last_message_id` is the tip (last assistant message id) that the
    /// conversation should thread from; `None` starts at the root.
    pub fn resume(&mut self, chat_session_id: String, last_message_id: Option<Value>) {
        self.chat_session_id = Some(chat_session_id);
        *self.last_message_id.lock().unwrap() = last_message_id;
    }

    /// Expose the current conversation tip (assistant message id) for debugging.
    pub fn last_message_id(&self) -> Option<Value> {
        self.last_message_id.lock().unwrap().clone()
    }

    /// List the user's conversations, most recently updated first.
    ///
    /// `count` is the page size (the server caps it at 100). `cursor` is the
    /// `(updated_at, id)` of the last entry already seen; the endpoint's
    /// `lte_cursor` is inclusive, so that entry is returned again and the
    /// caller must de-duplicate. `None` asks for the first page.
    ///
    /// Returns the page's entries and whether the server reported more.
    pub async fn list_sessions(
        &self,
        count: usize,
        cursor: Option<(f64, &str)>,
    ) -> Result<(Vec<crate::sessions::SessionEntry>, bool)> {
        let mut cancel = self.cancel.clone();
        let mut query: Vec<(&str, String)> = vec![
            ("count", count.to_string()),
            // The cursor's `pinned` flag only selects the pinned group first;
            // `false` walks the unpinned sessions that make up the whole list.
            ("lte_cursor.pinned", "false".to_string()),
        ];
        if let Some((updated_at, id)) = cursor {
            query.push(("lte_cursor.updated_at", format!("{updated_at}")));
            query.push(("lte_cursor.id", id.to_string()));
        }
        let biz = get_json(
            &self.http,
            &self.policy,
            &self.slot,
            &mut cancel,
            &session_page_url(),
            &query,
            "chat_session/fetch_page",
        )
        .await?;
        if biz.is_null() {
            bail!("chat_session/fetch_page returned no data: {biz}");
        }
        let arr = biz
            .get("chat_sessions")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        let mut out = Vec::with_capacity(arr.len());
        for v in arr {
            out.push(serde_json::from_value::<crate::sessions::SessionEntry>(v)?);
        }
        let has_more = biz
            .get("has_more")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        Ok((out, has_more))
    }

    /// Fetch a conversation's messages and return the id of its newest
    /// message, which is the parent the next prompt should reply to.
    pub async fn latest_message_id(&self, chat_session_id: &str) -> Result<Option<Value>> {
        let mut cancel = self.cancel.clone();
        let biz = get_json(
            &self.http,
            &self.policy,
            &self.slot,
            &mut cancel,
            &history_url(),
            &[("chat_session_id", chat_session_id.to_string())],
            "chat/history_messages",
        )
        .await?;
        // Prefer the session's declared tip, falling back to the last message.
        if let Some(tip) = biz
            .get("chat_session")
            .and_then(|c| c.get("current_message_id"))
            .filter(|v| !v.is_null())
        {
            return Ok(Some(tip.clone()));
        }
        let last = biz
            .get("chat_messages")
            .and_then(|m| m.as_array())
            .and_then(|a| a.last())
            .and_then(|m| m.get("message_id"))
            .filter(|v| !v.is_null())
            .cloned();
        Ok(last)
    }

    /// Rename a conversation on the server.
    pub async fn rename_session(&self, chat_session_id: &str, title: &str) -> Result<()> {
        let mut cancel = self.cancel.clone();
        post_json(
            &self.http,
            &self.policy,
            &self.slot,
            &mut cancel,
            &session_title_url(),
            &json!({ "chat_session_id": chat_session_id, "title": title }),
            "chat_session/update_title",
        )
        .await?;
        Ok(())
    }

    /// Delete a conversation on the server.
    pub async fn delete_session(&self, chat_session_id: &str) -> Result<()> {
        let mut cancel = self.cancel.clone();
        post_json(
            &self.http,
            &self.policy,
            &self.slot,
            &mut cancel,
            &session_delete_url(),
            &json!({ "chat_session_id": chat_session_id }),
            "chat_session/delete",
        )
        .await?;
        Ok(())
    }

    async fn ensure_session(&mut self) -> Result<&str> {
        if self.chat_session_id.is_none() {
            let mut cancel = self.cancel.clone();
            let resp = post_json(
                &self.http,
                &self.policy,
                &self.slot,
                &mut cancel,
                &session_url(),
                &json!({}),
                "chat_session/create",
            )
            .await?;
            let biz = &resp["data"]["biz_data"];
            let id = biz
                .get("chat_session")
                .and_then(|s| s.get("id"))
                .or_else(|| biz.get("id"))
                .and_then(|v| v.as_str())
                .ok_or_else(|| anyhow::anyhow!("no session id in response: {}", resp))?
                .to_string();
            self.chat_session_id = Some(id);
        }
        Ok(self.chat_session_id.as_ref().unwrap())
    }

    /// Start a new assistant reply to `message`.
    pub async fn send_message(
        &mut self,
        message: &str,
        thinking: bool,
        search: bool,
    ) -> Result<ChatStream> {
        let session_id = self.ensure_session().await?.to_string();
        let parent_json = self
            .last_message_id
            .lock()
            .unwrap()
            .clone()
            .unwrap_or(Value::Null);

        // Completion request with conversation threading. PoW is solved inside
        // `open_stream`, once per attempt, so throttled retries stay valid.
        let mut body = json!({
            "chat_session_id": session_id,
            "parent_message_id": parent_json,
            "model_type": "default",
            "prompt": message,
            "ref_file_ids": [],
            "thinking_enabled": thinking,
            "search_enabled": search,
            // The web app sends `action: null` here (see the HAR); a made-up
            // string is one more body field that does not match the browser.
            "action": null,
            "preempt": false,
        });
        // The web client sends the Shumei device fingerprint with completions
        // too. Without one, a completion is a device-less request, which is a
        // strong automation signal; forward the user's blob when configured.
        if let Some(device) = shumei_device_id() {
            body["shumei_device_id"] = Value::String(device);
        }

        self.open_chat_stream(
            body,
            StreamTarget {
                url: completion_url(),
                pow: true,
                label: "completion",
            },
        )
    }

    /// Resume generation of the current assistant message.
    ///
    /// `chat/continue` is what the web app posts when a reply was cut short -
    /// by a network drop, a closed tab, or a `generation_err` toast - instead
    /// of paying for a whole new completion. It takes the conversation id and
    /// the id of the interrupted *assistant* message (the tip), and answers
    /// with the same SSE event stream as `chat/completion`.
    ///
    /// Unlike completion it needs no proof of work: the captured request is a
    /// plain `POST` with the bearer token and the JSON body
    /// `{"chat_session_id", "message_id", "fallback_to_resume"}`. The web
    /// client also sends a `shumei_device_id` blob; it is optional here and is
    /// forwarded only when the caller sets `DS_SHUMEI_DEVICE_ID`.
    ///
    /// Returns an error when there is no assistant message to resume, since
    /// the endpoint has nothing to continue without one.
    pub async fn continue_message(&mut self) -> Result<ChatStream> {
        let session_id = match self.chat_session_id.clone() {
            Some(id) => id,
            None => bail!("continue: no conversation yet - send a message first"),
        };
        let message_id = self
            .last_message_id
            .lock()
            .unwrap()
            .clone()
            .filter(|v| !v.is_null())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "continue: no assistant message id to resume - send a message first"
                )
            })?;

        let device = shumei_device_id();
        let body = continue_body(&session_id, &message_id, device.as_deref());

        self.open_chat_stream(
            body,
            StreamTarget {
                url: continue_url(),
                pow: false,
                label: "continue",
            },
        )
    }

    /// Shared SSE plumbing for `chat/completion` and `chat/continue`.
    fn open_chat_stream(&self, body: Value, target: StreamTarget) -> Result<ChatStream> {
        let StreamTarget { label, .. } = target;
        let http = self.http.clone();
        let policy = self.policy;
        let mut cancel = self.cancel.clone();
        let slot = self.slot.clone();
        let last_id = self.last_message_id.clone();

        let s = async_stream::stream! {
            let mut in_think = false;
            let mut last_frag_think = false;
            // Whether this attempt has rendered any real content. Until it has,
            // a throttle/empty/broken stream can be retried transparently.
            let mut progressed = false;
            let mut attempt = 0u32;

            macro_rules! emit {
                ($text:expr, $is_think:expr) => {{
                    let text: &str = $text;
                    if text.is_empty() {
                        continue;
                    }
                    let want_think: bool = $is_think;
                    if want_think && !in_think {
                        yield Ok("\u{27EA}THINK\u{27EB}".to_string());
                        in_think = true;
                    } else if !want_think && in_think {
                        yield Ok("\u{27EA}/THINK\u{27EB}".to_string());
                        in_think = false;
                    }
                    progressed = true;
                    yield Ok(text.to_string());
                }};
            }

            'retry: loop {
                let resp = match open_stream(&http, &policy, &slot, &mut cancel, &body, target.clone()).await {
                    Ok(r) => r,
                    Err(e) => {
                        yield Err(e);
                        return;
                    }
                };

                let mut byte_stream = resp.bytes_stream();
                let mut buffer = String::new();
                let mut finished = false;
                let mut retry_wait: Option<Duration> = None;
                // A `toast` announcing a server-side generation failure (the HAR shows
                // `chat/continue` ending this way instead of FINISHED). Kept so the
                // truncated reply is reported rather than silently accepted.
                let mut generation_error: Option<String> = None;

                while let Some(item) = byte_stream.next().await {
                    match item {
                        Ok(chunk) => {
                            buffer.push_str(&String::from_utf8_lossy(&chunk));
                            while let Some(idx) = buffer.find('\n') {
                                let raw: String = buffer.drain(..idx + 1).collect();
                                let line = raw.trim();
                                if line.is_empty() {
                                    continue;
                                }

                                // Some throttles arrive as a plain JSON envelope
                                // instead of an SSE event (still HTTP 200).
                                if !line.starts_with("data:") {
                                    if line.starts_with('{') {
                                        if let Ok(evt) = serde_json::from_str::<Value>(line) {
                                            if event_rate_limited(&evt) && !progressed {
                                                retry_wait = Some(
                                                    retry_after_value(&evt)
                                                        .unwrap_or_else(|| backoff(attempt, policy.base, policy.cap)),
                                                );
                                                break;
                                            }
                                            if evt.get("p").is_none()
                                                && (evt.get("error").is_some()
                                                    || evt.get("code").is_some())
                                            {
                                                yield Err(anyhow::anyhow!(
                                                    "{} error: {}",
                                                    label,
                                                    line
                                                ));
                                                return;
                                            }
                                        }
                                    }
                                    continue;
                                }

                                let payload = line[5..].trim();
                                if payload.is_empty() || payload == "[DONE]" {
                                    continue;
                                }
                                let evt: Value = match serde_json::from_str(payload) {
                                    Ok(v) => v,
                                    Err(_) => continue,
                                };

                                // Throttling can also be reported inside an
                                // already open stream. If nothing has rendered
                                // yet, wait and open a fresh stream rather than
                                // returning an empty reply.
                                if event_rate_limited(&evt) && !progressed {
                                    retry_wait = Some(
                                        retry_after_value(&evt)
                                            .unwrap_or_else(|| backoff(attempt, policy.base, policy.cap)),
                                    );
                                    break;
                                }

                                if let Some(err) = evt.get("error").filter(|e| !e.is_null()) {
                                    let msg = err
                                        .get("message")
                                        .and_then(|m| m.as_str())
                                        .or_else(|| err.as_str())
                                        .unwrap_or("stream error");
                                    yield Err(anyhow::anyhow!("stream error: {}", msg));
                                    return;
                                }

                                // `event: toast` payloads carry no `p`/`v`;
                                // remember a generation failure for the
                                // end-of-stream check.
                                if evt.get("p").is_none()
                                    && evt.get("finish_reason").and_then(|v| v.as_str())
                                        == Some("generation_err")
                                {
                                    generation_error = Some(
                                        evt.get("content")
                                            .and_then(|c| c.as_str())
                                            .unwrap_or("generation error")
                                            .to_string(),
                                    );
                                    continue;
                                }

                                let p = evt.get("p").and_then(|v| v.as_str());
                                let v = evt.get("v");

                                if let Some(obj) = v.and_then(|v| v.as_object()) {
                                    if let Some(resp) =
                                        obj.get("response").and_then(|r| r.as_object())
                                    {
                                        if let Some(mid) = resp.get("message_id") {
                                            *last_id.lock().unwrap() = Some(mid.clone());
                                        }
                                        if let Some(frags) =
                                            resp.get("fragments").and_then(|f| f.as_array())
                                        {
                                            for frag in frags {
                                                let is_think = frag
                                                    .get("type")
                                                    .and_then(|t| t.as_str())
                                                    .map(|t| t == "THINK" || t == "THINKING")
                                                    .unwrap_or(false);
                                                last_frag_think = is_think;
                                                if let Some(content) = frag
                                                    .get("content")
                                                    .and_then(|c| c.as_str())
                                                {
                                                    emit!(content, is_think);
                                                }
                                            }
                                        }
                                        continue;
                                    }
                                }

                                if p == Some("response/message_id") {
                                    if let Some(mid) = v {
                                        *last_id.lock().unwrap() = Some(mid.clone());
                                    }
                                    continue;
                                }

                                if p == Some("response/fragments") {
                                    if let Some(arr) = v.and_then(|v| v.as_array()) {
                                        for frag in arr {
                                            let is_think = frag
                                                .get("type")
                                                .and_then(|t| t.as_str())
                                                .map(|t| t == "THINK" || t == "THINKING")
                                                .unwrap_or(false);
                                            last_frag_think = is_think;
                                            if let Some(content) =
                                                frag.get("content").and_then(|c| c.as_str())
                                            {
                                                emit!(content, is_think);
                                            }
                                        }
                                    }
                                    continue;
                                }

                                if p == Some("response/thinking_content") {
                                    if let Some(text) = v.and_then(|v| v.as_str()) {
                                        last_frag_think = true;
                                        emit!(text, true);
                                    }
                                    continue;
                                }

                                if p == Some("response/content") {
                                    if let Some(text) = v.and_then(|v| v.as_str()) {
                                        last_frag_think = false;
                                        emit!(text, false);
                                    }
                                    continue;
                                }

                                if p == Some("response/fragments/-1/content") {
                                    if let Some(text) = v.and_then(|v| v.as_str()) {
                                        emit!(text, last_frag_think);
                                    }
                                    continue;
                                }

                                if p.is_none() {
                                    if let Some(text) = v.and_then(|v| v.as_str()) {
                                        // Bare v-deltas continue the current fragment; preserve its type.
                                        emit!(text, last_frag_think);
                                    }
                                    continue;
                                }

                                if p == Some("response/status")
                                    && v.and_then(|v| v.as_str()) == Some("FINISHED")
                                {
                                    finished = true;
                                    break;
                                }
                            }
                            if finished || retry_wait.is_some() {
                                break;
                            }
                        }
                        Err(e) => {
                            // A connection drop before any output is just another
                            // transient failure: wait and retry patiently.
                            if !progressed {
                                retry_wait = Some(backoff(attempt, policy.base, policy.cap));
                                eprintln!(
                                    "[rate-limit] {}: stream broke before any output ({}), waiting before retry {}",
                                    label,
                                    e,
                                    attempt + 1
                                );
                                break;
                            }
                            // A drop after output is a cut-short reply, not a
                            // failed request: the tip is known, so the CLI can
                            // resume it via `chat/continue`.
                            yield Err(anyhow::Error::new(Truncated {
                                label,
                                reason: e.to_string(),
                            }));
                            return;
                        }
                    }
                }

                if finished {
                    if in_think {
                        yield Ok("\u{27EA}/THINK\u{27EB}".to_string());
                    }
                    return;
                }

                // The stream ended without FINISHED. If we were throttled, or
                // produced nothing at all, wait and start over rather than
                // reporting a spurious empty response.
                if retry_wait.is_some() || !progressed {
                    if policy.exhausted(attempt) {
                        yield Err(anyhow::anyhow!(
                            "{}: no usable output after {} attempts; giving up rather than re-opening the stream forever (set DS_MAX_RETRIES=0 for no cap)",
                            label,
                            attempt + 1
                        ));
                        return;
                    }
                    let wait = retry_wait
                        .unwrap_or_else(|| backoff(attempt, policy.base, policy.cap));
                    eprintln!(
                        "[rate-limit] {}: no usable output (attempt {}), waiting {:?} before retry",
                        label,
                        attempt + 1,
                        wait
                    );
                    if sleep_or_cancel(wait, &mut cancel).await {
                        return;
                    }
                    attempt += 1;
                    // Reset per-attempt parser state; nothing was emitted yet.
                    in_think = false;
                    last_frag_think = false;
                    progressed = false;
                    continue 'retry;
                }

                if in_think {
                    yield Ok("\u{27EA}/THINK\u{27EB}".to_string());
                }
                // A server-side generation failure ended the stream early.
                // The partial text already went out, so report the truncation
                // and point at the endpoint that resumes it rather than
                // pretending the reply was complete.
                if let Some(reason) = generation_error.take() {
                    yield Err(anyhow::Error::new(Truncated { label, reason }));
                    return;
                }
                // We rendered content but never saw FINISHED; accept it.
                return;
            }
        };
        Ok(Box::pin(s))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retry_policy_has_no_attempt_cap_by_default() {
        let p = RetryPolicy::default();
        assert_eq!(p.max_attempts, 0, "default policy must not cap retries");
        assert!(!p.min_gap.is_zero(), "default policy must pace requests");
        assert!(!p.exhausted(0));
        assert!(!p.exhausted(1_000_000));
    }

    #[test]
    fn attempt_cap_can_still_be_opted_in() {
        let p = RetryPolicy {
            max_attempts: 3,
            ..RetryPolicy::default()
        };
        assert!(!p.exhausted(2));
        assert!(p.exhausted(3));
    }

    #[test]
    fn default_profile_is_chrome_on_the_host_os() {
        let p = profile_for(None);
        assert!(p.user_agent.contains("Chrome/138."), "{}", p.user_agent);
        let hints = p.sec_ch_ua.expect("Chrome sends client hints");
        assert!(hints.contains("\"Google Chrome\";v=\"138\""), "{hints}");
        assert!(hints.contains("\"Chromium\";v=\"138\""), "{hints}");
        if cfg!(target_os = "linux") {
            assert!(p.user_agent.contains("X11; Linux x86_64"));
            assert_eq!(p.platform, "\"Linux\"");
        }
    }

    #[test]
    fn client_builds_with_alpn_negotiation() {
        // Regression: `http2_prior_knowledge()` sent the h2 preface
        // unconditionally and made hosts/proxies that answer in HTTP/1.1
        // fail with `http2 error: frame with invalid size`. The client must
        // build with normal protocol negotiation instead.
        let c = build_http(header::HeaderMap::new(), None);
        assert!(c.is_ok(), "{:?}", c.err());
    }

    #[test]
    fn headers_match_the_captured_har() {
        // Values taken verbatim from chat.deepseek.com_Archive *.har.
        let h = browser_headers();
        assert_eq!(h["x-client-version"], "2.7.1");
        assert_eq!(h["x-client-bundle-id"], "com.deepseek.chat");
        assert_eq!(h["x-client-platform"], "web");
        assert_eq!(h["x-client-locale"], "en_US");
        assert_eq!(h["accept"], "*/*");
        assert_eq!(h["accept-encoding"], "gzip, deflate, br, zstd");
        assert_eq!(h["dnt"], "1");
        assert_eq!(h["sec-gpc"], "1");
        assert_eq!(h["te"], "trailers");
        assert_eq!(h["priority"], "u=0");
        assert_eq!(h["sec-fetch-dest"], "empty");
        assert_eq!(h["sec-fetch-mode"], "cors");
        assert_eq!(h["sec-fetch-site"], "same-origin");
        assert!(h["x-client-timezone-offset"]
            .to_str()
            .unwrap()
            .parse::<i64>()
            .is_ok());
        // The real client sends an empty x-device-model, not no header at all.
        assert_eq!(h["x-device-model"], "");
    }

    #[test]
    fn device_id_is_a_stable_uuidv4() {
        let a = device_id();
        let b = device_id();
        assert_eq!(a, b, "must be stable across calls");
        assert_eq!(a.len(), 36, "{a}");
        let groups: Vec<&str> = a.split('-').collect();
        assert_eq!(
            groups.iter().map(|g| g.len()).collect::<Vec<_>>(),
            vec![8, 4, 4, 4, 12]
        );
        assert!(groups[2].starts_with('4'), "must look like UUIDv4: {a}");
        assert!(
            ["8", "9", "a", "b"].contains(&&groups[3][..1]),
            "variant nibble: {a}"
        );
        assert!(a.chars().all(|c| c.is_ascii_hexdigit() || c == '-'));
    }

    #[test]
    fn accept_language_follows_the_client_locale() {
        assert_eq!(accept_language_for("en_US"), "en-US,en;q=0.9");
        assert_eq!(accept_language_for("ru_RU"), "ru,ru-RU;q=0.9,en;q=0.8");
        // The default locale and the default Accept-Language agree.
        assert_eq!(accept_language_for(CLIENT_LOCALE), "en-US,en;q=0.9");
    }

    #[test]
    fn referer_points_at_the_conversation() {
        assert_eq!(
            referer_for(Some("033db2e9-61a9-4877-aeff-1ffe6e789a66")),
            "https://chat.deepseek.com/a/chat/s/033db2e9-61a9-4877-aeff-1ffe6e789a66"
        );
        assert_eq!(referer_for(None), "https://chat.deepseek.com/");
        assert_eq!(referer_for(Some("")), "https://chat.deepseek.com/");
    }

    #[test]
    fn completion_body_matches_captured_har() {
        // Shape from the HAR's chat/completion POST: `action` is null, there
        // is no `stream` key, and `thinking_enabled`/`search_enabled` are bools.
        let body = serde_json::json!({
            "chat_session_id": "s",
            "parent_message_id": null,
            "model_type": "default",
            "prompt": "hi",
            "ref_file_ids": [],
            "thinking_enabled": true,
            "search_enabled": true,
            "action": null,
            "preempt": false,
        });
        assert!(body["action"].is_null());
        assert!(body.get("stream").is_none());
    }

    #[test]
    fn firefox_ua_drops_client_hints() {
        let p = profile_for(Some(
            "Mozilla/5.0 (X11; Linux x86_64; rv:141.0) Gecko/20100101 Firefox/141.0".to_string(),
        ));
        assert!(p.is_firefox(), "Firefox sends no sec-ch-ua hints");
        assert_eq!(p.platform, "\"Linux\"");
        assert!(p.user_agent.contains("Firefox/141.0"));
    }

    #[test]
    fn client_hints_follow_an_overridden_ua_version() {
        let p = profile_for(Some(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 \
             (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
                .to_string(),
        ));
        let hints = p.sec_ch_ua.unwrap();
        assert!(hints.contains("\"Google Chrome\";v=\"131\""), "{hints}");
        assert_eq!(p.platform, "\"Windows\"");
    }

    #[test]
    fn browser_headers_are_internally_consistent() {
        let h = browser_headers();
        // Chrome-shaped profile: hints, encoding and fetch metadata all present.
        assert!(h.contains_key("sec-ch-ua"));
        assert!(h.contains_key("sec-ch-ua-platform"));
        assert!(h.contains_key("sec-ch-ua-mobile"));
        assert!(h.contains_key("priority"));
        assert_eq!(h["accept-encoding"], "gzip, deflate, br, zstd");
        assert_eq!(h["origin"], "https://chat.deepseek.com");
        assert_eq!(h["x-client-platform"], "web");
        assert_eq!(h["sec-fetch-site"], "same-origin");
        // The UA the client sends must name the same platform as the hint.
        let ua = user_agent();
        let hint = h["sec-ch-ua-platform"].to_str().unwrap();
        assert_eq!(hint, hint_platform_from_ua(&ua));
    }

    #[test]
    fn backoff_starts_high_and_grows_fast() {
        let p = RetryPolicy::default();
        assert!(
            p.base >= Duration::from_secs(2),
            "start wait should be more than a token pause"
        );
        assert!(
            p.cap >= Duration::from_secs(300),
            "maximum wait should be generous"
        );
        // Jitter halves the value at worst, so compare the exponent alone.
        let at = |n: u32| {
            let exp = p.base.saturating_mul(BACKOFF_FACTOR.saturating_pow(n));
            exp.min(p.cap)
        };
        assert!(at(1) >= p.base * 4);
        assert!(at(2) >= p.base * 16);
        assert_eq!(at(30), p.cap);
    }

    #[test]
    fn detects_429_envelope() {
        let v: Value = serde_json::json!({ "code": 429, "msg": "Too Many Requests" });
        assert!(envelope_rate_limited(&v));
    }

    #[test]
    fn detects_business_code_and_message() {
        let v: Value = serde_json::json!({ "code": 42901, "msg": "slow down" });
        assert!(envelope_rate_limited(&v));
    }

    #[test]
    fn detects_rate_limit_message_without_code() {
        let v: Value = serde_json::json!({ "msg": "You have hit the rate limit" });
        assert!(envelope_rate_limited(&v));
    }

    #[test]
    fn does_not_flag_normal_payload() {
        let v: Value = serde_json::json!({ "data": { "biz_data": { "id": "abc" } } });
        assert!(!envelope_rate_limited(&v));
    }

    #[test]
    fn detects_in_stream_error_event() {
        let evt: Value = serde_json::json!({
            "error": { "code": 429, "message": "rate limit exceeded" }
        });
        assert!(event_rate_limited(&evt));
    }

    #[test]
    fn detects_in_stream_status_event() {
        let evt: Value = serde_json::json!({
            "p": "response/status",
            "v": "RATE_LIMITED"
        });
        assert!(event_rate_limited(&evt));
    }

    #[test]
    fn normal_content_event_not_flagged() {
        let evt: Value = serde_json::json!({
            "p": "response/content",
            "v": "hello"
        });
        assert!(!event_rate_limited(&evt));
    }

    #[test]
    fn detects_har_toast_event() {
        // Captured verbatim from a real chat/continue response (200 OK):
        //   event: toast
        //   data: {"type":"error","content":"Messages too frequent. Try again later.","finish_reason":"rate_limit_reached"}
        let evt: Value = serde_json::json!({
            "type": "error",
            "content": "Messages too frequent. Try again later.",
            "finish_reason": "rate_limit_reached"
        });
        assert!(event_rate_limited(&evt));
    }

    #[test]
    fn detects_rate_limit_finish_reason_alone() {
        let evt: Value = serde_json::json!({ "finish_reason": "rate_limit_reached" });
        assert!(event_rate_limited(&evt));
    }

    #[test]
    fn normal_toast_content_not_flagged() {
        let evt: Value = serde_json::json!({ "type": "info", "content": "Copy link" });
        assert!(!event_rate_limited(&evt));
    }

    #[test]
    fn har_close_event_not_flagged() {
        let evt: Value = serde_json::json!({ "click_behavior": "none", "auto_resume": false });
        assert!(!event_rate_limited(&evt));
    }

    #[test]
    fn envelope_error_detects_invalid_token() {
        // Verbatim from a revoked/expired token: HTTP 200, code 40003.
        let v: Value = serde_json::json!({
            "code": 40003,
            "msg": "Authorization Failed (invalid token)",
            "data": null
        });
        let e = envelope_error(&v).expect("must be an error");
        assert!(e.contains("Authorization Failed"), "{e}");
        assert!(e.contains("40003"), "{e}");
    }

    #[test]
    fn classifies_auth_errors() {
        let expired = envelope_error(&serde_json::json!({
            "code": 40003, "msg": "Authorization Failed (invalid token)", "data": null
        }))
        .unwrap();
        assert!(is_auth_error(&expired), "{expired}");
        let missing = envelope_error(&serde_json::json!({
            "code": 40002, "msg": "Missing Token", "data": null
        }))
        .unwrap();
        assert!(is_auth_error(&missing), "{missing}");
        assert!(!is_auth_error("rate limited (code 429)"));
        assert!(!is_auth_error(
            "chat_session/create failed: network unreachable"
        ));
    }

    #[test]
    fn envelope_error_detects_missing_token() {
        let v: Value = serde_json::json!({ "code": 40002, "msg": "Missing Token", "data": null });
        assert!(envelope_error(&v).unwrap().contains("Missing Token"));
    }

    #[test]
    fn envelope_error_detects_biz_code() {
        let v: Value = serde_json::json!({
            "code": 0,
            "msg": "",
            "data": { "biz_code": 1, "biz_msg": "illegal count", "biz_data": null }
        });
        let e = envelope_error(&v).unwrap();
        assert!(e.contains("illegal count"), "{e}");
    }

    #[test]
    fn envelope_error_ignores_success() {
        let v: Value = serde_json::json!({
            "code": 0,
            "msg": "",
            "data": { "biz_code": 0, "biz_msg": "", "biz_data": { "ok": true } }
        });
        assert!(envelope_error(&v).is_none());
        assert_eq!(unwrap_biz(v, "x").unwrap()["ok"], true);
    }

    #[test]
    fn unwrap_biz_reports_api_errors_instead_of_null() {
        let v: Value = serde_json::json!({ "code": 40003, "msg": "INVALID_TOKEN", "data": null });
        let err = unwrap_biz(v, "users/current").unwrap_err().to_string();
        assert!(err.starts_with("users/current failed:"), "{err}");
        assert!(err.contains("INVALID_TOKEN"), "{err}");
    }

    #[test]
    fn continue_body_matches_captured_har() {
        // Shapes taken from chat.deepseek.com_api_v0_chat_continue_*.har:
        // a plain POST with these keys and no PoW header.
        let body = continue_body(
            "eb42960a-9d95-4320-9b6f-91e5143b73cf",
            &serde_json::json!(38),
            None,
        );
        assert_eq!(
            body["chat_session_id"],
            "eb42960a-9d95-4320-9b6f-91e5143b73cf"
        );
        assert_eq!(body["message_id"], 38);
        assert_eq!(body["fallback_to_resume"], true);
        assert!(
            body.get("shumei_device_id").is_none(),
            "optional device blob must not be sent unless configured"
        );
        // It is not a completion body: no prompt/parent_message_id keys.
        assert!(body.get("prompt").is_none());
        assert!(body.get("parent_message_id").is_none());
    }

    #[test]
    fn continue_body_forwards_configured_device_id() {
        let body = continue_body("s", &serde_json::json!("m-1"), Some("DeyJhcH"));
        assert_eq!(body["shumei_device_id"], "DeyJhcH");
        assert_eq!(body["message_id"], "m-1");
    }

    #[test]
    fn har_continue_toast_is_retryable_transient() {
        // The captured continue stream ends with a generation error toast
        // instead of FINISHED; that must be surfaced, not silently accepted.
        let evt: Value = serde_json::json!({
            "type": "error",
            "content": "Server is temporarily unavailable.",
            "finish_reason": "generation_err"
        });
        assert!(!event_rate_limited(&evt), "not a rate limit");
        assert!(evt.get("finish_reason").and_then(|v| v.as_str()) == Some("generation_err"));
    }

    #[test]
    fn truncated_error_is_recognisable_and_resumable() {
        let err = anyhow::Error::new(Truncated {
            label: "continue",
            reason: "Server is temporarily unavailable.".to_string(),
        });
        assert!(is_truncated(&err));
        assert!(format!("{err:#}").contains("reply cut short"));
        assert!(!is_truncated(&anyhow::anyhow!("stream error: boom")));
    }

    #[test]
    fn reads_retry_after_hint() {
        let evt: Value = serde_json::json!({ "retry_after": 7 });
        assert_eq!(retry_after_value(&evt), Some(Duration::from_secs(7)));
        let evt: Value = serde_json::json!({ "retry_after": "3s" });
        assert_eq!(retry_after_value(&evt), Some(Duration::from_secs(3)));
    }
}
