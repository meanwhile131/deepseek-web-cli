use anyhow::{bail, Context, Result};
use base64::{engine::general_purpose::STANDARD, Engine};
use futures_util::{Stream, StreamExt};
use reqwest::{header, Client, Response, StatusCode};
use serde_json::{json, Value};
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::pow::{ChatChallenge, SolvedChallenge};

pub type ChatStream = Pin<Box<dyn Stream<Item = Result<String>> + Send>>;

/// Shared pacing slot: the instant the next request is allowed to start.
type Slot = Arc<Mutex<Option<Instant>>>;

const COMPLETION_URL: &str = "https://chat.deepseek.com/api/v0/chat/completion";
const CHALLENGE_URL: &str = "https://chat.deepseek.com/api/v0/chat/create_pow_challenge";
const SESSION_URL: &str = "https://chat.deepseek.com/api/v0/chat_session/create";
const SESSION_PAGE_URL: &str = "https://chat.deepseek.com/api/v0/chat_session/fetch_page";
const SESSION_TITLE_URL: &str = "https://chat.deepseek.com/api/v0/chat_session/update_title";
const SESSION_DELETE_URL: &str = "https://chat.deepseek.com/api/v0/chat_session/delete";
const HISTORY_URL: &str = "https://chat.deepseek.com/api/v0/chat/history_messages";

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

/// Exponential backoff with jitter: base * 2^attempt, capped, then randomised
/// into [cap/2, cap] so parallel clients do not retry in lockstep.
fn backoff(attempt: u32, base: Duration, cap: Duration) -> Duration {
    let exp = base.saturating_mul(1u32 << attempt.min(16));
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
        || h.contains("ratelimit")
        || h.contains("too many requests")
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
    if let Some(err) = evt.get("error").filter(|e| !e.is_null()) {
        if is_rate_limit_code(err) || message_mentions_rate_limit(&err.to_string()) {
            return true;
        }
    }
    // Envelope-shaped event (no `p` path) carrying a code/msg.
    if evt.get("p").is_none() && (evt.get("code").is_some() || evt.get("msg").is_some()) {
        if envelope_rate_limited(evt) {
            return true;
        }
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
    /// First backoff step; doubles every attempt.
    pub base: Duration,
    /// Upper bound on any single wait.
    pub cap: Duration,
    /// Minimum spacing between consecutive requests (client-side pacing).
    pub min_gap: Duration,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            // Patient by default: throttling is normal and worth waiting out.
            base: Duration::from_millis(500),
            cap: Duration::from_secs(60),
            min_gap: Duration::ZERO,
        }
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
            let v: Value = serde_json::from_str(&text)
                .with_context(|| format!("{label}: invalid JSON response"))?;
            if envelope_rate_limited(&v) {
                eprintln!(
                    "[rate-limit] {label}: throttled in response body, waiting {:?} before retry {}",
                    wait,
                    attempt + 1
                );
                tokio::time::sleep(wait).await;
                attempt += 1;
                continue;
            }
            return Ok(v);
        }

        if !is_retryable(status) {
            let text = resp.text().await.unwrap_or_default();
            bail!(
                "{label} failed after {} attempt(s): {} - {}",
                attempt + 1,
                status,
                text
            );
        }

        eprintln!(
            "[rate-limit] {label}: {status}, retry {} in {:?}",
            attempt + 1,
            wait
        );
        tokio::time::sleep(wait).await;
        attempt += 1;
    }
}

/// GET a URL with retry on throttling / transient upstream failures and
/// unwrap the `data.biz_data` payload. `query` is appended verbatim.
async fn get_json(
    http: &Client,
    policy: &RetryPolicy,
    slot: &Slot,
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
            let v: Value = serde_json::from_str(&text)
                .with_context(|| format!("{label}: invalid JSON response"))?;
            if envelope_rate_limited(&v) {
                eprintln!(
                    "[rate-limit] {label}: throttled in response body, waiting {:?} before retry {}",
                    wait,
                    attempt + 1
                );
                tokio::time::sleep(wait).await;
                attempt += 1;
                continue;
            }
            return Ok(v["data"]["biz_data"].clone());
        }

        if !is_retryable(status) {
            let text = resp.text().await.unwrap_or_default();
            bail!(
                "{label} failed after {} attempt(s): {} - {}",
                attempt + 1,
                status,
                text
            );
        }

        eprintln!(
            "[rate-limit] {label}: {status}, retry {} in {:?}",
            attempt + 1,
            wait
        );
        tokio::time::sleep(wait).await;
        attempt += 1;
    }
}

async fn fetch_challenge(
    http: &Client,
    policy: &RetryPolicy,
    slot: &Slot,
) -> Result<ChatChallenge> {
    let raw = post_json(
        http,
        policy,
        slot,
        CHALLENGE_URL,
        &json!({ "target_path": "/api/v0/chat/completion" }),
        "create_pow_challenge",
    )
    .await?;
    let c: ChatChallenge = serde_json::from_value(raw["data"]["biz_data"].clone())?;
    Ok(c)
}

/// Open a completion stream, retrying throttled/transient responses.
///
/// A throttle does not always use a 4xx status: the server may answer `200 OK`
/// with a JSON error envelope. We inspect non-SSE bodies explicitly so those
/// never get mistaken for an empty reply.
async fn open_completion(
    http: &Client,
    policy: &RetryPolicy,
    slot: &Slot,
    body: &Value,
) -> Result<Response> {
    let mut attempt = 0u32;
    loop {
        pace(slot, policy.min_gap).await;

        let challenge = fetch_challenge(http, policy, slot).await?;
        let answer = challenge
            .solve()
            .ok_or_else(|| anyhow::anyhow!("PoW solving failed"))?;
        let solved: SolvedChallenge = challenge.to_solved(answer);
        let pow_b64 = STANDARD.encode(serde_json::to_string(&solved)?);

        let resp = http
            .post(COMPLETION_URL)
            .header("x-ds-pow-response", pow_b64)
            .json(body)
            .send()
            .await
            .context("completion: request failed")?;

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
            bail!(
                "completion failed after {} attempt(s): {} - {}",
                attempt + 1,
                status,
                text
            );
        }

        eprintln!(
            "[rate-limit] completion: {} ({}), waiting {:?} before retry {}",
            status,
            if rate_limited {
                "rate limited"
            } else {
                "transient"
            },
            wait,
            attempt + 1
        );
        tokio::time::sleep(wait).await;
        attempt += 1;
    }
}

pub struct DeepSeekClient {
    http: Client,
    chat_session_id: Option<String>,
    last_message_id: Arc<Mutex<Option<Value>>>,
    policy: RetryPolicy,
    /// Slot reservation for pacing: the instant the next request may start.
    slot: Slot,
}

impl DeepSeekClient {
    pub fn new(session_id: String, bearer: String) -> Result<Self> {
        let mut headers = header::HeaderMap::new();
        headers.insert(
            header::COOKIE,
            format!("ds_session_id={}", session_id).parse()?,
        );
        headers.insert(header::AUTHORIZATION, bearer.parse()?);
        let http = Client::builder()
            .default_headers(headers)
            .cookie_store(true)
            .user_agent(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 \
                 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            )
            .build()?;
        Ok(Self {
            http,
            chat_session_id: None,
            last_message_id: Arc::new(Mutex::new(None)),
            policy: RetryPolicy::default(),
            slot: Arc::new(Mutex::new(None)),
        })
    }

    pub fn set_policy(&mut self, policy: RetryPolicy) {
        self.policy = policy;
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
    /// `count` is the page size (the server caps it at 100) and `offset` the
    /// number already skipped. Returns the page's entries and whether the
    /// server reported more beyond it.
    pub async fn list_sessions(
        &self,
        count: usize,
        offset: usize,
    ) -> Result<(Vec<crate::sessions::SessionEntry>, bool)> {
        let biz = get_json(
            &self.http,
            &self.policy,
            &self.slot,
            SESSION_PAGE_URL,
            &[("count", count.to_string()), ("offset", offset.to_string())],
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
        let biz = get_json(
            &self.http,
            &self.policy,
            &self.slot,
            HISTORY_URL,
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
        post_json(
            &self.http,
            &self.policy,
            &self.slot,
            SESSION_TITLE_URL,
            &json!({ "chat_session_id": chat_session_id, "title": title }),
            "chat_session/update_title",
        )
        .await?;
        Ok(())
    }

    /// Delete a conversation on the server.
    pub async fn delete_session(&self, chat_session_id: &str) -> Result<()> {
        post_json(
            &self.http,
            &self.policy,
            &self.slot,
            SESSION_DELETE_URL,
            &json!({ "chat_session_id": chat_session_id }),
            "chat_session/delete",
        )
        .await?;
        Ok(())
    }

    async fn ensure_session(&mut self) -> Result<&str> {
        if self.chat_session_id.is_none() {
            let resp = post_json(
                &self.http,
                &self.policy,
                &self.slot,
                SESSION_URL,
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
        // `open_completion`, once per attempt, so throttled retries stay valid.
        let body = json!({
            "chat_session_id": session_id,
            "parent_message_id": parent_json,
            "model_type": "default",
            "prompt": message,
            "ref_file_ids": [],
            "thinking_enabled": thinking,
            "search_enabled": search,
            "action": "default",
            "preempt": false,
            "stream": true,
        });

        let http = self.http.clone();
        let policy = self.policy;
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
                let resp = match open_completion(&http, &policy, &slot, &body).await {
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
                                                    "completion error: {}",
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
                                    "[rate-limit] completion: stream broke before any output ({}), waiting before retry {}",
                                    e,
                                    attempt + 1
                                );
                                break;
                            }
                            yield Err(anyhow::Error::from(e));
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
                    let wait = retry_wait
                        .unwrap_or_else(|| backoff(attempt, policy.base, policy.cap));
                    eprintln!(
                        "[rate-limit] completion: no usable output (attempt {}), waiting {:?} before retry",
                        attempt + 1,
                        wait
                    );
                    tokio::time::sleep(wait).await;
                    attempt += 1;
                    // Reset per-attempt parser state; nothing was emitted yet.
                    in_think = false;
                    last_frag_think = false;
                    progressed = false;
                    continue 'retry;
                }

                // We rendered content but never saw FINISHED; accept it.
                if in_think {
                    yield Ok("\u{27EA}/THINK\u{27EB}".to_string());
                }
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
    fn reads_retry_after_hint() {
        let evt: Value = serde_json::json!({ "retry_after": 7 });
        assert_eq!(retry_after_value(&evt), Some(Duration::from_secs(7)));
        let evt: Value = serde_json::json!({ "retry_after": "3s" });
        assert_eq!(retry_after_value(&evt), Some(Duration::from_secs(3)));
    }
}
