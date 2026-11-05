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

// ---------------------------------------------------------------------------
// Rate-limit / transient-failure policy
// ---------------------------------------------------------------------------

/// Statuses worth retrying: explicit throttling plus transient upstream errors.
fn is_retryable(status: StatusCode) -> bool {
    matches!(status.as_u16(), 429 | 500 | 502 | 503 | 504)
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

/// Retry knobs, configurable from the environment in `main`.
#[derive(Clone, Copy, Debug)]
pub struct RetryPolicy {
    /// Retries after the first attempt (so total tries = max_retries + 1).
    pub max_retries: u32,
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
            max_retries: 6,
            base: Duration::from_millis(500),
            cap: Duration::from_secs(30),
            min_gap: Duration::ZERO,
        }
    }
}

pub struct DeepSeekClient {
    http: Client,
    chat_session_id: Option<String>,
    last_message_id: Arc<Mutex<Option<Value>>>,
    policy: RetryPolicy,
    /// Slot reservation for pacing: the instant the next request may start.
    next_slot: Mutex<Option<Instant>>,
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
            next_slot: Mutex::new(None),
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

    /// Expose the current conversation tip (assistant message id) for debugging.
    pub fn last_message_id(&self) -> Option<Value> {
        self.last_message_id.lock().unwrap().clone()
    }

    /// Client-side pacing: sleep until this request's slot opens, and reserve
    /// the following slot. Concurrent callers therefore queue up instead of
    /// bursting. No-op when `min_gap` is zero.
    async fn pace(&self) {
        let gap = self.policy.min_gap;
        if gap.is_zero() {
            return;
        }
        let wait = {
            let mut slot = self.next_slot.lock().unwrap();
            let now = Instant::now();
            let wait = match *slot {
                Some(at) => at.saturating_duration_since(now),
                None => Duration::ZERO,
            };
            *slot = Some(now + wait + gap);
            wait
        };
        if !wait.is_zero() {
            tokio::time::sleep(wait).await;
        }
    }

    /// POST JSON with retry on throttling / transient upstream failures.
    async fn post_json(&self, url: &str, body: &Value, label: &str) -> Result<Value> {
        let mut attempt = 0u32;
        loop {
            self.pace().await;
            let resp = self
                .http
                .post(url)
                .json(body)
                .send()
                .await
                .with_context(|| format!("{label}: request failed"))?;

            let status = resp.status();
            if status.is_success() {
                return resp
                    .json()
                    .await
                    .with_context(|| format!("{label}: invalid JSON response"));
            }

            let wait = retry_after(&resp)
                .unwrap_or_else(|| backoff(attempt, self.policy.base, self.policy.cap));

            if !is_retryable(status) || attempt >= self.policy.max_retries {
                let text = resp.text().await.unwrap_or_default();
                bail!(
                    "{label} failed after {} attempt(s): {} - {}",
                    attempt + 1,
                    status,
                    text
                );
            }

            eprintln!(
                "[rate-limit] {label}: {status}, retry {}/{} in {:?}",
                attempt + 1,
                self.policy.max_retries,
                wait
            );
            tokio::time::sleep(wait).await;
            attempt += 1;
        }
    }

    async fn ensure_session(&mut self) -> Result<&str> {
        if self.chat_session_id.is_none() {
            let resp = self
                .post_json(
                    "https://chat.deepseek.com/api/v0/chat_session/create",
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

    async fn fetch_challenge(&self, target: &str) -> Result<ChatChallenge> {
        let raw = self
            .post_json(
                "https://chat.deepseek.com/api/v0/chat/create_pow_challenge",
                &json!({ "target_path": target }),
                "create_pow_challenge",
            )
            .await?;
        let c: ChatChallenge = serde_json::from_value(raw["data"]["biz_data"].clone())?;
        Ok(c)
    }

    /// POST the completion, solving a *fresh* PoW for every attempt.
    ///
    /// The challenge is bound to `target_path` and expires, so it cannot be
    /// reused across a retry: each loop iteration fetches and solves a new one.
    async fn post_completion(&self, body: &Value) -> Result<Response> {
        let mut attempt = 0u32;
        loop {
            self.pace().await;

            let challenge = self.fetch_challenge("/api/v0/chat/completion").await?;
            let answer = challenge
                .solve()
                .ok_or_else(|| anyhow::anyhow!("PoW solving failed"))?;
            let solved: SolvedChallenge = challenge.to_solved(answer);
            let pow_b64 = STANDARD.encode(serde_json::to_string(&solved)?);

            let resp = self
                .http
                .post("https://chat.deepseek.com/api/v0/chat/completion")
                .header("x-ds-pow-response", pow_b64)
                .json(body)
                .send()
                .await
                .context("completion: request failed")?;

            let status = resp.status();
            if status.is_success() {
                return Ok(resp);
            }

            let wait = retry_after(&resp)
                .unwrap_or_else(|| backoff(attempt, self.policy.base, self.policy.cap));

            if !is_retryable(status) || attempt >= self.policy.max_retries {
                let text = resp.text().await.unwrap_or_default();
                bail!(
                    "completion failed after {} attempt(s): {} - {}",
                    attempt + 1,
                    status,
                    text
                );
            }

            eprintln!(
                "[rate-limit] completion: {status}, retry {}/{} in {:?}",
                attempt + 1,
                self.policy.max_retries,
                wait
            );
            tokio::time::sleep(wait).await;
            attempt += 1;
        }
    }

    pub async fn send_message(&mut self, message: &str, thinking: bool, search: bool) -> Result<ChatStream> {
        let session_id = self.ensure_session().await?.to_string();
        let parent_json = self
            .last_message_id
            .lock()
            .unwrap()
            .clone()
            .unwrap_or(Value::Null);

        // Completion request with conversation threading. PoW is solved inside
        // `post_completion`, once per attempt, so throttled retries stay valid.
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

        let resp = self.post_completion(&body).await?;

        let mut byte_stream = resp.bytes_stream();
        let last_id = self.last_message_id.clone();

        // SSE parser with message_id capture
        let s = async_stream::stream! {
            let mut buffer = String::new();
            let mut in_think = false;
            let mut last_frag_think = false;

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
                    yield Ok(text.to_string());
                }};
            }

            while let Some(item) = byte_stream.next().await {
                match item {
                    Ok(chunk) => {
                        buffer.push_str(&String::from_utf8_lossy(&chunk));
                        while let Some(idx) = buffer.find('\n') {
                            let line: String = buffer.drain(..idx + 1).collect();
                            let line = line.trim();
                            if !line.starts_with("data:") {
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
                            // Throttling can also be reported inside an already
                            // open stream; surface it as a normal error.
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
                                if let Some(resp) = obj.get("response").and_then(|r| r.as_object()) {
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
                                            if let Some(content) =
                                                frag.get("content").and_then(|c| c.as_str())
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
                                if in_think {
                                    yield Ok("\u{27EA}/THINK\u{27EB}".to_string());
                                }
                                return;
                            }
                        }
                    }
                    Err(e) => {
                        yield Err(anyhow::Error::from(e));
                        return;
                    }
                }
            }
        };
        Ok(Box::pin(s))
    }
}
