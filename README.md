# deepseek-web-cli

A terminal chat client for the DeepSeek web API. It keeps a conversation
threaded server-side, streams replies (including the reasoning region), and can
run shell commands the model proposes. Commands are executed automatically,
inside a sandbox rooted at the directory the CLI was started in.

## Install

Prebuilt binaries are attached to every [release], one archive per target
(`deepseek-<target>.tar.gz` on unix, `deepseek-<target>.zip` on Windows), each
containing the bare `deepseek` binary. If you have
[cargo-binstall](https://github.com/cargo-bins/cargo-binstall) installed, it
picks those up straight from the release page:

    cargo binstall deepseek-web-cli

(After a published release; until then, point it at the checkout with
`cargo binstall --manifest-path . deepseek-web-cli`.)

Otherwise build from source:

    cargo build --release
    install -m755 target/release/deepseek ~/.local/bin/deepseek

[release]: https://github.com/meanwhile131/deepseek-web-cli/releases

### Every commit

Every push to `main` publishes its own release, tagged
`build-<first 12 chars of the commit sha>`, containing both target archives. The newest build is
marked as the repository's "Latest" release; the newest 10 are kept and older
ones are pruned automatically. Grab the latest build with:

    gh release download --repo meanwhile131/deepseek-web-cli \
      --pattern '*linux-gnu.tar.gz' "$(gh release view --repo meanwhile131/deepseek-web-cli --json tagName -q .tagName)"

Or download a specific commit's build:

    gh release download build-<sha> --repo meanwhile131/deepseek-web-cli

`cargo binstall deepseek-web-cli` reads the `package.metadata.binstall` block
in `Cargo.toml`, which points at GitHub's `releases/latest/download/...` alias,
so it always installs the newest build. Version tags (`v*`) still publish their
own release, but they do not take over the "Latest" marker.

## Usage

    deepseek [OPTIONS]

| Option | Effect |
| --- | --- |
| `-u`, `--unsandboxed` | lift the sandbox: give shell commands full, unsandboxed filesystem access |
| `--login` | sign in through the browser even if a token is already saved |
| `--no-browser` | during sign-in, print the URL instead of opening a browser |
| `-h`, `--help` | show help and exit |
| `-V`, `--version` | print the version and exit |

## Signing in

You do not have to copy a token out of the browser. On first run, if no
credential is configured, the tool opens <https://chat.deepseek.com/sign_in>
in your browser and waits for you to sign in the way you normally do (password,
Google, phone, ...). As soon as the web app has stored its token, the CLI picks
it up, verifies it, and saves it to the user-wide secrets file so later runs
start straight away. Pass `--login` to repeat this at any time, or type
`/login` in the session; `/logout` drops the saved token.

How it works: after a sign-in the web app keeps the bearer token in the
browser's local storage under `userToken`, which is exactly the value the API
expects. The CLI reads that entry from the local profile - **read-only, from a
copy**, so a running browser is never disturbed and nothing is written back.

> **This is your browser session, not a separate API key.** The harvested
> token is the credential the web app is already using. Replaying it from the
> CLI means two clients share one session, and a session can only have one
> live token at a time: when the server rotates it for one client, the other
> is signed out. If you need the CLI and the browser at the same time, that is
> the trade-off - the CLI no longer papers over it by silently re-harvesting
> the token, which made the logouts worse (each re-read rotated the session
> again). A rejected token now fails once with an explanation instead.
Firefox (`storage/default/https+++chat.deepseek.com/ls/data.sqlite`) and
Chromium-family browsers (Chrome, Chromium, Edge, Brave, Vivaldi; the
`Local Storage/leveldb` database) are supported, including Snap/Flatpak paths
and the Windows locations.

If no browser data can be read (headless machine, unusual setup) the URL is
printed so you can sign in elsewhere, or you can set the token yourself as
described below.

## Credentials

| Variable | Meaning |
| --- | --- |
| `AUTHORIZATION_TOKEN` | The `Bearer ...` value (a bare token also works). Optional - without it the browser sign-in runs. |
| `DS_SESSION_ID` | optional. The `ds_session_id` cookie from chat.deepseek.com |

Only the bearer token is checked by the API. `DS_SESSION_ID` may be omitted
(or left empty); when set it is sent as a cookie, when unset no `Cookie`
header is sent at all. The tool keeps no cookie jar of its own, so it can never
replay a cookie the user did not configure.

At startup the token is verified against `users/current`. If it has been
revoked or has expired the tool **stops** and tells you where the secrets file
is; it does not loop through browser sign-ins. Re-harvesting a rejected token
rotates the shared session and signs you out of the browser, so signing in
again is something you ask for explicitly (`deepseek --login` or `/login`).

Credentials are read from the environment first, then from a user-wide secrets
file:

1. `$DS_SECRETS_FILE`
2. `$XDG_CONFIG_HOME/deepseek-web-cli/.env`
3. `~/.config/deepseek-web-cli/.env`

The file uses one `KEY=VALUE` per line; `#` comments, an `export ` prefix and
matching quotes are accepted. Real environment variables always win. A token
harvested by `--login`/`/login` is written here with mode `0600`, replacing any
previous `AUTHORIZATION_TOKEN` line and leaving the rest of the file untouched.

## Markdown rendering

Assistant replies are rendered as Markdown as they stream in: headings,
bullet and numbered lists, blockquotes, horizontal rules and fenced code
blocks are styled, and inline `**bold**`, `*italic*`, `~~strikethrough~~`,
`` `code` `` and `[text](url)` markers are replaced with their styled text
(links keep the URL in parentheses). The reasoning region and shell output
are left untouched. Styling is ANSI only and is suppressed when stdout is not
a terminal or `NO_COLOR` is set; set `CLICOLOR_FORCE=1` to keep it when piping.

## Commands

| Input | Effect |
| --- | --- |
| `exit` / Ctrl+D | quit |
| `/login` | sign in again through the browser (replaces the saved token) |
| `/logout` | remove the saved token from the secrets file (sign out in the browser too, or the next run re-reads that session) |
| `/new` | start a fresh conversation (the old one stays resumable) |
| `/continue` | resume the interrupted reply via `chat/continue` (no new completion charge, no proof of work) |
| `/id` | show the current conversation tip |
| `/goal <text>` | set a goal and let the model work until it reports completion |
| `/goal`, `/goal show`, `/goal status` | show the active goal |
| `/goal clear`, `none`, `off` | drop the active goal |
| `/sessions` | list conversations, oldest first |
| `/sessions more` | load the next (older) page |
| `/resume <n|id-prefix>` | continue a saved conversation (restores its goal and reprompts the model to keep working on it) |
| `/label <text>` | rename the current conversation |
| `/forget <n|id-prefix>` | delete a conversation on the server |

### Resuming a cut-short reply

`chat/completion` is not the only way to get assistant output: when a reply is
interrupted (network drop, closed tab, or a server-side `generation_err`), the
web client posts to `chat/continue` with the conversation id and the id of the
interrupted assistant message instead of paying for a whole new completion.
`/continue` does the same:

```json
{"chat_session_id": "<id>", "message_id": 38, "fallback_to_resume": true}
```

The endpoint returns the same SSE event stream as `chat/completion` and needs
no proof of work - only the bearer token. The client tracks the assistant
message id from the stream, so `/continue` resumes whatever reply was last cut
off in the current conversation.

When a reply is cut short mid-generation (a `generation_err` toast, or the
connection dropping after some output), the CLI resumes it **automatically** by
posting to `chat/continue` itself - no need to type `/continue`, and no new
completion is charged. The partial text stays on screen and streaming picks up
where it stopped. Resuming is unlimited: a truncated reply keeps being resumed
until it completes (Ctrl+C interrupts it). `/continue` is still useful when you
come back to a conversation whose reply was interrupted in an earlier session.

Ctrl+C cancels the current reply, command, or retry backoff and returns to the
prompt; at the prompt it just clears the line.

Up/down at the prompt cycle through previously entered lines - both ordinary
messages and slash commands - from this and earlier runs. History is kept in
`~/.config/deepseek-web-cli/history` (or `$DS_HISTORY_FILE`), capped at 1000
entries with consecutive duplicates and blank lines dropped. Prefix a line with
a space to keep it out of the history.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `DS_THINKING` | `1` | request the reasoning output |
| `DS_SEARCH` | `1` | enable web search |
| `DS_SHELL_TIMEOUT_SECS` | `120` | default per-command timeout |
| `DS_SHELL_MAX_OUTPUT_BYTES` | `16384` | cap kept per output stream |
| `DS_HTTP_TIMEOUT_MS` | `180000` | overall HTTP request timeout |
| `DS_RETRY_BASE_MS` | `2000` | first retry backoff step |
| `DS_RETRY_CAP_MS` | `300000` | backoff ceiling |
| `DS_MIN_GAP_MS` | `1500` | minimum spacing between requests |
| `DS_MAX_RETRIES` | `0` | retries per request before giving up (`0` = unlimited, the default) |
| `DS_MAX_GOAL_TURNS` | `25` | completion requests `/goal` may spend before pausing |
| `DS_MAX_SHELL_ITERS` | `50` | shell round-trips per turn before stopping |
| `DS_SHUMEI_DEVICE_ID` | unset | optional `shumei_device_id` blob forwarded to `chat/continue` (the web client sends one; the endpoint works without it) |
| `DS_FORCE_HTTP1` | `0` | pin the HTTP client to HTTP/1.1 (escape hatch for proxies that break h2 negotiation) |
| `DS_BROWSER_HEADERS` | `1` | send the web app's browser headers (`0`/`false`/`no`/`off` disables) |
| `DS_USER_AGENT` | Chrome on the host OS | override the `User-Agent`; the client hints are re-derived to match it |
| `DS_CLIENT_LOCALE` | `en_US` | `x-client-locale`; `Accept-Language` is derived from it |
| `DS_TIMEZONE_OFFSET` | local offset | `x-client-timezone-offset` in seconds east of UTC |
| `DS_DEVICE_ID` | derived, stable | `x-device-id`; must stay the same across runs |
| `DS_HIF_LEIM` / `DS_HIF_DLIQ` | unset | optional captured `x-hif-*` anti-bot tokens forwarded with completions |
| `DS_GOALS_FILE` | XDG data dir | where per-conversation goals live |
| `DS_SECRETS_FILE` | XDG config dir | where credentials are read from and saved |
| `DS_HISTORY_FILE` | XDG config dir | where prompt history is read from and saved |
| `DS_DEBUG_PENDING` | `0` | dump model output and extracted commands |
| `DS_DEBUG_SECRETS` | `0` | report which secrets file was loaded |
| `NO_COLOR` / `CLICOLOR_FORCE` | - | follow https://no-color.org |

Requests are paced and retries back off exponentially (base x4 per attempt,
jittered, up to the ceiling) with no fixed attempt cap by default. A client
that retries forever with no spacing looks like a scraper to abuse detection;
the growing wait is what keeps retries polite, so keep `DS_MIN_GAP_MS` above
zero and raise it rather than setting it to `0`.

### Blending in with browser traffic

Every request is dressed as the web app itself makes it, because a call that
sends only `Authorization` and a bare `User-Agent` is easy to tell apart from a
browser. The headers are built from one coherent profile, so nothing
contradicts anything else:

- `User-Agent` defaults to Chrome on the platform the binary is running on
  (Linux, macOS or Windows), not a hard-coded string.
- `sec-ch-ua` / `sec-ch-ua-platform` / `sec-ch-ua-mobile` carry the same Chrome
  major version and the same OS as the UA. A Firefox UA (`DS_USER_AGENT`) drops
  the `Sec-CH-UA*` hints entirely, since Firefox sends none.
- `Accept-Encoding: gzip, deflate, br, zstd` matches the encodings the client
  can actually decode, so compressed responses look (and are handled) like the
  browser's.
- `Origin`, `Accept-Language`, `Sec-Fetch-Site/Mode/Dest`, `DNT`, `Sec-GPC`,
  `TE: trailers`, `Priority: u=0`, and the app's own `x-client-version`
  (`2.7.1`), `x-client-bundle-id`, `x-client-locale`, `x-client-timezone-offset`,
  `x-device-id` and `x-device-model` headers are set exactly as the web app
  sends them in the captured HAR.
- `x-client-locale` and `Accept-Language` are derived from each other, and
  `x-client-timezone-offset` is the machine's real offset, so the locale and
  timezone in the request are not a UTC/English default that stands out.
- `x-device-id` is a stable UUIDv4 derived from the host and kept the same
  across runs, the way a browser keeps one per profile.
- `Referer` is the conversation URL (`/a/chat/s/<id>`) once a session exists,
  not the site root on every call.
- The completion body matches the HAR: `action: null`, no `stream` key.
- Connections negotiate HTTP/2 over ALPN and fall back to HTTP/1.1, with keep-alive pooling, like a browser tab. `DS_FORCE_HTTP1=1` forces HTTP/1.1.

Requests are also paced (`DS_MIN_GAP_MS`) and retries jittered, so the traffic
shape - not just the headers - looks like a person typing, not a scraper. Set
`DS_BROWSER_HEADERS=0` to send the credentials alone if a deployment objects.

## Shell commands

The model asks to run a command with a tagged block:

    <shell>command</shell>
    <shell timeout="5m">long running command</shell>

Commands are auto-approved: the model's `<shell>` blocks are executed as soon
as they arrive, without a confirmation prompt. What bounds them is the
**sandbox**, not an approval step.

By default every command runs confined to the directory the CLI was started
from:

* the working directory, `PWD` and `TMPDIR` are all set to that directory, so
  relative paths and scratch files resolve inside it. `HOME` is deliberately
  left at the user's real home (read-only under the OS sandbox), so tools that
  read config from `~` keep working;
* where an OS sandbox helper is installed the confinement is enforced rather
  than merely conventional. On Linux, [bubblewrap](https://github.com/containers/bubblewrap)
  (`bwrap`) is used: the whole filesystem is mounted read-only and only the
  sandbox root is writable. On macOS, `sandbox-exec` denies writes outside the
  root. If neither helper is present a warning is printed at startup and only
  the working directory is redirected - an absolute path can still reach
  outside, so install `bwrap` for real isolation.

Pass `-u` / `--unsandboxed` (`--no-sandbox` also works) to turn this off and
give commands full, unsandboxed access to the machine.

Commands run under `sh -c` with stdin at `/dev/null`. On timeout the child and
its whole process group are killed. Output is capped per stream (head and tail
are kept) before being sent back to the model. Interactive programs will see
EOF rather than the terminal.

## Development

    cargo test
    cargo clippy --all-targets
    cargo fmt --check
