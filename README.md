# deepseek-web-cli

A terminal chat client for the DeepSeek web API. It keeps a conversation
threaded server-side, streams replies (including the reasoning region), and can
run shell commands the model proposes, with your approval by default.

## Install

    cargo build --release
    install -m755 target/release/deepseek ~/.local/bin/deepseek

## Usage

    deepseek [OPTIONS]

| Option | Effect |
| --- | --- |
| `-y`, `--auto-approve` | run model-proposed shell commands without asking (opt-in) |
| `-h`, `--help` | show help and exit |

## Credentials

Two values are required:

| Variable | Meaning |
| --- | --- |
| `DS_SESSION_ID` | the `ds_session_id` cookie from chat.deepseek.com |
| `AUTHORIZATION_TOKEN` | the full `Bearer ...` value sent by the web app |

They are read from the environment first, then from a user-wide secrets file:

1. `$DS_SECRETS_FILE`
2. `$XDG_CONFIG_HOME/deepseek-web-cli/.env`
3. `~/.config/deepseek-web-cli/.env`

The file uses one `KEY=VALUE` per line; `#` comments, an `export ` prefix and
matching quotes are accepted. Real environment variables always win.

## Commands

| Input | Effect |
| --- | --- |
| `exit` / Ctrl+D | quit |
| `/new` | start a fresh conversation (the old one stays resumable) |
| `/id` | show the current conversation tip |
| `/goal <text>` | set a goal and let the model work until it reports completion |
| `/goal`, `/goal show`, `/goal status` | show the active goal |
| `/goal clear`, `none`, `off` | drop the active goal |
| `/sessions` | list conversations, newest first |
| `/sessions more` | load the next page |
| `/resume <n|id-prefix>` | continue a saved conversation (restores its goal) |
| `/label <text>` | rename the current conversation |
| `/forget <n|id-prefix>` | delete a conversation on the server |

Ctrl+C cancels the current reply, command, or retry backoff and returns to the
prompt; at the prompt it just clears the line.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `DS_THINKING` | `1` | request the reasoning output |
| `DS_SEARCH` | `1` | enable web search |
| `DS_SHELL_TIMEOUT_SECS` | `120` | default per-command timeout |
| `DS_SHELL_MAX_OUTPUT_BYTES` | `16384` | cap kept per output stream |
| `DS_HTTP_TIMEOUT_MS` | `180000` | overall HTTP request timeout |
| `DS_RETRY_BASE_MS` | `500` | first retry backoff step |
| `DS_RETRY_CAP_MS` | `60000` | backoff ceiling |
| `DS_MIN_GAP_MS` | `0` | minimum spacing between requests |
| `DS_GOALS_FILE` | XDG data dir | where per-conversation goals live |
| `DS_SECRETS_FILE` | XDG config dir | where credentials are read from |
| `DS_DEBUG_PENDING` | `0` | dump model output and extracted commands |
| `DS_DEBUG_SECRETS` | `0` | report which secrets file was loaded |
| `NO_COLOR` / `CLICOLOR_FORCE` | - | follow https://no-color.org |

## Shell commands

The model asks to run a command with a tagged block:

    <shell>command</shell>
    <shell timeout="5m">long running command</shell>

Commands run under `sh -c` with stdin at `/dev/null`. On timeout the child and
its whole process group are killed. Output is capped per stream (head and tail
are kept) and any tool-call tags inside it are defanged before being sent back
to the model. Interactive programs will see EOF rather than the terminal.

## Development

    cargo test
    cargo clippy --all-targets
    cargo fmt --check
