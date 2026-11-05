//! ANSI color helpers for the CLI.
//!
//! Colors are enabled only when stdout is a terminal and `NO_COLOR` is unset
//! (<https://no-color.org>). Set `CLICOLOR_FORCE=1` to force colors even when
//! stdout is piped (useful for testing).

use std::ffi::OsStr;
use std::io::IsTerminal;
use std::sync::OnceLock;

fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        if std::env::var_os("NO_COLOR").is_some() {
            return false;
        }
        if let Some(v) = std::env::var_os("CLICOLOR_FORCE") {
            if v != OsStr::new("0") {
                return true;
            }
        }
        std::io::stdout().is_terminal()
    })
}

/// Wrap `s` in the given SGR `code`, resetting afterwards.
pub fn paint(code: &str, s: &str) -> String {
    if enabled() {
        format!("\x1b[{code}m{s}\x1b[0m")
    } else {
        s.to_string()
    }
}

macro_rules! color_fns {
    ($( $name:ident => $code:literal ),* $(,)?) => {
        $(
            #[allow(dead_code)]
            pub fn $name() -> &'static str {
                if enabled() { $code } else { "" }
            }
        )*
    };
}

color_fns! {
    reset          => "\x1b[0m",
    bold           => "\x1b[1m",
    dim            => "\x1b[2m",
    italic         => "\x1b[3m",
    red            => "\x1b[31m",
    green          => "\x1b[32m",
    yellow         => "\x1b[33m",
    blue           => "\x1b[34m",
    magenta        => "\x1b[35m",
    cyan           => "\x1b[36m",
    gray           => "\x1b[90m",
    bright_red     => "\x1b[91m",
    bright_green   => "\x1b[92m",
    bright_yellow  => "\x1b[93m",
    bright_blue    => "\x1b[94m",
    bright_magenta => "\x1b[95m",
    bright_cyan    => "\x1b[96m",
}
