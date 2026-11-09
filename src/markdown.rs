//! Minimal streaming Markdown renderer for assistant replies.
//!
//! The API streams plain text, so rendering happens line by line: text is
//! buffered until a newline arrives, the completed line is styled with ANSI
//! escapes (via [`crate::color::paint`], which is a no-op when colors are
//! disabled), and printed. Whatever is left over at end of stream is flushed
//! by [`MarkdownStream::finish`].
//!
//! The styles are deliberately small: headings, bullet/ordered lists,
//! blockquotes, horizontal rules, fenced code blocks and inline
//! bold/italic/strikethrough/code/links. Anything unrecognised is passed
//! through unchanged, so this never loses text.

use crate::color::paint;

/// Streaming renderer state: the current partial line plus whether we are
/// inside a fenced code block.
pub struct MarkdownStream {
    line: String,
    in_code: bool,
}

impl MarkdownStream {
    pub fn new() -> Self {
        Self {
            line: String::new(),
            in_code: false,
        }
    }

    /// Feed a chunk of visible text. Complete lines are rendered immediately.
    pub fn push(&mut self, s: &str) {
        for c in s.chars() {
            if c == '\n' {
                let line = std::mem::take(&mut self.line);
                let rendered = render_line(line.trim_end_matches('\r'), &mut self.in_code);
                out(&rendered);
                out("\n");
            } else {
                self.line.push(c);
            }
        }
    }

    /// Flush a trailing partial line (without adding a newline; the caller
    /// emits one after the stream ends).
    pub fn finish(&mut self) {
        if !self.line.is_empty() {
            let line = std::mem::take(&mut self.line);
            let rendered = render_line(line.trim_end_matches('\r'), &mut self.in_code);
            out(&rendered);
        }
    }
}

fn out(s: &str) {
    use std::io::Write;
    print!("{s}");
    std::io::stdout().flush().ok();
}

/// Render one complete line, returning the styled text (no trailing newline).
fn render_line(line: &str, in_code: &mut bool) -> String {
    let trimmed = line.trim_start();
    let indent = &line[..line.len() - trimmed.len()];

    // Fenced code block delimiters.
    if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
        if *in_code {
            *in_code = false;
            return paint("2", trimmed);
        }
        *in_code = true;
        return paint("2", trimmed);
    }
    if *in_code {
        return paint("36", line);
    }

    // Horizontal rule.
    if trimmed.len() >= 3
        && trimmed
            .chars()
            .all(|c| c == '-' || c == '*' || c == '_' || c == ' ')
        && trimmed.chars().filter(|&c| c != ' ').count() >= 3
    {
        return paint("2", &"─".repeat(40));
    }

    // ATX heading.
    let hashes = trimmed.chars().take_while(|&c| c == '#').count();
    if (1..=6).contains(&hashes) && (trimmed[hashes..].starts_with(' ') || trimmed.len() == hashes)
    {
        let text = trimmed[hashes..].trim();
        let code = match hashes {
            1 => "1;35",
            2 => "1;36",
            3 => "1;34",
            _ => "1",
        };
        return format!("{indent}{}", paint(code, &render_inline(text)));
    }

    // Blockquote.
    if let Some(rest) = trimmed.strip_prefix('>') {
        let rest = rest.strip_prefix(' ').unwrap_or(rest);
        return format!("{indent}{} {}", paint("2", "│"), render_inline(rest));
    }

    // Unordered list item.
    if let Some(rest) = trimmed
        .strip_prefix("- ")
        .or_else(|| trimmed.strip_prefix("* "))
        .or_else(|| trimmed.strip_prefix("+ "))
    {
        return format!("{indent}{} {}", paint("36", "•"), render_inline(rest));
    }

    // Ordered list item.
    let digits = trimmed.chars().take_while(|c| c.is_ascii_digit()).count();
    if digits > 0 && trimmed[digits..].starts_with(". ") {
        return format!(
            "{indent}{} {}",
            paint("36", &trimmed[..digits + 1]),
            render_inline(&trimmed[digits + 2..])
        );
    }

    format!("{indent}{}", render_inline(trimmed))
}

/// Render inline spans: `**bold**`, `*italic*`, `~~strike~~`, `` `code` ``,
/// and `[text](url)`. Unmatched markers are emitted literally.
fn render_inline(s: &str) -> String {
    let mut out = String::new();
    let mut i = 0;
    while i < s.len() {
        let rest = &s[i..];

        // Backslash escape.
        if let Some(after) = rest.strip_prefix('\\') {
            if let Some(c) = after.chars().next() {
                if c.is_ascii_punctuation() {
                    out.push(c);
                    i += 1 + c.len_utf8();
                    continue;
                }
            }
        }

        // Bold (** or __).
        if rest.starts_with("**") || rest.starts_with("__") {
            let marker = &rest[..2];
            if let Some(end) = rest[2..].find(marker) {
                let inner = &rest[2..2 + end];
                if !inner.trim().is_empty() && !inner.contains('\n') {
                    out.push_str(&paint("1", &render_inline(inner)));
                    i += 2 + end + 2;
                    continue;
                }
            }
        }

        // Strikethrough.
        if let Some(inner_rest) = rest.strip_prefix("~~") {
            if let Some(end) = inner_rest.find("~~") {
                let inner = &inner_rest[..end];
                if !inner.trim().is_empty() && !inner.contains('\n') {
                    out.push_str(&paint("2", &render_inline(inner)));
                    i += 2 + end + 2;
                    continue;
                }
            }
        }

        // Inline code.
        if let Some(inner_rest) = rest.strip_prefix('`') {
            if let Some(end) = inner_rest.find('`') {
                let inner = &inner_rest[..end];
                if !inner.is_empty() && !inner.contains('\n') {
                    out.push_str(&paint("36", inner));
                    i += 1 + end + 1;
                    continue;
                }
            }
        }

        // Italic (* or _). Underscores only count at a word boundary, so
        // snake_case identifiers are left alone.
        let italic = match rest.chars().next() {
            Some('*') => true,
            Some('_') => {
                i == 0
                    || !s[..i]
                        .chars()
                        .next_back()
                        .is_some_and(|c| c.is_alphanumeric())
            }
            _ => false,
        };
        if italic {
            let marker = &rest[..1];
            if let Some(end) = rest[1..].find(marker) {
                let inner = &rest[1..1 + end];
                if !inner.trim().is_empty()
                    && !inner.starts_with(' ')
                    && !inner.ends_with(' ')
                    && !inner.contains('\n')
                {
                    out.push_str(&paint("3", &render_inline(inner)));
                    i += 1 + end + 1;
                    continue;
                }
            }
        }

        // Link [text](url).
        if rest.starts_with('[') {
            if let Some(close) = rest.find("](") {
                if let Some(paren) = rest[close + 2..].find(')') {
                    let text = &rest[1..close];
                    let url = &rest[close + 2..close + 2 + paren];
                    out.push_str(&paint("4;34", text));
                    if url != text {
                        out.push_str(&paint("2", &format!(" ({url})")));
                    }
                    i += close + 2 + paren + 1;
                    continue;
                }
            }
        }

        let c = rest.chars().next().expect("non-empty rest");
        out.push(c);
        i += c.len_utf8();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inline_styles_strip_markers() {
        assert_eq!(render_inline("**bold**"), "bold");
        assert_eq!(render_inline("*it*"), "it");
        assert_eq!(render_inline("`code`"), "code");
        assert_eq!(render_inline("~~gone~~"), "gone");
        assert_eq!(
            render_inline("a [link](https://x.y) b"),
            "a link (https://x.y) b"
        );
        assert_eq!(render_inline("snake_case_name"), "snake_case_name");
        assert_eq!(render_inline("2 * 3 * 4"), "2 * 3 * 4");
    }

    #[test]
    fn line_styles() {
        let mut in_code = false;
        assert_eq!(render_line("# Title", &mut in_code), "Title");
        assert_eq!(render_line("- item", &mut in_code), "• item");
        assert_eq!(render_line("1. item", &mut in_code), "1. item");
        assert_eq!(render_line("> quote", &mut in_code), "│ quote");
        assert_eq!(render_line("plain", &mut in_code), "plain");
    }

    #[test]
    fn code_fence_round_trip() {
        let mut in_code = false;
        assert_eq!(render_line("```rust", &mut in_code), "```rust");
        assert!(in_code);
        assert_eq!(render_line("let x = 1;", &mut in_code), "let x = 1;");
        assert_eq!(render_line("```", &mut in_code), "```");
        assert!(!in_code);
    }
}
