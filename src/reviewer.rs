//! Independent review of `/goal` completion claims.
//!
//! The worker model decides on its own when a goal is done, and a model that
//! wants to stop can simply say so: `[GOAL COMPLETE]` is a claim, not a
//! result.  `/goal` therefore hands every claim to a second, separate
//! conversation - the reviewer - which is told to check the workspace for
//! evidence (it has the same shell access) and must end with an explicit
//! verdict.  Only `VERDICT: ACHIEVED` closes the goal; anything else sends the
//! reviewer's findings back to the worker as its next instruction.

/// Verdict line that accepts a completion claim.
pub const VERDICT_ACHIEVED: &str = "VERDICT: ACHIEVED";

/// Verdict line that rejects a completion claim.
pub const VERDICT_NOT_ACHIEVED: &str = "VERDICT: NOT ACHIEVED";

/// Reminder appended to the reviewer's shell-output follow-up turns.
///
/// A review runs in a conversation with no system prompt, and follow-up turns
/// carry only `[SHELL OUTPUT]...`. Without this, a reviewer that ran a command
/// could answer with findings and no verdict line, and the claim would stay
/// unverified forever.
pub const VERDICT_REMINDER: &str = "Reminder: you are the reviewer. When you have \
checked enough, finish with exactly one line: VERDICT: ACHIEVED if the goal is \
genuinely achieved, or VERDICT: NOT ACHIEVED if it is not.";

/// How much of the worker's final report is quoted back to the reviewer.
/// Reports can be long; only the claim itself needs to be checked.
const MAX_REPORT_CHARS: usize = 8_000;

/// Keep the tail of `s`: a report's conclusion is at the end.
fn tail(s: &str, max: usize) -> &str {
    match s.char_indices().nth_back(max - 1) {
        Some((i, _)) => &s[i..],
        None => s,
    }
}

/// The prompt that turns a fresh conversation into a reviewer for `goal`.
///
/// The reviewer is given the worker's final report as *the claim to check*,
/// not as evidence, and is told how to end its reply so the verdict can be
/// parsed.
pub fn review_prompt(goal: &str, report: &str) -> String {
    let report = tail(report.trim(), MAX_REPORT_CHARS);
    format!(
        "You are a strict, independent reviewer. Another agent was asked to achieve this goal:\n\
         \n\
         Goal: {goal}\n\
         \n\
         You have a local shell, exactly like the worker had: you can run commands\n\
         yourself and read their output, and you are expected to do so rather than\n\
         judging the report on its own.\n\
         \n\
         {SHELL_INSTRUCTIONS}\n\
         \n\
         It now claims the goal is complete. Its final report:\n\
         \n\
         ---\n\
         {report}\n\
         ---\n\
         \n\
         Verify the claim against reality, not against the report. Read the changed\n\
         files, run the tests, check the artifacts. The goal text is the standard; a\n\
         claim without evidence is not an achievement, and partial work is not an\n\
         achievement.\n\
         \n\
         Command format reminder: emit commands only as\n\
         <shell>command</shell> blocks. Do not use DSML, JSON tool calls,\n\
         or any other tag - those are not executed here, and a review\n\
         that runs nothing must say so rather than guess.\n\
         \n\
         Be concise. Say what you checked and what is missing, then finish your reply with\n\
         exactly one line, on its own:\n\
         \n\
         {VERDICT_ACHIEVED}\n\
         \n\
         if the goal has genuinely been achieved, or\n\
         \n\
         {VERDICT_NOT_ACHIEVED}\n\
         \n\
         if it has not.",
        SHELL_INSTRUCTIONS = crate::SHELL_INSTRUCTIONS
    )
}

/// The message that sends a rejection back to the worker as its next task.
pub fn feedback_prompt(goal: &str, review: &str) -> String {
    let review = tail(review.trim(), MAX_REPORT_CHARS);
    format!(
        "An independent reviewer checked your claim that the goal below is complete and did not\n\
         accept it.\n\
         \n\
         Goal: {goal}\n\
         \n\
         Reviewer's verdict and findings:\n\
         \n\
         ---\n\
         {review}\n\
         ---\n\
         \n\
         Continue working toward the goal: address the reviewer's findings, using the shell as\n\
         needed. When the goal is fully accomplished, include the token [GOAL COMPLETE] in your\n\
         visible reply."
    )
}

/// Extract the reviewer's verdict from its reply.
///
/// `Some(true)` means the goal was confirmed achieved, `Some(false)` means it
/// was rejected, and `None` means the reply carried no readable verdict (which
/// callers must treat as a rejection: an unverified claim does not close a
/// goal).  The *last* verdict in the reply wins, so a reviewer that restates
/// an earlier draft cannot flip its conclusion.
pub fn parse_verdict(text: &str) -> Option<bool> {
    let mut found = None;
    for line in text.lines() {
        let clean = line.replace(['*', '`', '_', '#'], " ").to_ascii_lowercase();
        let Some(idx) = clean.find("verdict") else {
            continue;
        };
        let rest = &clean[idx + "verdict".len()..];
        let Some(colon) = rest.find(':') else {
            continue;
        };
        let verdict = rest[colon + 1..].trim();
        if verdict.contains("not achieved")
            || verdict.contains("not-achieved")
            || verdict.contains("unachieved")
        {
            found = Some(false);
        } else if verdict.contains("achieved") {
            found = Some(true);
        }
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn review_prompt_names_the_goal_and_demands_a_verdict() {
        let p = review_prompt("fix the bug", "I fixed it [GOAL COMPLETE]");
        assert!(p.contains("Goal: fix the bug"), "{p}");
        assert!(p.contains("strict, independent reviewer"), "{p}");
        assert!(p.contains(VERDICT_ACHIEVED), "{p}");
        assert!(p.contains(VERDICT_NOT_ACHIEVED), "{p}");
        assert!(p.contains("I fixed it"), "{p}");
    }

    #[test]
    fn review_prompt_tells_the_reviewer_how_to_run_shell_commands() {
        // The reviewer inspects the workspace with the same shell the worker
        // uses; without the command format it cannot act on "inspect the
        // workspace" and can only judge the report.
        let p = review_prompt("fix the bug", "I fixed it [GOAL COMPLETE]");
        assert!(p.contains("You have a local shell"), "{p}");
        assert!(p.contains(crate::SHELL_INSTRUCTIONS), "{p}");
        assert!(p.contains("<shell>"), "{p}");
        assert!(p.contains("ONLY way to run a command"), "{p}");
        assert!(p.contains("nothing runs"), "{p}");
        // A worked example is included, not just a placeholder.
        assert!(p.contains("git log --oneline -3"), "{p}");
        // A final reminder, near the verdict request, keeps the model on
        // the documented form at the point it actually acts.
        assert!(p.contains("Command format reminder"), "{p}");
        assert!(p.contains("[SHELL OUTPUT]"), "{p}");
    }

    #[test]
    fn review_prompt_truncates_huge_reports_but_keeps_the_tail() {
        let long = format!("{}{}", "x".repeat(20_000), "the end");
        let p = review_prompt("g", &long);
        assert!(p.contains("the end"), "tail must survive");
        assert!(!p.contains(&"x".repeat(MAX_REPORT_CHARS)), "head dropped");
    }

    #[test]
    fn feedback_prompt_restates_the_goal_and_the_reminder() {
        let p = feedback_prompt("fix the bug", "VERDICT: NOT ACHIEVED - tests fail");
        assert!(p.contains("Goal: fix the bug"), "{p}");
        assert!(p.contains("tests fail"), "{p}");
        assert!(p.contains("[GOAL COMPLETE]"), "{p}");
    }

    #[test]
    fn verdict_reminder_names_both_verdicts() {
        // Appended to the reviewer's shell follow-ups, which carry no system
        // prompt; without both lines a reply could be unreadable.
        assert!(
            VERDICT_REMINDER.contains(VERDICT_ACHIEVED),
            "{VERDICT_REMINDER}"
        );
        assert!(
            VERDICT_REMINDER.contains("VERDICT: NOT ACHIEVED"),
            "{VERDICT_REMINDER}"
        );
    }

    #[test]
    fn parses_both_verdicts() {
        assert_eq!(parse_verdict("stuff\nVERDICT: ACHIEVED"), Some(true));
        assert_eq!(
            parse_verdict("stuff\nVERDICT: NOT ACHIEVED\nbecause tests fail"),
            Some(false)
        );
        assert_eq!(parse_verdict("`VERDICT: ACHIEVED`"), Some(true));
        assert_eq!(parse_verdict("**Verdict:** not achieved"), Some(false));
    }

    #[test]
    fn last_verdict_wins() {
        assert_eq!(
            parse_verdict("VERDICT: NOT ACHIEVED\non reflection\nVERDICT: ACHIEVED"),
            Some(true)
        );
        assert_eq!(
            parse_verdict("VERDICT: ACHIEVED\nbut no\nVERDICT: NOT ACHIEVED"),
            Some(false)
        );
    }

    #[test]
    fn missing_verdict_is_none() {
        assert_eq!(parse_verdict("looks fine to me"), None);
        assert_eq!(parse_verdict(""), None);
        // "incomplete" must not be read as a verdict.
        assert_eq!(parse_verdict("the work is incomplete"), None);
    }
}
