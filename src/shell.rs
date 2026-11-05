use anyhow::Result;
use std::process::Stdio;
use tokio::process::Command;

pub async fn run_shell(cmd: &str) -> Result<String> {
    let output = Command::new("sh")
        .arg("-c")
        .arg(cmd)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await?;

    let mut result = String::new();
    if !output.stdout.is_empty() {
        result.push_str(&String::from_utf8_lossy(&output.stdout));
    }
    if !output.stderr.is_empty() {
        result.push_str("\n[stderr]\n");
        result.push_str(&String::from_utf8_lossy(&output.stderr));
    }
    Ok(result)
}

pub fn extract_shell_commands(text: &str) -> Vec<String> {
    let mut cmds = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find("<shell>") {
        if let Some(end) = rest[start..].find("</shell>") {
            let cmd = &rest[start + 7..start + end];
            cmds.push(cmd.trim().to_string());
            rest = &rest[start + end + 8..];
        } else {
            break;
        }
    }
    cmds
}
