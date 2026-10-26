use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

pub fn script_path(name: &str) -> PathBuf {
    let cwd = PathBuf::from("scripts").join(name);
    if cwd.exists() {
        return cwd;
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../scripts").join(name)
}

pub fn run_python(args: &[String]) -> anyhow::Result<Output> {
    let mut invocations: Vec<(String, Vec<String>)> = Vec::new();
    if cfg!(windows) {
        if let Ok(out) = Command::new("where.exe").arg("python").output() {
            for line in String::from_utf8_lossy(&out.stdout).lines() {
                let p = line.trim();
                if p.ends_with(".exe") && !p.contains("WindowsApps") {
                    invocations.push((p.to_string(), vec![]));
                }
            }
        }
        invocations.push(("py".into(), vec!["-3".into()]));
    }
    invocations.push(("python3".into(), vec![]));
    invocations.push(("python".into(), vec![]));
    let mut last_fail: Option<Output> = None;
    for (prog, prefix) in invocations {
        let probe = Command::new(&prog)
            .args(&prefix)
            .args(["-c", "import numpy"])
            .output();
        match probe {
            Ok(out) if !out.status.success() => continue,
            Err(_) => continue,
            _ => {}
        }
        let mut cmd = Command::new(&prog);
        cmd.args(&prefix).args(args);
        match cmd.output() {
            Ok(out) if out.status.success() => return Ok(out),
            Ok(out) => last_fail = Some(out),
            Err(err) if err.kind() == ErrorKind::NotFound => {}
            Err(err) => return Err(err.into()),
        }
    }
    if let Some(out) = last_fail {
        let err = String::from_utf8_lossy(&out.stderr);
        anyhow::bail!("python failed: {err}");
    }
    anyhow::bail!("no Python interpreter found (tried python, py -3, python3)")
}

pub fn run_script(script: &Path, args: &[String]) -> anyhow::Result<()> {
    if !script.exists() {
        anyhow::bail!("missing script {}", script.display());
    }
    let mut all = vec![script.display().to_string()];
    all.extend(args.iter().cloned());
    let out = run_python(&all)?;
    if !out.status.success() {
        anyhow::bail!(
            "{} failed: {}",
            script.display(),
            String::from_utf8_lossy(&out.stderr)
        );
    }
    Ok(())
}
