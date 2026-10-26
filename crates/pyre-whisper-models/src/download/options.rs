use std::sync::Arc;

pub type PrepProgressFn = Arc<dyn Fn(PrepProgress) + Send + Sync>;

#[derive(Clone)]
pub struct DownloadOptions {
    pub hf_token: Option<String>,
    pub verbose: bool,
    pub force: bool,
    pub progress: Option<PrepProgressFn>,
}

impl std::fmt::Debug for DownloadOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DownloadOptions")
            .field("hf_token", &self.hf_token.as_ref().map(|_| "***"))
            .field("verbose", &self.verbose)
            .field("force", &self.force)
            .field("progress", &self.progress.as_ref().map(|_| "set"))
            .finish()
    }
}

impl Default for DownloadOptions {
    fn default() -> Self {
        Self {
            hf_token: std::env::var("HF_TOKEN").ok(),
            verbose: false,
            force: false,
            progress: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PrepProgress {
    /// `download`, `convert`, or `load`
    pub stage: &'static str,
    pub label: String,
    /// 0.0–1.0 within this stage
    pub fraction: f64,
    pub bytes_done: Option<u64>,
    pub bytes_total: Option<u64>,
}

impl PrepProgress {
    pub fn download(label: impl Into<String>, fraction: f64, done: Option<u64>, total: Option<u64>) -> Self {
        Self {
            stage: "download",
            label: label.into(),
            fraction: fraction.clamp(0.0, 1.0),
            bytes_done: done,
            bytes_total: total,
        }
    }

    pub fn convert(label: impl Into<String>, fraction: f64) -> Self {
        Self {
            stage: "convert",
            label: label.into(),
            fraction: fraction.clamp(0.0, 1.0),
            bytes_done: None,
            bytes_total: None,
        }
    }

    pub fn load(label: impl Into<String>, fraction: f64) -> Self {
        Self {
            stage: "load",
            label: label.into(),
            fraction: fraction.clamp(0.0, 1.0),
            bytes_done: None,
            bytes_total: None,
        }
    }
}

impl DownloadOptions {
    pub fn report(&self, progress: PrepProgress) {
        match (progress.bytes_done, progress.bytes_total) {
            (Some(done), Some(total)) if total > 0 => tracing::info!(
                stage = progress.stage,
                "{} ({:.0}%) {} / {}",
                progress.label,
                progress.fraction * 100.0,
                format_bytes(done),
                format_bytes(total)
            ),
            (Some(done), _) => tracing::info!(
                stage = progress.stage,
                "{} {}",
                progress.label,
                format_bytes(done)
            ),
            _ => tracing::info!(
                stage = progress.stage,
                "{} ({:.0}%)",
                progress.label,
                progress.fraction * 100.0
            ),
        }
        if let Some(cb) = &self.progress {
            cb(progress);
        }
    }
}

pub fn format_bytes(n: u64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;
    const GB: f64 = MB * 1024.0;
    let n = n as f64;
    if n >= GB {
        format!("{:.2} GB", n / GB)
    } else if n >= MB {
        format!("{:.1} MB", n / MB)
    } else if n >= KB {
        format!("{:.0} KB", n / KB)
    } else {
        format!("{n:.0} B")
    }
}
