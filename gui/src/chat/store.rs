//! Conversations and settings, persisted as JSON in the per-user app data dir.
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Clone, Serialize, Deserialize, Default)]
pub struct Msg {
    pub role: String,
    pub content: String,
    #[serde(default)] pub reasoning: String,
    /// pyre's `reasoning` object from the final chunk (effort, thinking_tokens, auto, ...).
    #[serde(default)] pub info: serde_json::Value,
    #[serde(default)] pub stats: String,
    #[serde(default)] pub error: String,
    #[serde(default)] pub effort: String,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Conv {
    pub id: u64,
    pub title: String,
    #[serde(default)] pub messages: Vec<Msg>,
    #[serde(default)] pub updated: u64,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Settings {
    pub server: String,
    #[serde(default)] pub system: String,
    pub temperature: String,
    pub max_tokens: String,
    pub effort: String,
}
impl Default for Settings {
    fn default() -> Self {
        Settings { server: "http://127.0.0.1:8099".into(), system: String::new(), temperature: "0.7".into(), max_tokens: "8192".into(), effort: "medium".into() }
    }
}

pub fn dir() -> PathBuf {
    if let Some(d) = std::env::var_os("PYRE_CHAT_DIR") { return PathBuf::from(d); }
    let home = || std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")).map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
    let base = if cfg!(windows) {
        std::env::var_os("APPDATA").map(PathBuf::from).unwrap_or_else(home)
    } else if cfg!(target_os = "macos") {
        home().join("Library/Application Support")
    } else {
        std::env::var_os("XDG_DATA_HOME").map(PathBuf::from).unwrap_or_else(|| home().join(".local/share"))
    };
    base.join("pyre-chat")
}

fn load_json<T: for<'a> Deserialize<'a>>(name: &str) -> Option<T> {
    serde_json::from_slice(&std::fs::read(dir().join(name)).ok()?).ok()
}
fn save_json<T: Serialize>(name: &str, v: &T) {
    let d = dir();
    let _ = std::fs::create_dir_all(&d);
    let tmp = d.join(format!("{name}.tmp"));
    if let Ok(b) = serde_json::to_vec_pretty(v) {
        if std::fs::write(&tmp, b).is_ok() { let _ = std::fs::rename(&tmp, d.join(name)); }
    }
}
pub fn load_convs() -> Vec<Conv> { load_json("conversations.json").unwrap_or_default() }
pub fn save_convs(c: &[Conv]) { save_json("conversations.json", &c) }
pub fn load_settings() -> Settings { load_json("settings.json").unwrap_or_default() }
pub fn save_settings(s: &Settings) { save_json("settings.json", s) }
pub fn now() -> u64 { std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0) }
