//! Persistent voice identity library for cloning-capable TTS.
//!
//! Voices live on disk under `{models_root}/audio_voices/` (override with
//! `PYRE_AUDIO_VOICES_DIR` or `C:\Temp\qnx\audio_voices` when set via env).
//! Each voice is a folder: `meta.json` + `reference.wav` (mono, typically 16–24 kHz).
//!
//! Synthesis with a cloned voice requires the Burn Pocket TTS engine (in progress).
//! Until that engine is loaded, creating/listing/deleting voices works; speaking with
//! `voice_id` other than the built-in MMS voice returns a clear error.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceMeta {
    pub id: String,
    pub name: String,
    pub created_ms: u64,
    pub duration_secs: f64,
    pub sample_rate: u32,
    pub source: String,
    /// Optional notes / how the sample was captured.
    #[serde(default)]
    pub notes: String,
    /// Engine that can synthesize with this identity once ready.
    #[serde(default = "default_engine")]
    pub engine: String,
}

fn default_engine() -> String { "pocket-tts".into() }

#[derive(Debug, Clone, Serialize)]
pub struct VoiceInfo {
    #[serde(flatten)]
    pub meta: VoiceMeta,
    pub has_reference: bool,
    pub reference_bytes: u64,
    /// True when a Burn cloning engine is loaded and can speak as this voice.
    pub synthesizable: bool,
}

pub struct VoiceLib {
    root: PathBuf,
}

impl VoiceLib {
    pub fn open(root: PathBuf) -> Result<Self> {
        fs::create_dir_all(&root).with_context(|| format!("create voice lib {}", root.display()))?;
        Ok(Self { root })
    }

    pub fn root(&self) -> &Path { &self.root }

    pub fn list(&self, synthesizable: bool) -> Result<Vec<VoiceInfo>> {
        let mut out = Vec::new();
        if !self.root.exists() { return Ok(out); }
        let mut dirs: Vec<_> = fs::read_dir(&self.root)?.filter_map(|e| e.ok()).collect();
        dirs.sort_by_key(|e| e.file_name());
        for e in dirs {
            if !e.file_type().map(|t| t.is_dir()).unwrap_or(false) { continue; }
            let id = e.file_name().to_string_lossy().into_owned();
            if let Ok(info) = self.info(&id, synthesizable) { out.push(info); }
        }
        Ok(out)
    }

    pub fn info(&self, id: &str, synthesizable: bool) -> Result<VoiceInfo> {
        let dir = self.dir(id)?;
        let meta: VoiceMeta = serde_json::from_str(&fs::read_to_string(dir.join("meta.json"))?)?;
        let ref_path = dir.join("reference.wav");
        let reference_bytes = fs::metadata(&ref_path).map(|m| m.len()).unwrap_or(0);
        Ok(VoiceInfo {
            meta,
            has_reference: ref_path.is_file(),
            reference_bytes,
            synthesizable,
        })
    }

    pub fn reference_path(&self, id: &str) -> Result<PathBuf> {
        let p = self.dir(id)?.join("reference.wav");
        if !p.is_file() { bail!("voice '{id}' has no reference.wav"); }
        Ok(p)
    }

    pub fn create(&self, name: &str, wav_bytes: &[u8], source: &str, notes: &str) -> Result<VoiceInfo> {
        let name = name.trim();
        if name.is_empty() { bail!("voice name is required"); }
        if wav_bytes.len() < 44 { bail!("audio too short"); }
        // Basic WAV header check; full decode happens at clone/synth time.
        if &wav_bytes[..4] != b"RIFF" || &wav_bytes[8..12] != b"WAVE" {
            bail!("reference audio must be WAV (convert uploads server-side before create)");
        }
        let (sr, dur) = wav_rate_duration(wav_bytes)?;
        if !(dur >= 2.0 && dur <= 45.0) {
            bail!("reference should be about 2–45 s (aim for 5–30) of speech (got {dur:.1}s)");
        }
        let id = slug(name);
        let dir = self.root.join(&id);
        if dir.exists() { bail!("a voice named '{name}' (id '{id}') already exists"); }
        fs::create_dir_all(&dir)?;
        fs::write(dir.join("reference.wav"), wav_bytes)?;
        let meta = VoiceMeta {
            id: id.clone(),
            name: name.into(),
            created_ms: now_ms(),
            duration_secs: dur,
            sample_rate: sr,
            source: source.into(),
            notes: notes.into(),
            engine: default_engine(),
        };
        fs::write(dir.join("meta.json"), serde_json::to_vec_pretty(&meta)?)?;
        self.info(&id, false)
    }

    pub fn delete(&self, id: &str) -> Result<()> {
        let dir = self.dir(id)?;
        fs::remove_dir_all(dir)?;
        Ok(())
    }

    pub fn export_zip_bytes(&self, id: &str) -> Result<Vec<u8>> {
        // Cheap "export": just the wav + meta as a simple tar-less pair isn't zip without a crate.
        // Return the reference wav; meta is available via GET. Full zip can wait for a zip crate.
        fs::read(self.reference_path(id)?).map_err(Into::into)
    }

    pub fn import_wav(&self, name: &str, wav_bytes: &[u8], notes: &str) -> Result<VoiceInfo> {
        self.create(name, wav_bytes, "import", notes)
    }

    fn dir(&self, id: &str) -> Result<PathBuf> {
        let id = id.trim();
        if id.is_empty() || id.contains(['/', '\\']) || id.contains("..") || id == "." {
            bail!("invalid voice id");
        }
        let dir = self.root.join(id);
        if !dir.is_dir() { bail!("voice '{id}' not found"); }
        Ok(dir)
    }
}

fn slug(name: &str) -> String {
    let mut s: String = name.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { '-' })
        .collect();
    while s.contains("--") { s = s.replace("--", "-"); }
    let s = s.trim_matches('-').to_string();
    if s.is_empty() { format!("voice-{}", now_ms()) } else { s }
}

fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

fn wav_rate_duration(bytes: &[u8]) -> Result<(u32, f64)> {
    if bytes.len() < 44 { bail!("wav too short"); }
    let channels = u16::from_le_bytes([bytes[22], bytes[23]]) as u32;
    let sr = u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]);
    let bits = u16::from_le_bytes([bytes[34], bytes[35]]) as u32;
    // Find data chunk
    let mut i = 12usize;
    let mut data_len = 0u32;
    while i + 8 <= bytes.len() {
        let id = &bytes[i..i+4];
        let sz = u32::from_le_bytes([bytes[i+4], bytes[i+5], bytes[i+6], bytes[i+7]]);
        if id == b"data" { data_len = sz; break; }
        i += 8 + sz as usize;
        if sz % 2 == 1 { i += 1; }
    }
    if sr == 0 || channels == 0 || bits == 0 { bail!("bad wav header"); }
    let bytes_per_sec = sr.saturating_mul(channels).saturating_mul(bits / 8);
    if bytes_per_sec == 0 { bail!("bad wav format"); }
    let dur = data_len as f64 / bytes_per_sec as f64;
    Ok((sr, dur))
}

/// Process-wide library, opened lazily from env / default path.
pub fn global() -> Result<std::sync::MutexGuard<'static, VoiceLib>> {
    static LIB: std::sync::OnceLock<Mutex<VoiceLib>> = std::sync::OnceLock::new();
    let m = LIB.get_or_init(|| {
        let root = std::env::var_os("PYRE_AUDIO_VOICES_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                if Path::new(r"C:\Temp\qnx").is_dir() {
                    PathBuf::from(r"C:\Temp\qnx\audio_voices")
                } else {
                    dirs_fallback()
                }
            });
        Mutex::new(VoiceLib::open(root).expect("open voice library"))
    });
    Ok(m.lock().map_err(|_| anyhow::anyhow!("voice library lock poisoned"))?)
}

fn dirs_fallback() -> PathBuf {
    if let Ok(p) = std::env::var("LOCALAPPDATA") {
        return PathBuf::from(p).join("pyre").join("audio_voices");
    }
    PathBuf::from("audio_voices")
}

/// Built-in MMS voice entry for the picker (not a cloned identity).
pub fn builtin_mms() -> serde_json::Value {
    serde_json::json!({
        "id": "mms",
        "name": "MMS English (built-in)",
        "builtin": true,
        "engine": "mms-vits",
        "synthesizable": true,
        "has_reference": false,
        "notes": "facebook/mms-tts-eng Burn VITS. No cloning; supports SSML breaks/prosody."
    })
}
