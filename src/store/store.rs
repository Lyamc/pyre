//! pyre's own model store: `<root>/<kind>/<org>/<name>/<rev>/` with a `pyre-model.toml` manifest.
//! Entries are either *stored* (files live in the entry dir: hardlinked / symlinked / copied /
//! moved in) or *registered* (manifest only, `path` points at a folder used in place).
use super::{detect, hfcache, walk, Kind};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

pub const MANIFEST: &str = "pyre-model.toml";

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FileEntry { pub path: String, pub size: u64, pub sha256: String, #[serde(default)] pub link: String }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub name: String,
    #[serde(default)] pub alias: Option<String>,
    pub kind: Kind,
    #[serde(default)] pub architecture: String,
    #[serde(default)] pub dtype: String,
    #[serde(default)] pub source_repo: Option<String>,
    #[serde(default)] pub source_revision: Option<String>,
    /// "hf-cache", "local", "pull"
    #[serde(default)] pub source: String,
    /// Registered-in-place folder; None = files live next to the manifest.
    #[serde(default)] pub path: Option<PathBuf>,
    /// link | copy | move | register
    #[serde(default)] pub import_mode: String,
    #[serde(default)] pub created: u64,
    #[serde(default)] pub files: Vec<FileEntry>,
}

#[derive(Debug, Clone)]
pub struct Entry { pub dir: PathBuf, pub manifest: Manifest }
impl Entry {
    pub fn model_dir(&self) -> PathBuf { self.manifest.path.clone().unwrap_or_else(|| self.dir.clone()) }
    pub fn size(&self) -> u64 { self.manifest.files.iter().map(|f| f.size).sum() }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode { Link, Copy, Move }

#[derive(Debug, Default, Deserialize)]
pub struct ConfigFile {
    pub models_dir: Option<PathBuf>,
    pub search_order: Option<Vec<String>>,
    pub offline: Option<bool>,
}

pub fn config_path() -> PathBuf {
    if let Some(p) = std::env::var_os("PYRE_CONFIG") { return PathBuf::from(p); }
    #[cfg(windows)]
    if let Some(p) = std::env::var_os("APPDATA") { return PathBuf::from(p).join("pyre").join("config.toml"); }
    let base = std::env::var_os("XDG_CONFIG_HOME").map(PathBuf::from).unwrap_or_else(|| hfcache::home().join(".config"));
    base.join("pyre").join("config.toml")
}

pub fn load_config() -> ConfigFile {
    std::fs::read_to_string(config_path()).ok().and_then(|s| toml::from_str(&s).ok()).unwrap_or_default()
}

/// `--models-dir` > `PYRE_MODELS_DIR` > config file `models_dir` > platform default.
pub fn default_root(cli: Option<&Path>, cfg: &ConfigFile) -> PathBuf {
    if let Some(p) = cli { return p.to_path_buf(); }
    if let Some(p) = std::env::var_os("PYRE_MODELS_DIR") { return PathBuf::from(p); }
    if let Some(p) = &cfg.models_dir { return p.clone(); }
    #[cfg(windows)]
    if let Some(p) = std::env::var_os("LOCALAPPDATA") { return PathBuf::from(p).join("pyre").join("models"); }
    let base = std::env::var_os("XDG_DATA_HOME").map(PathBuf::from).unwrap_or_else(|| hfcache::home().join(".local").join("share"));
    base.join("pyre").join("models")
}

pub struct Store { pub root: PathBuf }

fn now() -> u64 { std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0) }

fn sanitize(s: &str) -> String { s.chars().map(|c| if c.is_ascii_alphanumeric() || "-_.".contains(c) { c } else { '_' }).collect() }

impl Store {
    pub fn new(root: PathBuf) -> Self { Self { root } }

    pub fn list(&self) -> Vec<Entry> {
        let mut out = vec![];
        for f in walk(&self.root) {
            if f.file_name().map(|n| n == MANIFEST).unwrap_or(false) {
                if let Ok(m) = std::fs::read_to_string(&f).map_err(anyhow::Error::from).and_then(|s| Ok(toml::from_str::<Manifest>(&s)?)) {
                    out.push(Entry { dir: f.parent().unwrap().to_path_buf(), manifest: m });
                }
            }
        }
        out.sort_by(|a, b| a.manifest.name.cmp(&b.manifest.name));
        out
    }

    /// Find by alias, exact name (`org/name[@rev]`), or name without revision (newest).
    pub fn find(&self, name: &str) -> Option<Entry> {
        let all = self.list();
        if let Some(e) = all.iter().find(|e| e.manifest.alias.as_deref() == Some(name)) { return Some(e.clone()); }
        let (n, rev) = hfcache::split_rev(name);
        let mut c: Vec<&Entry> = all.iter().filter(|e| e.manifest.name == n && rev.map(|r| e.manifest.source_revision.as_deref() == Some(r) || e.dir.ends_with(r)).unwrap_or(true)).collect();
        c.sort_by_key(|e| e.manifest.created);
        c.last().map(|e| (*e).clone())
    }

    fn entry_dir(&self, kind: Kind, name: &str, rev: &str) -> PathBuf {
        let (org, n) = name.split_once('/').unwrap_or(("local", name));
        self.root.join(kind.as_str()).join(sanitize(org)).join(sanitize(n)).join(sanitize(rev))
    }

    fn write(&self, dir: &Path, m: &Manifest) -> Result<()> {
        std::fs::create_dir_all(dir)?;
        std::fs::write(dir.join(MANIFEST), toml::to_string_pretty(m)?)?;
        Ok(())
    }

    /// Import a HF-cache snapshot (or any folder) into the store with link/copy/move semantics.
    /// Returns the entry and human-readable notes (e.g. link fallbacks).
    pub fn import(&self, src: &Path, name: &str, repo: Option<&str>, rev: Option<&str>, kind: Option<Kind>, mode: Mode) -> Result<(Entry, Vec<String>)> {
        let (dk, arch) = detect(src);
        let kind = kind.unwrap_or(dk);
        let rev = rev.map(String::from).or_else(|| src.file_name().map(|s| s.to_string_lossy().to_string())).unwrap_or_else(|| "main".into());
        let dir = self.entry_dir(kind, name, &rev);
        if dir.join(MANIFEST).exists() { bail!("{} already exists in the store ({})", name, dir.display()); }
        std::fs::create_dir_all(&dir)?;
        let blobs = hfcache::blobs_of(src);
        let mut notes = vec![];
        let mut files = vec![];
        let mut warned = false;
        for f in walk(src) {
            let rel = f.strip_prefix(src)?.to_path_buf();
            if rel.as_os_str() == MANIFEST { continue; }
            let (real, _, blob) = hfcache::resolve_entry(&f, &blobs);
            let dst = dir.join(&rel);
            std::fs::create_dir_all(dst.parent().unwrap())?;
            let size = std::fs::metadata(&real)?.len();
            // HF blob names are the sha256 for LFS files; reuse instead of rehashing GBs.
            let sha = blob.as_ref().and_then(|b| b.file_name()).map(|n| n.to_string_lossy().to_string()).filter(|n| n.len() == 64).map(Ok).unwrap_or_else(|| sha256(&real))?;
            let link = match mode {
                Mode::Copy => { std::fs::copy(&real, &dst)?; "copy" }
                Mode::Move => {
                    if std::fs::rename(&real, &dst).is_err() { std::fs::copy(&real, &dst)?; std::fs::remove_file(&real)?; }
                    if f != real { let _ = std::fs::remove_file(&f); }
                    "move"
                }
                Mode::Link => match std::fs::hard_link(&real, &dst) {
                    Ok(()) => "hardlink",
                    Err(e) => {
                        if symlink_file(&real, &dst).is_ok() {
                            if !warned { notes.push(format!("hardlink failed ({e}); used symlinks instead")); warned = true; }
                            "symlink"
                        } else {
                            if !warned { notes.push(format!("hardlink failed ({e}) and symlinks are not permitted (different volume / no Developer Mode?); COPIED files instead, which uses extra disk space")); warned = true; }
                            std::fs::copy(&real, &dst)?; "copy"
                        }
                    }
                },
            };
            files.push(FileEntry { path: rel.to_string_lossy().replace('\\', "/"), size, sha256: sha, link: link.into() });
        }
        let dtype = std::fs::read_to_string(src.join("config.json")).ok().and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
            .and_then(|j| j.get("torch_dtype").and_then(|v| v.as_str()).map(String::from)).unwrap_or_default();
        let m = Manifest {
            name: name.into(), alias: None, kind, architecture: arch, dtype,
            source_repo: repo.map(String::from), source_revision: Some(rev.clone()),
            source: if repo.is_some() { "hf-cache".into() } else { "local".into() },
            path: None, import_mode: format!("{mode:?}").to_lowercase(), created: now(), files,
        };
        self.write(&dir, &m)?;
        Ok((Entry { dir, manifest: m }, notes))
    }

    /// Register a local folder in place (manifest only; nothing copied).
    pub fn register(&self, folder: &Path, name: Option<&str>, kind: Option<Kind>) -> Result<Entry> {
        let folder = std::fs::canonicalize(folder).with_context(|| format!("{}", folder.display()))?;
        let (dk, arch) = detect(&folder);
        let kind = kind.unwrap_or(dk);
        let default_name = format!("local/{}", folder.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "model".into()));
        let name = name.map(String::from).unwrap_or(default_name);
        let dir = self.entry_dir(kind, &name, "local");
        if dir.join(MANIFEST).exists() { bail!("{name} already registered"); }
        let files = walk(&folder).into_iter().filter_map(|f| Some(FileEntry {
            path: f.strip_prefix(&folder).ok()?.to_string_lossy().replace('\\', "/"), size: std::fs::metadata(&f).ok()?.len(), sha256: String::new(), link: "in-place".into() })).collect();
        let m = Manifest { name, alias: None, kind, architecture: arch, dtype: String::new(), source_repo: None, source_revision: None,
            source: "local".into(), path: Some(folder), import_mode: "register".into(), created: now(), files };
        self.write(&dir, &m)?;
        Ok(Entry { dir, manifest: m })
    }

    pub fn set_alias(&self, name: &str, alias: &str) -> Result<Entry> {
        if let Some(o) = self.find(alias) { if o.manifest.alias.as_deref() == Some(alias) && o.manifest.name != name { bail!("alias {alias} already used by {}", o.manifest.name); } }
        let mut e = self.find(name).with_context(|| format!("{name} not in store"))?;
        e.manifest.alias = Some(alias.into());
        self.write(&e.dir, &e.manifest)?;
        Ok(e)
    }

    /// Remove a store entry. Never follows links into the HF cache: removing a hardlink or
    /// symlink only drops the store's reference; blob data stays in the cache.
    pub fn remove(&self, e: &Entry) -> Result<()> {
        if !e.dir.starts_with(&self.root) { bail!("refusing to delete outside the store: {}", e.dir.display()); }
        std::fs::remove_dir_all(&e.dir)?;
        // prune empty parents
        let mut p = e.dir.parent();
        while let Some(d) = p { if d == self.root || std::fs::remove_dir(d).is_err() { break; } p = d.parent(); }
        Ok(())
    }
}

#[cfg(windows)]
pub fn symlink_file(src: &Path, dst: &Path) -> std::io::Result<()> { std::os::windows::fs::symlink_file(src, dst) }
#[cfg(unix)]
pub fn symlink_file(src: &Path, dst: &Path) -> std::io::Result<()> { std::os::unix::fs::symlink(src, dst) }

pub fn sha256(p: &Path) -> Result<String> {
    let mut f = std::fs::File::open(p)?;
    let mut h = Sha256::new();
    std::io::copy(&mut f, &mut h)?;
    Ok(format!("{:x}", h.finalize()))
}
