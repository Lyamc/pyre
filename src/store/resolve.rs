//! Model reference resolution. Default order: explicit path -> pyre store (alias/name) ->
//! HF cache -> download (unless offline). Configurable via `--search-order` / config file.
use super::{hfcache, store::Store};
use anyhow::{bail, Result};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source { Path, Store, HfCache, Download }

pub fn parse_order(s: &[String]) -> Result<Vec<Source>> {
    s.iter().map(|x| Ok(match x.trim() {
        "path" => Source::Path, "store" => Source::Store, "hf" | "hf-cache" => Source::HfCache, "download" | "pull" => Source::Download,
        o => bail!("unknown search source {o} (path|store|hf|download)"),
    })).collect()
}

pub const DEFAULT_ORDER: [Source; 4] = [Source::Path, Source::Store, Source::HfCache, Source::Download];

pub struct Resolver<'a> { pub store: &'a Store, pub hf_cache: PathBuf, pub order: Vec<Source>, pub offline: bool }

impl Resolver<'_> {
    pub fn resolve(&self, r: &str) -> Result<(PathBuf, Source)> {
        let (repo, rev) = hfcache::split_rev(r);
        for s in &self.order {
            match s {
                Source::Path => { let p = Path::new(r); if p.is_dir() { return Ok((p.to_path_buf(), *s)); } }
                Source::Store => { if let Some(e) = self.store.find(r) { return Ok((e.model_dir(), *s)); } }
                Source::HfCache => { if repo.contains('/') { if let Some(p) = hfcache::find(&self.hf_cache, repo, rev) { return Ok((p, *s)); } } }
                Source::Download => {
                    if self.offline || hfcache::offline() { continue; }
                    if repo.contains('/') {
                        let p = hfcache::download(&self.hf_cache, repo, rev, &|f, i, n| tracing::info!("[{}/{}] {}", i + 1, n, f))?;
                        return Ok((p, *s));
                    }
                }
            }
        }
        bail!("could not resolve model '{r}' (searched {:?}{})", self.order, if self.offline || hfcache::offline() { "; offline mode" } else { "" })
    }
}
