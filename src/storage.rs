//! Wires CLI/config/env storage options into a Store + Resolver.
use crate::store::{hfcache, resolve::{self, Resolver}, store::{self, Store}};
use crate::StorageArgs;
use anyhow::Result;
use std::path::PathBuf;

pub struct Ctx { pub store: Store, pub hf_cache: PathBuf, pub offline: bool, pub order: Vec<resolve::Source> }

impl Ctx {
    pub fn new(a: &StorageArgs) -> Result<Self> {
        let cfg = store::load_config();
        let order = match a.search_order.as_ref().or(cfg.search_order.as_ref()) {
            Some(o) => resolve::parse_order(o)?,
            None => resolve::DEFAULT_ORDER.to_vec(),
        };
        Ok(Self {
            store: Store::new(store::default_root(a.models_dir.as_deref(), &cfg)),
            hf_cache: a.hf_cache.clone().unwrap_or_else(hfcache::cache_dir),
            offline: a.offline || cfg.offline.unwrap_or(false) || hfcache::offline(),
            order,
        })
    }
    pub fn resolver(&self) -> Resolver<'_> {
        Resolver { store: &self.store, hf_cache: self.hf_cache.clone(), order: self.order.clone(), offline: self.offline }
    }
}
