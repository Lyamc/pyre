use super::hfcache::{self, find, list, repo_folder};
use super::resolve::{Resolver, Source};
use super::store::{symlink_file, Mode, Store};
use super::Kind;
use std::fs;
use std::path::{Path, PathBuf};

const COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";
const SHA: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

/// Build `models--org--tiny` with blobs/refs/snapshots. `symlinks`: snapshot entries are
/// symlinks into blobs (returns false if the OS refused); otherwise plain copies.
fn fake_cache(root: &Path, symlinks: bool) -> bool {
    let repo = root.join(repo_folder("org/tiny"));
    fs::create_dir_all(repo.join("blobs")).unwrap();
    fs::create_dir_all(repo.join("refs")).unwrap();
    let snap = repo.join("snapshots").join(COMMIT);
    fs::create_dir_all(&snap).unwrap();
    fs::write(repo.join("refs").join("main"), COMMIT).unwrap();
    let files = [("config.json", "cfgblob", br#"{"architectures":["LlamaForCausalLM"],"model_type":"llama","num_hidden_layers":1,"torch_dtype":"bfloat16"}"#.to_vec()),
                 ("model.safetensors", SHA, vec![7u8; 4096])];
    for (name, blob, data) in files {
        fs::write(repo.join("blobs").join(blob), &data).unwrap();
        if symlinks {
            if symlink_file(&repo.join("blobs").join(blob), &snap.join(name)).is_err() { return false; }
        } else {
            fs::write(snap.join(name), &data).unwrap();
        }
    }
    true
}

fn tmp() -> tempfile::TempDir { tempfile::tempdir().unwrap() }

#[test]
fn finds_snapshot_via_refs_copy_mode() {
    let t = tmp();
    fake_cache(t.path(), false);
    let p = find(t.path(), "org/tiny", None).unwrap();
    assert!(p.ends_with(COMMIT));
    assert_eq!(find(t.path(), "org/tiny", Some(COMMIT)).unwrap(), p);
    assert!(find(t.path(), "org/other", None).is_none());
    let l = list(t.path());
    assert_eq!(l.len(), 1);
    assert_eq!(l[0].repo, "org/tiny");
    assert_eq!(l[0].mode, "copy");
    assert_eq!(l[0].refs, vec!["main".to_string()]);
}

#[test]
fn symlink_mode_resolves_to_blobs() {
    let t = tmp();
    if !fake_cache(t.path(), true) { eprintln!("symlinks not permitted here; skipping"); return; }
    let l = list(t.path());
    assert_eq!(l[0].mode, "symlink");
    assert!(l[0].size >= 4096);
    assert_eq!(super::detect(&l[0].path).0, Kind::Llm);
}

#[test]
fn dedupe_turns_copies_into_hardlinks() {
    let t = tmp();
    fake_cache(t.path(), false);
    let snap = find(t.path(), "org/tiny", None).unwrap();
    hfcache::dedupe_snapshot(&snap);
    let blob = t.path().join(repo_folder("org/tiny")).join("blobs").join(SHA);
    assert!(same_file::is_same_file(snap.join("model.safetensors"), &blob).unwrap());
    assert_eq!(list(t.path())[0].mode, "hardlink");
}

fn import(mode: Mode, symlinks: bool) -> Option<(tempfile::TempDir, PathBuf, PathBuf)> {
    let t = tmp();
    let cache = t.path().join("hf");
    if !fake_cache(&cache, symlinks) { return None; }
    let store = Store::new(t.path().join("store"));
    let snap = find(&cache, "org/tiny", None).unwrap();
    let (e, notes) = store.import(&snap, "org/tiny", Some("org/tiny"), Some(COMMIT), None, mode).unwrap();
    eprintln!("notes: {notes:?}");
    assert_eq!(e.manifest.kind, Kind::Llm);
    assert!(e.dir.starts_with(t.path().join("store").join("llm").join("org").join("tiny")));
    assert!(e.dir.join("pyre-model.toml").exists());
    let blob = cache.join(repo_folder("org/tiny")).join("blobs").join(SHA);
    let st = e.dir.join("model.safetensors");
    Some((t, blob, st))
}

#[test]
fn link_import_uses_no_extra_space() {
    for symlinks in [false, true] {
        let Some((_t, blob, st)) = import(Mode::Link, symlinks) else { continue };
        // copy-mode snapshot is its own file; link must share storage with the snapshot entry,
        // and for symlink caches with the blob itself.
        if symlinks { assert!(same_file::is_same_file(&st, &blob).unwrap(), "store file must be the same file as the blob"); }
        assert_eq!(fs::read(&st).unwrap(), vec![7u8; 4096]);
    }
}

#[test]
fn link_import_same_file_as_snapshot() {
    let t = tmp();
    let cache = t.path().join("hf");
    fake_cache(&cache, false);
    let snap = find(&cache, "org/tiny", None).unwrap();
    let store = Store::new(t.path().join("store"));
    let (e, _) = store.import(&snap, "org/tiny", Some("org/tiny"), None, None, Mode::Link).unwrap();
    assert!(same_file::is_same_file(e.dir.join("model.safetensors"), snap.join("model.safetensors")).unwrap());
    // manifest reused the blob sha only when symlinked; copy-mode hashes the file
    assert_eq!(e.manifest.files.iter().find(|f| f.path == "model.safetensors").unwrap().sha256.len(), 64);
}

#[test]
fn copy_import_is_independent() {
    let (_t, blob, st) = import(Mode::Copy, false).unwrap();
    assert!(!same_file::is_same_file(&st, &blob).unwrap());
    assert_eq!(fs::read(&st).unwrap().len(), 4096);
}

#[test]
fn move_import_removes_source() {
    let t = tmp();
    let cache = t.path().join("hf");
    fake_cache(&cache, false);
    let snap = find(&cache, "org/tiny", None).unwrap();
    let store = Store::new(t.path().join("store"));
    let (e, _) = store.import(&snap, "org/tiny", Some("org/tiny"), None, None, Mode::Move).unwrap();
    assert!(!snap.join("model.safetensors").exists());
    assert!(e.dir.join("model.safetensors").exists());
}

#[test]
fn register_alias_find_remove_and_resolve_order() {
    let t = tmp();
    let folder = t.path().join("mymodel");
    fs::create_dir_all(&folder).unwrap();
    fs::write(folder.join("config.json"), r#"{"model_type":"whisper"}"#).unwrap();
    let store = Store::new(t.path().join("store"));
    let e = store.register(&folder, None, None).unwrap();
    assert_eq!(e.manifest.kind, Kind::Stt);
    assert_eq!(e.manifest.name, "local/mymodel");
    store.set_alias("local/mymodel", "asr").unwrap();
    assert_eq!(store.find("asr").unwrap().model_dir(), fs::canonicalize(&folder).unwrap());

    // resolution: store alias wins; HF cache found without download; offline never downloads
    let cache = t.path().join("hf");
    fake_cache(&cache, false);
    let r = Resolver { store: &store, hf_cache: cache.clone(), order: super::resolve::DEFAULT_ORDER.to_vec(), offline: true };
    assert_eq!(r.resolve("asr").unwrap().1, Source::Store);
    assert_eq!(r.resolve("org/tiny").unwrap().1, Source::HfCache);
    assert_eq!(r.resolve(folder.to_str().unwrap()).unwrap().1, Source::Path);
    assert!(r.resolve("org/missing").is_err());

    store.remove(&store.find("asr").unwrap()).unwrap();
    assert!(store.find("asr").is_none());
    assert!(folder.join("config.json").exists(), "registered folder must never be deleted");
}
