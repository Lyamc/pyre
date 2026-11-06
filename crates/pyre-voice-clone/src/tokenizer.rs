//! GPT-2 byte-level BPE for Chatterbox-Nano / Turbo.
//!
//! This is the tokenizer `ChatterboxTurboTTS.generate` calls after `punc_norm`:
//! Hugging Face `GPT2Tokenizer`, `add_bos_token = false`, `add_prefix_space = false`,
//! vocab 50276. The 19 direction tags are added tokens and must stay one id.
//! `<|endoftext|>` is 50256 and is not prepended.
//!
//! Files, and only these files, are read from a directory or from an HF hub
//! snapshot (`models--ResembleAI--chatterbox-nano/snapshots/<rev>/`):
//! `vocab.json`, `merges.txt`, `added_tokens.json`, and optionally
//! `tokenizer_config.json`. Snapshot symlinks may resolve into the repo `blobs/`
//! directory. Anything else is refused. Nothing is downloaded.

use crate::directions::{self, DirectionKind, Script};
use anyhow::{anyhow, bail, Context, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use tokenizers::models::bpe::{BpeBuilder, Vocab, BPE};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::{AddedToken, Tokenizer};

pub const REPO_NANO: &str = "ResembleAI/chatterbox-nano";
pub const REPO_TURBO: &str = "ResembleAI/chatterbox-turbo";
pub const VOCAB_SIZE: usize = 50_276;
pub const EOS_ID: u32 = 50_256;

const REQUIRED: &[&str] = &["vocab.json", "merges.txt", "added_tokens.json"];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodedScript {
    pub script: Script,
    pub ids: Vec<u32>,
}

pub struct NanoTokenizer {
    tok: Tokenizer,
}

impl NanoTokenizer {
    pub fn load(dir: &Path) -> Result<Self> {
        if !dir.is_dir() {
            bail!("{} is not a directory", dir.display());
        }
        let vocab_path = safe_file(dir, "vocab.json")?;
        let merges_path = safe_file(dir, "merges.txt")?;
        let added_path = safe_file(dir, "added_tokens.json")?;
        let config_path = dir.join("tokenizer_config.json");
        let config_path = if config_path.exists() {
            Some(safe_file(dir, "tokenizer_config.json")?)
        } else {
            None
        };

        let added: HashMap<String, u32> = serde_json::from_reader(
            fs::File::open(&added_path).with_context(|| added_path.display().to_string())?,
        )
        .with_context(|| format!("parsing {}", added_path.display()))?;
        let specials = read_specials(config_path.as_deref())?;

        let (mut vocab, merges) = BPE::read_file(
            vocab_path.to_string_lossy().as_ref(),
            merges_path.to_string_lossy().as_ref(),
        )
        .map_err(|e| anyhow!("reading vocab/merges: {e}"))?;

        if vocab.get("<|endoftext|>") != Some(&EOS_ID) {
            bail!("vocab.json must map <|endoftext|> to {EOS_ID}");
        }
        for (token, id) in &added {
            if *id <= EOS_ID {
                bail!("added token {token} uses id {id}, which collides with the base GPT-2 vocab");
            }
            if vocab.insert(token.clone(), *id).is_some() {
                bail!("added token {token} is already in vocab.json");
            }
        }
        if vocab.len() != VOCAB_SIZE {
            bail!(
                "vocab is {} tokens after added tokens, expected {VOCAB_SIZE}",
                vocab.len()
            );
        }

        let bpe = BpeBuilder::new()
            .vocab_and_merges(vocab, merges)
            .unk_token("<|endoftext|>".to_string())
            .build()
            .map_err(|e| anyhow!("building GPT-2 BPE: {e}"))?;
        let mut tok = Tokenizer::new(bpe);
        let pretok = ByteLevel::new(false, false, true);
        tok.with_pre_tokenizer(Some(pretok));
        tok.with_decoder(Some(ByteLevel::new(false, false, true)));

        let mut extras = Vec::new();
        extras.push(added_token("<|endoftext|>", true));
        for (token, id) in &added {
            let special = specials.get(token).copied().unwrap_or(false);
            extras.push(added_token(token, special));
            let _ = id;
        }
        tok.add_tokens(extras)
            .map_err(|e| anyhow!("registering added tokens: {e}"))?;

        let loaded = Self { tok };
        if loaded.vocab_size() != VOCAB_SIZE {
            bail!(
                "tokenizer reports vocab {}, expected {VOCAB_SIZE}",
                loaded.vocab_size()
            );
        }
        for (token, id) in &added {
            match loaded.token_id(token) {
                Some(got) if got == *id => {}
                other => bail!("{token} resolved to {other:?}, file says {id}"),
            }
            let ids = loaded.encode(token)?;
            if ids != [*id] {
                bail!("{token} encoded as {ids:?}, not a single id {id}");
            }
        }
        Ok(loaded)
    }

    /// `PYRE_CHATTERBOX_TOKENIZER`, then `PYRE_CHATTERBOX_NANO`, then the HF hub
    /// cache for Nano and Turbo. Does not download.
    pub fn discover() -> Option<PathBuf> {
        discover_tokenizer_dir()
    }

    pub fn vocab_size(&self) -> usize {
        self.tok.get_vocab_size(true)
    }

    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.tok.token_to_id(token)
    }

    /// Encode `text` exactly. Does not run punctuation cleanup and does not add bos/eos.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self
            .tok
            .encode(text, false)
            .map_err(|e| anyhow!("encode: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Punctuation cleanup, then casefold a known tag, then BPE.
    /// `[CLEAR THROAT]` becomes the single trained id. [`Self::encode_like_generate`]
    /// does not casefold, and that is the stream that matches HF on the raw string.
    pub fn encode_script(&self, text: &str) -> Result<EncodedScript> {
        let script = directions::parse_script(text);
        // parse_script casefolds tags so the model sees the trained spelling.
        // The golden HF stream does not. Callers that need the HF bytes use `encode`
        // on `normalize_script` output. `encode_script` is the production path: a
        // recognized tag becomes its single id even if the user capitalized it.
        let ids = self.encode(&script.normalized)?;
        Ok(EncodedScript { script, ids })
    }

    /// Punctuation cleanup only, then BPE. Matches Chatterbox `generate` on the nose,
    /// including leaving an uppercase tag split into several ids.
    pub fn encode_like_generate(&self, text: &str) -> Result<(String, Vec<u32>)> {
        let normalized = directions::normalize_script(text);
        let ids = self.encode(&normalized)?;
        Ok((normalized, ids))
    }
}

/// `PYRE_CHATTERBOX_TOKENIZER`, then `PYRE_CHATTERBOX_NANO`, then the HF hub
/// cache for Nano and Turbo. Does not download.
pub fn discover_tokenizer_dir() -> Option<PathBuf> {
    for key in ["PYRE_CHATTERBOX_TOKENIZER", "PYRE_CHATTERBOX_NANO"] {
        if let Some(p) = std::env::var_os(key) {
            let p = PathBuf::from(p);
            if dir_has_tokenizer(&p) {
                return Some(p);
            }
        }
    }
    // Cabin catalog install path: <PYRE_STORE>/cabin/chatterbox-nano (HfFiles offer).
    if let Some(store) = std::env::var_os("PYRE_STORE") {
        let cabin = PathBuf::from(store).join("cabin").join("chatterbox-nano");
        if dir_has_tokenizer(&cabin) {
            return Some(cabin);
        }
    }
    find_in_hf_cache(&hf_cache_dir())
}

fn added_token(content: &str, special: bool) -> AddedToken {
    AddedToken::from(content, special)
        .single_word(false)
        .lstrip(false)
        .rstrip(false)
        .normalized(true)
}

#[derive(Deserialize)]
struct AddedEntry {
    content: String,
    special: Option<bool>,
}

fn read_specials(config: Option<&Path>) -> Result<HashMap<String, bool>> {
    let Some(config) = config else {
        return Ok(HashMap::new());
    };
    let raw = fs::read_to_string(config).with_context(|| config.display().to_string())?;
    let v: serde_json::Value = serde_json::from_str(&raw).context("tokenizer_config.json")?;
    let Some(map) = v.get("added_tokens_decoder").and_then(|m| m.as_object()) else {
        return Ok(HashMap::new());
    };
    let mut out = HashMap::new();
    for (id, entry) in map {
        let entry: AddedEntry =
            serde_json::from_value(entry.clone()).with_context(|| format!("added token {id}"))?;
        out.insert(entry.content, entry.special.unwrap_or(false));
    }
    Ok(out)
}

pub fn dir_has_tokenizer(dir: &Path) -> bool {
    dir.is_dir() && REQUIRED.iter().all(|name| safe_file(dir, name).is_ok())
}

pub(crate) fn safe_file(dir: &Path, name: &str) -> Result<PathBuf> {
    if name.is_empty() || name.contains(['/', '\\']) || name.contains("..") {
        bail!("refusing tokenizer path {name}");
    }
    let path = dir.join(name);
    let meta =
        fs::symlink_metadata(&path).with_context(|| format!("missing {}", path.display()))?;
    if !meta.file_type().is_file() && !meta.file_type().is_symlink() {
        bail!("{} is not a file", path.display());
    }
    let real = fs::canonicalize(&path).with_context(|| format!("resolve {}", path.display()))?;
    let dir_real = fs::canonicalize(dir).with_context(|| format!("resolve {}", dir.display()))?;
    if real.starts_with(&dir_real) {
        return Ok(real);
    }
    // HF snapshots store the bytes in `<repo>/blobs` and symlink them from the snapshot.
    if let Some(repo) = dir_real.parent().and_then(|p| p.parent()) {
        if real.starts_with(repo.join("blobs")) {
            return Ok(real);
        }
    }
    bail!(
        "{} resolves outside the tokenizer directory",
        path.display()
    );
}

fn hf_cache_dir() -> PathBuf {
    if let Some(p) = std::env::var_os("HF_HUB_CACHE") {
        return PathBuf::from(p);
    }
    if let Some(p) = std::env::var_os("HF_HOME") {
        return PathBuf::from(p).join("hub");
    }
    if let Some(p) = std::env::var_os("XDG_CACHE_HOME") {
        return PathBuf::from(p).join("huggingface").join("hub");
    }
    let home = std::env::var_os("USERPROFILE")
        .or_else(|| std::env::var_os("HOME"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    home.join(".cache").join("huggingface").join("hub")
}

pub fn find_in_hf_cache(cache: &Path) -> Option<PathBuf> {
    for repo in [REPO_NANO, REPO_TURBO] {
        let root = cache.join(format!("models--{}", repo.replace('/', "--")));
        if let Some(snap) = snapshot_with_tokenizer(&root) {
            return Some(snap);
        }
    }
    None
}

fn snapshot_with_tokenizer(repo_root: &Path) -> Option<PathBuf> {
    if let Ok(rev) = fs::read_to_string(repo_root.join("refs").join("main")) {
        let snap = repo_root.join("snapshots").join(rev.trim());
        if dir_has_tokenizer(&snap) {
            return Some(snap);
        }
    }
    let snaps = fs::read_dir(repo_root.join("snapshots")).ok()?;
    snaps
        .flatten()
        .map(|e| e.path())
        .find(|p| dir_has_tokenizer(p))
}

/// GPT-2's byte→unicode map. Used by the synthetic fixture, not by the real vocab.
pub fn bytes_to_unicode() -> Vec<char> {
    let mut bs: Vec<u32> = (b'!'..=b'~').map(|b| b as u32).collect();
    bs.extend((0xA1..=0xAC).map(|b| b as u32));
    bs.extend((0xAE..=0xFF).map(|b| b as u32));
    let mut cs = bs.clone();
    let mut n = 0u32;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut out = vec!['\0'; 256];
    for (b, c) in bs.into_iter().zip(cs) {
        out[b as usize] = char::from_u32(c).unwrap();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct Golden {
        vocab_size: usize,
        eos_id: u32,
        cases: Vec<Case>,
        scripts: Vec<ScriptCase>,
        tags: Vec<TagCase>,
    }
    #[derive(Deserialize)]
    struct Case {
        text: String,
        ids: Vec<u32>,
    }
    #[derive(Deserialize)]
    struct ScriptCase {
        input: String,
        normalized: String,
        ids: Vec<u32>,
    }
    #[derive(Deserialize)]
    struct TagCase {
        tag: String,
        id: u32,
    }

    fn golden() -> Golden {
        serde_json::from_str(include_str!("../testdata/tokenizer_golden.json")).unwrap()
    }

    #[test]
    fn golden_tags_are_one_id_and_match_the_parser() {
        let g = golden();
        assert_eq!(g.vocab_size, VOCAB_SIZE);
        assert_eq!(g.eos_id, EOS_ID);
        assert_eq!(g.tags.len(), 19);
        for tag in &g.tags {
            assert_eq!(
                directions::tag_token_id(&tag.tag),
                Some(tag.id),
                "{}",
                tag.tag
            );
        }
        for (tag, id, kind) in directions::all_tags() {
            let row = g.tags.iter().find(|t| t.tag == tag).unwrap();
            assert_eq!(row.id, id);
            if matches!(kind, DirectionKind::Event) {
                assert!((50267..=50275).contains(&id));
            }
        }
        let laugh = g.cases.iter().find(|c| c.text == "[laugh]").unwrap();
        assert_eq!(laugh.ids, vec![50275]);
        let throat = g.cases.iter().find(|c| c.text == "[clear throat]").unwrap();
        assert_eq!(throat.ids, vec![50267]);
        let shouted = g.cases.iter().find(|c| c.text == "[CLEAR THROAT]").unwrap();
        assert!(shouted.ids.len() > 1, "HF does not casefold tags");
    }

    /// A tiny byte-level BPE with one added tag, built the same way as `load` but
    /// without the 50276-size check. Proves a bracket tag is one id.
    fn load_synthetic(dir: &Path) -> Tokenizer {
        fs::create_dir_all(dir).unwrap();
        let unicode = bytes_to_unicode();
        let mut vocab_json = serde_json::Map::new();
        let mut vocab = Vocab::default();
        for (byte, ch) in unicode.iter().enumerate() {
            vocab_json.insert(ch.to_string(), serde_json::Value::from(byte as u32));
            vocab.insert(ch.to_string(), byte as u32);
        }
        vocab_json.insert("<|endoftext|>".into(), serde_json::Value::from(256));
        vocab.insert("<|endoftext|>".into(), 256);
        vocab.insert("[laugh]".into(), 257);
        fs::write(
            dir.join("vocab.json"),
            serde_json::to_string(&vocab_json).unwrap(),
        )
        .unwrap();
        fs::write(dir.join("merges.txt"), "#version: 0.2\n").unwrap();
        fs::write(dir.join("added_tokens.json"), r#"{ "[laugh]": 257 }"#).unwrap();

        let bpe = BpeBuilder::new()
            .vocab_and_merges(vocab, Vec::new())
            .unk_token("<|endoftext|>".into())
            .build()
            .unwrap();
        let mut tok = Tokenizer::new(bpe);
        tok.with_pre_tokenizer(Some(ByteLevel::new(false, false, true)));
        tok.add_tokens([
            added_token("[laugh]", false),
            added_token("<|endoftext|>", true),
        ])
        .unwrap();
        tok
    }

    #[test]
    fn synthetic_tag_is_one_id_and_outside_links_are_refused() {
        let tmp = std::env::temp_dir().join(format!("pyre-nano-tok-{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();
        let tok = load_synthetic(&tmp);
        let ids = tok.encode("[laugh]", false).unwrap().get_ids().to_vec();
        assert_eq!(ids, vec![257], "tag must not be byte-pieced");
        let around = tok.encode("a [laugh] b", false).unwrap().get_ids().to_vec();
        assert!(around.contains(&257), "{around:?}");
        assert_eq!(around.iter().filter(|id| **id == 257).count(), 1);

        let snap = tmp.join("snap");
        fs::create_dir_all(&snap).unwrap();
        assert!(safe_file(&snap, "../vocab.json").is_err());
        assert!(safe_file(&snap, "a/vocab.json").is_err());
        assert!(safe_file(&snap, "").is_err());
        #[cfg(unix)]
        {
            let outside = tmp.join("outside.txt");
            fs::write(&outside, "nope").unwrap();
            symlink_file(&outside, &snap.join("vocab.json"));
            assert!(safe_file(&snap, "vocab.json").is_err());
        }
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn hf_cache_layout_is_found_and_blobs_symlink_is_allowed() {
        let tmp = std::env::temp_dir().join(format!("pyre-nano-hf-{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        let repo = tmp.join("models--ResembleAI--chatterbox-nano");
        let snap = repo.join("snapshots").join("abc123");
        let blobs = repo.join("blobs");
        fs::create_dir_all(&snap).unwrap();
        fs::create_dir_all(&blobs).unwrap();
        fs::create_dir_all(repo.join("refs")).unwrap();
        fs::write(repo.join("refs").join("main"), "abc123\n").unwrap();
        for name in ["vocab.json", "merges.txt", "added_tokens.json"] {
            let blob = blobs.join(name);
            fs::write(&blob, b"{}\n").unwrap();
            symlink_file(&blob, &snap.join(name));
        }
        let found = find_in_hf_cache(&tmp).unwrap();
        assert_eq!(found, snap);
        assert!(safe_file(&snap, "vocab.json").is_ok());
        let _ = fs::remove_dir_all(&tmp);
    }

    fn symlink_file(src: &Path, dst: &Path) {
        #[cfg(unix)]
        std::os::unix::fs::symlink(src, dst).unwrap();
        #[cfg(windows)]
        std::os::windows::fs::symlink_file(src, dst).unwrap();
    }

    fn tokenizer_dir() -> Option<PathBuf> {
        if let Some(p) = std::env::var_os("PYRE_CHATTERBOX_TOKENIZER") {
            return Some(PathBuf::from(p));
        }
        if let Some(p) = std::env::var_os("PYRE_CHATTERBOX_NANO") {
            return Some(PathBuf::from(p));
        }
        None
    }

    #[test]
    #[ignore = "set PYRE_CHATTERBOX_TOKENIZER to the Nano tokenizer dir and pass --ignored"]
    fn real_tokenizer_matches_hf_golden() {
        let Some(dir) = tokenizer_dir() else {
            eprintln!("PYRE_CHATTERBOX_TOKENIZER unset; skipping HF compare");
            return;
        };
        let tok = NanoTokenizer::load(&dir).unwrap();
        let g = golden();
        assert_eq!(tok.vocab_size(), g.vocab_size);
        assert_eq!(tok.token_id("<|endoftext|>"), Some(g.eos_id));
        for case in &g.cases {
            let ids = tok.encode(&case.text).unwrap();
            assert_eq!(ids, case.ids, "encode {:?}", case.text);
        }
        for script in &g.scripts {
            let (norm, ids) = tok.encode_like_generate(&script.input).unwrap();
            assert_eq!(norm, script.normalized, "punc {:?}", script.input);
            assert_eq!(ids, script.ids, "ids {:?}", script.input);
        }
        for tag in &g.tags {
            assert_eq!(tok.encode(&tag.tag).unwrap(), vec![tag.id]);
        }
        // Production path casefolds a shouted tag onto the trained id.
        let shouted = tok.encode_script("wait [CLEAR THROAT] okay").unwrap();
        assert!(shouted.ids.contains(&50267), "{:?}", shouted.ids);
        assert_eq!(shouted.ids.iter().filter(|id| **id == 50267).count(), 1);
    }
}
