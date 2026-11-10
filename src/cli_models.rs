//! `pyre models ...` subcommands.
use crate::storage::Ctx;
use crate::store::{detect, dir_size, hfcache, human, store::Mode, Kind};
use crate::ModelsCmd;
use anyhow::{bail, Context, Result};
use serde_json::json;
use std::path::Path;

pub fn list_json(ctx: &Ctx) -> serde_json::Value {
    let mut rows = vec![];
    for e in ctx.store.list() {
        let m = &e.manifest;
        let links: std::collections::BTreeSet<&str> = m.files.iter().map(|f| f.link.as_str()).collect();
        let dir = e.model_dir();
        rows.push(json!({"name": m.name, "alias": m.alias, "kind": m.kind.as_str(), "source": "store", "architecture": m.architecture,
            "size": e.size(), "location": &dir, "storage": links.into_iter().collect::<Vec<_>>().join("+"),
            "revision": m.source_revision, "repo": m.source_repo,
            "capabilities": crate::capabilities::Capabilities::for_model(m.kind, &dir).to_value()}));
    }
    for c in hfcache::list(&ctx.hf_cache) {
        let (kind, arch) = detect(&c.path);
        rows.push(json!({"name": c.repo, "alias": null, "kind": kind.as_str(), "source": "hf-cache", "architecture": arch,
            "size": c.size, "location": c.path, "storage": format!("shared ({})", c.mode), "revision": c.revision, "refs": c.refs, "repo": c.repo,
            "capabilities": crate::capabilities::Capabilities::for_model(kind, &c.path).to_value()}));
    }
    json!({"store_root": ctx.store.root, "hf_cache": ctx.hf_cache, "models": rows})
}

pub fn parse_kind(k: &Option<String>) -> Result<Option<Kind>> {
    k.as_deref().map(|s| Kind::parse(s).with_context(|| format!("unknown kind {s}"))).transpose()
}

pub fn run(ctx: &Ctx, cmd: ModelsCmd) -> Result<()> {
    match cmd {
        ModelsCmd::List { json } => {
            let v = list_json(ctx);
            if json { println!("{}", serde_json::to_string_pretty(&v)?); return Ok(()); }
            println!("store:    {}\nhf cache: {}\n", ctx.store.root.display(), ctx.hf_cache.display());
            println!("{:<44} {:<8} {:<9} {:>10}  {:<28} {}", "NAME", "KIND", "SOURCE", "SIZE", "CAPABILITIES", "LOCATION");
            for r in v["models"].as_array().unwrap() {
                let name = match r["alias"].as_str() { Some(a) => format!("{} ({a})", r["name"].as_str().unwrap()), None => r["name"].as_str().unwrap().to_string() };
                println!("{:<44} {:<8} {:<9} {:>10}  {:<28} {}", name, r["kind"].as_str().unwrap(), r["source"].as_str().unwrap(),
                    human(r["size"].as_u64().unwrap_or(0)), r["capabilities"]["summary"].as_str().unwrap_or(""), r["location"].as_str().unwrap_or(""));
            }
        }
        ModelsCmd::Pull { repo, store, kind } => {
            let (r, rev) = hfcache::split_rev(&repo);
            if ctx.offline { bail!("offline mode: cannot pull"); }
            let snap = match hfcache::find(&ctx.hf_cache, r, rev) {
                Some(p) => { println!("already in HF cache: {}", p.display()); hfcache::dedupe_snapshot(&p); p }
                None => hfcache::download(&ctx.hf_cache, r, rev, &|f, i, n| eprintln!("[{}/{}] {f}", (i + 1).min(n), n))?,
            };
            println!("HF cache: {}", snap.display());
            if store == "pyre" {
                let (e, notes) = ctx.store.import(&snap, r, Some(r), rev, parse_kind(&kind)?, Mode::Link)?;
                for n in notes { eprintln!("note: {n}"); }
                println!("linked into pyre store: {}", e.dir.display());
            }
        }
        ModelsCmd::Import { source, link: _, copy, r#move, register, name, kind } => {
            let kind = parse_kind(&kind)?;
            let p = Path::new(&source);
            if register {
                if !p.is_dir() { bail!("--register needs a folder"); }
                let e = ctx.store.register(p, name.as_deref(), kind)?;
                println!("registered {} in place -> {}", e.manifest.name, p.display());
                return Ok(());
            }
            let mode = if copy { Mode::Copy } else if r#move { Mode::Move } else { Mode::Link };
            let (src, repo, rev) = if p.is_dir() { (p.to_path_buf(), None, None) } else {
                let (r, rev) = hfcache::split_rev(&source);
                (hfcache::find(&ctx.hf_cache, r, rev).with_context(|| format!("{source} not found in HF cache {} (run `pyre models pull {source}` first)", ctx.hf_cache.display()))?, Some(r.to_string()), rev.map(String::from))
            };
            if mode == Mode::Move && repo.is_some() {
                eprintln!("warning: --move takes the files out of the HF cache; huggingface_hub will re-download them if asked");
            }
            let nm = name.or(repo.clone()).unwrap_or_else(|| format!("local/{}", src.file_name().unwrap().to_string_lossy()));
            let (e, notes) = ctx.store.import(&src, &nm, repo.as_deref(), rev.as_deref(), kind, mode)?;
            for n in notes { eprintln!("note: {n}"); }
            println!("imported {} ({:?}, {}) -> {}", nm, mode, human(e.size()), e.dir.display());
        }
        ModelsCmd::Rm { name, from_hf_cache, yes } => {
            let e = ctx.store.find(&name);
            let hf = if from_hf_cache { Some(ctx.hf_cache.join(hfcache::repo_folder(hfcache::split_rev(&name).0))) } else { None };
            if e.is_none() && hf.as_ref().map(|p| !p.exists()).unwrap_or(true) { bail!("{name}: not found in store{}", if from_hf_cache { " or HF cache" } else { "" }); }
            if let Some(e) = &e { println!("store entry: {} ({})", e.dir.display(), if e.manifest.path.is_some() { "registered folder is kept" } else { "linked files only drop a reference" }); }
            if let Some(h) = hf.as_ref().filter(|p| p.exists()) { println!("HF cache repo (blobs will be DELETED): {} ({})", h.display(), human(dir_size(h))); }
            if !yes && !confirm("Delete?")? { println!("aborted"); return Ok(()); }
            if let Some(e) = e { ctx.store.remove(&e)?; println!("removed store entry"); }
            if let Some(h) = hf.filter(|p| p.exists()) { std::fs::remove_dir_all(&h)?; println!("removed {} from HF cache", h.display()); }
        }
        ModelsCmd::Alias { name, alias } => {
            let e = match ctx.store.find(&name) {
                Some(_) => ctx.store.set_alias(&name, &alias)?,
                None => {
                    // alias an HF-cache model: link it into the store first (no extra space)
                    let (r, rev) = hfcache::split_rev(&name);
                    let snap = hfcache::find(&ctx.hf_cache, r, rev).with_context(|| format!("{name} not found"))?;
                    let (_, notes) = ctx.store.import(&snap, r, Some(r), rev, None, Mode::Link)?;
                    for n in notes { eprintln!("note: {n}"); }
                    ctx.store.set_alias(r, &alias)?
                }
            };
            println!("{} -> alias {alias}", e.manifest.name);
        }
        ModelsCmd::Info { name } => {
            if let Some(e) = ctx.store.find(&name) {
                println!("# {}\n", e.dir.join(crate::store::store::MANIFEST).display());
                println!("{}", toml::to_string_pretty(&e.manifest)?);
                println!("capabilities: {}", crate::capabilities::Capabilities::for_model(e.manifest.kind, &e.model_dir()).summary());
            } else {
                let (r, rev) = hfcache::split_rev(&name);
                let p = hfcache::find(&ctx.hf_cache, r, rev).with_context(|| format!("{name} not found in store or HF cache"))?;
                let (k, a) = detect(&p);
                let caps = crate::capabilities::Capabilities::for_model(k, &p);
                println!("HF cache model {r}\npath: {}\nkind: {}\narchitecture: {a}\ncapabilities: {}\nsize: {}", p.display(), k.as_str(), caps.summary(), human(dir_size(&p)));
                let blobs = hfcache::blobs_of(&p);
                for f in crate::store::walk(&p) { let (_, m, _) = hfcache::resolve_entry(&f, &blobs); println!("  {:<40} {:>10} {m}", f.strip_prefix(&p)?.display(), human(std::fs::metadata(&f)?.len())); }
            }
        }
    }
    Ok(())
}

fn confirm(q: &str) -> Result<bool> {
    use std::io::Write;
    print!("{q} [y/N] ");
    std::io::stdout().flush()?;
    let mut s = String::new();
    std::io::stdin().read_line(&mut s)?;
    Ok(matches!(s.trim(), "y" | "Y" | "yes"))
}
