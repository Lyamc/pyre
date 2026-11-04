#![recursion_limit = "512"]
mod backend;
mod ctx;
mod effort;
mod engine;
mod api;
mod logbuf;
mod registry;
mod stt;
mod diar;
mod cabin;
mod esr;
mod isnet;
mod segformer;
mod lcm;
mod rife;
mod tts;
mod image;
mod tagger;
mod ui;
mod cli_models;
mod storage;
mod store;
mod model;
mod sampler;
mod server;
mod limits;
mod metrics;
mod template;
mod qn_cli;
mod apps;
mod engine_qn;
mod qn_find;
mod spec;

use anyhow::{Context, Result};
use clap::Parser;
use model::config::{ids, LlamaConfig};
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Parser, Debug)]
#[command(name = "pyre", version, about = "pyre: a llama-server-style inference server where Burn runs the models")]
pub struct Cli {
    #[command(flatten)]
    pub storage: StorageArgs,
    #[command(subcommand)]
    pub cmd: Cmd,
}

#[derive(clap::Args, Debug, Clone)]
pub struct StorageArgs {
    /// pyre model store (default: PYRE_MODELS_DIR, config file, %LOCALAPPDATA%\pyre\models or ~/.local/share/pyre/models)
    #[arg(long = "models-dir", global = true)]
    pub models_dir: Option<PathBuf>,
    /// Hugging Face hub cache (default: HF_HUB_CACHE, $HF_HOME/hub, ~/.cache/huggingface/hub)
    #[arg(long = "hf-cache", global = true)]
    pub hf_cache: Option<PathBuf>,
    /// Never download (also honours HF_HUB_OFFLINE=1)
    #[arg(long, global = true)]
    pub offline: bool,
    /// Comma-separated model search order: path,store,hf,download
    #[arg(long = "search-order", global = true, value_delimiter = ',')]
    pub search_order: Option<Vec<String>>,
}

#[derive(clap::Subcommand, Debug)]
pub enum Cmd {
    /// Run the HTTP server (OpenAI + llama.cpp compatible API)
    Serve(ServeArgs),
    /// Manage models in the pyre store and the HF cache
    Models {
        #[command(subcommand)]
        cmd: ModelsCmd,
    },
    /// Run a Qwen3.8-Flash-Next (qwen4_exp) checkpoint directly: greedy text or logits dump
    QwenNext(qn_cli::QnArgs),
    /// Speaker diarization (Nemotron-3-Diarization) of a 16 kHz-resampled audio file: prints segments as JSON
    Diarize { file: PathBuf, #[arg(long, default_value = "nvidia/Nemotron-3-Diarization")] model: String, #[arg(long, default_value = "cpu")] backend: String, /// also write raw logits [T,8] (f32 LE) here
        #[arg(long)] logits: Option<PathBuf> },
    /// Tag images (or folders of images) with an EVA02 Danbooru tagger (default Redstonexs/kagami-24k)
    Tag(tagger::TagArgs),
    /// Companion apps that use pyre (peal, cabin, ...): list | install (add) | update | remove | run | info
    App {
        #[command(subcommand)]
        cmd: apps::AppCmd,
    },
}

#[derive(clap::Args, Debug)]
pub struct ServeArgs {
    /// Text-generation model: alias, store name, HF repo id[@rev] or folder
    #[arg(short = 'm', long = "model")]
    pub model: Vec<String>,
    /// Extra models as kind=ref (kind: llm|image|stt|tts|tagger); repeatable
    #[arg(long = "load", value_name = "KIND=REF")]
    pub load: Vec<String>,
    #[arg(long = "image-model")]
    pub image_model: Option<String>,
    #[arg(long = "stt-model")]
    pub stt_model: Option<String>,
    #[arg(long = "tts-model")]
    pub tts_model: Option<String>,
    /// Image tagger (EVA02 Danbooru tagger, e.g. Redstonexs/kagami-24k) for POST /v1/images/tags
    #[arg(long = "tagger-model")]
    pub tagger_model: Option<String>,
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,
    #[arg(long, default_value_t = 8080)]
    pub port: u16,
    /// cuda | wgpu | vulkan | cpu (flex) | ndarray
    #[arg(short = 'b', long, default_value = "cuda")]
    pub backend: String,
    /// auto | f32 | f16 | bf16
    #[arg(long, default_value = "auto")]
    pub dtype: String,
    #[arg(short = 'c', long = "ctx-size", default_value_t = 4096)]
    pub ctx_size: usize,
    /// Default max tokens when the request sets none
    #[arg(short = 'n', long = "n-predict", default_value_t = 512)]
    pub n_predict: usize,
    /// Require `Authorization: Bearer <key>`
    #[arg(long = "api-key", env = "PYRE_API_KEY", hide_env_values = true)]
    pub api_key: Option<String>,
    /// Serve /metrics, /api/metrics and /api/logs on this extra address too (e.g. 0.0.0.0:9099); they are always on the main port
    #[arg(long = "metrics-bind", env = "PYRE_METRICS_BIND")]
    pub metrics_bind: Option<String>,
    /// Model name reported by /v1/models (defaults to -m)
    #[arg(short = 'a', long)]
    pub alias: Option<String>,
    /// Load + one greedy generation, print it and exit (for benchmarking / sanity checks)
    #[arg(long)]
    pub prompt: Option<String>,
    /// VRAM kept free for other apps, e.g. 1G or 512M (default max(1 GiB, 10% of VRAM); 0 disables). Env PYRE_VRAM_RESERVE
    #[arg(long = "vram-reserve")]
    pub vram_reserve: Option<String>,
    /// Max GPU duty cycle in percent (default 90; 100 or 0 disables). Env PYRE_GPU_UTIL_MAX
    #[arg(long = "gpu-util-max")]
    pub gpu_util_max: Option<u32>,
    /// Speculative decoding draft length for qwen-next (0 = off, the default; config `spec`, env PYRE_SPEC)
    #[arg(long = "spec")]
    pub spec: Option<usize>,
    /// Residency-biased expert routing strength for qwen-next (0 = off, the default; changes outputs; config `route_prior`, env PYRE_ROUTE_PRIOR)
    #[arg(long = "route-prior")]
    pub route_prior: Option<f32>,
    /// Lossy speed shortcuts for qwen-next (route prior 1.5 + route tail skip 7, from Strata); on by default, `--fast false` = exact outputs (env PYRE_QN_FAST)
    #[arg(long = "fast")]
    pub fast: Option<bool>,
    /// Disable the prompt-lookup (n-gram) drafter
    #[arg(long = "no-ngram")]
    pub no_ngram: bool,
}

#[derive(clap::Subcommand, Debug)]
pub enum ModelsCmd {
    /// List pyre-store and HF-cache models
    List { #[arg(long)] json: bool },
    /// Download repo[@rev] (into the HF cache by default)
    Pull { repo: String, #[arg(long, default_value = "hf")] store: String, #[arg(long)] kind: Option<String> },
    /// Import an HF repo (from the cache) or a local folder into the pyre store
    Import {
        source: String,
        /// hardlink (falls back to symlink, then copy) - default, no extra disk
        #[arg(long, group = "mode")] link: bool,
        #[arg(long, group = "mode")] copy: bool,
        #[arg(long, group = "mode")] r#move: bool,
        /// Register a local folder in place (no copy)
        #[arg(long, group = "mode")] register: bool,
        #[arg(long)] name: Option<String>,
        #[arg(long)] kind: Option<String>,
    },
    /// Remove a store entry (HF cache untouched unless --from-hf-cache)
    Rm { name: String, #[arg(long = "from-hf-cache")] from_hf_cache: bool, #[arg(short = 'y', long)] yes: bool },
    /// Set an alias for a store model
    Alias { name: String, alias: String },
    /// Show details for a model
    Info { name: String },
}

fn main() -> Result<()> {
    if std::env::var("PYRE_QN_KBENCH").is_ok() { crate::model::qdot::kbench(); return Ok(()); }
    if std::env::var("PYRE_QN_PBENCH").is_ok() { crate::model::qwen4exp_gguf::pool_bench(); return Ok(()); }
    if std::env::var("PYRE_QN_XBENCH").is_ok() { return crate::model::qn_cube::xbench(); }
    logbuf::init(); let _ = metrics::uptime_s();
    kernel_cache_config();
    let cli = Cli::parse();
    let ctx = storage::Ctx::new(&cli.storage)?;
    match cli.cmd {
        Cmd::Models { cmd } => cli_models::run(&ctx, cmd),
        Cmd::Serve(a) => serve(ctx, a),
        Cmd::QwenNext(a) => qn_cli::run(a),
        Cmd::Diarize { file, model, backend, logits } => diar::cli(&ctx.hf_cache, &file, &model, &backend, logits.as_deref()),
        Cmd::Tag(a) => tagger::run_cli(&ctx, a),
        Cmd::App { cmd } => apps::run_cli(&ctx, cmd),
    }
}

/// Load one text-generation model and start its worker.
pub fn load_llm(dir: &std::path::Path, id: &str, backend_name: &str, dtype: &str, ctx_size: usize, n_predict: usize) -> Result<server::Llm> {
    if qn_find::is_qwen4exp(dir) { return load_qn(dir, id, backend_name, n_predict); }
    let (llama, tok, template, eos, bname, fdt) = load_llm_parts(dir, backend_name, dtype, ctx_size)?;
    let n_ctx = llama.max_ctx;
    let engine = engine::spawn(llama, tok.clone(), eos);
    Ok(server::Llm { engine, tok, template, model_id: id.to_string(), model_path: dir.display().to_string(), n_ctx,
        backend: bname.to_string(), dtype: format!("{fdt:?}"), default_max_tokens: n_predict })
}

/// rayon pool for the CPU GGUF path (PYRE_QN_THREADS, default 16 of the 13900K's 32 threads).
/// Opt this process out of Windows power throttling (EcoQoS) unless PYRE_ECOQOS=1. When the user is at the
/// desktop, Windows may treat a background console job as low priority and run it at reduced clocks or on
/// E-cores; decode then slows ~30%. This is a per-process request, not a system setting.
pub fn no_power_throttle() {
    #[cfg(windows)] {
        if std::env::var("PYRE_ECOQOS").map(|v| v == "1").unwrap_or(false) { return; }
        use windows_sys::Win32::System::Threading::{GetCurrentProcess, SetProcessInformation, ProcessPowerThrottling, PROCESS_POWER_THROTTLING_STATE, PROCESS_POWER_THROTTLING_CURRENT_VERSION, PROCESS_POWER_THROTTLING_EXECUTION_SPEED, PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION};
        let st = PROCESS_POWER_THROTTLING_STATE { Version: PROCESS_POWER_THROTTLING_CURRENT_VERSION, ControlMask: PROCESS_POWER_THROTTLING_EXECUTION_SPEED | PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION, StateMask: 0 };
        let ok = unsafe { SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &st as *const _ as *const core::ffi::c_void, std::mem::size_of::<PROCESS_POWER_THROTTLING_STATE>() as u32) };
        if ok == 0 { eprintln!("power throttling opt-out failed"); }
    }
}
pub fn qn_threads() {
    no_power_throttle();
    let n = std::env::var("PYRE_QN_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(16);
    let _ = rayon::ThreadPoolBuilder::new().num_threads(n).build_global();
}

/// Tokenizer, chat template and EOS ids for a qwen4exp GGUF (from the base model's HF files).
pub fn qn_text_parts(f: &std::path::Path, m: &model::qwen4exp_gguf::Model) -> Result<(Arc<tokenizers::Tokenizer>, template::ChatTemplate, Vec<u32>)> {
    let base = m.g.meta.get("general.base_model.0.repo_url").and_then(|v| v.as_str()).map(String::from);
    let tdir = qn_find::tokenizer_dir(f, base.as_deref())?;
    let tok = Arc::new(tokenizers::Tokenizer::from_file(tdir.join("tokenizer.json")).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?);
    let template = template::ChatTemplate::from_model_dir(&tdir)?;
    let mut eos = vec![];
    if let Ok(g) = std::fs::read_to_string(tdir.join("generation_config.json")) {
        if let Ok(g) = serde_json::from_str::<serde_json::Value>(&g) { eos.extend(ids(&g.get("eos_token_id").cloned())); }
    }
    for t in [template.eos_token.as_str(), "<|im_end|>"] { if let Some(i) = tok.token_to_id(t) { eos.push(i); } }
    eos.sort(); eos.dedup();
    Ok((tok, template, eos))
}

/// Qwen3.8-Flash-Next GGUF: CPU engine, experts quantized in the mmap (see model::qwen4exp_gguf).
/// `-b cuda` (and a CUDA build): dense projections to the GPU (Strata's split); otherwise all-CPU.
pub fn qn_attach_gpu(m: &mut model::qwen4exp_gguf::Model, backend: &str) -> Result<String> {
    if backend != "cuda" || !cfg!(feature = "cuda") { return Ok("cpu".into()); }
    let t = std::time::Instant::now();
    let (on, off, gb) = m.attach_gpu(model::qn_gpu::Gpu::new()?)?;
    tracing::info!("qwen4exp: {on} projection groups on the GPU ({gb:.2} GiB Q8), {off} left on the CPU (VRAM reserve), in {:.1}s", t.elapsed().as_secs_f64());
    eprintln!("gpu: {on} groups on the GPU ({gb:.2} GiB Q8), {off} on the CPU, {:.1}s", t.elapsed().as_secs_f64());
    Ok(if off == 0 { "cuda+cpu-experts".into() } else { "cuda(partial)+cpu".into() })
}

fn load_qn(dir: &std::path::Path, id: &str, backend_name: &str, n_predict: usize) -> Result<server::Llm> {
    let f = qn_find::find_gguf(dir).context("no .gguf found")?;
    qn_threads();
    let backend_name = backend_name.to_string();
    // load, GPU attach, MTP + verify-graph prewarm all on the worker thread that will decode (see engine_qn::spawn_init)
    let (engine, (tok, template, n_ctx, label, path)) = engine_qn::spawn_init(move || {
        let t = std::time::Instant::now();
        let mut m = model::qwen4exp_gguf::Model::load(&f)?;
        let label = qn_attach_gpu(&mut m, &backend_name)?;
        m.load_mtp()?; m.prewarm_verify();
        let tp = std::time::Instant::now(); let gb = m.prefault_experts(); if gb > 0.0 { tracing::info!("qwen4exp: {gb:.1} GiB of routed experts made resident in {:.1}s", tp.elapsed().as_secs_f64()); }
        tracing::info!("qwen4exp GGUF {} ({:.1} GB mapped, {} layers) ready in {:.1}s | {label}, {} threads, routed experts stay quantized on the CPU (mmap)",
            f.display(), m.g.file_bytes() as f64 / 1e9, m.cfg.n_layers, t.elapsed().as_secs_f64(), rayon::current_num_threads());
        let (tok, template, eos) = qn_text_parts(&f, &m)?;
        tracing::info!("eos ids {eos:?}");
        let n_ctx = m.max_ctx();
        Ok((m, tok.clone(), eos, (tok, template, n_ctx, label, f.display().to_string())))
    })?;
    Ok(server::Llm { engine, tok, template, model_id: id.to_string(), model_path: path, n_ctx,
        backend: label, dtype: "gguf".into(), default_max_tokens: n_predict })
}

#[allow(clippy::type_complexity)]
fn load_llm_parts(dir: &std::path::Path, backend_name: &str, dtype: &str, ctx_size: usize)
    -> Result<(model::llama::Llama, Arc<tokenizers::Tokenizer>, template::ChatTemplate, Vec<u32>, &'static str, burn::tensor::FloatDType)> {
    let cfg: LlamaConfig = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).context("config.json")?)?;
    let arch = cfg.arch()?;
    let (device, fdt, bname) = backend::select(backend_name, dtype, cfg.torch_dtype.as_deref())?;
    let ctx = ctx_size.min(cfg.max_position_embeddings);
    tracing::info!("model {} arch {:?} | backend {bname} dtype {fdt:?} | ctx {ctx}", dir.display(), arch);
    let t = std::time::Instant::now();
    let ws = model::loader::Weights::open(dir, device.clone(), fdt.into())?;
    let llama = model::llama::Llama::load(cfg.clone(), &ws, device.clone(), fdt, ctx)?;
    drop(ws);
    tracing::info!("weights loaded in {:.1}s", t.elapsed().as_secs_f64());
    let tok = Arc::new(tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?);
    let template = template::ChatTemplate::from_model_dir(dir)?;
    let mut eos = cfg.eos_ids();
    if let Ok(g) = std::fs::read_to_string(dir.join("generation_config.json")) {
        if let Ok(g) = serde_json::from_str::<serde_json::Value>(&g) { eos.extend(ids(&g.get("eos_token_id").cloned())); }
    }
    for t in [template.eos_token.as_str(), "<|im_end|>", "<|eot_id|>"] {
        if let Some(id) = tok.token_to_id(t) { eos.push(id); }
    }
    eos.sort(); eos.dedup();
    tracing::info!("eos ids {eos:?}");
    Ok((llama, tok, template, eos, bname, fdt))
}

fn serve(sctx: storage::Ctx, args: ServeArgs) -> Result<()> {
    use registry::{Kind, LoadReq};
    limits::init(args.vram_reserve.as_deref(), args.gpu_util_max);
    if let Some(f) = args.fast { spec::set_fast(f); }
    { let cfg = store::store::load_config();
      if let Some(n) = args.spec.or(cfg.spec) { spec::set_draft_len(n); }
      if let Some(p) = args.route_prior.or(cfg.route_prior) { spec::set_route_prior(p); }
      if args.no_ngram || cfg.spec_ngram == Some(false) { spec::set_ngram(false); }
      if cfg.spec_mtp == Some(false) { spec::set_mtp(false); } }
    #[cfg(feature = "image")]
    let _ = pyre_sd::pipelines::stable_diffusion::STEP_HOOK.set(limits::pace);
    let defaults = registry::LoadDefaults { backend: args.backend.clone(), dtype: args.dtype.clone(), ctx_size: args.ctx_size, n_predict: args.n_predict };
    // collect (kind, ref)
    // (kind, ref, backend override): `--load stt:cpu=base` loads that model on its own backend
    let mut want: Vec<(Kind, String, Option<String>)> = args.model.iter().map(|m| (Kind::Llm, m.clone(), None)).collect();
    let mut diar_want: Option<(String, String)> = None;
    for l in &args.load {
        let (k, r) = l.split_once('=').context("--load expects kind[:backend]=ref")?;
        let (k, b) = match k.split_once(':') { Some((k, b)) => (k, Some(b.to_string())), None => (k, None) };
        if k == "diar" { diar_want = Some((r.to_string(), b.unwrap_or_else(|| "auto".into()))); continue; }
        want.push((Kind::parse(k).with_context(|| format!("unknown kind {k}"))?, r.to_string(), b));
    }
    if let Some(r) = &args.image_model { want.push((Kind::Image, r.clone(), None)); }
    if let Some(r) = &args.stt_model { want.push((Kind::Stt, r.clone(), None)); }
    if let Some(r) = &args.tts_model { want.push((Kind::Tts, r.clone(), None)); }
    if let Some(r) = &args.tagger_model { want.push((Kind::Tagger, r.clone(), None)); }

    if let Some(p) = &args.prompt {
        let r = args.model.first().context("--prompt needs -m")?;
        let (dir, _) = sctx.resolver().resolve(r)?;
        let (llama, tok, template, eos, _, _) = load_llm_parts(&dir, &args.backend, &args.dtype, args.ctx_size)?;
        return backend::oneshot(&llama, &tok, &template, &eos, p);
    }

    let app = Arc::new(registry::App { models: Default::default(), storage: sctx, defaults: defaults.clone(), api_key: args.api_key.clone(), started: std::time::Instant::now() });
    for (i, (kind, r, be)) in want.iter().enumerate() {
        let alias = if i == 0 && *kind == Kind::Llm { args.alias.clone() } else { None };
        let m = registry::load(&app.storage, &defaults, LoadReq { kind: *kind, r, alias, backend: be.clone(), dtype: None, ctx_size: None })?;
        tracing::info!("loaded {} model '{}' on {}", kind.as_str(), m.id, m.backend);
        app.insert(m)?;
    }
    if let Some((r, b)) = diar_want {
        match diar::Diarizer::load(&r, &b, &app.storage.hf_cache) { Ok(d) => { let _ = diar::DIAR.set(Arc::new(d)); } Err(e) => tracing::error!("diarizer {r} not loaded: {e:#}") }
    }
    if want.is_empty() { tracing::warn!("no models given; load them from the web UI or POST /api/models/load"); }
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async move {
        let addr = format!("{}:{}", args.host, args.port);
        let local = if args.host == "0.0.0.0" || args.host == "::" { "127.0.0.1".to_string() } else { args.host.clone() };
        apps::set_server(format!("http://{local}:{}", args.port));
        crate::metrics::set_instance(args.port); crate::metrics::start_sampler(); crate::limits::governor_start();
        if let Some(mb) = args.metrics_bind.clone().filter(|s| !s.is_empty()) {
            let ml = tokio::net::TcpListener::bind(&mb).await?;
            tracing::info!("metrics on http://{mb}/metrics (token {})", if std::env::var("PYRE_METRICS_TOKEN").map(|t| !t.is_empty()).unwrap_or(false) { "required" } else { "not set" });
            let mr = api::metrics_routes(axum::Router::new()).with_state(app.clone());
            tokio::spawn(async move { if let Err(e) = axum::serve(ml, mr).await { tracing::error!("metrics listener: {e}"); } });
        }
        if let Some(p) = crate::logbuf::file_path() { tracing::info!("structured log: {p}"); }
        let l = tokio::net::TcpListener::bind(&addr).await?;
        tracing::info!("pyre listening on http://{addr} (web UI at /)");
        for u in lan_urls(&args.host, args.port) { tracing::info!("  reachable at {u}"); }
        if args.host != "127.0.0.1" && args.host != "localhost" && args.api_key.is_none() {
            tracing::warn!("listening beyond localhost without --api-key: anyone on the network can use this server");
        }
        axum::serve(l, server::router(app)).await?;
        Ok(())
    })
}

/// cubecl persists autotune results but, by default, not the compiled kernels, so every start
/// recompiled (NVRTC) every kernel on first use: ~45 s before the first tiny.en transcription on
/// CUDA, ~0.7 s once cached. Turn the compiled-kernel cache on (opt out: PYRE_KERNEL_CACHE=0),
/// keeping anything else from cubecl.toml / CUBECL_* env. Must run before any device is created.
fn kernel_cache_config() {
    use burn_cubecl::cubecl::config::{CubeClRuntimeConfig, RuntimeConfig};
    let mut c = CubeClRuntimeConfig::from_current_dir().override_from_env();
    let off = matches!(std::env::var("PYRE_KERNEL_CACHE").as_deref(), Ok("0" | "false" | "off"));
    c.compilation.cache = !off;
    // cubecl's default root is <outermost Cargo workspace above cwd>/target/environment, else the user
    // cache dir, so the same binary used different caches depending on where it was started (repo vs
    // GUI/elsewhere). Pin it to the user cache (%LOCALAPPDATA%\cubecl) unless configured.
    if matches!(c.environment.path, burn_cubecl::cubecl::config::cache::CacheConfig::Target) {
        c.environment.path = burn_cubecl::cubecl::config::cache::CacheConfig::Global;
    }
    if !CubeClRuntimeConfig::try_set(c) { tracing::warn!("cubecl config already loaded; kernel cache setting not applied"); }
}
/// URLs other machines can use when bound to 0.0.0.0 / a specific address.
fn lan_urls(host: &str, port: u16) -> Vec<String> {
    let mut v = vec![];
    if host == "0.0.0.0" || host == "::" {
        v.push(format!("http://127.0.0.1:{port}"));
        // primary outbound interface (UDP connect sends no packets)
        if let Ok(s) = std::net::UdpSocket::bind("0.0.0.0:0") {
            if s.connect("8.8.8.8:80").is_ok() {
                if let Ok(a) = s.local_addr() { if !a.ip().is_loopback() { v.push(format!("http://{}:{port}  (LAN)", a.ip())); } }
            }
        }
        if let Ok(h) = std::env::var("COMPUTERNAME") { v.push(format!("http://{}:{port}", h.to_lowercase())); }
    }
    v
}
