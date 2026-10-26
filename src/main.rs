mod backend;
mod engine;
mod cli_models;
mod storage;
mod store;
mod model;
mod sampler;
mod server;
mod template;

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
}

#[derive(clap::Args, Debug)]
pub struct ServeArgs {
    /// Text-generation model: alias, store name, HF repo id[@rev] or folder
    #[arg(short = 'm', long = "model")]
    pub model: Vec<String>,
    /// Extra models as kind=ref (kind: llm|image|stt|tts); repeatable
    #[arg(long = "load", value_name = "KIND=REF")]
    pub load: Vec<String>,
    #[arg(long = "image-model")]
    pub image_model: Option<String>,
    #[arg(long = "stt-model")]
    pub stt_model: Option<String>,
    #[arg(long = "tts-model")]
    pub tts_model: Option<String>,
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
    /// Model name reported by /v1/models (defaults to -m)
    #[arg(short = 'a', long)]
    pub alias: Option<String>,
    /// Load + one greedy generation, print it and exit (for benchmarking / sanity checks)
    #[arg(long)]
    pub prompt: Option<String>,
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
    tracing_subscriber::fmt().with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| "info,wgpu_core=warn,wgpu_hal=warn,naga=warn,cubecl=warn,cubecl_runtime=warn,pliron=warn,cubecl_llvm=warn,xet_client=warn".into())).init();
    let cli = Cli::parse();
    let ctx = storage::Ctx::new(&cli.storage)?;
    match cli.cmd {
        Cmd::Models { cmd } => cli_models::run(&ctx, cmd),
        Cmd::Serve(a) => serve(&ctx, a),
    }
}

fn serve(sctx: &storage::Ctx, args: ServeArgs) -> Result<()> {
    let mref = args.model.first().cloned().context("serve needs -m <model>")?;
    let (dir, src) = sctx.resolver().resolve(&mref)?;
    tracing::info!("resolved {mref} -> {} (via {src:?})", dir.display());
    let cfg: LlamaConfig = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).context("config.json")?)?;
    let arch = cfg.arch()?;
    let (device, fdt, bname) = backend::select(&args.backend, &args.dtype, cfg.torch_dtype.as_deref())?;
    let ctx = args.ctx_size.min(cfg.max_position_embeddings);
    tracing::info!("model {} arch {:?} | backend {bname} dtype {fdt:?} | ctx {ctx}", dir.display(), arch);
    let t = std::time::Instant::now();
    let ws = model::loader::Weights::open(&dir, device.clone(), fdt.into())?;
    let llama = model::llama::Llama::load(cfg.clone(), &ws, device.clone(), fdt, ctx)?;
    drop(ws);
    tracing::info!("weights loaded in {:.1}s", t.elapsed().as_secs_f64());

    let tok = Arc::new(tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?);
    let template = template::ChatTemplate::from_model_dir(&dir)?;
    let mut eos = cfg.eos_ids();
    if let Ok(g) = std::fs::read_to_string(dir.join("generation_config.json")) {
        if let Ok(g) = serde_json::from_str::<serde_json::Value>(&g) { eos.extend(ids(&g.get("eos_token_id").cloned())); }
    }
    for t in [template.eos_token.as_str(), "<|im_end|>", "<|eot_id|>", "<|endoftext|>", "<|end_of_text|>"] {
        if let Some(id) = tok.token_to_id(t) { if t == template.eos_token || t == "<|im_end|>" || t == "<|eot_id|>" { eos.push(id); } }
    }
    eos.sort(); eos.dedup();
    tracing::info!("eos ids {eos:?}");

    if let Some(p) = &args.prompt {
        return backend::oneshot(&llama, &tok, &template, &eos, p);
    }

    let engine = engine::spawn(llama, tok.clone(), eos);
    let st = Arc::new(server::AppState {
        engine, tok, template,
        model_id: args.alias.clone().unwrap_or_else(|| mref.clone()),
        model_path: dir.display().to_string(),
        n_ctx: ctx, backend: bname.to_string(), dtype: format!("{fdt:?}"),
        api_key: args.api_key.clone(), default_max_tokens: args.n_predict,
    });
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async move {
        let addr = format!("{}:{}", args.host, args.port);
        let l = tokio::net::TcpListener::bind(&addr).await?;
        tracing::info!("pyre listening on http://{addr}");
        axum::serve(l, server::router(st)).await?;
        Ok(())
    })
}
