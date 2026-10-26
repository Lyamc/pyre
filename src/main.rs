mod backend;
mod engine;
mod hub;
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
#[command(name = "pyre", version, about = "llama-server-style LLM inference server powered by Burn")]
pub struct Args {
    /// Model directory (HF layout) or Hugging Face repo id, e.g. Qwen/Qwen2.5-0.5B-Instruct
    #[arg(short = 'm', long)]
    model: String,
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value_t = 8080)]
    port: u16,
    /// cuda | wgpu | vulkan | cpu (flex) | ndarray
    #[arg(short = 'b', long, default_value = "cuda")]
    backend: String,
    /// auto | f32 | f16 | bf16
    #[arg(long, default_value = "auto")]
    dtype: String,
    #[arg(short = 'c', long = "ctx-size", default_value_t = 4096)]
    ctx_size: usize,
    /// Default max tokens when the request sets none
    #[arg(short = 'n', long = "n-predict", default_value_t = 512)]
    n_predict: usize,
    /// Require `Authorization: Bearer <key>`
    #[arg(long = "api-key", env = "PYRE_API_KEY")]
    api_key: Option<String>,
    /// Model name reported by /v1/models (defaults to -m)
    #[arg(short = 'a', long)]
    alias: Option<String>,
    /// hf-hub cache directory
    #[arg(long = "hf-cache")]
    hf_cache: Option<PathBuf>,
    /// Load + one greedy generation, print it and exit (for benchmarking / sanity checks)
    #[arg(long)]
    prompt: Option<String>,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| "info,wgpu_core=warn,wgpu_hal=warn,naga=warn,cubecl=warn,cubecl_runtime=warn,pliron=warn,cubecl_llvm=warn".into())).init();
    let args = Args::parse();
    let dir = hub::resolve(&args.model, args.hf_cache.as_deref())?;
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
        model_id: args.alias.clone().unwrap_or_else(|| args.model.clone()),
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
