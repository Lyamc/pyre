//! `pyre transcribe file.mp4` demuxes audio locally and sends 16 kHz mono PCM to a pyre server.
//! Video bytes stay on this machine. Symphonia decodes the container; there is no ffmpeg step.
use anyhow::{bail, Context, Result};
use clap::Args;
use pyre_whisper_audio::{decode_to_mono_pcm, Waveform};
use std::io::{Read, Write};
use std::path::PathBuf;

#[derive(Args, Debug)]
pub struct TranscribeArgs {
    /// Audio or video file. Only the audio track is decoded and sent.
    pub file: PathBuf,
    /// Server root, for example http://127.0.0.1:8099
    #[arg(long, default_value = "http://127.0.0.1:8099")]
    pub server: String,
    /// Speech model id. fluid-speech-pro is the multilingual candidate.
    #[arg(long, default_value = "fluid-speech-pro")]
    pub model: String,
    /// Ask for an English translation.
    #[arg(long)]
    pub translate: bool,
    /// whisper (a loaded Whisper model) or llm (the loaded chat model rewrites the transcript).
    #[arg(long = "translate-with", default_value = "whisper")]
    pub translate_with: String,
    /// Fluid Speech Pro clean mode (drops fillers and repeated words).
    #[arg(long)]
    pub clean: bool,
    /// Token beam. Fluid Speech Pro uses 1..=4. Other models keep their own decoder.
    #[arg(long, default_value_t = 1)]
    pub beam: usize,
    /// json, text, srt, or vtt.
    #[arg(long, default_value = "text")]
    pub format: String,
    /// Write the transcript here. Prints it when omitted.
    #[arg(long)]
    pub out: Option<PathBuf>,
    /// Language code. Empty leaves detection to the model.
    #[arg(long, default_value = "")]
    pub language: String,
    /// Allow experimental speech models (Parakeet v3, Nemotron, Fluid Speech Pro).
    #[arg(long, default_value_t = true)]
    pub experimental: bool,
}

pub fn run(args: TranscribeArgs) -> Result<()> {
    let file_bytes = std::fs::metadata(&args.file).with_context(|| format!("stat {}", args.file.display()))?.len();
    let wave = decode_to_mono_pcm(&args.file).with_context(|| format!("decode {}", args.file.display()))?;
    let wave = wave.ensure_sample_rate(16_000).context("resample to 16 kHz")?;
    let pcm_bytes = wave.samples.len() * 4;
    eprintln!(
        "decoded {} ({file_bytes} bytes on disk) to {} of 16 kHz mono PCM ({:.1} s). video bytes are not uploaded.",
        args.file.display(),
        human(pcm_bytes as u64),
        wave.duration_secs(),
    );
    let body = post_chunks(&args, &wave)?;
    if let Some(path) = &args.out {
        std::fs::write(path, &body).with_context(|| format!("write {}", path.display()))?;
        eprintln!("wrote {}", path.display());
    } else {
        print!("{body}");
        if !body.ends_with('\n') { println!(); }
    }
    Ok(())
}

fn post_chunks(args: &TranscribeArgs, wave: &Waveform) -> Result<String> {
    let session = format!("cli-{}-{}", std::process::id(), wave.samples.len());
    let base = args.server.trim_end_matches('/');
    let agent: ureq::Agent = ureq::Agent::config_builder().http_status_as_error(false).timeout_global(None).build().into();
    let chunk_samples = 16_000 * 8;
    let mut sent = 0usize;
    while sent < wave.samples.len() {
        let end = (sent + chunk_samples).min(wave.samples.len());
        let last = end == wave.samples.len();
        let mut raw = Vec::with_capacity((end - sent) * 4);
        for s in &wave.samples[sent..end] {
            raw.extend_from_slice(&s.to_le_bytes());
        }
        let url = format!(
            "{base}/v1/audio/transcriptions/chunks?session={session}&final={final}&model={model}&response_format={fmt}&beam_size={beam}&clean_mode={clean}&experimental={exp}&task={task}&translate_with={via}&language={lang}&timestamps=true",
            final = if last { "1" } else { "0" },
            model = enc(&args.model),
            fmt = enc(&args.format),
            beam = args.beam,
            clean = if args.clean { "1" } else { "0" },
            exp = if args.experimental { "1" } else { "0" },
            task = if args.translate { "translations" } else { "transcriptions" },
            via = enc(&args.translate_with),
            lang = enc(&args.language),
        );
        eprint!("\rsending {:.0}%", end as f64 / wave.samples.len() as f64 * 100.0);
        let _ = std::io::stderr().flush();
        let mut resp = agent.post(&url)
            .header("content-type", "application/octet-stream")
            .send(&raw)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let status = resp.status();
        let mut text = String::new();
        resp.body_mut().as_reader().read_to_string(&mut text).context("read response")?;
        if !status.is_success() {
            eprintln!();
            bail!("HTTP {status}: {text}");
        }
        if last {
            eprintln!();
            return Ok(text);
        }
        sent = end;
    }
    bail!("empty audio")
}

fn enc(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' => out.push(b as char),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

fn human(n: u64) -> String {
    if n >= 1_048_576 { format!("{:.1} MiB", n as f64 / 1_048_576.0) } else { format!("{:.1} KiB", n as f64 / 1024.0) }
}
