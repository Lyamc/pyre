use std::sync::mpsc;
use std::time::Instant;
use std::iter;
use std::fs;
use std::path::{Path, PathBuf};
use std::io::Read;
use crate::model::{load_model, Model};
use crate::token::Gpt2Tokenizer;
use crate::transcribe::{waveform_to_text, get_output_path, format_srt, parse_time};
use crate::cli::{CommonArgs, get_device, get_backend_info, parse_language};

pub fn check_ffmpeg() -> bool {
    std::process::Command::new("ffmpeg")
        .arg("-version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok()
}

pub fn load_audio_waveform(filename: &Path, start: Option<f64>, duration: Option<f64>) -> anyhow::Result<(Vec<f32>, usize)> {
    if !check_ffmpeg() {
        anyhow::bail!("ffmpeg not found in PATH. Please install ffmpeg to process audio files.");
    }

    let mut args = Vec::new();
    if let Some(s) = start {
        args.push("-ss".to_string());
        args.push(s.to_string());
    }
    args.extend(["-i".to_string(), filename.to_str().unwrap().to_string()]);
    if let Some(d) = duration {
        args.push("-t".to_string());
        args.push(d.to_string());
    }
    args.extend([
        "-f".to_string(), "f32le".to_string(),
        "-ar".to_string(), "16000".to_string(),
        "-ac".to_string(), "1".to_string(),
        "-".to_string()
    ]);

    let output = std::process::Command::new("ffmpeg")
        .args(&args)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .output()?;

    if !output.status.success() {
        anyhow::bail!("ffmpeg failed to process the audio file: {}", filename.display());
    }

    let mut floats = Vec::new();
    let mut cursor = std::io::Cursor::new(output.stdout);
    let mut buffer = [0u8; 4];
    while cursor.read_exact(&mut buffer).is_ok() {
        floats.push(f32::from_le_bytes(buffer));
    }

    Ok((floats, 16000))
}

pub fn load_stereo_audio_waveform(filename: &Path, start: Option<f64>, duration: Option<f64>) -> anyhow::Result<Vec<[f32; 2]>> {
    if !check_ffmpeg() {
        anyhow::bail!("ffmpeg not found in PATH. Please install ffmpeg to process audio files.");
    }

    let mut args = Vec::new();
    if let Some(s) = start {
        args.push("-ss".to_string());
        args.push(s.to_string());
    }
    args.extend(["-i".to_string(), filename.to_str().unwrap().to_string()]);
    if let Some(d) = duration {
        args.push("-t".to_string());
        args.push(d.to_string());
    }
    args.extend([
        "-f".to_string(), "f32le".to_string(),
        "-ar".to_string(), "16000".to_string(),
        "-ac".to_string(), "2".to_string(),
        "-".to_string()
    ]);

    let output = std::process::Command::new("ffmpeg")
        .args(&args)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .output()?;

    if !output.status.success() {
        anyhow::bail!("ffmpeg failed to process the audio file: {}", filename.display());
    }

    let mut stereo_samples = Vec::new();
    let mut cursor = std::io::Cursor::new(output.stdout);
    let mut buffer = [0u8; 8];
    while cursor.read_exact(&mut buffer).is_ok() {
        let left = f32::from_le_bytes([buffer[0], buffer[1], buffer[2], buffer[3]]);
        let right = f32::from_le_bytes([buffer[4], buffer[5], buffer[6], buffer[7]]);
        stereo_samples.push([left, right]);
    }

    Ok(stereo_samples)
}

pub fn run_transcribe_file(
    common: &CommonArgs, 
    input: &Path, 
    output_arg: &Option<PathBuf>, 
    format: &str, 
    model: &Model, 
    bpe: &Gpt2Tokenizer, 
    device: &burn::tensor::Device,
    start_time: &Option<String>,
    duration_time: &Option<String>,
    end_time: &Option<String>,
    chunk_size: Option<f64>,
) {
    if !common.quiet {
        println!("Processing file: {}", input.display());
    }

    let start_sec = start_time.as_ref().and_then(|s| parse_time(s).ok());
    let end_sec = end_time.as_ref().and_then(|s| parse_time(s).ok());
    let mut duration_sec = duration_time.as_ref().and_then(|s| parse_time(s).ok());
    if let Some(e) = end_sec {
        let s = start_sec.unwrap_or(0.0);
        if e > s { duration_sec = Some(e - s); }
    }

    let (waveform, sample_rate) = match load_audio_waveform(input, start_sec, duration_sec) {
        Ok(res) => res,
        Err(e) => {
            eprintln!("Failed to load audio file {}: {}", input.display(), e);
            return;
        }
    };

    let stereo_audio = load_stereo_audio_waveform(input, start_sec, duration_sec).ok();

    let backend_info = get_backend_info(device);
    let lang = parse_language(&common.lang);
    let chunk_size_samples = chunk_size.map(|s| (s * 16000.0) as usize);

    let model_name = common.model.as_str();
    if let Err(e) = crate::transcribe::experimental_gate(model_name, waveform.len(), sample_rate) {
        eprintln!("{e}");
        std::process::exit(2);
    }
    let (text, segments) = match waveform_to_text(
        model, 
        model_name,
        bpe, 
        lang,
        &common.lang,
        waveform,
        stereo_audio,
        sample_rate, 
        false, 
        common.verbose, 
        common.debug,
        common.quiet, 
        format == "srt", 
        format!("Transcribe | {}", backend_info),
        common.beam_size,
        common.max_tokens,
        common.padding,
        chunk_size_samples,
        None,
    ) {
        Ok(res) => res,
        Err(e) => {
            eprintln!("Error transcribing {}: {}", input.display(), e);
            return;
        }
    };

    let output_file = get_output_path(input, output_arg, format);
    let content = if format == "srt" { format_srt(&segments) } else { text.clone() };

    if let Err(e) = fs::write(&output_file, content) {
        eprintln!("Failed to write output to {}: {}", output_file.display(), e);
    } else if !common.quiet {
        println!("Saved transcription to {}", output_file.display());
    }

    if common.verify {
        let expected = "hello i am the whisper machine learning model if you see this as text then i am working properly";
        let normalized = text.to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace())
            .collect::<String>()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");

        if normalized.contains(expected) {
            println!("Verification: SUCCESS - Model '{}' is working properly.", common.model);
        } else {
            println!("Verification: FAILED - Output does not match expected test string.");
            if common.verbose {
                println!("  Expected (subset): {}", expected);
                println!("  Actual (normalized): {}", normalized);
            }
            std::process::exit(1);
        }
    }
}

pub fn run_stream_logic(args: CommonArgs, backend_name: &str) {
    let device = get_device(&args.device);
    let (bpe, _model_config, model) = load_model(&args.model, &device, args.verbose)
        .expect("Failed to load model");
    let (sender, receiver) = mpsc::channel();
    let sender1 = sender.clone();
    std::thread::spawn(move || { crate::cli::audio::record_audio(sender1) });
    let backend_info = get_backend_info(&device);

    for (i, _) in iter::repeat(()).enumerate() {
        let audio_data_vectors = match receiver.recv() {
            Ok(data) => data,
            Err(e) => { eprintln!("Error receiving data: {}", e); std::process::exit(1); }
        };
        if args.verbose { println!("Received audio segment {}, length: {}", i, audio_data_vectors.len()); }
        let speech_segment_f32: Vec<f32> = audio_data_vectors.into_iter().map(|x| x as f32 / 32767.0).collect();
        if let Err(e) = crate::transcribe::experimental_gate(&args.model, speech_segment_f32.len(), 16000) {
            eprintln!("{e}");
            std::process::exit(2);
        }
        let start_time = Instant::now();
        let (text, _segments) = match waveform_to_text(
            &model, 
            &args.model,
            &bpe, 
            parse_language(&args.lang),
            &args.lang,
            speech_segment_f32,
            None,
            16000, 
            true, 
            args.verbose, 
            args.debug,
            args.quiet, 
            false, 
            format!("{} | {}", backend_name, backend_info),
            args.beam_size,
            args.max_tokens,
            args.padding,
            None,
            None,
        ) {
            Ok(res) => res,
            Err(e) => { eprintln!("Error during transcription: {}", e); continue; }
        };
        println!("\nText: {}, Iteration: {}, Time:{:?}", text, i, start_time.elapsed());
    }
}
