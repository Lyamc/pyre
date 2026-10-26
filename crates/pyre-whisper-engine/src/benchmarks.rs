#[cfg(test)]
mod tests {
    use crate::model::registry::MODEL_REGISTRY;
    use crate::model::load_model;
    use crate::transcribe::waveform_to_text;
    use crate::token::Language;
    use crate::cli::transcribe::load_audio_waveform;
        use std::path::Path;
    use std::time::Instant;
    use sysinfo::{System, ProcessRefreshKind, Pid, ProcessesToUpdate};
    use std::fs;

    fn get_dir_size<P: AsRef<Path>>(path: P) -> u64 {
        let mut size = 0;
        if let Ok(entries) = fs::read_dir(path) {
            for entry in entries.flatten() {
                if let Ok(meta) = entry.metadata() {
                    if meta.is_file() {
                        size += meta.len();
                    } else if meta.is_dir() {
                        size += get_dir_size(entry.path());
                    }
                }
            }
        }
        size
    }

    #[test]
    fn test_all_models_performance() {
        let mut sys = System::new_all();
        let pid = Pid::from(std::process::id() as usize);
        
        // Detect terminal width or use default
        let term_width = termsize::get().map(|t| t.cols as usize).unwrap_or(120);
        let static_cols_width = 25 + 12 + 12 + 12 + 12 + 10 + 5; // Fixed column widths + separators
        let preview_width = if term_width > static_cols_width {
            (term_width - static_cols_width).max(32)
        } else {
            32
        };

        println!("\n{:<25} | {:<10} | {:<10} | {:<10} | {:<10} | {:<8} | {:<width$}", 
            "Model", "Size (MB)", "Time (s)", "Mem (MB)", "CPU (%)", "Verify", "Preview", width = preview_width);
        println!("{:-<25}-|-{:-<10}-|-{:-<10}-|-{:-<10}-|-{:-<10}-|-{:-<8}-|-{:-<width$}", 
            "", "", "", "", "", "", "", width = preview_width);

        let audio_path = Path::new("audio.wav");
        if !audio_path.exists() {
            panic!("audio.wav not found! Benchmark requires a test audio file.");
        }

        let device = burn::tensor::Device::default();
        let mut pass = 0;
        let mut fail = 0;
        let mut skip = 0;
        let hf_token = std::env::var("HF_TOKEN").ok();

        for model_info in MODEL_REGISTRY {
            let model_dir = pyre_whisper_core::resolve_model_dir(model_info.name).display().to_string();
            if !Path::new(&model_dir).exists() {
                println!("Model '{}' not found. Attempting to download and convert...", model_info.name);
                if let Err(e) = crate::cli::download::run_download_logic(model_info.name, None, true, false, hf_token.clone()) {
                    println!("{:<25} | {:<10} | {:<10} | {:<10} | {:<10} | {:<8} | {:<width$}", 
                        model_info.name, "-", "-", "-", "-", "DL-FAIL", format!("DL Error: {}", e), width = preview_width);
                    fail += 1;
                    continue;
                }
            }

            let model_size_mb = get_dir_size(&model_dir) as f64 / 1024.0 / 1024.0;

            // Test basic loading for all categories
            let start = Instant::now();
            let panic_result = std::panic::catch_unwind(|| {
                load_model(model_info.name, &device, false)
            });

            if panic_result.is_err() {
                println!("{:<25} | {:<10.2} | {:<10} | {:<10} | {:<10} | {:<8} | {:<width$}", 
                    model_info.name, model_size_mb, "-", "-", "-", "PANIC", "Model load panicked", width = preview_width);
                fail += 1;
                continue;
            }
            
            let load_result = panic_result.unwrap();
            if let Err(e) = load_result {
                println!("{:<25} | {:<10.2} | {:<10} | {:<10} | {:<10} | {:<8} | {:<width$}", 
                    model_info.name, model_size_mb, "-", "-", "-", "L-FAIL", format!("Load error: {}", e), width = preview_width);
                fail += 1;
                continue;
            }
            let (bpe, _config, model) = load_result.unwrap();

            // ASR specific verification
            if matches!(model_info.category, crate::model::registry::ModelCategory::Asr) {
                let (waveform, sample_rate) = match load_audio_waveform(audio_path, None, None) {
                    Ok(res) => res,
                    Err(e) => {
                        println!("{:<25} | {:<10.2} | {:<10} | {:<10} | {:<10} | {:<8} | {:<width$}", 
                            model_info.name, model_size_mb, "-", "-", "-", "A-FAIL", format!("Audio err: {}", e), width = preview_width);
                        fail += 1;
                        continue;
                    }
                };

                sys.refresh_all();
                
                let transcription_result = waveform_to_text(
                    &model,
                    model_info.name,
                    &bpe,
                    Language::English,
                    "en",
                    waveform,
                    None,
                    sample_rate,
                    false, 
                    false, 
                    false, 
                    true,  
                    false, 
                    "Benchmark".to_string(),
                    1,     
                    448,   
                    200,   
                    None,
                    None,
                );

                let duration = start.elapsed().as_secs_f64();

                sys.refresh_processes_specifics(
                    ProcessesToUpdate::Some(&[pid]), 
                    true, 
                    ProcessRefreshKind::nothing().with_memory().with_cpu()
                );
                let process = sys.process(pid).expect("Failed to get process info");
                let mem_mb = process.memory() as f64 / 1024.0 / 1024.0;
                let cpu_pct = process.cpu_usage();

                let (verify_status, preview) = if let Ok((text, _)) = transcription_result {
                    let expected_words = ["machine", "learning", "properly"];
                    let normalized = text.to_lowercase();
                    
                    let status = if expected_words.iter().all(|w| normalized.contains(w)) { "PASS" } else { "FAIL" };
                    let mut p = text.trim().replace("\n", " ").replace("\r", "");
                    if p.len() > preview_width {
                        p.truncate(preview_width - 3);
                        p.push_str("...");
                    }
                    (status, p)
                } else {
                    ("ERROR", "Transcription failed".to_string())
                };

                if verify_status == "PASS" { pass += 1; } else { fail += 1; }

                println!("{:<25} | {:<10.2} | {:<10.2} | {:<10.2} | {:<10.2} | {:<8} | {:<width$}", 
                    model_info.name, model_size_mb, duration, mem_mb, cpu_pct, verify_status, preview, width = preview_width);
            } else {
                // Non-ASR models: PASS if they loaded successfully
                println!("{:<25} | {:<10.2} | {:<10.2} | {:<10} | {:<10} | {:<8} | {:<width$}", 
                    model_info.name, model_size_mb, start.elapsed().as_secs_f64(), "-", "-", "PASS", "Model loaded (VAD/Diarization)", width = preview_width);
                pass += 1;
            }
            
            drop(model);
            drop(bpe);
        }
        println!("\nBenchmark complete. Pass: {}, Fail: {}, Skip: {}", pass, fail, skip);
    }
}
