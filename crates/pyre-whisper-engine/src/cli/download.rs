use std::fs;
use std::path::Path;
use ureq;
use flate2::read::GzDecoder;
use indicatif::{ProgressBar, ProgressStyle};
use std::io;
use std::process::Command;
use crate::model::registry::find_model;

fn native_agent() -> ureq::Agent {
    use std::sync::OnceLock;
    use ureq::tls::{TlsConfig, TlsProvider};
    static AGENT: OnceLock<ureq::Agent> = OnceLock::new();
    AGENT
        .get_or_init(|| {
            let config = ureq::config::Config::builder()
                .tls_config(
                    TlsConfig::builder()
                        .provider(TlsProvider::NativeTls)
                        .build(),
                )
                .build();
            ureq::Agent::new_with_config(config)
        })
        .clone()
}



fn get_hf_id(name: &str) -> String {
    if let Some(info) = find_model(name) {
        return info.hf_id.to_string();
    }
    
    // Fallback for Whisper naming variations
    match name {
        "tiny.en" => "openai/whisper-tiny.en".to_string(),
        "base.en" => "openai/whisper-base.en".to_string(),
        "small.en" => "openai/whisper-small.en".to_string(),
        "medium.en" => "openai/whisper-medium.en".to_string(),
        "turbo" => "openai/whisper-large-v3-turbo".to_string(),
        n if n.contains('/') => n.to_string(),
        n => n.to_string(),
    }
}

pub fn try_convert_on_the_fly(model_name: &str, verbose: bool, hf_token: Option<String>) -> anyhow::Result<()> {
    let hf_id = get_hf_id(model_name);
    
    let mut cmd = Command::new("python");
    cmd.arg("convert_whisper.py")
        .arg(&hf_id)
        .arg("--name").arg(model_name);
    
    if let Some(ref token) = hf_token {
        cmd.arg("--hf-token").arg(token);
    }

    if verbose {
        cmd.arg("--verbose");
    }

    let status = cmd.status()?;

    if status.success() {
        Ok(())
    } else {
        anyhow::bail!("Conversion script failed for model '{}'.", hf_id)
    }
}

pub fn run_download_logic(
    model_name: &str, 
    base_url_opt: Option<String>, 
    decompress: bool, 
    verbose: bool,
    hf_token: Option<String>
) -> anyhow::Result<()> {
    let base_url = match base_url_opt {
        Some(url) => url,
        None => {
            // Attempt on-the-fly conversion if no URL is provided
            return try_convert_on_the_fly(model_name, verbose, hf_token);
        }
    };

    let output_dir = pyre_whisper_core::model_dir(model_name).display().to_string();
    fs::create_dir_all(&output_dir)?;

    let files = ["config.cfg", "tokenizer.json", "model.bpk"];

    for file in files {
        let download_file = file.to_string();
        let target_file = file.to_string();

        let url = if base_url.contains("huggingface.co") && !base_url.contains("/resolve/main") {
            format!("{}/resolve/main/{}/{}", base_url, model_name, download_file)
        } else {
            format!("{}/{}/{}", base_url, model_name, download_file)
        };

        if verbose { println!("Downloading {}...", url); }
        
        let mut request = native_agent().get(&url);
        if let Some(ref token) = hf_token {
            request = request.header("Authorization", format!("Bearer {token}"));
        }
        
        let response = request.call();

        match response {
            Ok(res) => {
                save_response(res, &output_dir, &target_file, decompress, verbose)?;
            }
            Err(_) => {
                // Try alternative: {model_name}.{ext} inside the folder
                let alt_download_file = if download_file == "config.cfg" {
                    format!("{}.cfg", model_name)
                } else if download_file == "model.bpk" {
                    format!("{}.bpk", model_name)
                } else {
                    download_file.clone()
                };

                if alt_download_file == download_file {
                    // Try .gz if it's .mpk
                    if download_file.ends_with(".bpk") {
                        let gz_url = format!("{}.gz", url);
                        if verbose { println!("Trying .gz: {}...", gz_url); }
                        
                        let mut request = native_agent().get(&gz_url);
                        if let Some(ref token) = hf_token {
                            request = request.header("Authorization", format!("Bearer {token}"));
                        }
                        
                        if let Ok(res) = request.call() {
                            save_response(res, &output_dir, &target_file, decompress, verbose)?;
                            continue;
                        }
                    }
                    continue;
                }

                let alt_url = if base_url.contains("huggingface.co") && !base_url.contains("/resolve/main") {
                    format!("{}/resolve/main/{}/{}", base_url, model_name, alt_download_file)
                } else {
                    format!("{}/{}/{}", base_url, model_name, alt_download_file)
                };

                if verbose { println!("Trying alternative {}...", alt_url); }
                
                let mut request = native_agent().get(&alt_url);
                if let Some(ref token) = hf_token {
                    request = request.header("Authorization", format!("Bearer {token}"));
                }
                
                match request.call() {
                    Ok(res) => {
                        save_response(res, &output_dir, &target_file, decompress, verbose)?;
                    }
                    Err(_) => {
                        // Try .gz for alternative
                        let gz_url = format!("{}.gz", alt_url);
                        if verbose { println!("Trying .gz: {}...", gz_url); }
                        
                        let mut request = native_agent().get(&gz_url);
                        if let Some(ref token) = hf_token {
                            request = request.header("Authorization", format!("Bearer {token}"));
                        }
                        
                        if let Ok(res) = request.call() {
                            save_response(res, &output_dir, &target_file, decompress, verbose)?;
                        } else {
                             // One last try: some repos have files in root
                             let root_url = if base_url.contains("huggingface.co") && !base_url.contains("/resolve/main") {
                                 format!("{}/resolve/main/{}", base_url, alt_download_file)
                             } else {
                                 format!("{}/{}", base_url, alt_download_file)
                             };
                             if verbose { println!("Trying root {}...", root_url); }
                             
                             let mut request = native_agent().get(&root_url);
                             if let Some(ref token) = hf_token {
                                 request = request.header("Authorization", format!("Bearer {token}"));
                             }
                             
                             if let Ok(res) = request.call() {
                                 save_response(res, &output_dir, &target_file, decompress, verbose)?;
                             } else {
                                 let root_gz_url = format!("{}.gz", root_url);
                                 
                                 let mut request = native_agent().get(&root_gz_url);
                                 if let Some(ref token) = hf_token {
                                     request = request.header("Authorization", format!("Bearer {token}"));
                                 }
                                 
                                 if let Ok(res) = request.call() {
                                     save_response(res, &output_dir, &target_file, decompress, verbose)?;
                                 } else {
                                     eprintln!("Failed to download {}", file);
                                 }
                             }
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

fn save_response(
    res: ureq::http::Response<ureq::Body>,
    output_dir: &str,
    target_file: &str,
    decompress: bool,
    verbose: bool,
) -> anyhow::Result<()> {
    let path = Path::new(output_dir).join(target_file);
    let mut file = fs::File::create(&path)?;
    
    let pb = if !verbose {
        let len = res
            .headers()
            .get("content-length")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0);
        let p = ProgressBar::new(len);
        p.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
            .unwrap()
            .progress_chars("#>-"));
        Some(p)
    } else {
        None
    };

    let reader = res.into_body().into_reader();
    let mut reader: Box<dyn io::Read> = if let Some(ref p) = pb {
        Box::new(p.wrap_read(reader))
    } else {
        Box::new(reader)
    };

    if decompress && target_file.ends_with(".gz") {
        let mut decoder = GzDecoder::new(reader);
        io::copy(&mut decoder, &mut file)?;
    } else {
        io::copy(&mut reader, &mut file)?;
    }
    
    if let Some(p) = pb {
        p.finish_and_clear();
    }
    Ok(())
}
