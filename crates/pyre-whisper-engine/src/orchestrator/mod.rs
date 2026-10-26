use crate::model::load_model;
use crate::token::Language;
use crate::transcribe::{waveform_to_text, DecodeTask, TranscriptSegment};
use crate::cli::CommonArgs;
use std::time::Instant;

pub struct Orchestrator {
    device: burn::tensor::Device,
    verbose: bool,
    debug: bool,
}

impl Orchestrator {
    pub fn new(device: burn::tensor::Device, verbose: bool, debug: bool) -> Self {
        Self { device, verbose, debug }
    }

    pub fn run_waveform(
        &self,
        waveform: Vec<f32>,
        sample_rate: usize,
        expert_args: CommonArgs,
        scout_model_name: &str,
        decode_task: DecodeTask,
    ) -> anyhow::Result<(String, Vec<TranscriptSegment>)> {
        let start_time = Instant::now();
        if !expert_args.quiet {
            println!("Starting orchestration pipeline...");
            println!(
                "Step 1: Scout segmentation using '{}'",
                scout_model_name
            );
        }

        let (scout_bpe, _scout_config, scout_model) =
            load_model(scout_model_name, &self.device, self.verbose).map_err(|e| {
                anyhow::anyhow!("Failed to load scout model '{}': {}", scout_model_name, e)
            })?;

        let (_scout_text, scout_segments) = waveform_to_text(
            &scout_model,
            scout_model_name,
            &scout_bpe,
            Language::English,
            "en",
            waveform.clone(),
            None,
            sample_rate,
            false,
            self.verbose,
            self.debug,
            true,
            true,
            "Scout".to_string(),
            1,
            expert_args.max_tokens,
            expert_args.padding,
            None,
            Some(decode_task),
        )
        .map_err(|e| anyhow::anyhow!("Scout failed: {}", e))?;

        if !expert_args.quiet {
            println!(
                "Scout found {} speech segments in {:?}",
                scout_segments.len(),
                start_time.elapsed()
            );
        }

        if scout_segments.is_empty() {
            return Ok(("".to_string(), vec![]));
        }

        if !expert_args.quiet {
            println!(
                "Step 2: Expert transcription using '{}'",
                expert_args.model
            );
        }

        drop(scout_model);

        let (expert_bpe, _expert_config, expert_model) =
            load_model(&expert_args.model, &self.device, self.verbose).map_err(|e| {
                anyhow::anyhow!(
                    "Failed to load expert model '{}': {}",
                    expert_args.model,
                    e
                )
            })?;
        let expert_lang = crate::cli::parse_language(&expert_args.lang);

        let mut final_segments = Vec::new();
        let mut final_text = String::new();

        for (i, segment) in scout_segments.iter().enumerate() {
            let start = (segment.start - 0.2).max(0.0);
            let end = segment.end + 0.2;
            let duration = end - start;

            if duration < 0.2 {
                continue;
            }

            if self.verbose || !expert_args.quiet {
                println!(
                    "  Expert segment {}/{}: {:.2}s - {:.2}s",
                    i + 1,
                    scout_segments.len(),
                    start,
                    end
                );
            }

            let start_sample = (start * sample_rate as f64) as usize;
            let end_sample = (end * sample_rate as f64).min(waveform.len() as f64) as usize;

            if start_sample >= waveform.len() {
                break;
            }
            let segment_waveform = waveform[start_sample..end_sample].to_vec();

            let (seg_text, _) = waveform_to_text(
                &expert_model,
                &expert_args.model,
                &expert_bpe,
                expert_lang,
                &expert_args.lang,
                segment_waveform,
                None,
                sample_rate,
                false,
                self.verbose,
                self.debug,
                false,
                false,
                "Expert".to_string(),
                expert_args.beam_size,
                expert_args.max_tokens,
                expert_args.padding,
                None,
                Some(decode_task),
            )
            .map_err(|e| anyhow::anyhow!("Expert failed on segment {}: {}", i, e))?;

            if !seg_text.trim().is_empty() {
                final_segments.push(TranscriptSegment {
                    start,
                    end,
                    text: seg_text.clone(),
                    diarization: None,
                    location: None,
                });
                if !final_text.is_empty() {
                    final_text.push(' ');
                }
                final_text.push_str(&seg_text);

                if !expert_args.quiet {
                    println!("  > {}", seg_text);
                }
            }
        }

        if !expert_args.quiet {
            println!("Orchestration complete in {:?}", start_time.elapsed());
        }

        Ok((final_text, final_segments))
    }
}