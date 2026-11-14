use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

use burn::tensor::Device;
use pyre_whisper_core::{
    SpeechTask, TaskOptions, TranscriptResult, TranscriptSegment, WhisburnError, WhisburnResult,
};

use crate::cli::{get_backend_info, get_device, parse_language, CommonArgs};
use crate::model::load_model;
use crate::model::Model;
use crate::orchestrator::Orchestrator;
use crate::token::{Gpt2Tokenizer, Language};
use crate::model::diarize::{is_diarize_model, run_diarize};
use crate::model::vad::{is_vad_model, run_vad};
use crate::transcribe::{waveform_to_text, DecodeTask};

struct LoadedModel {
    bpe: Gpt2Tokenizer,
    model: Model,
}

/// Burn dispatches on `Device` at runtime, so one cache serves every backend
/// (CUDA, wgpu, flex CPU) instead of one generic instantiation per backend.
pub struct InferenceRuntime {
    device: Device,
    cache: Mutex<HashMap<String, LoadedModel>>,
    verbose: bool,
    debug: bool,
    warmed: Mutex<HashSet<String>>,
}

fn summarize_on(
    model_name: &str,
    device: &Device,
    text: &str,
    verbose: bool,
    progress: Option<crate::model::qwen3::summarize::SummarizeProgress>,
) -> WhisburnResult<String> {
    let _ = verbose;
    tracing::info!(model = model_name, chars = text.len(), "loading offline summarizer");
    let (tokenizer, thinker) =
        crate::model::qwen3::summarize::load_summarizer(model_name, device, verbose)?;
    tracing::info!(model = model_name, "summarizer ready");
    crate::model::qwen3::summarize::summarize_with_thinker(&thinker, &tokenizer, text, device, progress)
}

impl InferenceRuntime {
    /// `device`: `None` (default wgpu adapter), `"cpu"`, `"cuda[:n]"`, or a discrete GPU index.
    pub fn new(device: Option<String>, verbose: bool, debug: bool) -> Self {
        Self::with_device(get_device(&device), verbose, debug)
    }

    /// Run on an explicit Burn device (pyre passes the device it selected for the model).
    pub fn with_device(device: Device, verbose: bool, debug: bool) -> Self {
        Self { device, cache: Mutex::new(HashMap::new()), verbose, debug, warmed: Mutex::new(HashSet::new()) }
    }

    pub fn backend_name(&self) -> String {
        get_backend_info(&self.device)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Offline English summarization with Qwen3-0.6B. Clears the ASR GPU cache first
    /// so a 4 GB card can load the small text model.
    pub fn summarize_text(
        &self,
        text: &str,
        model_name: &str,
        progress: Option<crate::model::qwen3::summarize::SummarizeProgress>,
    ) -> WhisburnResult<String> {
        {
            let mut warmed = self.warmed.lock().expect("warmed lock");
            warmed.clear();
        }
        self.cache.lock().expect("cache lock").clear();
        summarize_on(model_name, &self.device, text, self.verbose, progress)
    }

    pub fn loaded_models(&self) -> Vec<String> {
        let mut names: Vec<String> = self.cache.lock().expect("cache lock").keys().cloned().collect();
        names.sort();
        names
    }

    pub fn warm_model(&self, model_name: &str) -> WhisburnResult<()> {
        {
            let warmed = self.warmed.lock().expect("warmed lock");
            if warmed.contains(model_name) {
                return Ok(());
            }
        }
        if is_vad_model(model_name) || is_diarize_model(model_name) {
            self.warmed
                .lock()
                .expect("warmed lock")
                .insert(model_name.to_string());
            return Ok(());
        }
        Self::ensure_cached(model_name, &self.device, &self.cache, self.verbose)?;
        self.warmed
            .lock()
            .expect("warmed lock")
            .insert(model_name.to_string());
        Ok(())
    }

    fn ensure_cached(
        model_name: &str,
        device: &burn::tensor::Device,
        cache: &Mutex<HashMap<String, LoadedModel>>,
        verbose: bool,
    ) -> WhisburnResult<()> {
        let mut cache = cache.lock().expect("cache lock");
        if cache.contains_key(model_name) {
            return Ok(());
        }
        // A 12 GB GPU cannot hold several Whisper/Qwen checkpoints at once.
        // The HTTP server processes one job at a time, so keep only the active model.
        cache.clear();
        let (bpe, _config, model) = load_model(model_name, device, verbose)
            .map_err(|e| WhisburnError::Model(e.to_string()))?;
        cache.insert(model_name.to_string(), LoadedModel { bpe, model });
        Ok(())
    }

    pub fn transcribe_waveform(
        &self,
        model_name: &str,
        samples: Vec<f32>,
        sample_rate: usize,
        options: &TaskOptions,
    ) -> WhisburnResult<TranscriptResult> {
        if is_vad_model(model_name) {
            return run_vad(model_name, &samples, sample_rate)
                .map_err(|e| WhisburnError::Inference(e));
        }
        if is_diarize_model(model_name) {
            return run_diarize(model_name, &samples, sample_rate)
                .map_err(|e| WhisburnError::Inference(e));
        }

        let decode_task = decode_task_for(options.task);

        if options.orchestrate {
            return self.transcribe_orchestrated(model_name, samples, sample_rate, options, decode_task);
        }

        let lang = map_language(options.language.as_str());
        let backend_info = self.backend_name();

        Self::transcribe_with(
            model_name,
            &self.device,
            &self.cache,
                samples,
                sample_rate,
                options,
                lang,
                decode_task,
                backend_info,
                self.verbose,
                self.debug,
            )
    }

    fn transcribe_with(
        model_name: &str,
        device: &burn::tensor::Device,
        cache: &Mutex<HashMap<String, LoadedModel>>,
        samples: Vec<f32>,
        sample_rate: usize,
        options: &TaskOptions,
        lang: Language,
        decode_task: DecodeTask,
        backend_info: String,
        verbose: bool,
        debug: bool,
    ) -> WhisburnResult<TranscriptResult> {
        Self::ensure_cached(model_name, device, cache, verbose)?;
        let cache = cache.lock().expect("cache lock");
        let loaded = cache.get(model_name).expect("model inserted above");
        let beam = if model_name.contains("fluid-speech") {
            options.beam_size.clamp(1, 4)
        } else {
            1
        };
        let _pass = crate::model::fluid::PassGuard::new(options.clean_mode, beam);
        let (text, segments) = waveform_to_text(
            &loaded.model,
            model_name,
            &loaded.bpe,
            lang,
            options.language.as_str(),
            samples,
            None,
            sample_rate,
            false,
            verbose,
            debug,
            true,
            options.include_timestamps,
            backend_info,
            options.beam_size,
            options.max_tokens,
            options.padding,
            None,
            Some(decode_task),
        )
        .map_err(|e| WhisburnError::Inference(e.to_string()))?;

        Ok(TranscriptResult {
            text,
            segments: segments
                .into_iter()
                .map(|s| TranscriptSegment {
                    start: s.start,
                    end: s.end,
                    text: s.text,
                    speaker: s.diarization,
                })
                .collect(),
            language: Some(options.language.as_str().to_string()),
            model: model_name.to_string(),
            task: options.task.as_str().to_string(),
        })
    }

    fn transcribe_orchestrated(
        &self,
        model_name: &str,
        samples: Vec<f32>,
        sample_rate: usize,
        options: &TaskOptions,
        decode_task: DecodeTask,
    ) -> WhisburnResult<TranscriptResult> {
        let scout = options
            .scout_model
            .as_deref()
            .unwrap_or("tiny_en");

        let common = CommonArgs {
            model: model_name.to_string(),
            lang: options.language.as_str().to_string(),
            verbose: self.verbose,
            debug: self.debug,
            quiet: true,
            verify: false,
            hf_token: None,
            beam_size: options.beam_size,
            max_tokens: options.max_tokens,
            padding: options.padding,
            device: None,
        };

        let device = self.device.clone();
        let orchestrator = Orchestrator::new(device, self.verbose, self.debug);
        let (text, segments) = orchestrator
            .run_waveform(samples, sample_rate, common, scout, decode_task)
            .map_err(|e| WhisburnError::Inference(e.to_string()))?;

        Ok(TranscriptResult {
            text,
            segments: segments
                .into_iter()
                .map(|s| TranscriptSegment {
                    start: s.start,
                    end: s.end,
                    text: s.text,
                    speaker: s.diarization,
                })
                .collect(),
            language: Some(options.language.as_str().to_string()),
            model: model_name.to_string(),
            task: options.task.as_str().to_string(),
        })
    }
}

fn decode_task_for(task: SpeechTask) -> DecodeTask {
    match task {
        SpeechTask::Translate => DecodeTask::Translate,
        _ => DecodeTask::Transcribe,
    }
}

fn map_language(code: &str) -> Language {
    parse_language(code)
}