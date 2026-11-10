//! What a loaded or stored model can actually do.
//!
//! Flags describe the endpoints this process implements for that model. They are not a guess
//! from the model family name: Qwen-Next (the `qwen4exp` GGUF engine) is a text model with tool
//! calls, and it does not serve embeddings, images, or audio. Image, speech, tagging, and
//! diarization stay on the models that implement them.
use crate::store::Kind;
use serde_json::{json, Value};
use std::path::Path;

/// Nemotron-3-Diarization folders have no `Kind` of their own (they load via `--load diar=`).
pub fn diarizer_dir(dir: &Path) -> bool {
    let Ok(s) = std::fs::read_to_string(dir.join("config.json")) else { return false };
    let Ok(j) = serde_json::from_str::<Value>(&s) else { return false };
    j.pointer("/head_config/num_speakers").is_some() && j.pointer("/audio_config/num_mel_bins").is_some()
}

/// Modalities and endpoint features advertised on `/v1/models`, `/api/models`, and `/api/status`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Capabilities {
    pub text_input: bool,
    pub text_output: bool,
    pub image_input: bool,
    pub image_output: bool,
    pub audio_input: bool,
    pub audio_output: bool,
    /// OpenAI `tools` / `tool_calls` on `/v1/chat/completions`.
    pub tools: bool,
    /// `/v1/embeddings`. The Qwen-Next GGUF worker rejects this.
    pub embeddings: bool,
    /// Speaker turns from audio (`/v1/audio/diarize`), not a transcript.
    pub diarization: bool,
}

impl Capabilities {
    pub const fn none() -> Self {
        Self { text_input: false, text_output: false, image_input: false, image_output: false, audio_input: false, audio_output: false, tools: false, embeddings: false, diarization: false }
    }

    /// Text LLM. `embeddings` is true only for the safetensors Llama worker (`engine`), which
    /// pools a hidden state. The Qwen-Next GGUF worker does not.
    pub const fn text_llm(embeddings: bool) -> Self {
        Self { text_input: true, text_output: true, tools: true, embeddings, ..Self::none() }
    }

    /// Stable Diffusion: a text prompt, optional init/mask/control image, PNG out.
    pub const fn image() -> Self {
        Self { text_input: true, image_input: true, image_output: true, ..Self::none() }
    }

    /// Speech to text. Prompt text on the transcription API is accepted and ignored.
    pub const fn stt() -> Self {
        Self { audio_input: true, text_output: true, ..Self::none() }
    }

    /// VITS / MMS: characters in, waveform out.
    pub const fn tts() -> Self {
        Self { text_input: true, audio_output: true, ..Self::none() }
    }

    /// Danbooru tagger: image in, tag text out.
    pub const fn tagger() -> Self {
        Self { image_input: true, text_output: true, ..Self::none() }
    }

    /// Nemotron-3-Diarization: audio in, speaker turns out.
    pub const fn diarizer() -> Self {
        Self { audio_input: true, diarization: true, ..Self::none() }
    }

    /// Capabilities of a model folder that is not loaded yet. Qwen-Next GGUF is recognized by
    /// the same `qwen4exp` probe the loader uses, so a catalog row does not claim embeddings.
    pub fn for_model(kind: Kind, dir: &Path) -> Self {
        match kind {
            Kind::Llm => Self::text_llm(!crate::qn_find::is_qwen4exp(dir)),
            Kind::Image => Self::image(),
            Kind::Stt => Self::stt(),
            Kind::Tts => Self::tts(),
            Kind::Tagger => Self::tagger(),
            Kind::Unknown if crate::qn_find::is_qwen4exp(dir) => Self::text_llm(false),
            Kind::Unknown if diarizer_dir(dir) => Self::diarizer(),
            Kind::Unknown => Self::none(),
        }
    }

    pub fn input_modalities(&self) -> Vec<&'static str> {
        let mut v = Vec::new();
        if self.text_input { v.push("text"); }
        if self.image_input { v.push("image"); }
        if self.audio_input { v.push("audio"); }
        v
    }

    pub fn output_modalities(&self) -> Vec<&'static str> {
        let mut v = Vec::new();
        if self.text_output { v.push("text"); }
        if self.image_output { v.push("image"); }
        if self.audio_output { v.push("audio"); }
        if self.diarization { v.push("diarization"); }
        v
    }

    /// Short label for the Models tables and `pyre models list`.
    pub fn summary(&self) -> String {
        let mut bits = Vec::new();
        let mut io = |name: &str, i: bool, o: bool| match (i, o) {
            (true, true) => bits.push(format!("{name} in/out")),
            (true, false) => bits.push(format!("{name} in")),
            (false, true) => bits.push(format!("{name} out")),
            _ => {}
        };
        io("text", self.text_input, self.text_output);
        io("image", self.image_input, self.image_output);
        io("audio", self.audio_input, self.audio_output);
        if self.tools { bits.push("tools".into()); }
        if self.embeddings { bits.push("embeddings".into()); }
        if self.diarization { bits.push("diarization".into()); }
        if bits.is_empty() { "none".into() } else { bits.join(", ") }
    }

    pub fn to_value(&self) -> Value {
        json!({
            "text_input": self.text_input,
            "text_output": self.text_output,
            "image_input": self.image_input,
            "image_output": self.image_output,
            "audio_input": self.audio_input,
            "audio_output": self.audio_output,
            "tools": self.tools,
            "embeddings": self.embeddings,
            "diarization": self.diarization,
            "input_modalities": self.input_modalities(),
            "output_modalities": self.output_modalities(),
            "summary": self.summary(),
        })
    }

    /// Same record, plus a `speech` object when this id is a transcription model.
    pub fn to_value_for(&self, id: &str) -> Value {
        let mut v = self.to_value();
        if self.audio_input && self.text_output && !self.diarization {
            let speech = crate::speech_caps::profile(id);
            v["speech"] = speech.to_json();
            v["summary"] = json!(format!("{}, {}", self.summary(), speech.brief()));
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn flag(v: &Value, k: &str) -> bool { v[k].as_bool() == Some(true) }

    #[test]
    fn qwen_next_is_text_with_tools_and_nothing_else() {
        let c = Capabilities::text_llm(false);
        let v = c.to_value();
        assert!(c.text_input && c.text_output && c.tools);
        assert!(!c.image_input && !c.image_output && !c.audio_input && !c.audio_output);
        assert!(!c.embeddings && !c.diarization);
        assert_eq!(c.input_modalities(), ["text"]);
        assert_eq!(c.output_modalities(), ["text"]);
        assert_eq!(c.summary(), "text in/out, tools");
        assert!(flag(&v, "text_input") && flag(&v, "tools") && !flag(&v, "embeddings") && !flag(&v, "image_input"));
        assert_eq!(v["input_modalities"][0], "text");
        assert_eq!(v["output_modalities"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn safetensors_llm_adds_embeddings_only() {
        let c = Capabilities::text_llm(true);
        assert!(c.embeddings && c.tools && c.text_input && c.text_output);
        assert!(!c.image_input && !c.audio_output);
        assert_eq!(c.summary(), "text in/out, tools, embeddings");
    }

    #[test]
    fn other_kinds_do_not_inherit_llm_features() {
        let image = Capabilities::image();
        assert!(image.text_input && image.image_input && image.image_output);
        assert!(!image.text_output && !image.audio_input && !image.tools && !image.embeddings);
        assert_eq!(image.input_modalities(), ["text", "image"]);
        assert_eq!(image.output_modalities(), ["image"]);

        let stt = Capabilities::stt();
        assert!(stt.audio_input && stt.text_output);
        assert!(!stt.text_input && !stt.audio_output && !stt.diarization && !stt.tools);

        let tts = Capabilities::tts();
        assert!(tts.text_input && tts.audio_output && !tts.audio_input && !tts.text_output);

        let tag = Capabilities::tagger();
        assert!(tag.image_input && tag.text_output && !tag.image_output && !tag.text_input);

        let d = Capabilities::diarizer();
        assert!(d.audio_input && d.diarization);
        assert!(!d.text_output && !d.text_input && !d.audio_output);
        assert_eq!(d.output_modalities(), ["diarization"]);
        assert_eq!(d.summary(), "audio in, diarization");

        assert_eq!(Capabilities::none().summary(), "none");
        assert!(Capabilities::none().input_modalities().is_empty());
    }

    #[test]
    fn catalog_qwen_gguf_does_not_claim_embeddings() {
        let t = tempfile::tempdir().unwrap();
        let d = t.path().join("qn");
        fs::create_dir_all(&d).unwrap();
        fs::write(d.join("model.gguf"), b"GGUF....qwen4exp....").unwrap();
        let c = Capabilities::for_model(Kind::Unknown, &d);
        assert!(c.text_input && c.text_output && c.tools);
        assert!(!c.embeddings && !c.image_input && !c.audio_input && !c.diarization);

        let llama = t.path().join("llama");
        fs::create_dir_all(&llama).unwrap();
        fs::write(llama.join("config.json"), br#"{"architectures":["LlamaForCausalLM"],"model_type":"llama"}"#).unwrap();
        let c = Capabilities::for_model(Kind::Llm, &llama);
        assert!(c.embeddings && c.tools && !c.image_input);

        let diar = t.path().join("diar");
        fs::create_dir_all(&diar).unwrap();
        fs::write(diar.join("config.json"), br#"{"audio_config":{"num_mel_bins":128},"head_config":{"num_speakers":8}}"#).unwrap();
        let c = Capabilities::for_model(Kind::Unknown, &diar);
        assert!(c.audio_input && c.diarization);
        assert!(!c.text_output && !c.text_input && !c.tools && !c.embeddings);
        assert_eq!(c.output_modalities(), ["diarization"]);
    }
}
