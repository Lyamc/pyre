//! Per-model speech capabilities and the explicit Whisper fallback.
//!
//! Parakeet TDT v2 and Nemotron Speech Streaming are English transcription only.
//! Parakeet TDT v3 covers 25 European languages and does not translate.
//! A request those models cannot serve is answered by a loaded Whisper that can,
//! and the response carries a `routing` object. Nothing is swapped silently.

use serde_json::{json, Value};

pub const WHISPER_LANGS: &[&str] = &[
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it",
    "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur",
    "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn",
    "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si",
    "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
    "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "ln", "ha", "ba",
    "jw", "su",
];

/// Parakeet TDT 0.6B v3 language list (CC-BY-4.0 card).
pub const PARAKEET_V3_LANGS: &[&str] = &[
    "en", "es", "fr", "de", "bg", "hr", "cs", "da", "nl", "et", "fi", "el", "hu", "it", "lv", "lt",
    "mt", "pl", "pt", "ro", "sk", "sl", "sv", "ru", "uk",
];

const EN: &[&str] = &["en"];
const RU: &[&str] = &["ru"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LangSet {
    /// Do not redirect: this server does not have a finite list for the model.
    Unspecified,
    Listed(&'static [&'static str]),
}

#[derive(Clone, Copy, Debug)]
pub struct SpeechProfile {
    pub family: &'static str,
    pub languages: LangSet,
    pub translation: bool,
    pub timestamps: bool,
    /// Incremental audio over HTTP. File decode is not this.
    pub streaming: bool,
    /// The checkpoint was trained for chunked streaming (Nemotron lookahead).
    pub architecture_streaming: bool,
    pub lookahead_frames: Option<u32>,
    pub license: &'static str,
    /// Off unless `PYRE_STT_EXPERIMENTAL=1` or the request sets `experimental=true`.
    /// Parakeet v3 and Nemotron are not in the published WER range yet.
    pub experimental: bool,
    /// Checkpoint adapter that drops fillers and repeated words (Fluid Speech Pro).
    pub clean_mode: bool,
}

/// Single utterances at or under this length stay on for Parakeet v2.
/// Longer audio is chunked and is experimental for the new models.
pub const SHORT_CLIP_SECS: f64 = 60.0;

pub fn experimental_opt_in(request_flag: bool) -> bool {
    request_flag
        || matches!(
            std::env::var("PYRE_STT_EXPERIMENTAL").ok().as_deref(),
            Some("1" | "true" | "yes" | "on")
        )
}

pub fn chunks_long_audio(id: &str) -> bool {
    let family = profile(id).family;
    matches!(family, "parakeet" | "parakeet-tdt" | "nemotron-speech-rnnt")
}

/// v3 and Nemotron stay unloaded-from-the-default-path until their WER is in range.
pub fn gate_experimental_model(id: &str, opted_in: bool) -> Result<(), String> {
    if profile(id).experimental && !experimental_opt_in(opted_in) {
        Err(format!(
            "{id} is experimental and off by default (its WER is not in the published range yet). Set PYRE_STT_EXPERIMENTAL=1 or pass experimental=true to transcribe with it. Whisper and Parakeet v2 on clips up to {SHORT_CLIP_SECS:.0}s stay available."
        ))
    } else {
        Ok(())
    }
}

/// Joined audio for Parakeet and Nemotron still loses or repeats words at window edges.
pub fn gate_long_audio(id: &str, duration_secs: f64, opted_in: bool) -> Result<(), String> {
    if duration_secs <= SHORT_CLIP_SECS || !chunks_long_audio(id) || experimental_opt_in(opted_in) {
        return Ok(());
    }
    Err(format!(
        "long audio ({duration_secs:.0}s) on {id} is experimental and off by default. Clips up to {SHORT_CLIP_SECS:.0}s are transcribed in one window. Set PYRE_STT_EXPERIMENTAL=1 or pass experimental=true to run the chunked path."
    ))
}

impl SpeechProfile {
    pub fn supports_language(&self, language: Option<&str>) -> bool {
        let Some(lang) = language.map(normalize_lang).filter(|s| !s.is_empty()) else {
            return true;
        };
        match self.languages {
            LangSet::Unspecified => true,
            LangSet::Listed(list) => list.iter().any(|c| *c == lang),
        }
    }

    pub fn supports_translation(&self) -> bool {
        self.translation
    }

    pub fn brief(&self) -> String {
        let langs = match self.languages {
            LangSet::Unspecified => "languages unspecified".to_string(),
            LangSet::Listed(l) if l == EN => "English".into(),
            LangSet::Listed(l) if l == RU => "Russian".into(),
            LangSet::Listed(l) => format!("{} languages", l.len()),
        };
        let tr = if self.translation { "translation" } else { "no translation" };
        let ts = if self.timestamps { "timestamps" } else { "no timestamps" };
        let base = if self.architecture_streaming {
            format!("{langs}, {tr}, {ts}, chunked attention")
        } else {
            format!("{langs}, {tr}, {ts}")
        };
        if self.experimental {
            format!("{base}, experimental")
        } else {
            base
        }
    }

    pub fn to_json(&self) -> Value {
        let codes: Vec<&str> = match self.languages {
            LangSet::Unspecified => Vec::new(),
            LangSet::Listed(l) => l.to_vec(),
        };
        json!({
            "family": self.family,
            "languages": codes,
            "language_count": codes.len(),
            "translation": self.translation,
            "translation_target": if self.translation { Some("en") } else { None },
            "timestamps": self.timestamps,
            "streaming": self.streaming,
            "architecture_streaming": self.architecture_streaming,
            "lookahead_frames": self.lookahead_frames,
            "license": self.license,
            "experimental": self.experimental,
            "clean_mode": self.clean_mode,
            "beam_max": if self.clean_mode { Some(4) } else { None::<u8> },
            "translate_with": if self.clean_mode { vec!["whisper", "llm"] } else { Vec::<&str>::new() },
        })
    }
}

pub fn profile(id: &str) -> SpeechProfile {
    let key = model_key(id);
    if key.contains("fluid-speech") {
        return SpeechProfile {
            family: "parakeet-tdt",
            languages: LangSet::Listed(PARAKEET_V3_LANGS),
            translation: false,
            timestamps: true,
            streaming: true,
            architecture_streaming: false,
            lookahead_frames: None,
            license: "CC-BY-4.0",
            experimental: true,
            clean_mode: true,
        };
    }
    if key.contains("nemotron-speech") || key.contains("nemotron-3.5-asr") {
        return SpeechProfile {
            family: "nemotron-speech-rnnt",
            languages: LangSet::Listed(EN),
            translation: false,
            timestamps: false,
            streaming: false,
            architecture_streaming: true,
            lookahead_frames: Some(13),
            license: "NVIDIA Open Model License",
            experimental: true,
            clean_mode: false,
        };
    }
    if key.contains("parakeet-tdt-0.6b-v3") {
        return SpeechProfile {
            family: "parakeet-tdt",
            languages: LangSet::Listed(PARAKEET_V3_LANGS),
            translation: false,
            timestamps: false,
            streaming: false,
            architecture_streaming: false,
            lookahead_frames: None,
            license: "CC-BY-4.0",
            experimental: true,
            clean_mode: false,
        };
    }
    if key.contains("parakeet-tdt-0.6b-v2") || key.contains("parakeet-ctc") {
        return SpeechProfile {
            family: "parakeet",
            languages: LangSet::Listed(EN),
            translation: false,
            timestamps: false,
            streaming: false,
            architecture_streaming: false,
            lookahead_frames: None,
            license: "CC-BY-4.0",
            experimental: false,
            clean_mode: false,
        };
    }
    if key.contains("t-one") {
        return SpeechProfile {
            family: "t-one",
            languages: LangSet::Listed(RU),
            translation: false,
            timestamps: false,
            streaming: false,
            architecture_streaming: false,
            lookahead_frames: None,
            license: "Apache-2.0",
            experimental: false,
            clean_mode: false,
        };
    }
    if key.contains("moonshine") {
        return SpeechProfile {
            family: "moonshine",
            languages: LangSet::Listed(EN),
            translation: false,
            timestamps: false,
            streaming: false,
            architecture_streaming: false,
            lookahead_frames: None,
            license: "MIT",
            experimental: false,
            clean_mode: false,
        };
    }
    if is_english_whisper(&key) {
        return whisper(true, false);
    }
    if is_multilingual_whisper(&key) {
        let translation = !key.contains("distil-large");
        return whisper(false, translation);
    }
    SpeechProfile {
        family: "speech",
        languages: LangSet::Unspecified,
        translation: false,
        timestamps: false,
        streaming: false,
        architecture_streaming: false,
        lookahead_frames: None,
        license: "",
        experimental: false,
        clean_mode: false,
    }
}

fn whisper(english_only: bool, translation: bool) -> SpeechProfile {
    SpeechProfile {
        family: "whisper",
        languages: if english_only { LangSet::Listed(EN) } else { LangSet::Listed(WHISPER_LANGS) },
        translation,
        timestamps: true,
        streaming: false,
        architecture_streaming: false,
        lookahead_frames: None,
        license: "Apache-2.0",
        experimental: false,
        clean_mode: false,
    }
}

fn is_english_whisper(key: &str) -> bool {
    const NAMES: &[&str] = &["tiny_en", "base_en", "small_en", "medium_en", "distil-medium-en", "distil-medium.en"];
    NAMES.iter().any(|n| key == *n || key.ends_with(n)) || key.ends_with(".en") || key.ends_with("_en")
}

fn is_multilingual_whisper(key: &str) -> bool {
    const NAMES: &[&str] = &[
        "tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3", "large-v3-turbo", "distil-large-v3",
    ];
    NAMES.iter().any(|n| key == *n) || key.contains("whisper")
}

pub fn model_key(id: &str) -> String {
    let tail = id.rsplit(['/', '\\']).next().unwrap_or(id);
    tail.strip_prefix("whisburn:").unwrap_or(tail).to_string()
}

pub fn normalize_lang(raw: &str) -> String {
    let s = raw.trim().to_ascii_lowercase();
    let s = s.split(['-', '_']).next().unwrap_or(&s);
    match s {
        "english" => "en",
        "french" | "francais" => "fr",
        "german" | "deutsch" => "de",
        "spanish" | "espanol" => "es",
        "italian" => "it",
        "portuguese" => "pt",
        "dutch" => "nl",
        "polish" => "pl",
        "russian" => "ru",
        "ukrainian" => "uk",
        "swedish" => "sv",
        "danish" => "da",
        "finnish" => "fi",
        "hungarian" => "hu",
        "czech" => "cs",
        "romanian" => "ro",
        "greek" => "el",
        "bulgarian" => "bg",
        "croatian" => "hr",
        "slovak" => "sk",
        "slovenian" | "slovene" => "sl",
        "estonian" => "et",
        "latvian" => "lv",
        "lithuanian" => "lt",
        "maltese" => "mt",
        "chinese" => "zh",
        "japanese" => "ja",
        "korean" => "ko",
        "arabic" => "ar",
        "hindi" => "hi",
        "turkish" => "tr",
        other => other,
    }
    .to_string()
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RouteDecision {
    pub requested: String,
    pub served: String,
    pub reason: String,
    pub redirected: bool,
}

/// Pick the model that can actually do this request.
/// `loaded` is the set of STT ids already in memory. When the requested model can do the
/// job it is returned even if it is not loaded yet (the caller downloads it). When it
/// cannot, a loaded model that can is required.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TranslateWith {
    /// Existing path: a Whisper model that can translate speech to English.
    Whisper,
    /// Transcribe with the requested model, then translate the text with the loaded chat model.
    Llm,
}

pub fn route(requested: &str, loaded: &[String], language: Option<&str>, translate: bool) -> Result<RouteDecision, String> {
    route_with(requested, loaded, language, translate, TranslateWith::Whisper)
}

pub fn route_with(
    requested: &str,
    loaded: &[String],
    language: Option<&str>,
    translate: bool,
    via: TranslateWith,
) -> Result<RouteDecision, String> {
    let want = profile(requested);
    let lang = language.map(normalize_lang).filter(|s| !s.is_empty());
    let lang_ok = want.supports_language(lang.as_deref());
    let tr_ok = !translate || want.supports_translation();
    if translate && !want.supports_translation() && via == TranslateWith::Llm && lang_ok {
        return Ok(RouteDecision {
            requested: requested.to_string(),
            served: requested.to_string(),
            reason: format!(
                "{requested} only transcribes. The transcript is then translated to English by the loaded chat model."
            ),
            redirected: false,
        });
    }
    if lang_ok && tr_ok {
        let why = match lang.as_deref() {
            Some(code) => format!("{requested} transcribes {code}"),
            None if translate => format!("{requested} translates to English"),
            None => format!("{requested} handles this request"),
        };
        return Ok(RouteDecision {
            requested: requested.to_string(),
            served: requested.to_string(),
            reason: why,
            redirected: false,
        });
    }
    let reason_gap = gap_reason(requested, &want, lang.as_deref(), translate);
    let served = loaded.iter().find(|id| {
        let p = profile(id);
        p.supports_language(lang.as_deref()) && (!translate || p.supports_translation()) && p.family == "whisper"
    }).or_else(|| {
        loaded.iter().find(|id| {
            let p = profile(id);
            p.supports_language(lang.as_deref()) && (!translate || p.supports_translation())
        })
    });
    match served {
        Some(id) => Ok(RouteDecision {
            requested: requested.to_string(),
            served: id.clone(),
            reason: format!("{reason_gap} Served by {id}."),
            redirected: id != requested,
        }),
        None => Err(format!(
            "{reason_gap} Load a Whisper model that can (for example `base` or `large-v3-turbo`) and call again. Nothing was substituted."
        )),
    }
}

fn gap_reason(requested: &str, want: &SpeechProfile, lang: Option<&str>, translate: bool) -> String {
    if translate && !want.translation {
        return format!("{requested} does not translate.");
    }
    if let Some(code) = lang {
        let have = match want.languages {
            LangSet::Listed(l) if l == EN => "English".to_string(),
            LangSet::Listed(l) => format!("{} languages ({})", l.len(), l.join(", ")),
            LangSet::Unspecified => "an unspecified set".into(),
        };
        format!("{requested} does not transcribe '{code}' (it covers {have}).")
    } else {
        format!("{requested} cannot serve this request.")
    }
}

pub fn decision_json(d: &RouteDecision) -> Value {
    json!({
        "requested": d.requested,
        "served": d.served,
        "redirected": d.redirected,
        "reason": d.reason,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parakeet_v3_lists_the_european_set_and_does_not_translate() {
        let p = profile("parakeet-tdt-0.6b-v3");
        assert!(!p.translation);
        assert!(!p.timestamps);
        assert!(p.supports_language(Some("fr")));
        assert!(p.supports_language(Some("French")));
        assert!(!p.supports_language(Some("ja")));
        match p.languages {
            LangSet::Listed(l) => assert_eq!(l.len(), 25),
            LangSet::Unspecified => panic!("v3 list missing"),
        }
    }

    #[test]
    fn v3_and_nemotron_are_experimental_v2_and_whisper_are_not() {
        assert!(profile("parakeet-tdt-0.6b-v3").experimental);
        assert!(profile("nemotron-speech-streaming-en-0.6b").experimental);
        assert!(!profile("parakeet-tdt-0.6b-v2").experimental);
        assert!(!profile("base").experimental);
        assert!(profile("parakeet-tdt-0.6b-v3").brief().contains("experimental"));
        let err = gate_experimental_model("parakeet-tdt-0.6b-v3", false).unwrap_err();
        assert!(err.contains("experimental"));
        assert!(gate_experimental_model("parakeet-tdt-0.6b-v2", false).is_ok());
        assert!(gate_long_audio("parakeet-tdt-0.6b-v2", 20.0, false).is_ok());
        assert!(gate_long_audio("parakeet-tdt-0.6b-v2", 360.0, false).is_err());
        assert!(gate_long_audio("base", 360.0, false).is_ok());
    }

    #[test]
    fn nemotron_speech_is_english_with_chunked_attention() {
        let p = profile("nemotron-speech-streaming-en-0.6b");
        assert!(p.architecture_streaming);
        assert!(!p.streaming);
        assert!(!p.translation);
        assert!(p.supports_language(Some("en")));
        assert!(!p.supports_language(Some("de")));
        assert_eq!(p.license, "NVIDIA Open Model License");
        assert_eq!(p.lookahead_frames, Some(13));
    }

    #[test]
    fn french_on_parakeet_v2_routes_to_a_loaded_whisper() {
        let loaded = vec!["parakeet-tdt-0.6b-v2".into(), "base".into()];
        let d = route("parakeet-tdt-0.6b-v2", &loaded, Some("fr"), false).unwrap();
        assert!(d.redirected);
        assert_eq!(d.served, "base");
        assert!(d.reason.contains("parakeet-tdt-0.6b-v2"));
        assert!(d.reason.contains("base"));
    }

    #[test]
    fn translation_on_v3_routes_to_multilingual_whisper() {
        let loaded = vec!["parakeet-tdt-0.6b-v3".into(), "tiny_en".into(), "small".into()];
        let d = route("parakeet-tdt-0.6b-v3", &loaded, Some("de"), true).unwrap();
        assert_eq!(d.served, "small");
        assert!(d.redirected);
    }

    #[test]
    fn no_capable_whisper_is_an_error_not_a_silent_fallback() {
        let loaded = vec!["parakeet-tdt-0.6b-v2".into()];
        let err = route("parakeet-tdt-0.6b-v2", &loaded, Some("ja"), false).unwrap_err();
        assert!(err.contains("Load a Whisper"));
        assert!(err.contains("ja") || err.contains("Japanese") || err.contains("'ja'"));
    }

    #[test]
    fn english_on_v2_stays_on_v2() {
        let d = route("parakeet-tdt-0.6b-v2", &[], Some("en"), false).unwrap();
        assert!(!d.redirected);
        assert_eq!(d.served, "parakeet-tdt-0.6b-v2");
    }

    #[test]
    fn fluid_speech_pro_lists_the_v3_languages_and_offers_llm_translation() {
        let p = profile("fluid-speech-pro");
        assert!(p.clean_mode);
        assert!(p.timestamps);
        assert!(!p.translation);
        assert!(p.experimental);
        assert!(p.supports_language(Some("fr")));
        assert!(p.supports_language(Some("de")));
        let loaded = vec!["fluid-speech-pro".into(), "small".into()];
        let whisper = route("fluid-speech-pro", &loaded, Some("de"), true).unwrap();
        assert_eq!(whisper.served, "small");
        assert!(whisper.redirected);
        let llm = route_with("fluid-speech-pro", &loaded, Some("de"), true, TranslateWith::Llm).unwrap();
        assert_eq!(llm.served, "fluid-speech-pro");
        assert!(llm.reason.contains("chat model"));
        assert!(!llm.redirected);
    }

    #[test]
    fn whisper_base_translates_and_lists_98_languages() {
        let p = profile("base");
        assert!(p.translation && p.timestamps);
        match p.languages {
            LangSet::Listed(l) => assert_eq!(l.len(), 98),
            LangSet::Unspecified => panic!("whisper list"),
        }
        assert!(!profile("tiny_en").translation);
        assert!(!profile("distil-large-v3").translation);
        assert!(profile("distil-large-v3").supports_language(Some("fr")));
    }
}
