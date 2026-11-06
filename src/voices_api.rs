//! Voice library + speech markup HTTP handlers (used from api.rs).
use crate::registry::{App, Kind};
use crate::server::err;
use axum::extract::{Multipart, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use serde_json::{json, Value};
use std::path::Path;
use std::sync::Arc;

type S = State<Arc<App>>;

pub fn routes(r: Router<Arc<App>>) -> Router<Arc<App>> {
    r.route("/v1/audio/speech", post(speech))
        .route("/v1/audio/speech/preview", post(speech_preview))
        .route("/v1/audio/directions", get(directions_help))
        .route("/v1/audio/voices", get(voices_list).post(voices_create).layer(axum::extract::DefaultBodyLimit::max(32 << 20)))
        .route("/v1/audio/voices/{id}", get(voices_get).delete(voices_delete))
        .route("/v1/audio/voices/{id}/reference", get(voices_reference))
}

async fn directions_help() -> Response {
    let tag_table: Vec<Value> = pyre_voice_clone::directions::all_tags().into_iter().map(|(tag, id, kind)| json!({
        "tag": tag,
        "token_id": id,
        "kind": format!("{:?}", kind).to_ascii_lowercase(),
    })).collect();
    let tokenizer_dir = pyre_voice_clone::discover_tokenizer_dir().map(|p| p.display().to_string());
    let chatterbox_events = [
        "[clear throat]", "[sigh]", "[shush]", "[cough]", "[groan]",
        "[sniff]", "[gasp]", "[chuckle]", "[laugh]",
    ];
    let chatterbox_delivery = [
        "[angry]", "[fear]", "[surprised]", "[whispering]", "[advertisement]",
        "[dramatic]", "[narration]", "[crying]", "[happy]", "[sarcastic]",
    ];
    let stages: Vec<Value> = pyre_voice_clone::stages().iter().map(|s| json!({
        "id": s.id,
        "state": format!("{:?}", s.state).to_ascii_lowercase(),
        "weights": s.weights,
        "missing": s.missing,
    })).collect();
    Json(json!({
        "engine": {
            "id": "chatterbox-nano",
            "licence": "MIT (ResembleAI/chatterbox-nano)",
            "status": if pyre_voice_clone::text_to_cloned_speech_ready() { "ready" } else { "porting" },
            "text_to_cloned_speech_ready": pyre_voice_clone::text_to_cloned_speech_ready(),
            "stages": stages,
        },
        "native_syntax": {
            "family": "Chatterbox-Nano / Turbo bracket tokens (GPT-2 added tokens 50257-50275)",
            "example": "Hi there [chuckle], have you got one minute?",
            "event_tags": chatterbox_events,
            "delivery_tags": chatterbox_delivery,
            "tag_token_ids": tag_table,
            "tokenizer_dir": tokenizer_dir,
            "notes": [
                "Official Nano UI / added_tokens.json tags. Unknown brackets are reported, never silently remapped.",
                "Nano generate() ignores exaggeration and cfg_weight.",
                "Until Burn T3+HiFT land, MMS approximates event tags as silences and maps some delivery tags to rate/noise; the rest are stripped so they are not spoken."
            ]
        },
        "portable_timing": {
            "family": "SSML-compatible (applied by pyre on every engine, including MMS)",
            "example": "Hello. <break time=\"500ms\"/> <prosody rate=\"slow\">Take your time.</prosody>",
            "tags": crate::tts_markup::supported_tags(),
        },
        "builtin_voice": "mms"
    })).into_response()
}

async fn voices_list() -> Response {
    let synth = false;
    let clone_ready = pyre_voice_clone::text_to_cloned_speech_ready();
    match crate::voices::global() {
        Ok(lib) => match lib.list(synth) {
            Ok(v) => {
                let mut voices = vec![crate::voices::builtin_mms()];
                for info in v {
                    voices.push(json!({
                        "id": info.meta.id, "name": info.meta.name, "builtin": false,
                        "engine": info.meta.engine, "synthesizable": info.synthesizable,
                        "has_reference": info.has_reference, "reference_bytes": info.reference_bytes,
                        "duration_secs": info.meta.duration_secs, "sample_rate": info.meta.sample_rate,
                        "created_ms": info.meta.created_ms, "source": info.meta.source, "notes": info.meta.notes,
                    }));
                }
                Json(json!({
                    "voices": voices,
                    "clone_engine": {
                        "id": "chatterbox-nano",
                        "licence": "MIT",
                        "status": "porting",
                        "text_to_cloned_speech_ready": clone_ready,
                        "message": "Burn speaker encoder and GPT-2 text tokenizer are in-tree. text_to_cloned_speech_ready stays false until T3, S3, and HiFT emit audio. Bracket tags on the built-in voice are MMS silence/rate approximations, not Nano token ids. voice_id!=mms synthesis is not ready yet."
                    },
                    "directions": "/v1/audio/directions"
                })).into_response()
            }
            Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")),
        },
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")),
    }
}

async fn voices_get(axum::extract::Path(id): axum::extract::Path<String>) -> Response {
    if id == "mms" { return Json(crate::voices::builtin_mms()).into_response(); }
    match crate::voices::global().and_then(|lib| lib.info(&id, false)) {
        Ok(info) => Json(json!(info)).into_response(),
        Err(e) => err(StatusCode::NOT_FOUND, format!("{e:#}")),
    }
}

async fn voices_delete(axum::extract::Path(id): axum::extract::Path<String>) -> Response {
    if id == "mms" { return err(StatusCode::BAD_REQUEST, "cannot delete the built-in MMS voice"); }
    match crate::voices::global().and_then(|lib| lib.delete(&id)) {
        Ok(()) => Json(json!({"ok": true, "id": id})).into_response(),
        Err(e) => err(StatusCode::BAD_REQUEST, format!("{e:#}")),
    }
}

async fn voices_reference(axum::extract::Path(id): axum::extract::Path<String>) -> Response {
    if id == "mms" { return err(StatusCode::NOT_FOUND, "built-in MMS has no reference clip"); }
    match crate::voices::global().and_then(|lib| Ok(std::fs::read(lib.reference_path(&id)?)?)) {
        Ok(b) => ([("content-type", "audio/wav")], b).into_response(),
        Err(e) => err(StatusCode::NOT_FOUND, format!("{e:#}")),
    }
}

async fn voices_create(mp: Multipart) -> Response {
    #[cfg(not(feature = "stt"))]
    { return err(StatusCode::NOT_IMPLEMENTED, "voice create needs the stt feature (audio decode)"); }
    #[cfg(feature = "stt")]
    {
        let mut name = String::new();
        let mut notes = String::new();
        let mut source = "upload".to_string();
        let mut file: Option<(String, Vec<u8>)> = None;
        let mut mp = mp;
        loop {
            match mp.next_field().await {
                Ok(Some(f)) => {
                    let fname = f.name().unwrap_or("").to_string();
                    let filename = f.file_name().map(|s| s.to_string()).unwrap_or_default();
                    let Ok(bytes) = f.bytes().await else { return err(StatusCode::BAD_REQUEST, "reading multipart field"); };
                    match fname.as_str() {
                        "name" => name = String::from_utf8_lossy(&bytes).trim().to_string(),
                        "notes" => notes = String::from_utf8_lossy(&bytes).to_string(),
                        "source" => source = String::from_utf8_lossy(&bytes).trim().to_string(),
                        "file" | "audio" | "reference" => file = Some((filename, bytes.to_vec())),
                        _ => {}
                    }
                }
                Ok(None) => break,
                Err(e) => return err(StatusCode::BAD_REQUEST, e.to_string()),
            }
        }
        let Some((filename, bytes)) = file else { return err(StatusCode::BAD_REQUEST, "multipart field 'file' (reference audio) is required"); };
        if name.is_empty() { return err(StatusCode::BAD_REQUEST, "'name' is required"); }
        let ext = Path::new(&filename).extension().and_then(|e| e.to_str()).unwrap_or("wav").to_ascii_lowercase();
        let wav = match pyre_whisper_audio::decode_bytes_to_mono_pcm(&bytes, &ext) {
            Ok(w) => w,
            Err(e) => return err(StatusCode::BAD_REQUEST, format!("decode audio: {e}")),
        };
        let target = if wav.sample_rate >= 24_000 { 24_000 } else { 16_000 };
        let wav = match wav.ensure_sample_rate(target) {
            Ok(w) => w,
            Err(e) => return err(StatusCode::BAD_REQUEST, format!("resample: {e}")),
        };
        let wav_bytes = match pyre_whisper_audio::encode_wav(&wav) {
            Ok(b) => b,
            Err(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, format!("encode wav: {e}")),
        };
        match crate::voices::global().and_then(|lib| lib.create(&name, &wav_bytes, &source, &notes)) {
            Ok(info) => Json(json!({
                "ok": true, "voice": info, "synthesizable": false,
                "message": "Voice identity saved. Cloned synthesis waits on the native Burn Chatterbox-Nano port; TTS/STS still use built-in MMS until then."
            })).into_response(),
            Err(e) => err(StatusCode::BAD_REQUEST, format!("{e:#}")),
        }
    }
}

async fn speech_preview(State(_app): S, body: Option<Json<Value>>) -> Response {
    let Some(Json(b)) = body else { return err(StatusCode::BAD_REQUEST, "expected JSON body with 'input'"); };
    let input = b.get("input").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let speed = b.get("speed").and_then(|v| v.as_f64()).unwrap_or(1.0);
    let noise = b.get("noise_scale").and_then(|v| v.as_f64()).unwrap_or(1.0);
    let (_segs, meta) = crate::tts_markup::parse(&input, speed, noise);
    // Nano boundary: strip portable SSML, keep bracket tags, encode with GPT-2 so known
    // event/delivery tags become exact single token ids (not later string rewrites).
    let nano = nano_encode_boundary(&input);
    Json(json!({
        "meta": meta,
        "mms_approximation": true,
        "supported": crate::tts_markup::supported_tags(),
        "voice_id": b.get("voice_id").and_then(|v| v.as_str()).unwrap_or("mms"),
        "nano": nano,
        "text_to_cloned_speech_ready": pyre_voice_clone::text_to_cloned_speech_ready(),
    })).into_response()
}

/// Strip SSML timing tags so only Nano-facing text+brackets remain, then encode_script
/// when the tokenizer is discoverable. Without tokenizer files, returns status only.
fn nano_encode_boundary(input: &str) -> Value {
    let without_ssml = strip_ssml_for_nano(input);
    let script = pyre_voice_clone::parse_script(&without_ssml);
    let dirs: Vec<Value> = script.directions.iter().map(|d| json!({
        "tag": d.tag,
        "token_id": d.token_id,
        "kind": format!("{:?}", d.kind).to_ascii_lowercase(),
        "start": d.start,
        "end": d.end,
    })).collect();
    let tok_dir = pyre_voice_clone::discover_tokenizer_dir();
    match tok_dir.as_ref().and_then(|d| pyre_voice_clone::NanoTokenizer::load(d).ok()) {
        Some(tok) => match tok.encode_script(&without_ssml) {
            Ok(enc) => json!({
                "tokenizer": "ready",
                "tokenizer_dir": tok_dir.as_ref().map(|p| p.display().to_string()),
                "normalized": enc.script.normalized,
                "ids": enc.ids,
                "directions": dirs,
                "unknown": enc.script.unknown,
                "note": "Known Nano tags are exact single GPT-2 added-token ids. SSML was stripped before encode; apply SSML silences/rate on the waveform path (MMS today)."
            }),
            Err(e) => json!({
                "tokenizer": "error",
                "error": format!("{e:#}"),
                "directions": dirs,
                "unknown": script.unknown,
            }),
        },
        None => json!({
            "tokenizer": "missing",
            "directions": dirs,
            "unknown": script.unknown,
            "note": "Place ResembleAI/chatterbox-nano tokenizer files in the HF hub cache (or set PYRE_CHATTERBOX_TOKENIZER). Direction token_ids above are from the hardcoded Nano table; encode_script needs the files for the full id stream."
        }),
    }
}

fn strip_ssml_for_nano(input: &str) -> String {
    // Drop portable SSML so Nano encode only sees text + bracket tags.
    let re_break = regex::Regex::new(r#"(?i)<break\b[^>]*/?>"#).unwrap();
    let re_prosody_open = regex::Regex::new(r#"(?i)<prosody\b[^>]*>"#).unwrap();
    let re_prosody_close = regex::Regex::new(r"(?i)</prosody>").unwrap();
    let s = re_break.replace_all(input, " ");
    let s = re_prosody_open.replace_all(&s, "");
    let s = re_prosody_close.replace_all(&s, "");
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}


async fn speech(State(app): S, body: Option<Json<Value>>) -> Response {
    let Some(Json(b)) = body else { return err(StatusCode::BAD_REQUEST, "expected a JSON body with 'input'"); };
    let voice_id = b.get("voice_id").or_else(|| b.get("voice")).and_then(|v| v.as_str()).unwrap_or("mms");
    let voice_id = if voice_id.is_empty() || voice_id == "default" { "mms" } else { voice_id };

    if voice_id != "mms" {
        let exists = crate::voices::global().ok().and_then(|lib| lib.info(voice_id, false).ok()).is_some();
        let msg = if exists {
            format!("voice '{voice_id}' is saved, but native Burn Chatterbox-Nano cloning is not ready yet; use voice_id=mms (built-in) for now")
        } else {
            format!("unknown voice_id '{voice_id}' (built-in: mms; create cloned voices via POST /v1/audio/voices)")
        };
        return err(StatusCode::NOT_IMPLEMENTED, msg);
    }

    let name = b.get("model").and_then(|m| m.as_str()).map(String::from);
    let m = match app.find(Kind::Tts, name.as_deref()) { Err(r) => return r, Ok(m) => m };
    let t = match &m.inner {
        crate::registry::Inner::Tts(t) => t.clone(),
        crate::registry::Inner::Placeholder(r) => return err(StatusCode::NOT_IMPLEMENTED, r.clone()),
        _ => return err(StatusCode::INTERNAL_SERVER_ERROR, "registry mismatch"),
    };
    let input = b.get("input").and_then(|v| v.as_str()).unwrap_or("").to_string();
    if input.trim().is_empty() { return err(StatusCode::BAD_REQUEST, "'input' is required"); }
    let fmt = b.get("response_format").and_then(|v| v.as_str()).unwrap_or("wav").to_string();
    if !["wav", "pcm"].contains(&fmt.as_str()) { return err(StatusCode::BAD_REQUEST, format!("response_format '{fmt}' not supported (wav, pcm)")); }
    let speed = b.get("speed").and_then(|v| v.as_f64()).unwrap_or(1.0);
    let noise = b.get("noise_scale").and_then(|v| v.as_f64()).unwrap_or(1.0);
    let seed = b.get("seed").and_then(|v| v.as_u64()).unwrap_or(42);
    let (segs, meta) = crate::tts_markup::parse(&input, speed, noise);
    if segs.is_empty() { return err(StatusCode::BAD_REQUEST, "no speakable text after parsing directions"); }

    t.queued.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let r = tokio::task::spawn_blocking(move || {
        let v = t.vits.lock().unwrap();
        t.queued.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
        t.busy.store(true, std::sync::atomic::Ordering::SeqCst);
        let st = std::time::Instant::now();
        let rate = v.cfg.sampling_rate;
        let mut pcm: Vec<f32> = Vec::new();
        let mut err_out: Option<anyhow::Error> = None;
        for (i, seg) in segs.iter().enumerate() {
            match seg {
                crate::tts_markup::Seg::Speak { text, speed, noise, .. } => {
                    match v.synth(text, *speed, *noise, seed.wrapping_add(i as u64)) {
                        Ok(mut s) => pcm.append(&mut s),
                        Err(e) => { err_out = Some(e); break; }
                    }
                }
                crate::tts_markup::Seg::Silence { seconds } => {
                    let n = ((*seconds) * rate as f64).round().max(1.0) as usize;
                    pcm.extend(std::iter::repeat(0.0f32).take(n));
                }
            }
        }
        t.busy.store(false, std::sync::atomic::Ordering::SeqCst);
        if let Some(e) = err_out { return Err(e); }
        tracing::info!("tts: {} segs -> {:.2}s audio in {:.2}s (ignored {})",
            segs.len(), pcm.len() as f64 / rate as f64, st.elapsed().as_secs_f64(), meta.ignored.len());
        Ok((pcm, rate, meta))
    }).await;
    match r {
        Ok(Ok((s, rate, meta))) => {
            let meta_hdr = serde_json::to_string(&meta).unwrap_or_else(|_| "{}".into());
            let mut headers = axum::http::HeaderMap::new();
            let ct = if fmt == "pcm" { "audio/pcm" } else { "audio/wav" };
            headers.insert(axum::http::header::CONTENT_TYPE, ct.parse().unwrap());
            if let Ok(v) = axum::http::HeaderValue::from_str(&meta_hdr) {
                headers.insert("x-pyre-speech-meta", v);
            }
            let body = if fmt == "pcm" {
                let mut v = Vec::with_capacity(s.len() * 2);
                for x in &s { v.extend_from_slice(&((x.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes()); }
                v
            } else {
                crate::tts::wav_bytes(&s, rate)
            };
            (headers, body).into_response()
        }
        Ok(Err(e)) => err(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}
