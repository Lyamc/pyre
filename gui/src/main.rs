//! pyre-gui: a native GPUI (Lyamc fork) desktop front end for a pyre server.
//! It talks to pyre only over HTTP (OpenAI-compatible + `/api/*`) and never links Burn.
//!
//! Pages: Chat (streaming), Models (HF cache + store, pull/import/alias/remove, load/unload),
//! Images (txt2img), Audio (transcribe a file, TTS playback), Status.
mod net;

use base64::Engine as _;
use gpui::{
    div, img, prelude::*, px, rgb, size, App, Application, Bounds, ClickEvent, Context, FocusHandle,
    Focusable, Image, ImageFormat, ImageSource, KeystrokeEvent, ObjectFit, SharedString, Subscription,
    Window, WindowBounds, WindowOptions,
};
use net::Msg;
use serde_json::{json, Value};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::time::Duration;

const BG: u32 = 0x16181d;
const PANEL: u32 = 0x1f2229;
const RAISED: u32 = 0x2a2e37;
const ACCENT: u32 = 0xe8743b;
const TEXT: u32 = 0xe6e6e6;
const MUTED: u32 = 0x9aa0aa;
const DANGER: u32 = 0xe05a5a;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Page { Chat, Models, Images, Audio, Status }

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Field { Server, Chat, Pull, Import, AliasName, AliasTo, LoadRef, Prompt, Steps, AudioPath, Tts }

struct PyreGui {
    focus: FocusHandle,
    _keys: Subscription,
    tx: Sender<Msg>,
    rx: Receiver<Msg>,
    page: Page,
    field: Field,
    server: String,
    chat_input: String,
    messages: Vec<(String, String)>,
    streaming: bool,
    pull: String,
    import: String,
    alias_name: String,
    alias_to: String,
    load_ref: String,
    load_kind: &'static str,
    prompt: String,
    steps: String,
    audio_path: String,
    tts_text: String,
    models: Value,
    status: Value,
    health: String,
    image: Option<Arc<Image>>,
    image_info: String,
    transcript: String,
    log: Vec<String>,
}

impl PyreGui {
    fn new(window: &mut Window, cx: &mut Context<Self>) -> Self {
        let (tx, rx) = channel();
        let keys = cx.observe_keystrokes(|this, ev, _w, cx| this.on_key(ev, cx));
        let focus = cx.focus_handle();
        window.focus(&focus);
        let server = std::env::args().nth(1).or_else(|| std::env::var("PYRE_URL").ok()).unwrap_or_else(|| "http://127.0.0.1:8080".into());
        let mut me = Self {
            focus, _keys: keys, tx, rx, page: Page::Chat, field: Field::Chat, server,
            chat_input: String::new(), messages: vec![], streaming: false,
            pull: String::new(), import: String::new(), alias_name: String::new(), alias_to: String::new(),
            load_ref: String::new(), load_kind: "llm",
            prompt: "a lighthouse on a cliff at sunset, oil painting".into(), steps: "20".into(),
            audio_path: String::new(), tts_text: "Hello from pyre.".into(),
            models: Value::Null, status: Value::Null, health: "?".into(), image: None, image_info: String::new(),
            transcript: String::new(), log: vec![],
        };
        // Drain worker results on the UI thread.
        cx.spawn(async move |this, cx| loop {
            cx.background_executor().timer(Duration::from_millis(30)).await;
            if this.update(cx, |this, cx| this.drain(cx)).is_err() { break; }
        })
        .detach();
        // Poll server status every 3 s.
        cx.spawn(async move |this, cx| loop {
            if this.update(cx, |this, _| this.refresh_status()).is_err() { break; }
            cx.background_executor().timer(Duration::from_secs(3)).await;
        })
        .detach();
        me.refresh_models();
        // Scripted start for demos/screenshots: PYRE_GUI_PAGE=chat|models|images|audio|status, PYRE_GUI_SEND=<chat text>.
        me.page = match std::env::var("PYRE_GUI_PAGE").unwrap_or_default().as_str() {
            "models" => Page::Models, "images" => Page::Images, "audio" => Page::Audio, "status" => Page::Status, _ => Page::Chat,
        };
        if let Ok(s) = std::env::var("PYRE_GUI_SEND") { me.chat_input = s; me.send_chat(); }
        if let Ok(s) = std::env::var("PYRE_GUI_TTS") { me.page = Page::Audio; me.tts_text = s; me.do_tts(); }
        me
    }

    fn base(&self) -> String { self.server.trim_end_matches('/').to_string() }
    fn note(&mut self, s: impl Into<String>) { self.log.push(s.into()); if self.log.len() > 8 { self.log.remove(0); } }

    fn refresh_status(&self) {
        net::get(self.base(), "/api/status", "status", self.tx.clone());
        net::get(self.base(), "/health", "health", self.tx.clone());
    }
    fn refresh_models(&self) { net::get(self.base(), "/api/models", "models", self.tx.clone()); }

    fn drain(&mut self, cx: &mut Context<Self>) {
        let mut dirty = false;
        while let Ok(m) = self.rx.try_recv() {
            dirty = true;
            match m {
                Msg::ChatDelta(d) => { if let Some(last) = self.messages.last_mut() { last.1.push_str(&d); } }
                Msg::ChatDone(r) => { self.streaming = false; if let Err(e) = r { self.note(format!("chat: {e}")); } }
                Msg::Pull(p) => {
                    let v: Value = serde_json::from_str(&p).unwrap_or(Value::Null);
                    let s = if let Some(e) = v["error"].as_str() { format!("pull error: {e}") }
                        else if v["done"].as_bool() == Some(true) { format!("pulled -> {}", v["path"].as_str().unwrap_or("")) }
                        else { format!("pull {}/{}: {}", v["index"], v["total"], v["file"].as_str().unwrap_or("")) };
                    self.note(s);
                }
                Msg::Bytes(tag, r) => match r {
                    Ok(b) => {
                        let path = std::env::temp_dir().join("pyre-gui-tts.wav");
                        match std::fs::write(&path, &b) {
                            Ok(()) => { self.note(format!("{tag}: {} bytes -> {}", b.len(), path.display())); play(&path); }
                            Err(e) => self.note(format!("{tag}: {e}")),
                        }
                    }
                    Err(e) => self.note(format!("{tag}: {e}")),
                },
                Msg::Json(tag, r) => match (tag, r) {
                    ("status", Ok(v)) => self.status = v,
                    ("status", Err(_)) => self.status = Value::Null,
                    ("health", Ok(v)) => self.health = v["status"].as_str().unwrap_or("ok").to_string(),
                    ("health", Err(e)) => self.health = format!("unreachable ({e})"),
                    ("models", Ok(v)) => self.models = v,
                    ("image", Ok(v)) => {
                        if let Some(b) = v["data"][0]["b64_json"].as_str() {
                            if let Ok(png) = base64::engine::general_purpose::STANDARD.decode(b) {
                                self.image = Some(Arc::new(Image::from_bytes(ImageFormat::Png, png)));
                            }
                        }
                        self.image_info = format!("{:.1}s", v["timings"][0]["seconds"].as_f64().unwrap_or(0.0));
                    }
                    ("transcribe", Ok(v)) => self.transcript = v["text"].as_str().map(str::to_string).unwrap_or_else(|| v.to_string()),
                    (t, Ok(v)) => { self.note(format!("{t}: {}", short(&v))); if t != "pull-done" { self.refresh_models(); } else { self.refresh_models(); } }
                    (t, Err(e)) => self.note(format!("{t}: {e}")),
                },
            }
        }
        if dirty { cx.notify(); }
    }

    fn buf(&mut self, f: Field) -> &mut String {
        match f {
            Field::Server => &mut self.server, Field::Chat => &mut self.chat_input, Field::Pull => &mut self.pull,
            Field::Import => &mut self.import, Field::AliasName => &mut self.alias_name, Field::AliasTo => &mut self.alias_to,
            Field::LoadRef => &mut self.load_ref, Field::Prompt => &mut self.prompt, Field::Steps => &mut self.steps,
            Field::AudioPath => &mut self.audio_path, Field::Tts => &mut self.tts_text,
        }
    }

    fn on_key(&mut self, ev: &KeystrokeEvent, cx: &mut Context<Self>) {
        let k = &ev.keystroke;
        let f = self.field;
        if k.modifiers.control && k.key == "v" {
            if let Some(t) = cx.read_from_clipboard().and_then(|c| c.text()) { self.buf(f).push_str(t.trim_end_matches(['\r', '\n'])); }
        } else if k.key == "backspace" {
            self.buf(f).pop();
        } else if k.key == "enter" {
            self.submit(f);
        } else if k.modifiers.control || k.modifiers.alt || k.modifiers.platform {
            return;
        } else if let Some(c) = k.key_char.as_deref() {
            if !c.is_empty() && !c.chars().any(char::is_control) { self.buf(f).push_str(c); } else { return; }
        } else {
            return;
        }
        cx.notify();
    }

    fn submit(&mut self, f: Field) {
        match f {
            Field::Chat => self.send_chat(),
            Field::Pull => self.do_pull(),
            Field::Import => self.do_import(),
            Field::AliasName | Field::AliasTo => self.do_alias(),
            Field::LoadRef => self.do_load(),
            Field::Prompt | Field::Steps => self.do_image(),
            Field::AudioPath => self.do_transcribe(),
            Field::Tts => self.do_tts(),
            Field::Server => { self.refresh_status(); self.refresh_models(); }
        }
    }

    fn send_chat(&mut self) {
        let text = self.chat_input.trim().to_string();
        if text.is_empty() || self.streaming { return; }
        self.chat_input.clear();
        self.messages.push(("user".into(), text));
        let history = self.messages.clone();
        self.messages.push(("assistant".into(), String::new()));
        self.streaming = true;
        net::chat_stream(self.base(), None, history, self.tx.clone());
    }
    fn do_pull(&mut self) { let r = self.pull.trim().to_string(); if !r.is_empty() { self.note(format!("pulling {r}...")); net::pull(self.base(), r, self.tx.clone()); } }
    fn do_import(&mut self) { let s = self.import.trim().to_string(); if !s.is_empty() { net::post(self.base(), "/api/models/import", json!({"source": s, "mode": "link"}), "import", self.tx.clone()); } }
    fn do_alias(&mut self) {
        let (n, a) = (self.alias_name.trim().to_string(), self.alias_to.trim().to_string());
        if !n.is_empty() && !a.is_empty() { net::post(self.base(), "/api/models/alias", json!({"name": n, "alias": a}), "alias", self.tx.clone()); }
    }
    fn do_load(&mut self) {
        let r = self.load_ref.trim().to_string();
        if !r.is_empty() { self.note(format!("loading {r} as {}...", self.load_kind)); net::post(self.base(), "/api/models/load", json!({"model": r, "kind": self.load_kind}), "load", self.tx.clone()); }
    }
    fn do_image(&mut self) {
        let steps: u64 = self.steps.trim().parse().unwrap_or(20);
        self.image_info = "generating...".into();
        net::post(self.base(), "/v1/images/generations", json!({"prompt": self.prompt, "size": "512x512", "steps": steps, "response_format": "b64_json"}), "image", self.tx.clone());
    }
    fn do_transcribe(&mut self) {
        let p = self.audio_path.trim().trim_matches('"').to_string();
        if !p.is_empty() { self.transcript = "transcribing...".into(); net::transcribe(self.base(), p, self.tx.clone()); }
    }
    fn do_tts(&mut self) {
        net::post_bytes(self.base(), "/v1/audio/speech", json!({"model": "tts-1", "input": self.tts_text, "voice": "default", "response_format": "wav"}), "tts", self.tx.clone());
    }

    // ---------- view helpers ----------
    fn input(&self, f: Field, value: &str, placeholder: &str, cx: &mut Context<Self>) -> impl IntoElement {
        let active = self.field == f;
        let shown: SharedString = if value.is_empty() { placeholder.to_string().into() } else if active { format!("{value}|").into() } else { value.to_string().into() };
        div()
            .id(SharedString::from(format!("field-{f:?}")))
            .flex_1()
            .min_w(px(120.))
            .px(px(10.)).py(px(7.))
            .rounded_md()
            .bg(rgb(RAISED))
            .border_1()
            .border_color(rgb(if active { ACCENT } else { RAISED }))
            .text_color(rgb(if value.is_empty() { MUTED } else { TEXT }))
            .child(shown)
            .on_click(cx.listener(move |this, _: &ClickEvent, _, cx| { this.field = f; cx.notify(); }))
    }

    fn button(&self, id: &str, label: &str, primary: bool, cx: &mut Context<Self>, on: impl Fn(&mut Self) + 'static) -> impl IntoElement {
        div()
            .id(SharedString::from(id.to_string()))
            .px(px(12.)).py(px(7.))
            .rounded_md()
            .bg(rgb(if primary { ACCENT } else { RAISED }))
            .text_color(rgb(if primary { 0x101010 } else { TEXT }))
            .cursor_pointer()
            .child(label.to_string())
            .on_click(cx.listener(move |this, _: &ClickEvent, _, cx| { on(this); cx.notify(); }))
    }

    fn nav(&self, p: Page, label: &str, cx: &mut Context<Self>) -> impl IntoElement {
        let cur = self.page == p;
        div()
            .id(SharedString::from(format!("nav-{label}")))
            .px(px(14.)).py(px(9.))
            .rounded_md()
            .bg(rgb(if cur { RAISED } else { PANEL }))
            .text_color(rgb(if cur { ACCENT } else { TEXT }))
            .cursor_pointer()
            .child(label.to_string())
            .on_click(cx.listener(move |this, _: &ClickEvent, _, cx| {
                this.page = p;
                this.field = match p { Page::Chat => Field::Chat, Page::Models => Field::LoadRef, Page::Images => Field::Prompt, Page::Audio => Field::AudioPath, Page::Status => Field::Server };
                if p == Page::Models { this.refresh_models(); }
                cx.notify();
            }))
    }

    fn chat_page(&self, cx: &mut Context<Self>) -> gpui::AnyElement {
        let mut list = div().id("chat-scroll").flex().flex_col().gap(px(10.)).flex_1().overflow_y_scroll().p(px(4.));
        if self.messages.is_empty() {
            list = list.child(div().text_color(rgb(MUTED)).child("Type a message and press Enter. Replies stream from the loaded LLM."));
        }
        for (i, (role, content)) in self.messages.iter().enumerate() {
            let user = role == "user";
            let body = if content.is_empty() && !user && self.streaming { "...".to_string() } else { content.clone() };
            list = list.child(
                div().id(SharedString::from(format!("msg-{i}"))).flex().flex_col().gap(px(2.))
                    .child(div().text_sm().text_color(rgb(if user { ACCENT } else { MUTED })).child(role.clone()))
                    .child(div().p(px(10.)).rounded_md().bg(rgb(if user { RAISED } else { PANEL })).child(body)),
            );
        }
        div().flex().flex_col().size_full().gap(px(10.))
            .child(list)
            .child(div().flex().gap(px(8.))
                .child(self.input(Field::Chat, &self.chat_input, "Message pyre...", cx))
                .child(self.button("send", if self.streaming { "Streaming..." } else { "Send" }, true, cx, |t| t.send_chat()))
                .child(self.button("clear", "Clear", false, cx, |t| { if !t.streaming { t.messages.clear(); } })))
            .into_any_element()
    }

    fn models_page(&self, cx: &mut Context<Self>) -> gpui::AnyElement {
        let mut rows = div().id("models-scroll").flex().flex_col().gap(px(4.)).flex_1().overflow_y_scroll();
        let loaded = self.models["loaded"].as_array().cloned().unwrap_or_default();
        rows = rows.child(div().text_color(rgb(ACCENT)).child(format!("Loaded ({})", loaded.len())));
        for (i, m) in loaded.iter().enumerate() {
            let id = m["id"].as_str().unwrap_or("").to_string();
            let id2 = id.clone();
            rows = rows.child(div().flex().gap(px(8.)).items_center().p(px(6.)).rounded_md().bg(rgb(PANEL))
                .child(div().flex_1().child(format!("{}  [{}] on {}{}", id, m["kind"].as_str().unwrap_or(""), m["backend"].as_str().unwrap_or(""), if m["busy"].as_bool() == Some(true) { "  (busy)" } else { "" })))
                .child(self.button(&format!("unload-{i}"), "Unload", false, cx, move |t| net::post(t.base(), "/api/models/unload", json!({"id": id2}), "unload", t.tx.clone()))));
        }
        let all = self.models["models"].as_array().cloned().unwrap_or_default();
        rows = rows.child(div().pt(px(8.)).text_color(rgb(ACCENT)).child(format!("Available ({}) - store: {}  hf cache: {}", all.len(), self.models["store_root"].as_str().unwrap_or("?"), self.models["hf_cache"].as_str().unwrap_or("?"))));
        for (i, m) in all.iter().enumerate() {
            let name = m["name"].as_str().unwrap_or("").to_string();
            let kind = m["kind"].as_str().unwrap_or("llm").to_string();
            let (n1, n2, k1) = (name.clone(), name.clone(), kind.clone());
            let alias = m["alias"].as_str().map(|a| format!("  alias={a}")).unwrap_or_default();
            rows = rows.child(div().flex().gap(px(8.)).items_center().p(px(6.)).rounded_md().bg(rgb(PANEL))
                .child(div().flex_1().child(format!("{name}  [{kind}]  {}  {}{alias}", m["source"].as_str().unwrap_or(""), m["architecture"].as_str().unwrap_or(""))))
                .child(self.button(&format!("load-{i}"), "Load", true, cx, move |t| {
                    let kind: &'static str = match k1.as_str() { "image" => "image", "stt" => "stt", "tts" => "tts", _ => "llm" };
                    t.load_kind = kind; t.load_ref = n1.clone(); t.do_load();
                }))
                .child(self.button(&format!("alias-{i}"), "Alias...", false, cx, move |t| { t.alias_name = n2.clone(); t.field = Field::AliasTo; })));
        }
        let kinds = ["llm", "image", "stt", "tts"];
        let mut kind_row = div().flex().gap(px(6.)).items_center().child(div().text_color(rgb(MUTED)).child("kind:"));
        for k in kinds {
            let on = self.load_kind == k;
            kind_row = kind_row.child(div().id(SharedString::from(format!("kind-{k}"))).px(px(8.)).py(px(4.)).rounded_md()
                .bg(rgb(if on { ACCENT } else { RAISED })).text_color(rgb(if on { 0x101010 } else { TEXT })).cursor_pointer().child(k)
                .on_click(cx.listener(move |this, _: &ClickEvent, _, cx| { this.load_kind = k; cx.notify(); })));
        }
        div().flex().flex_col().size_full().gap(px(8.))
            .child(div().flex().gap(px(8.)).child(self.input(Field::LoadRef, &self.load_ref, "model ref to load (alias, org/name, folder)", cx)).child(kind_row)
                .child(self.button("load", "Load", true, cx, |t| t.do_load())).child(self.button("refresh", "Refresh", false, cx, |t| t.refresh_models())))
            .child(div().flex().gap(px(8.)).child(self.input(Field::Pull, &self.pull, "HF repo to pull, e.g. Qwen/Qwen2.5-0.5B-Instruct", cx)).child(self.button("pull", "Pull", false, cx, |t| t.do_pull())))
            .child(div().flex().gap(px(8.)).child(self.input(Field::Import, &self.import, "import (HF repo id or folder) into pyre store (link)", cx)).child(self.button("import", "Import", false, cx, |t| t.do_import())))
            .child(div().flex().gap(px(8.)).child(self.input(Field::AliasName, &self.alias_name, "store model name", cx)).child(self.input(Field::AliasTo, &self.alias_to, "alias", cx))
                .child(self.button("alias", "Set alias", false, cx, |t| t.do_alias()))
                .child(self.button("rm", "Remove from store", false, cx, |t| {
                    let n = t.alias_name.trim().to_string();
                    if !n.is_empty() { net::post(t.base(), "/api/models/rm", json!({"name": n, "confirm": true}), "rm", t.tx.clone()); }
                })))
            .child(rows)
            .into_any_element()
    }

    fn images_page(&self, cx: &mut Context<Self>) -> gpui::AnyElement {
        let mut out = div().flex().flex_col().size_full().gap(px(10.))
            .child(div().flex().gap(px(8.)).child(self.input(Field::Prompt, &self.prompt, "prompt", cx)).child(div().w(px(80.)).flex().child(self.input(Field::Steps, &self.steps, "steps", cx)))
                .child(self.button("gen", "Generate 512x512", true, cx, |t| t.do_image())))
            .child(div().text_color(rgb(MUTED)).child(self.image_info.clone()));
        if let Some(i) = &self.image {
            out = out.child(div().w(px(512.)).h(px(512.)).rounded_md().overflow_hidden().child(img(ImageSource::Image(i.clone())).size_full().object_fit(ObjectFit::Contain)));
        }
        out.into_any_element()
    }

    fn audio_page(&self, cx: &mut Context<Self>) -> gpui::AnyElement {
        div().flex().flex_col().size_full().gap(px(10.))
            .child(div().text_color(rgb(ACCENT)).child("Speech to text"))
            .child(div().flex().gap(px(8.)).child(self.input(Field::AudioPath, &self.audio_path, "path to an audio file (wav, mp3, flac, ogg, webm)", cx)).child(self.button("tr", "Transcribe", true, cx, |t| t.do_transcribe())))
            .child(div().p(px(10.)).rounded_md().bg(rgb(PANEL)).min_h(px(60.)).child(if self.transcript.is_empty() { "(transcript appears here)".to_string() } else { self.transcript.clone() }))
            .child(div().pt(px(12.)).text_color(rgb(ACCENT)).child("Text to speech"))
            .child(div().flex().gap(px(8.)).child(self.input(Field::Tts, &self.tts_text, "text to speak", cx)).child(self.button("tts", "Speak", true, cx, |t| t.do_tts())))
            .into_any_element()
    }

    fn status_page(&self, cx: &mut Context<Self>) -> gpui::AnyElement {
        let pretty = serde_json::to_string_pretty(&self.status).unwrap_or_default();
        div().flex().flex_col().size_full().gap(px(10.))
            .child(div().flex().gap(px(8.)).child(self.input(Field::Server, &self.server, "server URL", cx)).child(self.button("connect", "Connect", true, cx, |t| { t.refresh_status(); t.refresh_models(); })))
            .child(div().id("status-scroll").flex_1().overflow_y_scroll().p(px(10.)).rounded_md().bg(rgb(PANEL)).text_sm().child(pretty))
            .into_any_element()
    }
}

fn short(v: &Value) -> String { let s = v.to_string(); if s.len() > 160 { format!("{}...", &s[..160]) } else { s } }

fn play(path: &std::path::Path) {
    #[cfg(windows)]
    { let _ = std::process::Command::new("cmd").args(["/c", "start", ""]).arg(path).spawn(); }
    #[cfg(target_os = "macos")]
    { let _ = std::process::Command::new("afplay").arg(path).spawn(); }
    #[cfg(all(unix, not(target_os = "macos")))]
    { let _ = std::process::Command::new("xdg-open").arg(path).spawn(); }
}

impl Focusable for PyreGui {
    fn focus_handle(&self, _: &App) -> FocusHandle { self.focus.clone() }
}

impl Render for PyreGui {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let page = match self.page {
            Page::Chat => self.chat_page(cx),
            Page::Models => self.models_page(cx),
            Page::Images => self.images_page(cx),
            Page::Audio => self.audio_page(cx),
            Page::Status => self.status_page(cx),
        };
        let loaded = self.status["models"].as_array().map(|a| a.iter().map(|m| format!("{} ({})", m["id"].as_str().unwrap_or(""), m["kind"].as_str().unwrap_or(""))).collect::<Vec<_>>().join(", ")).unwrap_or_default();
        let mut log = div().flex().flex_col().text_xs().text_color(rgb(MUTED));
        for l in self.log.iter().rev().take(3) { log = log.child(l.clone()); }
        div()
            .track_focus(&self.focus)
            .size_full().flex().flex_col()
            .bg(rgb(BG)).text_color(rgb(TEXT)).text_size(px(15.))
            .child(div().flex().items_center().gap(px(6.)).px(px(12.)).py(px(8.)).bg(rgb(PANEL))
                .child(div().text_color(rgb(ACCENT)).text_size(px(20.)).pr(px(12.)).child("pyre"))
                .child(self.nav(Page::Chat, "Chat", cx)).child(self.nav(Page::Models, "Models", cx)).child(self.nav(Page::Images, "Images", cx))
                .child(self.nav(Page::Audio, "Audio", cx)).child(self.nav(Page::Status, "Status", cx))
                .child(div().flex_1())
                .child(div().text_sm().text_color(rgb(if self.health == "ok" { 0x7bd88f } else { DANGER })).child(format!("{}  {}", self.server, self.health))))
            .child(div().flex_1().p(px(14.)).child(page))
            .child(div().px(px(12.)).py(px(6.)).bg(rgb(PANEL)).flex().flex_col()
                .child(div().text_xs().text_color(rgb(MUTED)).child(format!("loaded: {}", if loaded.is_empty() { "none".into() } else { loaded })))
                .child(log))
    }
}

fn main() {
    Application::new().run(|cx: &mut App| {
        let bounds = Bounds::centered(None, size(px(1100.), px(760.)), cx);
        cx.open_window(
            WindowOptions { window_bounds: Some(WindowBounds::Windowed(bounds)), app_id: Some("pyre-gui".into()), ..Default::default() },
            |window, cx| cx.new(|cx| PyreGui::new(window, cx)),
        )
        .expect("open pyre window");
        cx.activate(true);
    });
}
