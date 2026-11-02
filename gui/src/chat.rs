//! pyre-chat: a native GPUI chat client for a pyre server (OpenAI-compatible streaming API).
//! Conversations sidebar, reasoning-effort picker (none/low/medium/high/very high/auto) with a
//! collapsible thinking section (tokens used, auto's choice), Markdown + code blocks with copy,
//! stop, settings (server URL, system prompt, temperature, max tokens), tok/s and context readout.
//! Layout ideas (sidebar, toolbar picker, context readout, settings drawer) after Strata's host UI.
#[path = "chat/md.rs"] mod md;
#[path = "chat/net.rs"] mod net;
#[path = "chat/store.rs"] mod store;

use gpui::{
    div, prelude::*, px, rgb, size, App, Application, Bounds, ClickEvent, ClipboardItem, Context, FocusHandle,
    Focusable, FontWeight, KeystrokeEvent, ScrollHandle, SharedString, Subscription, Window, WindowBounds, WindowOptions,
};
use net::Ev;
use serde_json::{json, Value};
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::time::{Duration, Instant};
use store::{Conv, Msg, Settings};

const BG: u32 = 0x16181d;
const PANEL: u32 = 0x1f2229;
const RAISED: u32 = 0x2a2e37;
const CODE_BG: u32 = 0x111317;
const ACCENT: u32 = 0xe8743b;
const TEXT: u32 = 0xe6e6e6;
const MUTED: u32 = 0x9aa0aa;
const DANGER: u32 = 0xe05a5a;
const GOOD: u32 = 0x7bd88f;
const MONO: &str = if cfg!(windows) { "Consolas" } else if cfg!(target_os = "macos") { "Menlo" } else { "DejaVu Sans Mono" };

const EFFORTS: [(&str, &str); 6] = [("none", "none"), ("low", "low"), ("medium", "medium"), ("high", "high"), ("very_high", "very high"), ("auto", "auto")];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Field { Composer, Server, System, Temp, MaxTok, Rename }

struct Live { conv: u64, cancel: Arc<AtomicBool>, started: Instant, first: Option<Instant>, chunks: usize }

struct ChatApp {
    focus: FocusHandle,
    _keys: Subscription,
    tx: Sender<Ev>,
    rx: Receiver<Ev>,
    convs: Vec<Conv>,
    cur: u64,
    s: Settings,
    field: Field,
    input: String,
    rename: String,
    renaming: Option<u64>,
    confirm_delete: Option<u64>,
    show_settings: bool,
    open_think: HashSet<(u64, usize)>,
    live: Option<Live>,
    health: String,
    n_ctx: u64,
    model: String,
    notice: String,
    scroll: ScrollHandle,
    stick: bool,
    think_all: bool,
}

fn effort_label(e: &str) -> &str { EFFORTS.iter().find(|(k, _)| *k == e).map(|(_, l)| *l).unwrap_or(e) }

fn num(n: u64) -> String {
    let s = n.to_string();
    let mut o = String::new();
    for (i, c) in s.chars().enumerate() { if i > 0 && (s.len() - i) % 3 == 0 { o.push(','); } o.push(c); }
    o
}

impl ChatApp {
    fn new(window: &mut Window, cx: &mut Context<Self>) -> Self {
        let (tx, rx) = channel();
        let keys = cx.observe_keystrokes(|this, ev, _w, cx| this.on_key(ev, cx));
        let focus = cx.focus_handle();
        window.focus(&focus);
        let mut s = store::load_settings();
        if let Some(u) = std::env::args().nth(1).or_else(|| std::env::var("PYRE_URL").ok()) { s.server = u; }
        if let Ok(e) = std::env::var("PYRE_CHAT_EFFORT") { s.effort = e; }
        let mut convs = store::load_convs();
        convs.sort_by(|a, b| b.updated.cmp(&a.updated));
        let mut me = Self {
            focus, _keys: keys, tx, rx, cur: 0, convs, s, field: Field::Composer, input: String::new(), rename: String::new(),
            renaming: None, confirm_delete: None, show_settings: std::env::var("PYRE_CHAT_SETTINGS").is_ok(), open_think: HashSet::new(),
            live: None, health: "connecting...".into(), n_ctx: 0, model: String::new(), notice: String::new(), scroll: ScrollHandle::new(), stick: true, think_all: std::env::var("PYRE_CHAT_OPEN_THINK").is_ok(),
        };
        if me.convs.is_empty() { me.new_conv(); } else { me.cur = me.convs[0].id; }
        cx.spawn(async move |this, cx| loop {
            cx.background_executor().timer(Duration::from_millis(30)).await;
            if this.update(cx, |this, cx| this.drain(cx)).is_err() { break; }
        }).detach();
        cx.spawn(async move |this, cx| loop {
            if this.update(cx, |this, _| this.poll()).is_err() { break; }
            cx.background_executor().timer(Duration::from_secs(4)).await;
        }).detach();
        // Scripted start for demos/screenshots.
        if let Ok(t) = std::env::var("PYRE_CHAT_SEND") { me.new_conv(); me.input = t; me.send(); }
        if let Some(secs) = std::env::var("PYRE_CHAT_STOP_AFTER").ok().and_then(|v| v.parse::<u64>().ok()) {
            cx.spawn(async move |this, cx| { cx.background_executor().timer(Duration::from_secs(secs)).await; let _ = this.update(cx, |this, cx| { this.stop(); cx.notify(); }); }).detach();
        }
        me
    }

    fn base(&self) -> String { self.s.server.trim().trim_end_matches('/').trim_end_matches("/v1").to_string() }
    fn poll(&self) {
        net::get(self.base(), "/health", "health", self.tx.clone());
        net::get(self.base(), "/api/metrics", "metrics", self.tx.clone());
    }
    fn conv(&self) -> Option<&Conv> { self.convs.iter().find(|c| c.id == self.cur) }
    fn conv_mut(&mut self, id: u64) -> Option<&mut Conv> { self.convs.iter_mut().find(|c| c.id == id) }
    fn save(&self) { store::save_convs(&self.convs); }

    fn new_conv(&mut self) {
        if let Some(c) = self.conv() { if c.messages.is_empty() { self.field = Field::Composer; return; } }
        let id = self.convs.iter().map(|c| c.id).max().unwrap_or(0).max(store::now() * 1000) + 1;
        self.convs.insert(0, Conv { id, title: "New chat".into(), messages: vec![], updated: store::now() });
        self.cur = id;
        self.field = Field::Composer;
        self.stick = true;
    }

    fn delete_conv(&mut self, id: u64) {
        if self.live.as_ref().map(|l| l.conv) == Some(id) { self.stop(); }
        self.convs.retain(|c| c.id != id);
        if self.cur == id { self.cur = self.convs.first().map(|c| c.id).unwrap_or(0); }
        if self.convs.is_empty() { self.new_conv(); }
        self.confirm_delete = None;
        self.save();
    }

    fn drain(&mut self, cx: &mut Context<Self>) {
        let mut dirty = false;
        while let Ok(ev) = self.rx.try_recv() {
            dirty = true;
            match ev {
                Ev::Delta(id, c, r) => {
                    if let Some(l) = self.live.as_mut() { if l.first.is_none() { l.first = Some(Instant::now()); } l.chunks += 1; }
                    if let Some(m) = self.conv_mut(id).and_then(|c| c.messages.last_mut()) { m.content.push_str(&c); m.reasoning.push_str(&r); }
                }
                Ev::Extra(id, info, timings, usage) => {
                    if let Some(m) = self.conv_mut(id).and_then(|c| c.messages.last_mut()) {
                        if !info.is_null() { m.info = info; }
                        if let Some(tps) = timings["predicted_per_second"].as_f64() {
                            m.stats = format!("{} tok · {:.1} tok/s · prompt {} tok ({:.0} tok/s, {} cached)", timings["predicted_n"], tps,
                                timings["prompt_n"], timings["prompt_per_second"].as_f64().unwrap_or(0.0), timings["cached_n"].as_u64().unwrap_or(0));
                        }
                        if !usage.is_null() { m.info["usage"] = usage; }
                    }
                }
                Ev::Done(id, r) => {
                    let stopped = self.live.as_ref().map(|l| l.cancel.load(Ordering::Relaxed)).unwrap_or(false);
                    let fallback = self.live.as_ref().and_then(|l| l.first.map(|f| (l.chunks, f.elapsed().as_secs_f64())));
                    if self.live.as_ref().map(|l| l.conv) == Some(id) { self.live = None; }
                    if let Some(m) = self.conv_mut(id).and_then(|c| c.messages.last_mut()) {
                        if let Err(e) = r { m.error = e; }
                        if stopped { m.stats = format!("stopped{}", if m.stats.is_empty() { String::new() } else { format!(" · {}", m.stats) }); }
                        else if m.stats.is_empty() { if let Some((n, s)) = fallback { m.stats = format!("{n} chunks · {:.1}/s", n as f64 / s.max(0.001)); } }
                    }
                    if let Some(c) = self.conv_mut(id) { c.updated = store::now(); }
                    self.save();
                }
                Ev::Json(tag, r) => match (tag, r) {
                    ("health", Ok(_)) => self.health = "ok".into(),
                    ("health", Err(e)) => self.health = format!("unreachable: {e}"),
                    ("metrics", Ok(v)) => {
                        if let Some(m) = v["models"].as_array().and_then(|a| a.iter().find(|m| m["kind"] == "llm")) {
                            self.n_ctx = m["meta"]["n_ctx"].as_u64().unwrap_or(self.n_ctx);
                            self.model = m["id"].as_str().unwrap_or("").to_string();
                        }
                    }
                    _ => {}
                },
            }
        }
        if dirty {
            if self.stick { self.scroll.scroll_to_bottom(); }
            cx.notify();
        }
    }

    fn buf(&mut self, f: Field) -> &mut String {
        match f {
            Field::Composer => &mut self.input, Field::Server => &mut self.s.server, Field::System => &mut self.s.system,
            Field::Temp => &mut self.s.temperature, Field::MaxTok => &mut self.s.max_tokens, Field::Rename => &mut self.rename,
        }
    }

    fn on_key(&mut self, ev: &KeystrokeEvent, cx: &mut Context<Self>) {
        let k = &ev.keystroke;
        let f = self.field;
        let m = &k.modifiers;
        let multiline = matches!(f, Field::Composer | Field::System);
        if m.control && k.key == "n" { self.new_conv(); }
        else if m.control && k.key == "v" {
            if let Some(t) = cx.read_from_clipboard().and_then(|c| c.text()) {
                let t = t.replace("\r\n", "\n");
                let t = if multiline { t } else { t.trim_end_matches(['\r', '\n']).replace('\n', " ") };
                self.buf(f).push_str(&t);
            }
        }
        else if k.key == "escape" {
            if self.live.is_some() { self.stop(); } else if self.renaming.is_some() { self.renaming = None; self.field = Field::Composer; } else { self.show_settings = false; self.field = Field::Composer; }
        }
        else if k.key == "backspace" {
            if m.control { let b = self.buf(f); let t = b.trim_end_matches(|c: char| !c.is_whitespace()).len(); let t = b[..t].trim_end().len(); b.truncate(t); } else { self.buf(f).pop(); }
        }
        else if k.key == "enter" {
            if m.shift && multiline { self.buf(f).push('\n'); } else { self.submit(f); }
        }
        else if m.control || m.alt || m.platform { return; }
        else if let Some(c) = k.key_char.as_deref() {
            if !c.is_empty() && !c.chars().any(char::is_control) { self.buf(f).push_str(c); } else { return; }
        } else { return; }
        cx.notify();
    }

    fn submit(&mut self, f: Field) {
        match f {
            Field::Composer => self.send(),
            Field::Rename => {
                let (t, id) = (self.rename.trim().to_string(), self.renaming.take());
                if let (Some(id), false) = (id, t.is_empty()) { if let Some(c) = self.conv_mut(id) { c.title = t; } self.save(); }
                self.field = Field::Composer;
            }
            Field::System => { self.s.system.push('\n'); }
            _ => { store::save_settings(&self.s); self.poll(); self.field = Field::Composer; }
        }
    }

    fn send(&mut self) {
        let text = self.input.trim().to_string();
        if text.is_empty() || self.live.is_some() { return; }
        self.input.clear();
        let id = self.cur;
        let effort = self.s.effort.clone();
        let (sys, temp, max) = (self.s.system.trim().to_string(), self.s.temperature.trim().parse::<f64>().unwrap_or(0.7), self.s.max_tokens.trim().parse::<u64>().unwrap_or(8192));
        let Some(c) = self.conv_mut(id) else { return };
        if c.messages.is_empty() { c.title = text.chars().take(48).collect::<String>().replace('\n', " "); }
        c.messages.push(Msg { role: "user".into(), content: text, ..Default::default() });
        let mut msgs: Vec<Value> = vec![];
        if !sys.is_empty() { msgs.push(json!({"role": "system", "content": sys})); }
        for m in &c.messages {
            if m.role == "assistant" && m.content.is_empty() { continue; }
            let content = if m.role == "assistant" { md::split_think(&m.content).1 } else { m.content.clone() };
            msgs.push(json!({"role": m.role, "content": content}));
        }
        c.messages.push(Msg { role: "assistant".into(), effort: effort.clone(), ..Default::default() });
        c.updated = store::now();
        let body = json!({"messages": msgs, "stream": true, "stream_options": {"include_usage": true}, "temperature": temp, "max_tokens": max, "reasoning_effort": effort});
        let cancel = Arc::new(AtomicBool::new(false));
        self.live = Some(Live { conv: id, cancel: cancel.clone(), started: Instant::now(), first: None, chunks: 0 });
        self.stick = true;
        store::save_settings(&self.s);
        self.save();
        net::stream(self.base(), id, body, cancel, self.tx.clone());
    }

    fn stop(&mut self) { if let Some(l) = &self.live { l.cancel.store(true, Ordering::Relaxed); } }

    // ---------- views ----------
    fn button(&self, id: impl Into<SharedString>, label: impl Into<SharedString>, kind: u8, cx: &mut Context<Self>, on: impl Fn(&mut Self) + 'static) -> gpui::Stateful<gpui::Div> {
        let (bg, fg) = match kind { 1 => (ACCENT, 0x101010), 2 => (DANGER, 0x101010), 3 => (PANEL, MUTED), _ => (RAISED, TEXT) };
        div().id(id.into()).px(px(10.)).py(px(5.)).rounded_md().bg(rgb(bg)).text_color(rgb(fg)).cursor_pointer()
            .hover(|s| s.opacity(0.85))
            .child(label.into())
            .on_click(cx.listener(move |this, _: &ClickEvent, _, cx| { on(this); cx.notify(); }))
    }

    fn field_box(&self, f: Field, value: &str, placeholder: &str, cx: &mut Context<Self>) -> gpui::Stateful<gpui::Div> {
        let active = self.field == f;
        let shown: SharedString = if value.is_empty() && !active { placeholder.to_string().into() } else if active { format!("{value}▏").into() } else { value.to_string().into() };
        div().id(SharedString::from(format!("field-{f:?}"))).px(px(10.)).py(px(7.)).rounded_md().bg(rgb(RAISED)).border_1()
            .border_color(rgb(if active { ACCENT } else { RAISED }))
            .text_color(rgb(if value.is_empty() && !active { MUTED } else { TEXT }))
            .cursor_text()
            .child(shown)
            .on_click(cx.listener(move |this, _: &ClickEvent, _, cx| { this.field = f; cx.notify(); }))
    }

    fn sidebar(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let mut list = div().id("conv-list").flex().flex_col().gap(px(2.)).flex_1().overflow_y_scroll();
        for c in &self.convs {
            let (id, cur) = (c.id, c.id == self.cur);
            let row = if self.renaming == Some(id) {
                div().id(SharedString::from(format!("conv-{id}"))).child(self.field_box(Field::Rename, &self.rename, "title", cx))
            } else {
                let title = if c.title.is_empty() { "untitled".to_string() } else { c.title.clone() };
                let busy = self.live.as_ref().map(|l| l.conv) == Some(id);
                div().id(SharedString::from(format!("conv-{id}"))).flex().items_center().gap(px(4.)).px(px(8.)).py(px(6.)).rounded_md()
                    .bg(rgb(if cur { RAISED } else { PANEL })).cursor_pointer().hover(|s| s.bg(rgb(RAISED)))
                    .child(div().flex_1().overflow_hidden().whitespace_nowrap().text_ellipsis().text_sm().text_color(rgb(if cur { TEXT } else { MUTED }))
                        .child(format!("{}{}", if busy { "● " } else { "" }, title)))
                    .when(cur, |d| d
                        .child(self.button(SharedString::from(format!("ren-{id}")), "✎", 3, cx, move |t| {
                            t.rename = t.conv().map(|c| c.title.clone()).unwrap_or_default(); t.renaming = Some(id); t.field = Field::Rename; }))
                        .child(self.button(SharedString::from(format!("del-{id}")), if self.confirm_delete == Some(id) { "delete?" } else { "✕" }, if self.confirm_delete == Some(id) { 2 } else { 3 }, cx, move |t| {
                            if t.confirm_delete == Some(id) { t.delete_conv(id) } else { t.confirm_delete = Some(id) } })))
                    .on_click(cx.listener(move |this, _: &ClickEvent, _, cx| { if this.cur != id { this.cur = id; this.confirm_delete = None; this.stick = true; this.scroll.scroll_to_bottom(); } cx.notify(); }))
            };
            list = list.child(row);
        }
        div().w(px(250.)).h_full().flex().flex_col().gap(px(8.)).p(px(10.)).bg(rgb(PANEL))
            .child(div().flex().items_center().gap(px(8.))
                .child(div().text_color(rgb(ACCENT)).text_size(px(20.)).font_weight(FontWeight::BOLD).child("pyre"))
                .child(div().text_sm().text_color(rgb(MUTED)).child("chat")))
            .child(self.button("new-chat", "+ New chat   (Ctrl+N)", 1, cx, |t| t.new_conv()))
            .child(list)
            .child(div().text_xs().text_color(rgb(MUTED)).child(format!("saved in {}", store::dir().display())))
    }

    fn toolbar(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let mut seg = div().flex().items_center().gap(px(2.)).p(px(2.)).rounded_md().bg(rgb(PANEL));
        for (k, label) in EFFORTS {
            let on = self.s.effort == k;
            seg = seg.child(div().id(SharedString::from(format!("eff-{k}"))).px(px(9.)).py(px(4.)).rounded_md()
                .bg(rgb(if on { ACCENT } else { PANEL })).text_color(rgb(if on { 0x101010 } else { TEXT })).text_sm().cursor_pointer()
                .child(label)
                .on_click(cx.listener(move |this, _: &ClickEvent, _, cx| { this.s.effort = k.to_string(); store::save_settings(&this.s); cx.notify(); })));
        }
        let ok = self.health == "ok";
        div().flex().items_center().gap(px(10.)).px(px(14.)).py(px(8.)).bg(rgb(PANEL)).border_b_1().border_color(rgb(RAISED))
            .child(div().text_sm().text_color(rgb(MUTED)).child("Reasoning"))
            .child(seg)
            .child(div().flex_1())
            .child(div().text_xs().text_color(rgb(MUTED)).child(self.model.rsplit('/').next().unwrap_or("").to_string()))
            .child(div().text_sm().text_color(rgb(if ok { GOOD } else { DANGER })).child(format!("● {}", if ok { self.base() } else { format!("{} ({})", self.base(), self.health) })))
            .child(self.button("settings", if self.show_settings { "Settings ▴" } else { "Settings ▾" }, 0, cx, |t| { t.show_settings = !t.show_settings; if !t.show_settings { store::save_settings(&t.s); t.field = Field::Composer; } }))
    }

    fn settings(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let lab = |t: &str| div().w(px(110.)).text_sm().text_color(rgb(MUTED)).child(t.to_string());
        div().flex().flex_col().gap(px(8.)).p(px(14.)).bg(rgb(PANEL)).border_b_1().border_color(rgb(RAISED))
            .child(div().flex().items_center().gap(px(8.)).child(lab("Server URL")).child(self.field_box(Field::Server, &self.s.server, "http://127.0.0.1:8099", cx).flex_1()))
            .child(div().flex().items_start().gap(px(8.)).child(lab("System prompt")).child(self.field_box(Field::System, &self.s.system, "(none)  Shift+Enter for newline", cx).flex_1()))
            .child(div().flex().items_center().gap(px(8.)).child(lab("Temperature")).child(self.field_box(Field::Temp, &self.s.temperature, "0.7", cx).w(px(90.)))
                .child(lab("Max tokens")).child(self.field_box(Field::MaxTok, &self.s.max_tokens, "8192", cx).w(px(110.)))
                .child(div().flex_1())
                .child(self.button("save-settings", "Save", 1, cx, |t| { store::save_settings(&t.s); t.poll(); t.show_settings = false; t.field = Field::Composer; })))
            .child(div().text_xs().text_color(rgb(MUTED)).child("Enter saves a field. Reasoning effort is sent as reasoning_effort; pyre's default is very high."))
    }

    fn markdown(&self, key: &str, text: &str, cx: &mut Context<Self>) -> gpui::Div {
        let mut col = div().flex().flex_col().gap(px(4.));
        for (i, b) in md::parse(text).into_iter().enumerate() {
            col = col.child(match b {
                md::Block::Para(t) => div().child(t),
                md::Block::Heading(n, t) => div().pt(px(4.)).font_weight(FontWeight::BOLD).text_size(px(match n { 1 => 21., 2 => 18., _ => 16. })).child(t),
                md::Block::Bullet(m, t) => div().flex().gap(px(6.)).child(div().text_color(rgb(ACCENT)).child(m)).child(div().flex_1().child(t)),
                md::Block::Quote(t) => div().pl(px(10.)).border_l_2().border_color(rgb(MUTED)).text_color(rgb(MUTED)).child(t),
                md::Block::Rule => div().h(px(1.)).bg(rgb(RAISED)),
                md::Block::Gap => div().h(px(4.)),
                md::Block::Code { lang, code } => {
                    let copy = code.clone();
                    div().flex().flex_col().rounded_md().bg(rgb(CODE_BG)).border_1().border_color(rgb(RAISED))
                        .child(div().flex().items_center().px(px(10.)).py(px(3.)).bg(rgb(RAISED)).text_xs().text_color(rgb(MUTED))
                            .child(div().flex_1().child(if lang.is_empty() { "code".to_string() } else { lang }))
                            .child(div().id(SharedString::from(format!("copy-{key}-{i}"))).cursor_pointer().hover(|s| s.text_color(rgb(ACCENT))).child("Copy")
                                .on_click(cx.listener(move |this, _: &ClickEvent, _, cx| { cx.write_to_clipboard(ClipboardItem::new_string(copy.clone())); this.notice = "copied code".into(); cx.notify(); }))))
                        .child(div().px(px(10.)).py(px(8.)).font_family(MONO).text_size(px(13.5)).child(code))
                }
            });
        }
        col
    }

    fn message(&self, cid: u64, i: usize, m: &Msg, cx: &mut Context<Self>) -> gpui::AnyElement {
        if m.role == "user" {
            return div().flex().justify_end().child(div().max_w(px(760.)).px(px(12.)).py(px(9.)).rounded_lg().bg(rgb(RAISED)).child(m.content.clone())).into_any_element();
        }
        let streaming = self.live.as_ref().map(|l| l.conv) == Some(cid) && i + 1 == self.conv().map(|c| c.messages.len()).unwrap_or(0);
        let (t2, answer) = md::split_think(&m.content);
        let think = format!("{}{}", m.reasoning, t2);
        let r = &m.info;
        let mut head = String::new();
        let eff = r["effort"].as_str().unwrap_or(&m.effort).to_string();
        if !eff.is_empty() { head.push_str(effort_label(&eff)); }
        if let Some(a) = r["auto"]["level"].as_str() { head.push_str(&format!(" → {} (rating {:.2})", effort_label(a), r["auto"]["rating"].as_f64().unwrap_or(0.0))); }
        let tt = r["thinking_tokens"].as_u64();
        let open = self.open_think.contains(&(cid, i)) != self.think_all;
        let mut col = div().flex().flex_col().gap(px(6.)).max_w(px(900.));
        if !think.trim().is_empty() || tt.unwrap_or(0) > 0 || (streaming && answer.is_empty()) {
            let label = if streaming && answer.is_empty() { format!("Thinking… {} words", think.split_whitespace().count()) } else { "Thought".to_string() };
            let detail = match tt { Some(n) => format!("{} tokens · {}{}", num(n), head, if r["forced_close"] == true { " · budget reached" } else { "" }), None => head.clone() };
            col = col.child(div().flex().flex_col().rounded_md().bg(rgb(PANEL)).border_1().border_color(rgb(RAISED))
                .child(div().id(SharedString::from(format!("think-{cid}-{i}"))).flex().gap(px(8.)).px(px(10.)).py(px(5.)).cursor_pointer().text_sm()
                    .child(div().text_color(rgb(ACCENT)).child(if open { "▾" } else { "▸" }))
                    .child(div().text_color(rgb(TEXT)).child(label))
                    .child(div().text_color(rgb(MUTED)).child(detail))
                    .on_click(cx.listener(move |this, _: &ClickEvent, _, cx| { if !this.open_think.remove(&(cid, i)) { this.open_think.insert((cid, i)); } cx.notify(); })))
                .when(open, |d| d.child(div().px(px(12.)).pb(px(8.)).text_sm().text_color(rgb(MUTED)).child(think.trim().to_string()))));
        } else if !head.is_empty() && !streaming {
            col = col.child(div().text_xs().text_color(rgb(MUTED)).child(format!("reasoning: {head}")));
        }
        if !answer.is_empty() { col = col.child(self.markdown(&format!("{cid}-{i}"), &answer, cx)); }
        if streaming && answer.is_empty() && think.is_empty() {
            let waited = self.live.as_ref().map(|l| l.started.elapsed().as_secs()).unwrap_or(0);
            col = col.child(div().text_color(rgb(MUTED)).child(format!("waiting for pyre… {waited}s (prefill{})", if self.s.effort == "auto" { " + auto probe" } else { "" })));
        }
        if !m.error.is_empty() { col = col.child(div().text_color(rgb(DANGER)).child(format!("error: {}", m.error))); }
        let mut foot = div().flex().items_center().gap(px(10.)).text_xs().text_color(rgb(MUTED));
        if !m.stats.is_empty() { foot = foot.child(m.stats.clone()); }
        if !answer.is_empty() && !streaming {
            let a = answer.clone();
            foot = foot.child(div().id(SharedString::from(format!("copymsg-{cid}-{i}"))).cursor_pointer().hover(|s| s.text_color(rgb(ACCENT))).child("Copy reply")
                .on_click(cx.listener(move |this, _: &ClickEvent, _, cx| { cx.write_to_clipboard(ClipboardItem::new_string(a.clone())); this.notice = "copied reply".into(); cx.notify(); })));
        }
        col.child(foot).into_any_element()
    }

    fn context_line(&self) -> String {
        let used = self.conv().and_then(|c| c.messages.iter().rev().find_map(|m| m.info["usage"]["total_tokens"].as_u64()));
        let live = self.live.as_ref().and_then(|l| l.first.map(|f| format!("  ·  live {:.1} chunks/s", l.chunks as f64 / f.elapsed().as_secs_f64().max(0.001))));
        let ctx = match (used, self.n_ctx) {
            (Some(u), n) if n > 0 => format!("context {} / {} ({:.1}%)", num(u), num(n), u as f64 * 100.0 / n as f64),
            (Some(u), _) => format!("context {} tokens", num(u)),
            (None, n) if n > 0 => format!("context 0 / {}", num(n)),
            _ => "context ?".into(),
        };
        format!("{ctx}{}", live.unwrap_or_default())
    }
}

impl Focusable for ChatApp {
    fn focus_handle(&self, _: &App) -> FocusHandle { self.focus.clone() }
}

impl Render for ChatApp {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let cid = self.cur;
        let msgs = self.conv().map(|c| c.messages.clone()).unwrap_or_default();
        let mut list = div().id("messages").track_scroll(&self.scroll).flex().flex_col().gap(px(14.)).flex_1().overflow_y_scroll().px(px(24.)).py(px(16.));
        if msgs.is_empty() {
            list = list.child(div().flex().flex_col().gap(px(6.)).pt(px(80.)).items_center().text_color(rgb(MUTED))
                .child(div().text_size(px(22.)).text_color(rgb(TEXT)).child("What can pyre help with?"))
                .child(format!("Reasoning: {}  ·  Enter to send, Shift+Enter for a new line, Esc to stop", effort_label(&self.s.effort))));
        }
        for (i, m) in msgs.iter().enumerate() { list = list.child(self.message(cid, i, m, cx)); }
        let busy = self.live.is_some();
        let composer = div().flex().items_end().gap(px(8.)).px(px(24.)).py(px(12.)).bg(rgb(PANEL)).border_t_1().border_color(rgb(RAISED))
            .child(self.field_box(Field::Composer, &self.input, "Message pyre…  (Enter to send, Shift+Enter for newline)", cx).flex_1().min_h(px(40.)))
            .child(if busy { self.button("stop", "■ Stop", 2, cx, |t| t.stop()) } else { self.button("send", "Send", 1, cx, |t| t.send()) });
        let status = div().flex().gap(px(16.)).px(px(24.)).py(px(4.)).bg(rgb(PANEL)).text_xs().text_color(rgb(MUTED))
            .child(self.context_line())
            .child(div().flex_1())
            .child(self.notice.clone());
        div().track_focus(&self.focus).size_full().flex().bg(rgb(BG)).text_color(rgb(TEXT)).text_size(px(15.))
            .child(self.sidebar(cx))
            .child(div().flex_1().h_full().flex().flex_col()
                .child(self.toolbar(cx))
                .when(self.show_settings, |d| d.child(self.settings(cx)))
                .child(list)
                .child(composer)
                .child(status))
    }
}

fn main() {
    Application::new().run(|cx: &mut App| {
        let bounds = Bounds::centered(None, size(px(1240.), px(860.)), cx);
        cx.open_window(
            WindowOptions { window_bounds: Some(WindowBounds::Windowed(bounds)), app_id: Some("pyre-chat".into()), titlebar: Some(gpui::TitlebarOptions { title: Some("pyre chat".into()), ..Default::default() }), ..Default::default() },
            |window, cx| cx.new(|cx| ChatApp::new(window, cx)),
        )
        .expect("open pyre-chat window");
        cx.activate(true);
    });
}
