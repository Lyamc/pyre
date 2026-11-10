//! Live dashboard page: polls `/api/metrics` once a second. Layout idea (gauges + sparklines +
//! request table at a glance) inspired by Strata's TUI (github.com/Niko1221/Strata); no code shared.
use crate::{PyreGui, ACCENT, MUTED, PANEL, RAISED, TEXT, DANGER};
use gpui::{div, prelude::*, px, rgb, Context};
use serde_json::{json, Value};

const GOOD: u32 = 0x7bd88f;
const WARN: u32 = 0xe5c07b;
pub const HIST: usize = 90;

fn f(v: &Value) -> f64 { v.as_f64().unwrap_or(0.0) }
fn gib(b: f64) -> String { format!("{:.1} GiB", b / (1u64 << 30) as f64) }
fn num(n: f64) -> String { if n >= 1e6 { format!("{:.2}M", n / 1e6) } else if n >= 1e4 { format!("{:.1}k", n / 1e3) } else { format!("{n:.0}") } }

/// Horizontal gauge: label, fraction 0..1, right-hand text. `mark` draws a reserve tick.
fn gauge(label: &str, frac: f64, text: String, mark: Option<f64>) -> impl IntoElement {
    let frac = frac.clamp(0.0, 1.0);
    let col = if frac > 0.9 { DANGER } else if frac > 0.75 { WARN } else { ACCENT };
    let mut bar = div().relative().h(px(12.)).flex_1().rounded_sm().bg(rgb(RAISED))
        .child(div().absolute().left_0().top_0().h_full().w(gpui::relative(frac as f32)).rounded_sm().bg(rgb(col)));
    if let Some(m) = mark {
        bar = bar.child(div().absolute().top_0().h_full().w(px(2.)).left(gpui::relative(m.clamp(0.0, 1.0) as f32)).bg(rgb(TEXT)));
    }
    div().flex().items_center().gap(px(10.)).text_sm()
        .child(div().w(px(92.)).text_color(rgb(MUTED)).child(label.to_string()))
        .child(bar)
        .child(div().w(px(300.)).child(text))
}

fn spark(title: String, h: &std::collections::VecDeque<f64>, col: u32) -> impl IntoElement {
    let max = h.iter().cloned().fold(1e-9, f64::max);
    let mut row = div().flex().items_end().gap(px(1.)).h(px(46.));
    for _ in h.len()..HIST { row = row.child(div().w(px(3.)).h(px(1.)).bg(rgb(RAISED))); }
    for v in h { row = row.child(div().w(px(3.)).h(px((v / max * 44.0).max(1.0) as f32)).bg(rgb(col))); }
    div().flex().flex_col().gap(px(4.)).p(px(10.)).rounded_md().bg(rgb(PANEL))
        .child(div().text_xs().text_color(rgb(MUTED)).child(title)).child(row)
}

fn card(title: &str, value: String, sub: String) -> impl IntoElement {
    div().flex().flex_col().flex_1().p(px(10.)).rounded_md().bg(rgb(PANEL))
        .child(div().text_xs().text_color(rgb(MUTED)).child(title.to_string()))
        .child(div().text_size(px(22.)).text_color(rgb(TEXT)).child(value))
        .child(div().text_xs().text_color(rgb(MUTED)).child(sub))
}

fn cells(cols: &[(f32, String)], color: u32) -> impl IntoElement {
    let mut r = div().flex().gap(px(8.)).text_xs().text_color(rgb(color));
    for (w, s) in cols { r = r.child(div().w(px(*w)).overflow_hidden().child(s.clone())); }
    r
}

impl PyreGui {
    pub fn dashboard_page(&self, cx: &mut Context<Self>) -> gpui::AnyElement {
        let m = &self.metrics;
        if m.is_null() {
            return div().text_color(rgb(MUTED)).child(format!("no /api/metrics from {} (older pyre or not running)", self.server)).into_any_element();
        }
        let l = &m["limits"];
        let mib = |k: &str| f(&l[k]) * (1u64 << 20) as f64;
        let (vt, vu, vf, vr) = (mib("vram_total_mib"), mib("vram_used_mib"), mib("vram_free_mib"), mib("vram_reserve_mib"));
        let ram = &m["ram"];
        let active = m["active"].as_array().cloned().unwrap_or_default();
        let recent = m["recent"].as_array().cloned().unwrap_or_default();
        let queued: u64 = m["models"].as_array().map(|a| a.iter().map(|x| x["queued"].as_u64().unwrap_or(0)).sum()).unwrap_or(0);
        let live = active.iter().find(|a| a["phase"] == "decode").or(active.first());
        let last = recent.first();
        let (ctx_used, n_ctx) = live.or(last).map(|a| (f(&a["ctx_used"]), f(&a["n_ctx"]))).unwrap_or((0.0, 0.0));
        let n_ctx = if n_ctx > 0.0 { n_ctx } else { m["models"].as_array().and_then(|a| a.iter().find_map(|x| x["meta"]["n_ctx"].as_f64())).unwrap_or(0.0) };
        let dec = live.map(|a| f(&a["decode_tps"])).filter(|v| *v > 0.0).or(last.map(|a| f(&a["decode_tps"]))).unwrap_or(0.0);
        let pre = live.and_then(|a| a["prefill_tps"].as_f64()).or(last.map(|a| f(&a["prefill_tps"]))).unwrap_or(0.0);
        let comp = &m["compute"];
        let (cpu_ms, gpu_ms) = (f(&comp["cpu_expert_ms"]), f(&comp["gpu_ms"]));
        let hit = m["expert_cache"]["hit_rate"].as_f64();
        let port = self.server.rsplit(':').next().unwrap_or("").trim_end_matches('/').to_string();
        let lan = m["lan_ip"].as_str().map(|ip| format!("http://{ip}:{port}/v1")).unwrap_or_else(|| "?".into());

        let models = m["models"].as_array().map(|a| a.iter().map(|x| format!("{} [{}{}{}]",
            x["id"].as_str().unwrap_or(""), x["kind"].as_str().unwrap_or(""),
            x["meta"]["n_ctx"].as_u64().map(|n| format!(", ctx {}", num(n as f64))).unwrap_or_default(),
            if x["busy"].as_bool() == Some(true) { ", busy" } else { "" })).collect::<Vec<_>>().join("   ")).unwrap_or_default();

        let header = div().flex().gap(px(16.)).text_sm().p(px(10.)).rounded_md().bg(rgb(PANEL))
            .child(div().text_color(rgb(GOOD)).child(format!("● {}", self.server)))
            .child(div().text_color(rgb(MUTED)).child(format!("LAN {lan}")))
            .child(div().text_color(rgb(MUTED)).child(format!("pyre {}  up {}m", m["version"].as_str().unwrap_or("?"), f(&m["uptime_s"]) as u64 / 60)))
            .child(div().flex_1().text_color(rgb(TEXT)).child(if models.is_empty() { "no models loaded".into() } else { models }));

        let cards = div().flex().gap(px(10.))
            .child(card("decode", format!("{dec:.1} tok/s"), live.map(|a| format!("{} live", a["phase"].as_str().unwrap_or(""))).unwrap_or("last request".into())))
            .child(card("prefill", format!("{pre:.0} tok/s"), last.map(|a| format!("last {:.0} ms", f(&a["prefill_ms"]))).unwrap_or_default()))
            .child(card("requests", format!("{} active", active.len()), format!("{queued} queued · {} done", m["totals"]["requests"])))
            .child(card("tokens", num(f(&m["totals"]["generated_tokens"])), format!("{} prompt", num(f(&m["totals"]["prompt_tokens"])))))
            .child(card("expert cache", hit.map(|h| format!("{:.1}%", h * 100.0)).unwrap_or("n/a".into()),
                format!("CPU experts {:.1}s · GPU {:.1}s", cpu_ms / 1e3, gpu_ms / 1e3)));

        let gauges = div().flex().flex_col().gap(px(8.)).p(px(10.)).rounded_md().bg(rgb(PANEL))
            .child(gauge("context", if n_ctx > 0.0 { ctx_used / n_ctx } else { 0.0 }, format!("{} / {}", num(ctx_used), num(n_ctx)), None))
            .child(gauge("VRAM", if vt > 0.0 { vu / vt } else { 0.0 }, format!("{} used · {} free · rsv {}", gib(vu), gib(vf), gib(vr)), if vt > 0.0 { Some(1.0 - vr / vt) } else { None }))
            .child(gauge("GPU util", f(&l["gpu_util_now_pct"]) / 100.0, format!("{}% (cap {}%)", l["gpu_util_now_pct"], l["gpu_util_max_pct"]), Some(f(&l["gpu_util_max_pct"]) / 100.0)))
            .child(gauge("RAM", if f(&ram["total"]) > 0.0 { f(&ram["used"]) / f(&ram["total"]) } else { 0.0 }, format!("{} / {} (pyre {})", gib(f(&ram["used"])), gib(f(&ram["total"])), gib(f(&ram["process_rss"]))), None))
            .child(gauge("CPU", f(&m["cpu_pct"]) / 100.0, format!("{:.0}%", f(&m["cpu_pct"])), None))
            .child(gauge("CPU vs GPU", if cpu_ms + gpu_ms > 0.0 { cpu_ms / (cpu_ms + gpu_ms) } else { 0.0 }, if cpu_ms + gpu_ms > 0.0 { format!("{:.0}% expert time on CPU", cpu_ms / (cpu_ms + gpu_ms) * 100.0) } else { "no data (backend hook)".into() }, None));

        let sparks = div().flex().gap(px(10.))
            .child(spark(format!("decode tok/s (now {dec:.1})"), &self.hist_decode, ACCENT))
            .child(spark(format!("GPU util % (now {})", l["gpu_util_now_pct"]), &self.hist_gpu, GOOD))
            .child(spark(format!("VRAM used (now {})", gib(vu)), &self.hist_vram, WARN));

        let w = [(46., "id"), (240., "model"), (70., "state"), (70., "prompt"), (60., "gen"), (110., "ctx"), (90., "prefill t/s"), (90., "decode t/s"), (70., "total")];
        let mut table = div().id("dash-req").flex().flex_col().gap(px(3.)).flex_1().overflow_y_scroll().p(px(10.)).rounded_md().bg(rgb(PANEL))
            .child(cells(&w.iter().map(|(a, b)| (*a, b.to_string())).collect::<Vec<_>>(), MUTED));
        let row = |a: &Value, state: String, total: String, col: u32| cells(&[
            (46., a["id"].to_string()), (240., a["model"].as_str().unwrap_or("").to_string()), (70., state),
            (70., a["prompt_n"].to_string()), (60., a["gen_n"].to_string()),
            (110., format!("{} / {}", num(f(&a["ctx_used"])), num(f(&a["n_ctx"])))),
            (90., format!("{:.0}", f(&a["prefill_tps"]))), (90., format!("{:.1}", f(&a["decode_tps"]))), (70., total)], col);
        for a in &active { table = table.child(row(a, a["phase"].as_str().unwrap_or("").into(), format!("{:.1}s", f(&a["elapsed_s"])), ACCENT)); }
        for a in recent.iter().take(30) {
            let ok = a["outcome"] == "ok";
            table = table.child(row(a, a["outcome"].as_str().unwrap_or("").into(), format!("{:.1}s", f(&a["total_ms"]) / 1e3), if ok { TEXT } else { DANGER }));
        }

        let qc = &m["qwen_context"];
        let at = &m["autotune"];
        let prof = &m["profiles"];
        let prof_line = if prof.is_null() { String::new() } else {
            let src = prof["active"]["source"].as_str().unwrap_or("default");
            let name = prof["active"]["name"].as_str().unwrap_or("");
            let shown = if src == "custom" && !name.is_empty() { format!("custom “{name}”") } else { src.to_string() };
            let stale = if prof["tune_stale"].as_bool() == Some(true) { " · tune was kept; hardware changed" } else { "" };
            format!("settings source {shown}{stale}")
        };
        let ctx_line = if qc.is_null() { String::new() } else {
            format!("context {} · KV {}-bit · {}", qc["context_label"].as_str().unwrap_or("?"), qc["kv_bits"].as_u64().unwrap_or(4), qc["fit"]["message"].as_str().unwrap_or(""))
        };
        let suggest = at["suggest"].as_bool() == Some(true);
        let tune_msg = at["message"].as_str().unwrap_or("").to_string();
        let mut page = div().flex().flex_col().size_full().gap(px(10.)).child(header);
        if suggest {
            page = page.child(div().p(px(10.)).rounded_md().bg(rgb(PANEL)).text_sm().text_color(rgb(WARN)).child(tune_msg.clone())
                .child(div().flex().gap(px(8.)).pt(px(8.))
                    .child(self.button("tune-keep", "Keep these results", false, cx, |t| { crate::net::post(t.base(), "/api/autotune/ack", json!({}), "autotune", t.tx.clone()); }))
                    .child(self.button("tune-re", "Retune on next start", true, cx, |t| { crate::net::post(t.base(), "/api/autotune/retune", json!({}), "autotune", t.tx.clone()); }))));
        }
        if !ctx_line.is_empty() {
            page = page.child(div().p(px(10.)).rounded_md().bg(rgb(PANEL)).text_sm().child(ctx_line)
                .when(!prof_line.is_empty(), |d| d.child(div().pt(px(4.)).text_color(rgb(MUTED)).child(prof_line)))
                .child(div().flex().gap(px(8.)).pt(px(8.))
                    .child(self.button("ctx128", "128k", false, cx, |t| { crate::net::post(t.base(), "/api/qwen/context", json!({"context":"128k"}), "qwen-context", t.tx.clone()); }))
                    .child(self.button("ctx256", "256k", false, cx, |t| { crate::net::post(t.base(), "/api/qwen/context", json!({"context":"256k"}), "qwen-context", t.tx.clone()); }))
                    .child(self.button("ctx384", "384k", false, cx, |t| { crate::net::post(t.base(), "/api/qwen/context", json!({"context":"384k"}), "qwen-context", t.tx.clone()); }))
                    .child(self.button("ctx512", "512k", false, cx, |t| { crate::net::post(t.base(), "/api/qwen/context", json!({"context":"512k"}), "qwen-context", t.tx.clone()); }))
                    .child(self.button("kv4", "KV 4-bit", false, cx, |t| { crate::net::post(t.base(), "/api/qwen/context", json!({"kv_bits":4}), "qwen-kv", t.tx.clone()); }))
                    .child(self.button("kv8", "KV 8-bit", false, cx, |t| { crate::net::post(t.base(), "/api/qwen/context", json!({"kv_bits":8}), "qwen-kv", t.tx.clone()); }))));
        }
        page.child(cards).child(gauges).child(sparks).child(table).into_any_element()
    }
}

impl PyreGui {
    pub fn push_hist(&mut self, v: &Value) {
        let dec = v["active"].as_array().and_then(|a| a.iter().find(|x| x["phase"] == "decode")).map(|x| f(&x["decode_tps"])).unwrap_or(0.0);
        let push = |h: &mut std::collections::VecDeque<f64>, x: f64| { h.push_back(x); while h.len() > HIST { h.pop_front(); } };
        push(&mut self.hist_decode, dec);
        push(&mut self.hist_gpu, f(&v["limits"]["gpu_util_now_pct"]));
        push(&mut self.hist_vram, f(&v["limits"]["vram_used_mib"]));
    }
}
