//! PYRE_QN_XCHAIN=1 (measurement only, no effect on results): per XGATE-gated layer, the chain after the CPU pool's
//! done flag, from device %globaltimer stamps (cubecl-pipe pyre_stamp, one-thread kernels writing mapped pinned
//! memory) and host clocks. Stamps: S0 before the gated row pull (GPU reaches the gate), S1 after it (rows on the
//! GPU), S2 after the sum/combine, S3 after the next layer's dense half; host: Hf = pool done (flag store),
//! Hd = next layer's rd doorbell seen. GPU->host clock offset = min over rd readbacks of (doorbell seen - stamp just
//! before the readback kernel), i.e. biased late by that kernel's few us.
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

pub fn on() -> bool { static K: OnceLock<bool> = OnceLock::new(); *K.get_or_init(|| std::env::var("PYRE_QN_XCHAIN").map(|v| v == "1").unwrap_or(false)) }
fn hns() -> i64 { static B: OnceLock<Instant> = OnceLock::new(); B.get_or_init(Instant::now).elapsed().as_nanos() as i64 }

const NR: usize = 256;
struct St { page: usize, ring: usize, cur: Option<(usize, usize)>, pend: Option<(usize, usize, i64)>, cal: usize, off: i64,
    // per record: wait_gpu, rows, sum, dense, door, total (us), kb
    rs: Vec<[f64; 7]>, pool: Vec<f64>, nprint: usize }
static S: Mutex<St> = Mutex::new(St { page: 0, ring: 0, cur: None, pend: None, cal: 0, off: i64::MAX, rs: Vec::new(), pool: Vec::new(), nprint: 0 });

fn page(cb: &super::qn_cube::Cube, s: &mut St) -> Option<usize> {
    if s.page == 0 { match cb.client.pyre_host_pinned(NR * 64 + 4096) { Ok(p) => { unsafe { std::ptr::write_bytes(p as *mut u8, 0, NR * 64 + 4096); } s.page = p; } Err(e) => { eprintln!("PYRE_QN_XCHAIN: pinned alloc failed ({e})"); return None; } } }
    Some(s.page)
}
fn rd(a: usize) -> i64 { unsafe { std::ptr::read_volatile(a as *const u64) as i64 } }
/// A new record for a gated layer (rows bytes `bytes`); returns its base address (stamps at +0, +8, +16, +24).
pub fn rec(cb: &super::qn_cube::Cube, bytes: usize) -> Option<usize> {
    if !on() { return None; }
    let mut s = S.lock().unwrap(); let p = page(cb, &mut s)?;
    let a = p + (s.ring % NR) * 64; s.ring += 1; unsafe { std::ptr::write_bytes(a as *mut u8, 0, 64); }
    s.cur = Some((a, bytes)); Some(a)
}
pub fn stamp(cb: &super::qn_cube::Cube, a: usize) { unsafe { cb.client.pyre_stamp(a) } }
/// The pool is done and the flag is about to be stored.
pub fn flag() { if !on() { return; } let t = hns(); let mut s = S.lock().unwrap(); if let Some((a, b)) = s.cur.take() { s.pend = Some((a, b, t)); } }
pub fn pool(ns: u64) { if !on() { return; } S.lock().unwrap().pool.push(ns as f64 / 1000.0); }
/// Before an rd readback kernel: stamp a calibration slot.
pub fn cal(cb: &super::qn_cube::Cube) -> Option<usize> {
    if !on() { return None; }
    let mut s = S.lock().unwrap(); let p = page(cb, &mut s)?; let a = p + NR * 64 + (s.cal % 8) * 8; s.cal += 1;
    unsafe { std::ptr::write_volatile(a as *mut u64, 0); } drop(s); stamp(cb, a); Some(a)
}
/// The rd doorbell was seen.
pub fn door(c: Option<usize>) {
    let Some(c) = c else { return; }; let hd = hns();
    let mut s = S.lock().unwrap(); let sc = rd(c); if sc != 0 { s.off = s.off.min(hd - sc); }
    if let Some((a, kb, hf)) = s.pend.take() {
        let st = [rd(a), rd(a + 8), rd(a + 16), rd(a + 24)];
        if st.iter().all(|&x| x != 0) && s.off != i64::MAX {
            let g = |x: i64| x + s.off; let (s0, s1, s2, s3) = (g(st[0]), g(st[1]), g(st[2]), g(st[3])); let u = |x: i64| x as f64 / 1000.0;
            s.rs.push([u(hf - s0), u(s1 - s0.max(hf)), u(s2 - s1), u(s3 - s2), u(hd - s3), u(hd - hf), kb as f64 / 1024.0]);
        }
    }
    if s.rs.len() >= 256 * (s.nprint + 1) { s.nprint += 1; report(&s); }
}
fn pct(v: &[f64], p: f64) -> f64 { if v.is_empty() { return 0.0; } let mut w = v.to_vec(); w.sort_by(|a, b| a.partial_cmp(b).unwrap()); w[((w.len() - 1) as f64 * p).round() as usize] }
fn report(s: &St) {
    let names = ["gpu_wait", "rows", "sum", "dense", "door", "flag->door"];
    let line = |sel: &dyn Fn(&[f64; 7]) -> bool, tag: &str| { let r: Vec<&[f64; 7]> = s.rs.iter().filter(|x| sel(x)).collect(); if r.is_empty() { return format!("{tag} n 0"); }
        let mut o = format!("{tag} n {} kb {:.0}", r.len(), r.iter().map(|x| x[6]).sum::<f64>() / r.len() as f64);
        for (i, nm) in names.iter().enumerate() { let v: Vec<f64> = r.iter().map(|x| x[i]).collect(); o += &format!(" | {nm} {:.1} ({:.0}/{:.0}/{:.0})", v.iter().sum::<f64>() / v.len() as f64, pct(&v, 0.1), pct(&v, 0.5), pct(&v, 0.9)); } o };
    let p = &s.pool;
    eprintln!("xchain all: {}", line(&|_| true, "all"));
    eprintln!("xchain cpu-crit (gpu waited at gate): {}", line(&|x| x[0] > 0.0, "cc"));
    eprintln!("xchain gpu-late (rows ready first): {}", line(&|x| x[0] <= 0.0, "gl"));
    eprintln!("xchain pool us/layer n {} mean {:.0} p10 {:.0} p50 {:.0} p90 {:.0} p99 {:.0} | clock off {} ns", p.len(), p.iter().sum::<f64>() / p.len().max(1) as f64, pct(p, 0.1), pct(p, 0.5), pct(p, 0.9), pct(p, 0.99), s.off);
}
