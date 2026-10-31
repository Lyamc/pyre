//! A small persistent worker pool for the per-layer expert phases of a decode step. Workers spin
//! for a short while after each job before parking, so the ~100 short jobs per token do not pay a
//! thread wake-up each (Strata keeps a dedicated pinned pool for the same reason; this is pyre's own).
use std::sync::atomic::{AtomicPtr, AtomicU64, AtomicUsize, Ordering::*};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

/// PYRE_QN_WBUSY=1: per-worker busy ns (inside tasks) vs summed job wall time, printed every 2000 jobs
fn wbusy() -> bool { static P: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *P.get_or_init(|| std::env::var("PYRE_QN_WBUSY").map(|v| v == "1").unwrap_or(false)) }
static WB: [AtomicU64; 64] = [const { AtomicU64::new(0) }; 64]; static WJ: AtomicU64 = AtomicU64::new(0); static WN: AtomicU64 = AtomicU64::new(0);
thread_local! { static WID: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }
/// PYRE_QN_JPROF=1: per job (split: with side / plain), mean us from publish to: first worker claim, caller side end,
/// caller own work end, last task claim, last task finish, caller return; plus busy % of (threads x wall)
fn jprof() -> bool { static P: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *P.get_or_init(|| std::env::var("PYRE_QN_JPROF").map(|v| v == "1").unwrap_or(false)) }
static EP: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
fn now_ns() -> u64 { EP.get_or_init(Instant::now).elapsed().as_nanos() as u64 }
static JFC: AtomicU64 = AtomicU64::new(0); static JLC: AtomicU64 = AtomicU64::new(0); static JLF: AtomicU64 = AtomicU64::new(0); static JB: AtomicU64 = AtomicU64::new(0);
static JA: [[AtomicU64; 8]; 2] = [const { [const { AtomicU64::new(0) }; 8] }; 2];
/// PYRE_QN_XPSTAT=1 (measurement only): per routed-pool job (tagged by XP_TAG = misses/experts of the layer), per worker first claim,
/// last finish, busy ns and task count; the distribution of job wall and where the slow jobs lose time (wake / P vs E / tail) is
/// printed every 600 jobs ('xpstat'). Which worker runs a task is unchanged -> results identical.
pub fn xpstat() -> bool { static P: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *P.get_or_init(|| std::env::var("PYRE_QN_XPSTAT").map(|v| v == "1").unwrap_or(false)) }
pub static XP_TAG: AtomicUsize = AtomicUsize::new(0);
static XS_FC: [AtomicU64; 64] = [const { AtomicU64::new(u64::MAX) }; 64]; static XS_LF: [AtomicU64; 64] = [const { AtomicU64::new(0) }; 64];
static XS_MX: [AtomicU64; 64] = [const { AtomicU64::new(0) }; 64]; static XS_MI: [AtomicU64; 64] = [const { AtomicU64::new(0) }; 64];
static XS_BZ: [AtomicU64; 64] = [const { AtomicU64::new(0) }; 64]; static XS_NT: [AtomicU64; 64] = [const { AtomicU64::new(0) }; 64];
#[derive(Clone, Copy, Default)] struct XRec { wall: u64, tag: u64, slp: u64, cfc: u64, pfc: u64, efc: u64, late: u64, last: u64, lid: usize, med: u64, cend: u64, mx: u64, mxw: usize, mxi: u64, n: u64, bz: [u64; 3], nt: [u64; 3] }
static XRECS: Mutex<Vec<XRec>> = Mutex::new(Vec::new());
fn xp_print(v: &mut Vec<XRec>, threads: usize) {
    let mut w: Vec<u64> = v.iter().map(|r| r.wall).collect(); w.sort(); let q = |f: f64| w[((w.len() - 1) as f64 * f) as usize] as f64 / 1e3;
    let thr = w[(w.len() - 1) * 9 / 10];
    let agg = |sel: &dyn Fn(&XRec) -> bool| { let s: Vec<&XRec> = v.iter().filter(|r| sel(r)).collect(); let n = s.len().max(1) as f64; let m = |f: &dyn Fn(&XRec) -> f64| s.iter().map(|r| f(r)).sum::<f64>() / n;
        let (mut lcc, mut lcp, mut lce) = (0, 0, 0); for r in &s { match r.lid { 0 => lcc += 1, x if x >= 1000 => lce += 1, _ => lcp += 1 } }
        format!("n {} wall {:.0} misses {:.2} sleepers@pub {:.2} callerStart {:.0} maxClaimP {:.0} maxClaimE {:.0} late>50us {:.2} lastEnd {:.0} medEnd {:.0} tail {:.0} | last=C/P/E {}/{}/{} | busy% {:.0}",
            s.len(), m(&|r| r.wall as f64 / 1e3), m(&|r| r.tag as f64), m(&|r| r.slp as f64), m(&|r| r.cfc as f64 / 1e3), m(&|r| r.pfc as f64 / 1e3), m(&|r| r.efc as f64 / 1e3), m(&|r| r.late as f64),
            m(&|r| r.last as f64 / 1e3), m(&|r| r.med as f64 / 1e3), m(&|r| (r.last - r.med.min(r.last)) as f64 / 1e3), lcc, lcp, lce,
            100.0 * s.iter().map(|r| r.bz.iter().sum::<u64>()).sum::<u64>() as f64 / (s.iter().map(|r| r.wall).sum::<u64>().max(1) as f64 * threads as f64)) };
    let pt = |c: usize| { let b: u64 = v.iter().map(|r| r.bz[c]).sum(); let n: u64 = v.iter().map(|r| r.nt[c]).sum(); b as f64 / n.max(1) as f64 / 1e3 };
    let mut per: Vec<(u64, u64)> = vec![(0, 0); 40]; for r in v.iter() { let k = (r.tag as usize).min(39); per[k].0 += r.wall; per[k].1 += 1; }
    let pm: String = per.iter().enumerate().filter(|x| x.1 .1 >= 20).map(|(k, x)| format!(" {k}:{:.0}", x.0 as f64 / x.1 as f64 / 1e3)).collect();
    eprintln!("xpstat jobs {} wall us p10 {:.0} p50 {:.0} p90 {:.0} p99 {:.0} max {:.0} | us/task C {:.1} P {:.1} E {:.1} (E/P {:.2})\n  all : {}\n  slow: {}\n  wall by misses:{pm}",
        w.len(), q(0.1), q(0.5), q(0.9), q(0.99), q(1.0), pt(0), pt(1), pt(2), pt(2) / pt(1).max(1e-9), agg(&|_| true), agg(&|r| r.wall >= thr));
    let mut ix: Vec<usize> = (0..v.len()).collect(); ix.sort_by_key(|&i| std::cmp::Reverse(v[i].wall));
    eprintln!("  worst:{}", ix.iter().take(4).map(|&i| { let r = &v[i]; format!(" [wall {:.0} callerStart {:.0} maxClaimP {:.0} maxClaimE {:.0} last w{} at {:.0} med {:.0} callerEnd {:.0} misses {} longestTask {:.0} w{} task {}/{}]", r.wall as f64 / 1e3, r.cfc as f64 / 1e3, r.pfc as f64 / 1e3, r.efc as f64 / 1e3, r.lid, r.last as f64 / 1e3, r.med as f64 / 1e3, r.cend as f64 / 1e3, r.tag, r.mx as f64 / 1e3, r.mxw, r.mxi, r.n) }).collect::<String>());
    v.clear();
}
#[inline] fn call(f: &(dyn Fn(usize) + Sync), i: usize) { if xpstat() { let w = WID.with(|w| w.get()).min(63); let t = now_ns(); XS_FC[w].fetch_min(t, Relaxed); f(i); let e = now_ns(); XS_LF[w].fetch_max(e, Relaxed); XS_BZ[w].fetch_add(e - t, Relaxed); XS_NT[w].fetch_add(1, Relaxed); if e - t > XS_MX[w].load(Relaxed) { XS_MX[w].store(e - t, Relaxed); XS_MI[w].store(i as u64, Relaxed); } return; } if jprof() { let t = now_ns(); if WID.with(|w| w.get()) != 0 { JFC.fetch_min(t, Relaxed); } JLC.fetch_max(t, Relaxed); f(i); let e = now_ns(); JLF.fetch_max(e, Relaxed); JB.fetch_add(e - t, Relaxed); return; } if wbusy() { let t = Instant::now(); f(i); WB[WID.with(|w| w.get()).min(63)].fetch_add(t.elapsed().as_nanos() as u64, Relaxed); } else { f(i) } }
struct Job<'a> { f: &'a (dyn Fn(usize) + Sync), n: usize }

struct Shared {
    gen: AtomicU64,
    ticket: AtomicU64, // (gen << 32) | next task index
    job: AtomicPtr<Job<'static>>,
    done: AtomicUsize,
    n: AtomicUsize,
    sleepers: AtomicUsize,
    lock: Mutex<()>,
    cv: Condvar,
    spin: Duration,
    /// the last `etail` tasks of a job are left to the fast workers (slow cores would finish last)
    etail: usize,
    /// bumped by prewake(): spinning workers restart their spin window, parked ones wake and spin
    kick: AtomicU64,
    pin0: Option<usize>,
    nfast: usize,
}

pub struct Pool { sh: Arc<Shared>, pub threads: usize }

impl Pool {
    /// `pin`: logical CPU per worker (index 0 is the caller and is not pinned).
    pub fn new(threads: usize, spin: Duration) -> Self { Self::new_pinned(threads, spin, &[]) }
    pub fn new_pinned(threads: usize, spin: Duration, pin: &[usize]) -> Self { Self::new_hybrid(threads, spin, pin, usize::MAX, 0) }
    /// Workers with index >= `n_fast` are slow (E-cores): they skip a job's last `etail` tasks.
    pub fn new_hybrid(threads: usize, spin: Duration, pin: &[usize], n_fast: usize, etail: usize) -> Self {
        let sh = Arc::new(Shared { gen: AtomicU64::new(0), ticket: AtomicU64::new(0), job: AtomicPtr::new(std::ptr::null_mut()), done: AtomicUsize::new(0), n: AtomicUsize::new(0),
            sleepers: AtomicUsize::new(0), lock: Mutex::new(()), cv: Condvar::new(), spin, etail, kick: AtomicU64::new(0),
            pin0: if std::env::var("PYRE_QN_XPIN0").map(|v| v == "1").unwrap_or(xexact_env()) { pin.first().copied() } else { None }, nfast: n_fast });
        for i in 1..threads.max(1) {
            let sh = sh.clone(); let cpu = pin.get(i).copied();
            std::thread::Builder::new().name(format!("pyre-xp{i}")).spawn(move || { WID.with(|w| w.set(i)); if let Some(c) = cpu { pin_to(c); } thr_tune(); worker(&sh, i >= n_fast) }).expect("spawn pool worker");
        }
        proc_tune(); thr_tune();
        Pool { sh, threads: threads.max(1) }
    }

    /// Wake parked workers and restart every worker's spin window now, so they are already spinning when the
    /// next job arrives (call just before blocking on the GPU readback that precedes a job).
    pub fn prewake(&self) {
        let sh = &*self.sh; sh.kick.fetch_add(1, Release);
        if sh.sleepers.load(Acquire) > 0 { let _l = sh.lock.lock().unwrap(); sh.cv.notify_all(); }
    }
    /// Run f(0..n) across the pool (the caller works too) and return when all are done.
    pub fn run(&self, n: usize, f: &(dyn Fn(usize) + Sync)) { self.run_impl(n, f, &mut || {}, false) }
    /// Like `run`, but the caller first does `side` (e.g. enqueueing GPU work) while the workers start.
    pub fn run_side(&self, n: usize, f: &(dyn Fn(usize) + Sync), side: &mut dyn FnMut()) { self.run_impl(n, f, side, true) }
    fn run_impl(&self, n: usize, f: &(dyn Fn(usize) + Sync), side: &mut dyn FnMut(), has_side: bool) {
        if n == 0 { side(); return; }
        let sh = &*self.sh;
        let js = if jprof() { let t = now_ns(); JFC.store(u64::MAX, Relaxed); JLC.store(0, Relaxed); JLF.store(0, Relaxed); JB.store(0, Relaxed); t } else { 0 };
        // PYRE_QN_XPIN0=1 (opt-in): the calling thread (pool index 0) is pinned to pin[0], the one core with no worker, instead
        // of floating (it could share a core with a pinned worker and halve that core's speed for the whole job)
        thread_local! { static P0: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }
        if let Some(c) = sh.pin0 { if !P0.with(|p| p.replace(true)) { pin_to(c); } }
        let job = Job { f, n };
        let xtag = if xpstat() { XP_TAG.load(Relaxed) } else { 0 };
        let (xjs, xslp) = if xtag > 0 { for w in 0..self.threads.min(64) { XS_FC[w].store(u64::MAX, Relaxed); XS_LF[w].store(0, Relaxed); XS_BZ[w].store(0, Relaxed); XS_NT[w].store(0, Relaxed); XS_MX[w].store(0, Relaxed); } (now_ns(), sh.sleepers.load(Relaxed) as u64) } else { (0, 0) };
        let g = sh.gen.load(Relaxed) + 1;
        // CAS path (TFA off or etail): the ticket moves to generation g FIRST: a worker still in work() for the previous job
        // may have read that job's exhausted ticket and then read the new (larger) n; its compare_exchange on the old ticket
        // must fail, or it would run task i twice through a stale job pointer and over-count `done`. (TFA claims carry
        // their own generation and n and need job/n published BEFORE the ticket, so they keep that order.)
        let tf = tfa() && sh.etail == 0;
        if !tf { sh.ticket.store(g << 32, SeqCst); }
        sh.done.store(0, Relaxed);
        sh.n.store(n, Release);
        sh.job.store(&job as *const Job as *mut Job<'static>, Relaxed);
        if tf { assert!(n + PB < 1 << 20); sh.ticket.store(((g & 0xfffff) << 44) | (((n + if pedq() { PB } else { 0 }) as u64) << 24), Release); }
        sh.gen.store(g, Release);
        if sh.sleepers.load(Acquire) > 0 { let _l = sh.lock.lock().unwrap(); sh.cv.notify_all(); }
        let tj = Instant::now();
        side(); let ts = if jprof() { now_ns() } else { 0 };
        if !cnw() { work(sh, g, &job, false); } let tc = if jprof() { now_ns() } else { 0 };
        while sh.done.load(Acquire) < n { std::hint::spin_loop(); }
        if xtag > 0 { let te = now_ns(); let nt = self.threads.min(64); let mut r = XRec { wall: te - xjs, tag: xtag as u64, slp: xslp, ..Default::default() };
            let mut ends: Vec<u64> = vec![];
            r.n = n as u64; for w in 0..nt { let m = XS_MX[w].load(Relaxed); if m > r.mx { r.mx = m; r.mxw = w; r.mxi = XS_MI[w].load(Relaxed); } }
            r.cend = XS_LF[0].load(Relaxed).saturating_sub(xjs);
            for w in 0..nt { let fc = XS_FC[w].load(Relaxed); let lf = XS_LF[w].load(Relaxed); let c = if w == 0 { 0 } else if w < sh.nfast { 1 } else { 2 };
                r.bz[c] += XS_BZ[w].load(Relaxed); r.nt[c] += XS_NT[w].load(Relaxed);
                if fc != u64::MAX { let d = fc.saturating_sub(xjs); if w == 0 { r.cfc = d } else if c == 1 { r.pfc = r.pfc.max(d) } else { r.efc = r.efc.max(d) } if w > 0 && d > 50_000 { r.late += 1; }
                    let le = lf.saturating_sub(xjs); ends.push(le); if le > r.last { r.last = le; r.lid = if c == 2 { 1000 + w } else { w }; } } }
            ends.sort(); r.med = ends.get(ends.len() / 2).copied().unwrap_or(0);
            let mut v = XRECS.lock().unwrap(); v.push(r); if v.len() >= 600 { xp_print(&mut v, self.threads); } }
        if jprof() { let te = now_ns(); let a = &JA[has_side as usize]; let fc = JFC.load(Relaxed); let fc = if fc == u64::MAX { te } else { fc };
            let v = [fc.saturating_sub(js), ts - js, tc - js, JLC.load(Relaxed).saturating_sub(js), JLF.load(Relaxed).saturating_sub(js), te - js, 1, JB.load(Relaxed)];
            for k in 0..8 { a[k].fetch_add(v[k], Relaxed); }
            if a[6].load(Relaxed) % 500 == 0 { let c = a[6].load(Relaxed) as f64; let m = |k: usize| a[k].load(Relaxed) as f64 / c / 1000.0;
                eprintln!("jprof side={} n={} tasks={} us: firstclaim {:.1} sideEnd {:.1} callerWorkEnd {:.1} lastClaim {:.1} lastFinish {:.1} wall {:.1} busy% {:.0}", has_side, c, n, m(0), m(1), m(2), m(3), m(4), m(5), a[7].load(Relaxed) as f64 * 100.0 / (a[5].load(Relaxed) as f64 * self.threads as f64)); } }
        if wbusy() { WJ.fetch_add(tj.elapsed().as_nanos() as u64, Relaxed); if WN.fetch_add(1, Relaxed) % 2000 == 1999 { let j = WJ.load(Relaxed).max(1) as f64;
            eprintln!("wbusy % of job wall ({} ms, side incl.): {}", j as u64 / 1_000_000, (0..self.threads.min(64)).map(|w| format!("{:.0}", WB[w].load(Relaxed) as f64 * 100.0 / j)).collect::<Vec<_>>().join(" ")); } }
    }
}

/// PYRE_QN_XCNW=1 (opt-in): the calling thread claims no tasks (only its side closure, then waits), so a preempted
/// caller can not hold a gate/up task the down barrier waits on. Which thread runs a task does not change its result.
fn cnw() -> bool { static P: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *P.get_or_init(|| std::env::var("PYRE_QN_XCNW").map(|v| v == "1").unwrap_or(false)) }
const PB: usize = 256;
fn pedq() -> bool { static P: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *P.get_or_init(|| std::env::var("PYRE_QN_PEDQ").map(|v| v == "1").unwrap_or(false)) }
fn tfa() -> bool { static P: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *P.get_or_init(|| std::env::var("PYRE_QN_TFA").map(|v| v != "0").unwrap_or(true)) }
fn work(sh: &Shared, g: u64, job: &Job, slow: bool) {
    // PYRE_QN_TFA (default on): claim tasks with one fetch_add instead of a CAS retry loop (24 workers on one line).
    // A worker that raced into a newer generation g2 still owns ticket i of g2: it runs that task with g2's job
    // (job/n are published before the ticket and stay valid until all g2 tasks, including this one, are done).
    // ticket layout under TFA: gen(20) | n(20) | index(24), so a claim carries its own generation and task count
    // PYRE_QN_PEDQ=1 (opt-in; b409 slower: pool ms A 28.6->33.3 B 35.9->50.4 C 39.4->53.8, E-cores on the small tail tiers delay per-expert GUQ/chain completion): double-ended claims. Tasks are ordered big first (tiered blocks), so fast (P-core)
    // workers take indices from the front and slow (E-core) workers from the back (the small tiers): the middle
    // field holds back+PB (bias PB > threads so failed back claims never borrow into gen). Which worker runs a task
    // does not change what it computes -> bit-identical.
    if tfa() && sh.etail == 0 && pedq() { let gm = g & 0xfffff;
        loop {
            let t0 = sh.ticket.load(Acquire); if t0 >> 44 != gm || (t0 & 0xff_ffff) as usize + PB >= ((t0 >> 24) & 0xfffff) as usize { return; }
            let (t, i) = if slow { let t = sh.ticket.fetch_sub(1 << 24, AcqRel); (t, (((t >> 24) & 0xfffff) as usize).wrapping_sub(PB + 1)) } else { let t = sh.ticket.fetch_add(1, AcqRel); (t, (t & 0xff_ffff) as usize) };
            let (tg, f, b) = (t >> 44, (t & 0xff_ffff) as usize, ((t >> 24) & 0xfffff) as usize);
            if f + PB >= b { return; }
            if tg == gm { call(job.f, i); } else { let j = sh.job.load(Acquire); unsafe { call((*j).f, i); } }
            sh.done.fetch_add(1, Release);
            if tg != gm { return; }
        } }
    if tfa() && sh.etail == 0 { let _ = slow; let gm = g & 0xfffff;
        loop {
            let t0 = sh.ticket.load(Acquire); if t0 >> 44 != gm || (t0 & 0xff_ffff) >= ((t0 >> 24) & 0xfffff) { return; }
            let t = sh.ticket.fetch_add(1, AcqRel); let (tg, tn, i) = (t >> 44, ((t >> 24) & 0xfffff) as usize, (t & 0xff_ffff) as usize);
            if i >= tn { return; }
            if tg == gm { call(job.f, i); } else { let j = sh.job.load(Acquire); unsafe { call((*j).f, i); } }
            sh.done.fetch_add(1, Release);
            if tg != gm { return; }
        } }
    loop {
        let t = sh.ticket.load(Acquire);
        if t >> 32 != g { return; }
        let i = (t & 0xffff_ffff) as usize;
        // n comes from the shared slot, not the job: the job may already be gone once all tasks are done
        let n = sh.n.load(Acquire);
        if i >= n || (slow && i + sh.etail >= n) { return; }
        if sh.ticket.compare_exchange_weak(t, t + 1, AcqRel, Relaxed).is_err() { continue; }
        call(job.f, i);
        sh.done.fetch_add(1, Release);
    }
}

fn worker(sh: &Shared, slow: bool) {
    let mut seen = 0u64; let mut kseen = sh.kick.load(Relaxed);
    let mut idle = Instant::now(); let mut ny = 0usize;
    loop {
        let g = sh.gen.load(Acquire);
        if g != seen {
            seen = g;
            let p = sh.job.load(Relaxed);
            // only runs tasks whose ticket still carries generation g, so a stale pointer is never called
            if !p.is_null() { work(sh, g, unsafe { &*p }, slow); }
            idle = Instant::now();
            continue;
        }
        let kk = sh.kick.load(Acquire); if kk != kseen { kseen = kk; idle = Instant::now(); continue; }
        if idle.elapsed() < sh.spin { if umwait_on() { umwait_line(&sh.gen as *const AtomicU64 as *const u8, || sh.gen.load(Acquire) == seen && sh.kick.load(Acquire) == kseen); } else { for _ in 0..64 { std::hint::spin_loop(); } } ny += 1; if spiny() > 0 && ny % spiny() == 0 { yield_now(); } continue; }
        sh.sleepers.fetch_add(1, AcqRel);
        let mut l = sh.lock.lock().unwrap();
        while sh.gen.load(Acquire) == seen && sh.kick.load(Acquire) == kseen { l = sh.cv.wait_timeout(l, Duration::from_millis(50)).unwrap().0; }
        drop(l);
        sh.sleepers.fetch_sub(1, AcqRel);
    }
}

/// p1-p8 (2026-10-31, XPSTAT): under PYRE_QN_XEXACT=1 the pool defaults to XPIN0=1 (caller on the one P-core without a worker),
/// SPIN_US=800 (workers keep spinning through the ~0.5 ms GPU dense half between layers instead of parking: 22-23 of 23 were
/// parked at every publish, ~55 us condvar wake) and UMWAIT=1 (that spin in C0.1). Each =0 / other value restores the old one.
pub fn xexact_env() -> bool { std::env::var("PYRE_QN_XEXACT").map(|v| v == "1").unwrap_or(false) }
/// PYRE_QN_SPINY=k (opt-in measurement/fix): an idle spinning worker calls SwitchToThread every k spin rounds (~0.8 us
/// each), so a ready thread of this process or the OS takes an idle spinner's core instead of preempting a worker that is
/// inside a task (a preempted gate/up task stalls every worker at the down barrier for a 15-31 ms quantum).
fn spiny() -> usize { static P: std::sync::OnceLock<usize> = std::sync::OnceLock::new(); *P.get_or_init(|| std::env::var("PYRE_QN_SPINY").ok().and_then(|v| v.parse().ok()).unwrap_or(0)) }
#[cfg(windows)] fn yield_now() { unsafe { windows_sys::Win32::System::Threading::SwitchToThread(); } }
#[cfg(not(windows))] fn yield_now() { std::thread::yield_now(); }
/// PYRE_QN_UMWAIT=1 (opt-in, x86 WAITPKG only): a spinning worker waits with UMONITOR/UMWAIT (light C0.1 state, ~7 us TSC
/// deadline) on the cache line of the job generation instead of a pause loop; it wakes on the publish's write. Less power
/// than 23 pause loops (more turbo headroom for the P-cores). After Strata's STRATA_POOL_UMWAIT (MIT). No CPU support: pause loop.
fn umwait_on() -> bool { static P: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *P.get_or_init(|| {
    let want = std::env::var("PYRE_QN_UMWAIT").map(|v| v == "1").unwrap_or(xexact_env()); if !want { return false; }
    #[cfg(target_arch = "x86_64")] { let ok = unsafe { (core::arch::x86_64::__cpuid_count(7, 0).ecx >> 5) & 1 == 1 }; eprintln!("expert pool: UMWAIT {}", if ok { "on" } else { "requested, no WAITPKG -> pause loop" }); ok }
    #[cfg(not(target_arch = "x86_64"))] { false } }) }
#[inline] fn umwait_line(addr: *const u8, still: impl Fn() -> bool) {
    #[cfg(target_arch = "x86_64")] unsafe {
        std::arch::asm!("umonitor {0}", in(reg) addr, options(nostack, preserves_flags));
        if !still() { return; }
        let dl = core::arch::x86_64::_rdtsc() + 20_000;
        std::arch::asm!("umwait {0:e}", in(reg) 1u32, in("edx") (dl >> 32) as u32, in("eax") dl as u32, options(nostack));
    }
    #[cfg(not(target_arch = "x86_64"))] { let _ = (addr, still); std::hint::spin_loop(); }
}

/// A raw pointer that tasks writing disjoint ranges may share.
#[derive(Clone, Copy)]
pub struct SyncPtr(pub *mut f32);
unsafe impl Send for SyncPtr {}
unsafe impl Sync for SyncPtr {}
impl SyncPtr { #[allow(clippy::mut_from_ref)] pub unsafe fn slice(&self, off: usize, n: usize) -> &mut [f32] { std::slice::from_raw_parts_mut(self.0.add(off), n) } }

#[cfg(windows)]
pub fn pin_to(cpu: usize) {
    use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadAffinityMask};
    if cpu < usize::BITS as usize { unsafe { SetThreadAffinityMask(GetCurrentThread(), 1usize << cpu); } }
}
#[cfg(not(windows))]
pub fn pin_to(_cpu: usize) {}

/// PYRE_QN_XQOS=1 (opt-in): opt this process (and each pool thread / the caller) out of Windows power throttling
/// (EcoQoS: a windowless background process may be scheduled for efficiency) via SetProcessInformation /
/// SetThreadInformation on our own handles - a per-process hint, nothing system-wide is changed.
/// PYRE_QN_XPRIO=N: SetThreadPriority(N) for the pool workers and the calling thread (1 above normal, 2 highest; 0 normal);
/// default 1 under PYRE_QN_XEXACT=1 (fewer preempted-worker tails: e62 maxlay 1.8 vs 2.0-3.2 ms), else unchanged.
fn xqos() -> bool { std::env::var("PYRE_QN_XQOS").map(|v| v == "1").unwrap_or(false) }
fn xprio() -> Option<i32> { match std::env::var("PYRE_QN_XPRIO") { Ok(v) => v.parse().ok(), Err(_) => if std::env::var("PYRE_QN_XEXACT").map(|v| v == "1").unwrap_or(false) { Some(1) } else { None } } }
#[cfg(windows)]
fn proc_tune() {
    use windows_sys::Win32::System::Threading::*;
    if xqos() { let st = PROCESS_POWER_THROTTLING_STATE { Version: PROCESS_POWER_THROTTLING_CURRENT_VERSION, ControlMask: PROCESS_POWER_THROTTLING_EXECUTION_SPEED, StateMask: 0 };
        let ok = unsafe { SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &st as *const _ as *const core::ffi::c_void, std::mem::size_of::<PROCESS_POWER_THROTTLING_STATE>() as u32) };
        eprintln!("expert pool: XQOS process power throttling off: {}", ok != 0); }
    if let Some(p) = xprio() { eprintln!("expert pool: XPRIO thread priority {p}"); }
}
#[cfg(windows)]
fn thr_tune() {
    use windows_sys::Win32::System::Threading::*;
    if xqos() { let st = THREAD_POWER_THROTTLING_STATE { Version: THREAD_POWER_THROTTLING_CURRENT_VERSION, ControlMask: 1, StateMask: 0 };
        unsafe { SetThreadInformation(GetCurrentThread(), ThreadPowerThrottling, &st as *const _ as *const core::ffi::c_void, std::mem::size_of::<THREAD_POWER_THROTTLING_STATE>() as u32); } }
    if let Some(p) = xprio() { unsafe { SetThreadPriority(GetCurrentThread(), p); } }
}
#[cfg(not(windows))]
fn proc_tune() {}
#[cfg(not(windows))]
fn thr_tune() {}

/// One logical CPU per physical core of the fastest efficiency class (P-cores on a hybrid Intel
/// part), or empty when that cannot be read. Strata likewise runs one pool worker per physical core.
/// Number of leading entries of fast_cores() that are fast (P) cores.
pub fn n_fast() -> usize { FASTN.load(Relaxed) }
static FASTN: AtomicUsize = AtomicUsize::new(usize::MAX);
#[cfg(windows)]
pub fn fast_cores() -> Vec<usize> {
    use windows_sys::Win32::System::SystemInformation::{GetLogicalProcessorInformationEx, RelationProcessorCore, SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX};
    let mut len = 0u32;
    unsafe { GetLogicalProcessorInformationEx(RelationProcessorCore, std::ptr::null_mut(), &mut len); }
    if len == 0 { return vec![]; }
    let mut buf = vec![0u8; len as usize];
    if unsafe { GetLogicalProcessorInformationEx(RelationProcessorCore, buf.as_mut_ptr() as *mut SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, &mut len) } == 0 { return vec![]; }
    let mut cores: Vec<(u8, usize)> = vec![]; let mut off = 0usize;
    while off < len as usize {
        let p = unsafe { &*(buf.as_ptr().add(off) as *const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) };
        let pr = unsafe { &p.Anonymous.Processor };
        let g = pr.GroupMask[0];
        if g.Group == 0 && g.Mask != 0 { cores.push((pr.EfficiencyClass, g.Mask.trailing_zeros() as usize)); }
        off += p.Size as usize;
        if p.Size == 0 { break; }
    }
    let best = cores.iter().map(|c| c.0).max().unwrap_or(0);
    let mut v: Vec<usize> = cores.iter().filter(|c| c.0 == best).map(|c| c.1).collect(); FASTN.store(v.len(), Relaxed);
    // PYRE_QN_ECORES (default on): also one worker per slower physical core (hybrid CPUs' E-cores) after the fast ones.
    // Measured on an i9-13900K: 8 P + 16 E workers decode ~5% faster than 8 P alone; same results.
    // PYRE_QN_NECORES=k: at most k of those slower-core workers (worker-mix sweeps, e.g. 8 P + 8 E).
    let ne = std::env::var("PYRE_QN_NECORES").ok().and_then(|x| x.parse().ok()).unwrap_or(usize::MAX);
    if std::env::var("PYRE_QN_ECORES").map(|x| x != "0").unwrap_or(true) { v.extend(cores.iter().filter(|c| c.0 != best).map(|c| c.1).take(ne)); }
    v
}
#[cfg(not(windows))]
pub fn fast_cores() -> Vec<usize> { vec![] }
