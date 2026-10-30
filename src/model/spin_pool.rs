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
#[inline] fn call(f: &(dyn Fn(usize) + Sync), i: usize) { if wbusy() { let t = Instant::now(); f(i); WB[WID.with(|w| w.get()).min(63)].fetch_add(t.elapsed().as_nanos() as u64, Relaxed); } else { f(i) } }
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
}

pub struct Pool { sh: Arc<Shared>, pub threads: usize }

impl Pool {
    /// `pin`: logical CPU per worker (index 0 is the caller and is not pinned).
    pub fn new(threads: usize, spin: Duration) -> Self { Self::new_pinned(threads, spin, &[]) }
    pub fn new_pinned(threads: usize, spin: Duration, pin: &[usize]) -> Self { Self::new_hybrid(threads, spin, pin, usize::MAX, 0) }
    /// Workers with index >= `n_fast` are slow (E-cores): they skip a job's last `etail` tasks.
    pub fn new_hybrid(threads: usize, spin: Duration, pin: &[usize], n_fast: usize, etail: usize) -> Self {
        let sh = Arc::new(Shared { gen: AtomicU64::new(0), ticket: AtomicU64::new(0), job: AtomicPtr::new(std::ptr::null_mut()), done: AtomicUsize::new(0), n: AtomicUsize::new(0),
            sleepers: AtomicUsize::new(0), lock: Mutex::new(()), cv: Condvar::new(), spin, etail, kick: AtomicU64::new(0) });
        for i in 1..threads.max(1) {
            let sh = sh.clone(); let cpu = pin.get(i).copied();
            std::thread::Builder::new().name(format!("pyre-xp{i}")).spawn(move || { WID.with(|w| w.set(i)); if let Some(c) = cpu { pin_to(c); } worker(&sh, i >= n_fast) }).expect("spawn pool worker");
        }
        Pool { sh, threads: threads.max(1) }
    }

    /// Wake parked workers and restart every worker's spin window now, so they are already spinning when the
    /// next job arrives (call just before blocking on the GPU readback that precedes a job).
    pub fn prewake(&self) {
        let sh = &*self.sh; sh.kick.fetch_add(1, Release);
        if sh.sleepers.load(Acquire) > 0 { let _l = sh.lock.lock().unwrap(); sh.cv.notify_all(); }
    }
    /// Run f(0..n) across the pool (the caller works too) and return when all are done.
    pub fn run(&self, n: usize, f: &(dyn Fn(usize) + Sync)) { self.run_side(n, f, &mut || {}) }
    /// Like `run`, but the caller first does `side` (e.g. enqueueing GPU work) while the workers start.
    pub fn run_side(&self, n: usize, f: &(dyn Fn(usize) + Sync), side: &mut dyn FnMut()) {
        if n == 0 { side(); return; }
        let sh = &*self.sh;
        let job = Job { f, n };
        let g = sh.gen.load(Relaxed) + 1;
        sh.done.store(0, Relaxed);
        sh.n.store(n, Relaxed);
        sh.job.store(&job as *const Job as *mut Job<'static>, Relaxed);
        if tfa() && sh.etail == 0 { assert!(n < 1 << 20); sh.ticket.store(((g & 0xfffff) << 44) | ((n as u64) << 24), Release); } else { sh.ticket.store(g << 32, Relaxed); }
        sh.gen.store(g, Release);
        if sh.sleepers.load(Acquire) > 0 { let _l = sh.lock.lock().unwrap(); sh.cv.notify_all(); }
        let tj = Instant::now();
        side();
        work(sh, g, &job, false);
        while sh.done.load(Acquire) < n { std::hint::spin_loop(); }
        if wbusy() { WJ.fetch_add(tj.elapsed().as_nanos() as u64, Relaxed); if WN.fetch_add(1, Relaxed) % 2000 == 1999 { let j = WJ.load(Relaxed).max(1) as f64;
            eprintln!("wbusy % of job wall ({} ms, side incl.): {}", j as u64 / 1_000_000, (0..self.threads.min(64)).map(|w| format!("{:.0}", WB[w].load(Relaxed) as f64 * 100.0 / j)).collect::<Vec<_>>().join(" ")); } }
    }
}

fn tfa() -> bool { static P: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *P.get_or_init(|| std::env::var("PYRE_QN_TFA").map(|v| v != "0").unwrap_or(true)) }
fn work(sh: &Shared, g: u64, job: &Job, slow: bool) {
    // PYRE_QN_TFA (default on): claim tasks with one fetch_add instead of a CAS retry loop (24 workers on one line).
    // A worker that raced into a newer generation g2 still owns ticket i of g2: it runs that task with g2's job
    // (job/n are published before the ticket and stay valid until all g2 tasks, including this one, are done).
    // ticket layout under TFA: gen(20) | n(20) | index(24), so a claim carries its own generation and task count
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
    let mut idle = Instant::now();
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
        if idle.elapsed() < sh.spin { for _ in 0..64 { std::hint::spin_loop(); } continue; }
        sh.sleepers.fetch_add(1, AcqRel);
        let mut l = sh.lock.lock().unwrap();
        while sh.gen.load(Acquire) == seen && sh.kick.load(Acquire) == kseen { l = sh.cv.wait_timeout(l, Duration::from_millis(50)).unwrap().0; }
        drop(l);
        sh.sleepers.fetch_sub(1, AcqRel);
    }
}

/// A raw pointer that tasks writing disjoint ranges may share.
#[derive(Clone, Copy)]
pub struct SyncPtr(pub *mut f32);
unsafe impl Send for SyncPtr {}
unsafe impl Sync for SyncPtr {}
impl SyncPtr { #[allow(clippy::mut_from_ref)] pub unsafe fn slice(&self, off: usize, n: usize) -> &mut [f32] { std::slice::from_raw_parts_mut(self.0.add(off), n) } }

#[cfg(windows)]
fn pin_to(cpu: usize) {
    use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadAffinityMask};
    if cpu < usize::BITS as usize { unsafe { SetThreadAffinityMask(GetCurrentThread(), 1usize << cpu); } }
}
#[cfg(not(windows))]
fn pin_to(_cpu: usize) {}

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
    if std::env::var("PYRE_QN_ECORES").map(|x| x != "0").unwrap_or(true) { v.extend(cores.iter().filter(|c| c.0 != best).map(|c| c.1)); }
    v
}
#[cfg(not(windows))]
pub fn fast_cores() -> Vec<usize> { vec![] }
