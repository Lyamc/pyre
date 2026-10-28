//! A small persistent worker pool for the per-layer expert phases of a decode step. Workers spin
//! for a short while after each job before parking, so the ~100 short jobs per token do not pay a
//! thread wake-up each (Strata keeps a dedicated pinned pool for the same reason; this is pyre's own).
use std::sync::atomic::{AtomicPtr, AtomicU64, AtomicUsize, Ordering::*};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

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
}

pub struct Pool { sh: Arc<Shared>, pub threads: usize }

impl Pool {
    /// `pin`: logical CPU per worker (index 0 is the caller and is not pinned).
    pub fn new(threads: usize, spin: Duration) -> Self { Self::new_pinned(threads, spin, &[]) }
    pub fn new_pinned(threads: usize, spin: Duration, pin: &[usize]) -> Self {
        let sh = Arc::new(Shared { gen: AtomicU64::new(0), ticket: AtomicU64::new(0), job: AtomicPtr::new(std::ptr::null_mut()), done: AtomicUsize::new(0), n: AtomicUsize::new(0),
            sleepers: AtomicUsize::new(0), lock: Mutex::new(()), cv: Condvar::new(), spin });
        for i in 1..threads.max(1) {
            let sh = sh.clone(); let cpu = pin.get(i).copied();
            std::thread::Builder::new().name(format!("pyre-xp{i}")).spawn(move || { if let Some(c) = cpu { pin_to(c); } worker(&sh) }).expect("spawn pool worker");
        }
        Pool { sh, threads: threads.max(1) }
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
        sh.ticket.store(g << 32, Relaxed);
        sh.gen.store(g, Release);
        if sh.sleepers.load(Acquire) > 0 { let _l = sh.lock.lock().unwrap(); sh.cv.notify_all(); }
        side();
        work(sh, g, &job);
        while sh.done.load(Acquire) < n { std::hint::spin_loop(); }
    }
}

fn work(sh: &Shared, g: u64, job: &Job) {
    loop {
        let t = sh.ticket.load(Acquire);
        if t >> 32 != g { return; }
        let i = (t & 0xffff_ffff) as usize;
        // n comes from the shared slot, not the job: the job may already be gone once all tasks are done
        if i >= sh.n.load(Acquire) { return; }
        if sh.ticket.compare_exchange_weak(t, t + 1, AcqRel, Relaxed).is_err() { continue; }
        (job.f)(i);
        sh.done.fetch_add(1, Release);
    }
}

fn worker(sh: &Shared) {
    let mut seen = 0u64;
    let mut idle = Instant::now();
    loop {
        let g = sh.gen.load(Acquire);
        if g != seen {
            seen = g;
            let p = sh.job.load(Relaxed);
            // only runs tasks whose ticket still carries generation g, so a stale pointer is never called
            if !p.is_null() { work(sh, g, unsafe { &*p }); }
            idle = Instant::now();
            continue;
        }
        if idle.elapsed() < sh.spin { for _ in 0..64 { std::hint::spin_loop(); } continue; }
        sh.sleepers.fetch_add(1, AcqRel);
        let mut l = sh.lock.lock().unwrap();
        while sh.gen.load(Acquire) == seen { l = sh.cv.wait_timeout(l, Duration::from_millis(50)).unwrap().0; }
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
    cores.iter().filter(|c| c.0 == best).map(|c| c.1).collect()
}
#[cfg(not(windows))]
pub fn fast_cores() -> Vec<usize> { vec![] }
