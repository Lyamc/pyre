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
    pub fn new(threads: usize, spin: Duration) -> Self {
        let sh = Arc::new(Shared { gen: AtomicU64::new(0), ticket: AtomicU64::new(0), job: AtomicPtr::new(std::ptr::null_mut()), done: AtomicUsize::new(0), n: AtomicUsize::new(0),
            sleepers: AtomicUsize::new(0), lock: Mutex::new(()), cv: Condvar::new(), spin });
        for i in 1..threads.max(1) {
            let sh = sh.clone();
            std::thread::Builder::new().name(format!("pyre-xp{i}")).spawn(move || worker(&sh)).expect("spawn pool worker");
        }
        Pool { sh, threads: threads.max(1) }
    }

    /// Run f(0..n) across the pool (the caller works too) and return when all are done.
    pub fn run(&self, n: usize, f: &(dyn Fn(usize) + Sync)) {
        if n == 0 { return; }
        let sh = &*self.sh;
        let job = Job { f, n };
        let g = sh.gen.load(Relaxed) + 1;
        sh.done.store(0, Relaxed);
        sh.n.store(n, Relaxed);
        sh.job.store(&job as *const Job as *mut Job<'static>, Relaxed);
        sh.ticket.store(g << 32, Relaxed);
        sh.gen.store(g, Release);
        if sh.sleepers.load(Acquire) > 0 { let _l = sh.lock.lock().unwrap(); sh.cv.notify_all(); }
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
