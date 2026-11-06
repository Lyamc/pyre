//! Process-wide signal that in-flight SSE streams should emit an abort frame and finish.
//!
//! `limits::stop_now` and the HTTP graceful-shutdown task both call [`Drain::fire_and_wait`].
//! The wait is on a std condvar so the model thread (which is not on the tokio runtime) can
//! block until the response tasks have dropped their [`TrackGuard`].

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::Duration;

static GLOBAL: OnceLock<Arc<Drain>> = OnceLock::new();

/// The process-wide drain used by the pyre binary.
pub fn drain() -> Arc<Drain> {
    GLOBAL.get_or_init(|| Drain::new()).clone()
}

pub struct Drain {
    tx: tokio::sync::broadcast::Sender<()>,
    inflight: AtomicUsize,
    fired: AtomicBool,
    mu: Mutex<()>,
    cv: Condvar,
}

impl Drain {
    pub fn new() -> Arc<Self> {
        let (tx, _) = tokio::sync::broadcast::channel(16);
        Arc::new(Self { tx, inflight: AtomicUsize::new(0), fired: AtomicBool::new(false), mu: Mutex::new(()), cv: Condvar::new() })
    }

    /// Register one response. The returned receiver fires when the process is draining.
    /// Drop the guard when the response has finished writing (or the client has gone).
    pub fn track(self: &Arc<Self>) -> (tokio::sync::broadcast::Receiver<()>, TrackGuard) {
        let rx = self.tx.subscribe();
        self.inflight.fetch_add(1, Ordering::SeqCst);
        if self.fired.load(Ordering::SeqCst) {
            let _ = self.tx.send(());
        }
        (rx, TrackGuard { drain: Arc::clone(self) })
    }

    /// Tell every tracked stream to abort, then wait until they drop or `timeout` elapses.
    pub fn fire_and_wait(&self, timeout: Duration) {
        self.fired.store(true, Ordering::SeqCst);
        let _ = self.tx.send(());
        let start = std::time::Instant::now();
        let mut guard = self.mu.lock().unwrap_or_else(|e| e.into_inner());
        while self.inflight.load(Ordering::SeqCst) > 0 {
            let left = timeout.saturating_sub(start.elapsed());
            if left.is_zero() {
                break;
            }
            let (g, result) = self.cv.wait_timeout(guard, left).unwrap_or_else(|e| e.into_inner());
            guard = g;
            if result.timed_out() {
                break;
            }
        }
        drop(guard);
        // Let the runtime push the last SSE chunk out of userspace before `process::exit`.
        std::thread::sleep(Duration::from_millis(50).min(timeout));
    }

    pub fn fired(&self) -> bool {
        self.fired.load(Ordering::SeqCst)
    }
}

pub struct TrackGuard {
    drain: Arc<Drain>,
}

impl Drop for TrackGuard {
    fn drop(&mut self) {
        if self.drain.inflight.fetch_sub(1, Ordering::SeqCst) == 1 {
            let _g = self.drain.mu.lock().unwrap_or_else(|e| e.into_inner());
            self.drain.cv.notify_all();
        }
    }
}

/// Resolves on Ctrl-C, and on Unix also on SIGTERM (the watchdog / service stop path).
pub async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    {
        let mut term = match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(s) => s,
            Err(_) => {
                ctrl_c.await;
                return;
            }
        };
        tokio::select! {
            _ = ctrl_c => {}
            _ = term.recv() => {}
        }
    }
    #[cfg(not(unix))]
    {
        ctrl_c.await;
    }
}
