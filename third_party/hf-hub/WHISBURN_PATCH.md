# Vendored hf-hub (0.5.0)

Patched for whisburn so the default build does not pull the `cc` or `libc` crates:

- Upstream `ureq` 3 defaults enable rustls → ring → cc.
- This copy sets `ureq` to `default-features = false` with `native-tls` only.
- `native-tls` no longer enables `reqwest` (that pulled rustls/ring/quinn even for the ureq API).
- Unix file locking calls `flock` via `extern "C"` instead of the `libc` crate.

Use via workspace path dependency in the root `Cargo.toml`.
