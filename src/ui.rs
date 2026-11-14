//! Built-in web UI, embedded in the binary (vanilla HTML/JS/CSS in `ui/`; no Node build step).
use axum::http::header;
use axum::response::IntoResponse;
use axum::routing::get;
use axum::Router;

const INDEX: &str = include_str!("../ui/index.html");
const APP_JS: &str = concat!(include_str!("../ui/stt_progress.js"), "\n", include_str!("../ui/media_extract.js"), "\n", include_str!("../ui/app.js"));
/// Mediabunny (MPL-2.0) prebuilt ESM bundle from the vendor/mediabunny submodule. The Audio tab imports it
/// on demand to pull the audio track out of video files in the browser, so only audio is uploaded.
const MEDIABUNNY_JS: &str = include_str!("../vendor/mediabunny/dist/bundles/mediabunny.min.mjs");
const MEDIABUNNY_PATH: &str = "/ui/vendor/mediabunny-1.66.0.min.mjs";
const STYLE: &str = include_str!("../ui/style.css");
const ICON: &str = include_str!("../ui/favicon.svg");
const CABIN_JS: &str = include_str!("../ui/cabin.js");
const CABIN_CSS: &str = include_str!("../ui/cabin.css");

pub fn is_static(p: &str) -> bool { matches!(p, "/" | "/index.html" | "/ui/app.js" | "/ui/style.css" | "/favicon.svg") || p == MEDIABUNNY_PATH }

pub fn routes<S: Clone + Send + Sync + 'static>() -> Router<S> {
    Router::new()
        .route("/", get(|| async { ([(header::CONTENT_TYPE, "text/html; charset=utf-8"), (header::CACHE_CONTROL, "no-cache")], INDEX).into_response() }))
        .route("/index.html", get(|| async { ([(header::CONTENT_TYPE, "text/html; charset=utf-8"), (header::CACHE_CONTROL, "no-cache")], INDEX).into_response() }))
        .route("/ui/app.js", get(|| async { ([(header::CONTENT_TYPE, "application/javascript; charset=utf-8"), (header::CACHE_CONTROL, "no-cache")], APP_JS).into_response() }))
        .route("/ui/cabin.js", get(|| async { ([(header::CONTENT_TYPE, "application/javascript; charset=utf-8"), (header::CACHE_CONTROL, "no-cache")], CABIN_JS).into_response() }))
        .route("/ui/cabin.css", get(|| async { ([(header::CONTENT_TYPE, "text/css; charset=utf-8"), (header::CACHE_CONTROL, "no-cache")], CABIN_CSS).into_response() }))
        .route("/ui/style.css", get(|| async { ([(header::CONTENT_TYPE, "text/css; charset=utf-8"), (header::CACHE_CONTROL, "no-cache")], STYLE).into_response() }))
        .route(MEDIABUNNY_PATH, get(|| async { ([(header::CONTENT_TYPE, "text/javascript; charset=utf-8"), (header::CACHE_CONTROL, "public, max-age=31536000, immutable")], MEDIABUNNY_JS).into_response() }))
        .route("/favicon.svg", get(|| async { ([(header::CONTENT_TYPE, "image/svg+xml")], ICON).into_response() }))
}
