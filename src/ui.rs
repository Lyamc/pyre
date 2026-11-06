//! Built-in web UI, embedded in the binary (vanilla HTML/JS/CSS in `ui/`; no Node build step).
use axum::http::header;
use axum::response::IntoResponse;
use axum::routing::get;
use axum::Router;

const INDEX: &str = include_str!("../ui/index.html");
const APP_JS: &str = concat!(include_str!("../ui/stt_progress.js"), "\n", include_str!("../ui/app.js"));
const STYLE: &str = include_str!("../ui/style.css");
const ICON: &str = include_str!("../ui/favicon.svg");
const CABIN_JS: &str = include_str!("../ui/cabin.js");
const CABIN_CSS: &str = include_str!("../ui/cabin.css");

pub fn is_static(p: &str) -> bool { matches!(p, "/" | "/index.html" | "/ui/app.js" | "/ui/style.css" | "/favicon.svg") }

pub fn routes<S: Clone + Send + Sync + 'static>() -> Router<S> {
    Router::new()
        .route("/", get(|| async { ([(header::CONTENT_TYPE, "text/html; charset=utf-8"), (header::CACHE_CONTROL, "no-cache")], INDEX).into_response() }))
        .route("/index.html", get(|| async { ([(header::CONTENT_TYPE, "text/html; charset=utf-8"), (header::CACHE_CONTROL, "no-cache")], INDEX).into_response() }))
        .route("/ui/app.js", get(|| async { ([(header::CONTENT_TYPE, "application/javascript; charset=utf-8"), (header::CACHE_CONTROL, "no-cache")], APP_JS).into_response() }))
        .route("/ui/cabin.js", get(|| async { ([(header::CONTENT_TYPE, "application/javascript; charset=utf-8"), (header::CACHE_CONTROL, "no-cache")], CABIN_JS).into_response() }))
        .route("/ui/cabin.css", get(|| async { ([(header::CONTENT_TYPE, "text/css; charset=utf-8"), (header::CACHE_CONTROL, "no-cache")], CABIN_CSS).into_response() }))
        .route("/ui/style.css", get(|| async { ([(header::CONTENT_TYPE, "text/css; charset=utf-8"), (header::CACHE_CONTROL, "no-cache")], STYLE).into_response() }))
        .route("/favicon.svg", get(|| async { ([(header::CONTENT_TYPE, "image/svg+xml")], ICON).into_response() }))
}
