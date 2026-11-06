//! OpenAI-shaped SSE frames. Every data frame carries the request id.
//! A stream is unfinished until [`frames_finished`] or [`frames_failed`] has been written.

use crate::journal::Status;
use serde_json::{json, Value};

#[derive(Clone, Debug)]
pub struct Envelope {
    pub id: String,
    pub created: u64,
    pub model: String,
}

impl Envelope {
    pub fn new(id: impl Into<String>, created: u64, model: impl Into<String>) -> Self {
        Self { id: id.into(), created, model: model.into() }
    }

    pub fn resume_token(&self) -> &str {
        &self.id
    }

    /// One `chat.completion.chunk`. `finish_reason` null means "still running".
    pub fn chunk(&self, delta: Value, finish_reason: Option<&str>, status: Status, replay: bool, message: Option<&str>) -> Value {
        let mut pyre = json!({
            "resume_token": self.id,
            "status": status.as_str(),
            "replay": replay,
        });
        if let Some(m) = message {
            pyre["message"] = json!(m);
        }
        json!({
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
                "logprobs": null,
            }],
            "pyre": pyre,
        })
    }

    pub fn error_body(&self, message: &str, code: &str) -> Value {
        json!({
            "error": {
                "message": message,
                "type": "server_error",
                "code": code,
                "resume_token": self.id,
            }
        })
    }
}

#[derive(Clone, Debug)]
pub struct Frame {
    pub data: String,
}

impl Frame {
    pub fn json(v: &Value) -> Self {
        Self { data: v.to_string() }
    }
    pub fn done() -> Self {
        Self { data: "[DONE]".into() }
    }
    pub fn is_done(&self) -> bool {
        self.data == "[DONE]"
    }
}

pub fn role_frame(env: &Envelope) -> Frame {
    Frame::json(&env.chunk(json!({"role": "assistant", "content": ""}), None, Status::Running, false, None))
}

pub fn data_frame(env: &Envelope, delta: Value, replay: bool) -> Frame {
    Frame::json(&env.chunk(delta, None, Status::Running, replay, None))
}

pub fn error_frame(env: &Envelope, message: &str, code: &str) -> Frame {
    Frame::json(&env.error_body(message, code))
}

/// Successful terminal chunk (caller merges timings / usage into `extra`) plus `[DONE]`.
pub fn frames_finished(env: &Envelope, finish_reason: &str, extra: Value) -> Vec<Frame> {
    let mut chunk = env.chunk(json!({}), Some(finish_reason), Status::Finished, false, None);
    if let (Some(obj), Some(extra_obj)) = (chunk.as_object_mut(), extra.as_object()) {
        for (k, v) in extra_obj {
            obj.insert(k.clone(), v.clone());
        }
    }
    vec![Frame::json(&chunk), Frame::done()]
}

/// Error object, then a chunk with `finish_reason: "error"` and `pyre.status`, then `[DONE]`.
/// `[DONE]` is included so clients that block until the sentinel unblock; the finish reason
/// is what tells them the choice did not succeed.
pub fn frames_failed(env: &Envelope, status: Status, message: &str, code: &str) -> Vec<Frame> {
    let st = match status {
        Status::Error => Status::Error,
        _ => Status::Aborted,
    };
    let mut chunk = env.chunk(json!({}), Some("error"), st, false, Some(message));
    chunk["pyre"]["resumable"] = json!(st == Status::Aborted);
    vec![error_frame(env, message, code), Frame::json(&chunk), Frame::done()]
}

pub fn code_for(status: Status, shutdown: bool) -> &'static str {
    if shutdown {
        "server_shutdown"
    } else if status == Status::Error {
        "generation_error"
    } else {
        "generation_aborted"
    }
}
