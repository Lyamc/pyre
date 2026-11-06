//! Append-only JSONL journal. One file per generation: an `open` line, `delta` lines,
//! and a single winning `close` line. A crash leaves no `close`; [`Store::recover`]
//! appends `status: aborted`. A partial trailing line (kill mid-write) is ignored.

use crate::protocol::{frames_failed, frames_finished, Envelope, Frame};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub const SUCCESS_FINISH: &[&str] = &["stop", "length", "tool_calls", "content_filter", "function_call"];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    Running,
    Finished,
    Aborted,
    Error,
}

impl Status {
    pub fn as_str(self) -> &'static str {
        match self {
            Status::Running => "running",
            Status::Finished => "finished",
            Status::Aborted => "aborted",
            Status::Error => "error",
        }
    }
    pub fn parse(s: &str) -> Self {
        match s {
            "finished" => Status::Finished,
            "aborted" => Status::Aborted,
            "error" => Status::Error,
            _ => Status::Running,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OpenMeta {
    pub id: String,
    pub resume_token: String,
    pub model: String,
    pub created: u64,
    pub endpoint: String,
    pub prompt_ids: Vec<u32>,
    pub max_tokens: usize,
    pub stop: Vec<String>,
    #[serde(default)]
    pub sampling: Value,
    #[serde(default)]
    pub messages: Value,
    #[serde(default)]
    pub tools: Option<Value>,
    #[serde(default)]
    pub extra: Value,
}

impl OpenMeta {
    pub fn envelope(&self) -> Envelope {
        Envelope::new(self.id.clone(), self.created, self.model.clone())
    }
}

#[derive(Clone, Debug)]
pub struct Record {
    pub meta: OpenMeta,
    pub text: String,
    pub ids: Vec<u32>,
    pub status: Status,
    pub finish_reason: Option<String>,
    pub message: Option<String>,
    pub content: Option<String>,
    pub reasoning: Option<String>,
    pub tool_calls: Vec<Value>,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
}

impl Record {
    pub fn snapshot(&self) -> Snapshot {
        let resumable = self.status == Status::Aborted && self.ids.len() < self.meta.max_tokens;
        Snapshot {
            id: self.meta.id.clone(),
            resume_token: self.meta.resume_token.clone(),
            model: self.meta.model.clone(),
            created: self.meta.created,
            status: self.status,
            finish_reason: self.finish_reason.clone(),
            message: self.message.clone(),
            text: self.text.clone(),
            content: self.content.clone(),
            reasoning: self.reasoning.clone(),
            tool_calls: self.tool_calls.clone(),
            prompt_tokens: if self.prompt_tokens == 0 { self.meta.prompt_ids.len() } else { self.prompt_tokens },
            completion_tokens: if self.completion_tokens == 0 { self.ids.len() } else { self.completion_tokens },
            resumable,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Snapshot {
    pub id: String,
    pub resume_token: String,
    pub model: String,
    pub created: u64,
    pub status: Status,
    pub finish_reason: Option<String>,
    pub message: Option<String>,
    pub text: String,
    pub content: Option<String>,
    pub reasoning: Option<String>,
    pub tool_calls: Vec<Value>,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub resumable: bool,
}

#[derive(Clone, Debug)]
pub struct Continue {
    pub prompt_ids: Vec<u32>,
    pub max_tokens: usize,
    pub sampling: Value,
    pub stop: Vec<String>,
    pub messages: Value,
    pub tools: Option<Value>,
    pub extra: Value,
    pub model: String,
    pub generated_text: String,
    pub generated_ids: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct Plan {
    pub record: Record,
    /// Replay already-generated text to the client.
    pub replay: bool,
    /// Present when the caller should run the model again for the remainder.
    pub cont: Option<Continue>,
}

#[derive(Debug)]
pub enum PlanError {
    NotFound,
    /// A live process holds the generation (the original stream is still attached).
    Busy,
    BadId,
    Io(io::Error),
}

impl std::fmt::Display for PlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlanError::NotFound => write!(f, "generation not found"),
            PlanError::Busy => write!(f, "generation is still running on its original stream"),
            PlanError::BadId => write!(f, "invalid generation id"),
            PlanError::Io(e) => write!(f, "{e}"),
        }
    }
}

impl From<io::Error> for PlanError {
    fn from(e: io::Error) -> Self {
        PlanError::Io(e)
    }
}

#[derive(Clone, Debug)]
pub struct Store {
    dir: PathBuf,
}

impl Store {
    pub fn open(dir: impl Into<PathBuf>) -> io::Result<Self> {
        let dir = dir.into();
        fs::create_dir_all(&dir)?;
        Ok(Self { dir })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Drop stale lock files and close journals a crash left running.
    /// Returns how many journals were marked aborted.
    pub fn recover(&self) -> io::Result<usize> {
        let mut n = 0;
        let ttl = crate::ttl_hours();
        for ent in fs::read_dir(&self.dir)? {
            let ent = ent?;
            let name = ent.file_name();
            let name = name.to_string_lossy();
            if name.ends_with(".lock") {
                let _ = fs::remove_file(ent.path());
                continue;
            }
            if !name.ends_with(".jsonl") {
                continue;
            }
            if ttl > 0 {
                if let Ok(meta) = ent.metadata() {
                    if let Ok(modified) = meta.modified() {
                        if let Ok(age) = modified.elapsed() {
                            if age.as_secs() > ttl.saturating_mul(3600) {
                                let _ = fs::remove_file(ent.path());
                                continue;
                            }
                        }
                    }
                }
            }
            let rec = read_record(&ent.path())?;
            if rec.status == Status::Running {
                append_close(
                    &ent.path(),
                    &CloseBody {
                        status: Status::Aborted,
                        finish_reason: "error",
                        message: Some("server restarted before the generation finished".into()),
                        content: None,
                        reasoning: None,
                        tool_calls: vec![],
                        prompt_tokens: rec.meta.prompt_ids.len(),
                        completion_tokens: rec.ids.len(),
                    },
                )?;
                n += 1;
            }
        }
        Ok(n)
    }

    pub fn load(&self, id: &str) -> Result<Option<Record>, PlanError> {
        let id = safe_id(id).map_err(|_| PlanError::BadId)?;
        let path = self.path(&id);
        if !path.exists() {
            return Ok(None);
        }
        Ok(Some(read_record(&path)?))
    }

    pub fn is_live(&self, id: &str) -> bool {
        safe_id(id).map(|id| self.lock_path(&id).exists()).unwrap_or(false)
    }

    pub fn snapshot(&self, id: &str) -> Result<Snapshot, PlanError> {
        match self.load(id)? {
            Some(r) => Ok(r.snapshot()),
            None => Err(PlanError::NotFound),
        }
    }

    /// Decide what a resume call should do.
    /// `cont`: `None` means "continue only if the generation was aborted and tokens remain".
    pub fn plan(&self, id: &str, replay: bool, cont: Option<bool>) -> Result<Plan, PlanError> {
        if self.is_live(id) {
            return Err(PlanError::Busy);
        }
        let rec = self.load(id)?.ok_or(PlanError::NotFound)?;
        if rec.status == Status::Running {
            return Err(PlanError::Busy);
        }
        let want = cont.unwrap_or(rec.status == Status::Aborted);
        let cont = if want && rec.status == Status::Aborted {
            let used = rec.ids.len();
            let rem = rec.meta.max_tokens.saturating_sub(used);
            if rem == 0 {
                None
            } else {
                let mut prompt_ids = rec.meta.prompt_ids.clone();
                prompt_ids.extend_from_slice(&rec.ids);
                Some(Continue {
                    prompt_ids,
                    max_tokens: rem,
                    sampling: rec.meta.sampling.clone(),
                    stop: rec.meta.stop.clone(),
                    messages: rec.meta.messages.clone(),
                    tools: rec.meta.tools.clone(),
                    extra: rec.meta.extra.clone(),
                    model: rec.meta.model.clone(),
                    generated_text: rec.text.clone(),
                    generated_ids: rec.ids.clone(),
                })
            }
        } else {
            None
        };
        Ok(Plan { record: rec, replay, cont })
    }

    fn path(&self, id: &str) -> PathBuf {
        self.dir.join(format!("{id}.jsonl"))
    }
    fn lock_path(&self, id: &str) -> PathBuf {
        self.dir.join(format!("{id}.lock"))
    }
}

/// One open generation. Dropping it without [`Session::close_finished`] / [`Session::close_failed`]
/// writes an aborted close, so a client disconnect still leaves a resumable journal.
pub struct Session {
    lock_path: PathBuf,
    file: File,
    pub meta: OpenMeta,
    text: String,
    ids: Vec<u32>,
    terminal: bool,
}

impl Session {
    pub fn create(store: &Store, meta: OpenMeta) -> io::Result<Self> {
        let id = safe_id(&meta.id)?;
        if id != meta.id {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "id changed"));
        }
        let path = store.path(&id);
        let lock_path = store.lock_path(&id);
        let mut lock = OpenOptions::new().write(true).create_new(true).open(&lock_path)?;
        let _ = writeln!(lock, "{}", std::process::id());
        let mut file = OpenOptions::new().create(true).append(true).open(&path)?;
        let line = json!({
            "op": "open",
            "v": 1,
            "id": meta.id,
            "resume_token": meta.resume_token,
            "model": meta.model,
            "created": meta.created,
            "endpoint": meta.endpoint,
            "prompt_ids": meta.prompt_ids,
            "max_tokens": meta.max_tokens,
            "stop": meta.stop,
            "sampling": meta.sampling,
            "messages": meta.messages,
            "tools": meta.tools,
            "extra": meta.extra,
        });
        write_line(&mut file, &line)?;
        Ok(Self { lock_path, file, meta, text: String::new(), ids: Vec::new(), terminal: false })
    }

    /// Continue an aborted journal in place. The resume token does not change.
    pub fn resume_append(store: &Store, id: &str) -> Result<(Self, Record), PlanError> {
        let id = safe_id(id).map_err(|_| PlanError::BadId)?;
        let rec = store.load(&id)?.ok_or(PlanError::NotFound)?;
        if rec.status == Status::Running || store.is_live(&id) {
            return Err(PlanError::Busy);
        }
        if rec.status != Status::Aborted {
            return Err(PlanError::Busy);
        }
        let lock_path = store.lock_path(&id);
        let mut lock = OpenOptions::new().write(true).create_new(true).open(&lock_path).map_err(|e| {
            if e.kind() == io::ErrorKind::AlreadyExists { PlanError::Busy } else { PlanError::Io(e) }
        })?;
        let _ = writeln!(lock, "{}", std::process::id());
        let path = store.path(&id);
        let mut file = OpenOptions::new().create(true).append(true).open(&path)?;
        write_line(&mut file, &json!({"op": "reopen", "at": now_ms()}))?;
        Ok((
            Self { lock_path, file, meta: rec.meta.clone(), text: rec.text.clone(), ids: rec.ids.clone(), terminal: false },
            rec,
        ))
    }

    pub fn env(&self) -> Envelope {
        self.meta.envelope()
    }
    pub fn text(&self) -> &str {
        &self.text
    }
    pub fn ids(&self) -> &[u32] {
        &self.ids
    }

    pub fn note_token(&mut self, text: &str, ids: &[u32]) -> io::Result<()> {
        if self.terminal || (text.is_empty() && ids.is_empty()) {
            return Ok(());
        }
        write_line(&mut self.file, &json!({"op": "delta", "text": text, "ids": ids}))?;
        self.text.push_str(text);
        self.ids.extend_from_slice(ids);
        Ok(())
    }

    pub fn close_finished(
        &mut self,
        finish_reason: &str,
        content: &str,
        reasoning: &str,
        tool_calls: Vec<Value>,
        prompt_tokens: usize,
        completion_tokens: usize,
        extra: Value,
    ) -> io::Result<Vec<Frame>> {
        self.write_close(&CloseBody {
            status: Status::Finished,
            finish_reason,
            message: None,
            content: Some(content.to_string()),
            reasoning: Some(reasoning.to_string()),
            tool_calls,
            prompt_tokens,
            completion_tokens,
        })?;
        Ok(frames_finished(&self.env(), finish_reason, extra))
    }

    pub fn close_failed(&mut self, status: Status, message: &str, code: &str) -> io::Result<Vec<Frame>> {
        let st = if status == Status::Error { Status::Error } else { Status::Aborted };
        self.write_close(&CloseBody {
            status: st,
            finish_reason: "error",
            message: Some(message.to_string()),
            content: None,
            reasoning: None,
            tool_calls: vec![],
            prompt_tokens: self.meta.prompt_ids.len(),
            completion_tokens: self.ids.len(),
        })?;
        Ok(frames_failed(&self.env(), st, message, code))
    }

    fn write_close(&mut self, body: &CloseBody) -> io::Result<()> {
        if self.terminal {
            return Ok(());
        }
        let line = json!({
            "op": "close",
            "status": body.status.as_str(),
            "finish_reason": body.finish_reason,
            "message": body.message,
            "content": body.content,
            "reasoning": body.reasoning,
            "tool_calls": body.tool_calls,
            "prompt_tokens": body.prompt_tokens,
            "completion_tokens": body.completion_tokens,
        });
        write_line(&mut self.file, &line)?;
        self.terminal = true;
        Ok(())
    }
}

struct CloseBody<'a> {
    status: Status,
    finish_reason: &'a str,
    message: Option<String>,
    content: Option<String>,
    reasoning: Option<String>,
    tool_calls: Vec<Value>,
    prompt_tokens: usize,
    completion_tokens: usize,
}

impl Drop for Session {
    fn drop(&mut self) {
        if !self.terminal {
            let _ = self.write_close(&CloseBody {
                status: Status::Aborted,
                finish_reason: "error",
                message: Some("stream dropped before a terminal event".into()),
                content: None,
                reasoning: None,
                tool_calls: vec![],
                prompt_tokens: self.meta.prompt_ids.len(),
                completion_tokens: self.ids.len(),
            });
        }
        let _ = fs::remove_file(&self.lock_path);
    }
}

fn append_close(path: &Path, body: &CloseBody) -> io::Result<()> {
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    let line = json!({
        "op": "close",
        "status": body.status.as_str(),
        "finish_reason": body.finish_reason,
        "message": body.message,
        "content": body.content,
        "reasoning": body.reasoning,
        "tool_calls": body.tool_calls,
        "prompt_tokens": body.prompt_tokens,
        "completion_tokens": body.completion_tokens,
    });
    write_line(&mut file, &line)
}

fn write_line(file: &mut File, v: &Value) -> io::Result<()> {
    writeln!(file, "{v}")?;
    file.flush()
}

fn read_record(path: &Path) -> io::Result<Record> {
    let raw = fs::read_to_string(path)?;
    let mut meta: Option<OpenMeta> = None;
    let mut text = String::new();
    let mut ids = Vec::new();
    let mut status = Status::Running;
    let mut finish_reason = None;
    let mut message = None;
    let mut content = None;
    let mut reasoning = None;
    let mut tool_calls = Vec::new();
    let mut prompt_tokens = 0;
    let mut completion_tokens = 0;
    let mut closed = false;
    for line in raw.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let v: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue, // torn tail from a kill mid-write
        };
        match v.get("op").and_then(|o| o.as_str()) {
            Some("open") => {
                meta = Some(OpenMeta {
                    id: v["id"].as_str().unwrap_or_default().to_string(),
                    resume_token: v["resume_token"].as_str().unwrap_or_default().to_string(),
                    model: v["model"].as_str().unwrap_or_default().to_string(),
                    created: v["created"].as_u64().unwrap_or(0),
                    endpoint: v["endpoint"].as_str().unwrap_or("chat").to_string(),
                    prompt_ids: u32s(&v["prompt_ids"]),
                    max_tokens: v["max_tokens"].as_u64().unwrap_or(0) as usize,
                    stop: v["stop"].as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect()).unwrap_or_default(),
                    sampling: v.get("sampling").cloned().unwrap_or(Value::Null),
                    messages: v.get("messages").cloned().unwrap_or(Value::Null),
                    tools: v.get("tools").cloned().filter(|t| !t.is_null()),
                    extra: v.get("extra").cloned().unwrap_or(Value::Null),
                });
            }
            Some("delta") => {
                if let Some(t) = v.get("text").and_then(|t| t.as_str()) {
                    text.push_str(t);
                }
                ids.extend(u32s(&v["ids"]));
                if closed {
                    // deltas after a close only happen once `reopen` has run; a torn
                    // file that somehow deltas first is treated as running again below.
                }
            }
            Some("reopen") => {
                closed = false;
                status = Status::Running;
                finish_reason = None;
                message = None;
            }
            Some("close") => {
                status = Status::parse(v["status"].as_str().unwrap_or("aborted"));
                finish_reason = v["finish_reason"].as_str().map(str::to_string);
                message = v["message"].as_str().map(str::to_string);
                content = v["content"].as_str().map(str::to_string);
                reasoning = v["reasoning"].as_str().map(str::to_string);
                tool_calls = v["tool_calls"].as_array().cloned().unwrap_or_default();
                prompt_tokens = v["prompt_tokens"].as_u64().unwrap_or(0) as usize;
                completion_tokens = v["completion_tokens"].as_u64().unwrap_or(0) as usize;
                closed = true;
            }
            _ => {}
        }
    }
    let meta = meta.ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "journal has no open record"))?;
    if !closed {
        status = Status::Running;
    }
    Ok(Record { meta, text, ids, status, finish_reason, message, content, reasoning, tool_calls, prompt_tokens, completion_tokens })
}

fn u32s(v: &Value) -> Vec<u32> {
    v.as_array().map(|a| a.iter().filter_map(|x| x.as_u64().map(|n| n as u32)).collect()).unwrap_or_default()
}

pub fn safe_id(id: &str) -> io::Result<String> {
    if (8..=80).contains(&id.len()) && id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') {
        Ok(id.to_string())
    } else {
        Err(io::Error::new(io::ErrorKind::InvalidInput, "invalid generation id"))
    }
}

fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{data_frame, role_frame, Envelope};

    fn store() -> (tempfile::TempDir, Store) {
        let d = tempfile::tempdir().unwrap();
        let s = Store::open(d.path()).unwrap();
        (d, s)
    }

    fn meta(id: &str, max_tokens: usize) -> OpenMeta {
        OpenMeta {
            id: id.into(),
            resume_token: id.into(),
            model: "qwen".into(),
            created: 10,
            endpoint: "chat".into(),
            prompt_ids: vec![1, 2, 3],
            max_tokens,
            stop: vec!["</s>".into()],
            sampling: json!({"temperature": 0.0, "seed": 7}),
            messages: json!([{"role": "user", "content": "hi"}]),
            tools: None,
            extra: json!({"prompt_open_think": false}),
        }
    }

    #[test]
    fn deltas_survive_reopen_and_recover_marks_aborted() {
        let (_d, s) = store();
        let mut sess = Session::create(&s, meta("chatcmpl-aaaaaaaaaaaaaaaa", 32)).unwrap();
        sess.note_token("hel", &[10]).unwrap();
        sess.note_token("lo", &[11, 12]).unwrap();
        drop(sess); // client gone: aborted close + lock released
        let rec = s.load("chatcmpl-aaaaaaaaaaaaaaaa").unwrap().unwrap();
        assert_eq!(rec.status, Status::Aborted);
        assert_eq!(rec.text, "hello");
        assert_eq!(rec.ids, vec![10, 11, 12]);
        let plan = s.plan("chatcmpl-aaaaaaaaaaaaaaaa", true, None).unwrap();
        let c = plan.cont.unwrap();
        assert_eq!(c.prompt_ids, vec![1, 2, 3, 10, 11, 12]);
        assert_eq!(c.max_tokens, 29);
        assert_eq!(c.generated_text, "hello");

        // Simulate a kill: strip the close line and recover.
        let path = s.path("chatcmpl-aaaaaaaaaaaaaaaa");
        let raw = fs::read_to_string(&path).unwrap();
        let kept: String = raw.lines().filter(|l| !l.contains("\"op\":\"close\"")).map(|l| format!("{l}\n")).collect();
        fs::write(&path, kept).unwrap();
        assert_eq!(s.load("chatcmpl-aaaaaaaaaaaaaaaa").unwrap().unwrap().status, Status::Running);
        assert_eq!(s.recover().unwrap(), 1);
        let rec = s.load("chatcmpl-aaaaaaaaaaaaaaaa").unwrap().unwrap();
        assert_eq!(rec.status, Status::Aborted);
        assert_eq!(rec.text, "hello");
        assert!(rec.message.unwrap().contains("restarted"));
    }

    #[test]
    fn torn_tail_is_ignored_and_finished_is_not_recoverable() {
        let (_d, s) = store();
        let mut sess = Session::create(&s, meta("chatcmpl-bbbbbbbbbbbbbbbb", 8)).unwrap();
        sess.note_token("ok", &[4]).unwrap();
        let frames = sess.close_finished("stop", "ok", "", vec![], 3, 1, json!({})).unwrap();
        assert_eq!(frames.last().unwrap().data, "[DONE]");
        assert!(frames.iter().filter(|f| f.data == "[DONE]").count() == 1);
        drop(sess);
        let path = s.path("chatcmpl-bbbbbbbbbbbbbbbb");
        let mut raw = fs::read_to_string(&path).unwrap();
        raw.push_str("{\"op\":\"delta\",\"text\":\"PARTIAL\""); // no newline, invalid
        fs::write(&path, raw).unwrap();
        let rec = s.load("chatcmpl-bbbbbbbbbbbbbbbb").unwrap().unwrap();
        assert_eq!(rec.status, Status::Finished);
        assert_eq!(rec.text, "ok");
        assert_eq!(rec.finish_reason.as_deref(), Some("stop"));
        assert!(s.plan("chatcmpl-bbbbbbbbbbbbbbbb", true, Some(true)).unwrap().cont.is_none());
        assert_eq!(s.recover().unwrap(), 0);
    }

    #[test]
    fn rejects_path_ids_and_busy_lock() {
        let (_d, s) = store();
        assert!(s.load("../etc/passwd").is_err());
        assert!(s.load("short").is_err());
        let sess = Session::create(&s, meta("chatcmpl-cccccccccccccccc", 4)).unwrap();
        assert!(matches!(s.plan("chatcmpl-cccccccccccccccc", true, None), Err(PlanError::Busy)));
        drop(sess);
        assert!(s.plan("chatcmpl-cccccccccccccccc", true, None).is_ok());
    }

    #[test]
    fn every_frame_carries_the_request_id() {
        let env = Envelope::new("chatcmpl-dddddddddddddddd", 1, "qwen");
        let frames = vec![
            role_frame(&env),
            data_frame(&env, json!({"content": "hi"}), false),
        ];
        let failed = frames_failed(&env, Status::Aborted, "server shutting down", "server_shutdown");
        frames.iter().chain(failed.iter()).filter(|f| !f.is_done()).for_each(|f| {
            let v: Value = serde_json::from_str(&f.data).unwrap();
            let id = v.get("id").and_then(|i| i.as_str()).or_else(|| v.pointer("/error/resume_token").and_then(|i| i.as_str()));
            assert_eq!(id, Some("chatcmpl-dddddddddddddddd"));
        });
        let term = failed.iter().find(|f| f.data.contains("finish_reason")).unwrap();
        let v: Value = serde_json::from_str(&term.data).unwrap();
        assert_eq!(v["choices"][0]["finish_reason"], "error");
        assert_eq!(v["pyre"]["status"], "aborted");
        assert_eq!(v["pyre"]["message"], "server shutting down");
        assert_eq!(failed.last().unwrap().data, "[DONE]");
    }

    #[test]
    fn resume_append_keeps_the_prefix() {
        let (_d, s) = store();
        let id = "chatcmpl-eeeeeeeeeeeeeeee";
        let mut sess = Session::create(&s, meta(id, 16)).unwrap();
        sess.note_token("ab", &[9]).unwrap();
        drop(sess);
        let (mut sess, rec) = Session::resume_append(&s, id).unwrap();
        assert_eq!(rec.text, "ab");
        sess.note_token("c", &[10]).unwrap();
        sess.close_finished("stop", "abc", "", vec![], 3, 2, json!({})).unwrap();
        drop(sess);
        let rec = s.load(id).unwrap().unwrap();
        assert_eq!(rec.status, Status::Finished);
        assert_eq!(rec.text, "abc");
        assert_eq!(rec.ids, vec![9, 10]);
        assert!(s.plan(id, true, Some(true)).unwrap().cont.is_none());
    }
}
