//! HF chat templates (tokenizer_config.json / chat_template.jinja) rendered with minijinja.
use anyhow::{Context, Result};
use minijinja::{context, Environment, Error, ErrorKind, Value};
use serde_json::Value as Json;
use std::path::Path;

pub struct ChatTemplate {
    env: Environment<'static>,
    pub source: String,
    pub bos_token: String,
    pub eos_token: String,
}

fn token_str(v: Option<&Json>) -> String {
    match v {
        Some(Json::String(s)) => s.clone(),
        Some(Json::Object(o)) => o.get("content").and_then(|c| c.as_str()).unwrap_or_default().to_string(),
        _ => String::new(),
    }
}

/// Fallback ChatML template (Qwen-style) used when the model ships none.
pub const CHATML: &str = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}";

impl ChatTemplate {
    pub fn new(source: String, bos_token: String, eos_token: String) -> Result<Self> {
        let mut env = Environment::new();
        env.set_syntax(minijinja::syntax::SyntaxConfig::builder().trim_blocks(true).lstrip_blocks(true).build().expect("syntax"));
        minijinja_contrib::add_to_environment(&mut env);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function("raise_exception", |msg: String| -> Result<Value, Error> {
            Err(Error::new(ErrorKind::InvalidOperation, msg))
        });
        env.add_function("strftime_now", |fmt: String| -> String {
            // Minimal: only used by Llama 3.x templates for "Today Date".
            let _ = fmt;
            "26 Jul 2024".to_string()
        });
        env.add_template_owned("chat", source.clone()).context("parsing chat template")?;
        Ok(Self { env, source, bos_token, eos_token })
    }

    /// Load from a model directory: chat_template.jinja, else tokenizer_config.json `chat_template`.
    pub fn from_model_dir(dir: &Path) -> Result<Self> {
        let cfg: Json = std::fs::read_to_string(dir.join("tokenizer_config.json"))
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or(Json::Null);
        let bos = token_str(cfg.get("bos_token"));
        let eos = token_str(cfg.get("eos_token"));
        let mut src = std::fs::read_to_string(dir.join("chat_template.jinja")).ok();
        if src.is_none() {
            src = match cfg.get("chat_template") {
                Some(Json::String(s)) => Some(s.clone()),
                Some(Json::Array(a)) => a
                    .iter()
                    .find(|t| t.get("name").and_then(|n| n.as_str()) == Some("default"))
                    .or_else(|| a.first())
                    .and_then(|t| t.get("template")).and_then(|t| t.as_str()).map(String::from),
                _ => None,
            };
        }
        let src = src.unwrap_or_else(|| {
            tracing::warn!("model has no chat template; falling back to ChatML");
            CHATML.to_string()
        });
        Self::new(src, bos, eos)
    }

    pub fn render(&self, messages: &[Json], add_generation_prompt: bool, tools: Option<&Json>) -> Result<String> {
        let tmpl = self.env.get_template("chat")?;
        let out = tmpl.render(context! {
            messages => Value::from(minijinja::value::Serde(messages)),
            add_generation_prompt => add_generation_prompt,
            tools => tools.map(|t| Value::from(minijinja::value::Serde(t))),
            bos_token => self.bos_token.clone(),
            eos_token => self.eos_token.clone(),
        })?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // Trimmed Qwen2.5 template (system default + ChatML).
    const QWEN: &str = r#"{%- if messages[0]['role'] == 'system' %}
    {{- '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}
{%- else %}
    {{- '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n' }}
{%- endif %}
{%- for message in messages %}
    {%- if not (message.role == "system" and loop.first) %}
        {{- '<|im_start|>' + message.role + '\n' + message.content.strip() + '<|im_end|>' + '\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}
"#;

    #[test]
    fn qwen_template_renders() {
        let t = ChatTemplate::new(QWEN.into(), "".into(), "<|im_end|>".into()).unwrap();
        let out = t.render(&[json!({"role":"user","content":" Hi! "})], true, None).unwrap();
        assert_eq!(out, "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nHi!<|im_end|>\n<|im_start|>assistant\n");
    }

    #[test]
    fn llama3_style_template_with_bos() {
        let src = "{{ bos_token }}{% for m in messages %}{{ '<|start_header_id|>' + m['role'] + '<|end_header_id|>\n\n' + m['content'] | trim + '<|eot_id|>' }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}";
        let t = ChatTemplate::new(src.into(), "<|begin_of_text|>".into(), "<|eot_id|>".into()).unwrap();
        let out = t.render(&[json!({"role":"system","content":"S"}), json!({"role":"user","content":"U"})], true, None).unwrap();
        assert_eq!(out, "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nS<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nU<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n");
    }

    #[test]
    fn raise_exception_surfaces() {
        let t = ChatTemplate::new("{{ raise_exception('nope') }}".into(), "".into(), "".into()).unwrap();
        assert!(t.render(&[], true, None).is_err());
    }

    #[test]
    fn chatml_fallback() {
        let t = ChatTemplate::new(CHATML.into(), "".into(), "".into()).unwrap();
        let out = t.render(&[json!({"role":"user","content":"x"})], false, None).unwrap();
        assert_eq!(out, "<|im_start|>user\nx<|im_end|>\n");
    }
}
