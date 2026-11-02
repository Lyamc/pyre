//! A small Markdown subset for chat replies: fenced code blocks, headings, bullets, numbered
//! lists, quotes, rules, and inline `code` / **bold** (markers removed).
pub enum Block {
    Para(String),
    Heading(u8, String),
    Bullet(String, String),
    Quote(String),
    Code { lang: String, code: String },
    Rule,
    Gap,
}

pub fn strip_inline(s: &str) -> String {
    s.replace("**", "").replace("__", "").replace('`', "")
}

pub fn parse(src: &str) -> Vec<Block> {
    let mut out = vec![];
    let mut code: Option<(String, Vec<&str>)> = None;
    let mut para: Vec<String> = vec![];
    let flush = |para: &mut Vec<String>, out: &mut Vec<Block>| {
        if !para.is_empty() { out.push(Block::Para(para.join("\n"))); para.clear(); }
    };
    for line in src.lines() {
        let t = line.trim_start();
        if let Some((lang, body)) = code.as_mut() {
            if t.starts_with("```") {
                out.push(Block::Code { lang: lang.clone(), code: body.join("\n") });
                code = None;
            } else {
                body.push(line);
            }
            continue;
        }
        if let Some(rest) = t.strip_prefix("```") {
            flush(&mut para, &mut out);
            code = Some((rest.trim().to_string(), vec![]));
        } else if t.is_empty() {
            flush(&mut para, &mut out);
            if !matches!(out.last(), Some(Block::Gap) | None) { out.push(Block::Gap); }
        } else if t.starts_with('#') {
            flush(&mut para, &mut out);
            let n = t.chars().take_while(|c| *c == '#').count().min(6) as u8;
            out.push(Block::Heading(n, strip_inline(t[n as usize..].trim())));
        } else if t == "---" || t == "***" || t == "___" {
            flush(&mut para, &mut out);
            out.push(Block::Rule);
        } else if let Some(r) = t.strip_prefix("- ").or_else(|| t.strip_prefix("* ")).or_else(|| t.strip_prefix("+ ")) {
            flush(&mut para, &mut out);
            let indent = (line.len() - t.len()) / 2;
            out.push(Block::Bullet(format!("{}•", "    ".repeat(indent)), strip_inline(r)));
        } else if let Some((num, r)) = t.split_once(". ").filter(|(n, _)| !n.is_empty() && n.len() <= 3 && n.chars().all(|c| c.is_ascii_digit())) {
            flush(&mut para, &mut out);
            out.push(Block::Bullet(format!("{num}."), strip_inline(r)));
        } else if let Some(r) = t.strip_prefix('>') {
            flush(&mut para, &mut out);
            out.push(Block::Quote(strip_inline(r.trim())));
        } else {
            para.push(strip_inline(line));
        }
    }
    if let Some((lang, body)) = code { out.push(Block::Code { lang, code: body.join("\n") }); }
    flush(&mut para, &mut out);
    while matches!(out.last(), Some(Block::Gap)) { out.pop(); }
    out
}

/// Splits a raw `<think>..</think>` prefix out of content (when the server's reasoning split is off).
pub fn split_think(content: &str) -> (String, String) {
    if let Some(i) = content.find("</think>") {
        let think = content[..i].trim_start_matches("<think>").to_string();
        return (think, content[i + 8..].trim_start().to_string());
    }
    if let Some(r) = content.strip_prefix("<think>") { return (r.to_string(), String::new()); }
    (String::new(), content.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn blocks() {
        let b = parse("# Title\nsome **bold** text\n\n```python\nprint(1)\n```\n- a\n1. b");
        assert!(matches!(&b[0], Block::Heading(1, t) if t == "Title"));
        assert!(matches!(&b[1], Block::Para(t) if t == "some bold text"));
        assert!(matches!(&b[3], Block::Code { lang, code } if lang == "python" && code == "print(1)"));
        assert!(matches!(&b[4], Block::Bullet(_, t) if t == "a"));
        assert!(matches!(&b[5], Block::Bullet(n, t) if n == "1." && t == "b"));
        assert_eq!(split_think("<think>hm</think>\n\nok"), ("hm".into(), "ok".into()));
    }
}
