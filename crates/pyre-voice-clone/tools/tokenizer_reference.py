#!/usr/bin/env python3
"""Dev-time check of the Nano GPT-2 tokenizer against Hugging Face.

Not used by pyre. Needs `transformers` (torch is not required):

    python crates/pyre-voice-clone/tools/tokenizer_reference.py \\
        --dir /path/to/chatterbox-nano

Prints ids for the strings locked in testdata/tokenizer_golden.json and
exits non-zero if they drifted. Point PYRE_CHATTERBOX_TOKENIZER at the same
directory and run:

    cargo test -p pyre-voice-clone -- --ignored real_tokenizer_matches_hf_golden
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def punc(text: str) -> str:
    if len(text) == 0:
        return "You need to add some text for me to talk."
    if text[0].islower():
        text = text[0].upper() + text[1:]
    text = " ".join(text.split())
    for old, new in (
        ("…", ", "),
        (":", ","),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        ("“", '"'),
        ("”", '"'),
        ("‘", "'"),
        ("’", "'"),
    ):
        text = text.replace(old, new)
    text = text.rstrip(" ")
    if not text.endswith((".", "!", "?", "-", ",")):
        text += "."
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument(
        "--golden",
        default=str(Path(__file__).resolve().parents[1] / "testdata" / "tokenizer_golden.json"),
    )
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.dir, local_files_only=True)
    golden = json.loads(Path(args.golden).read_text())
    bad = 0
    if len(tok) != golden["vocab_size"] or tok.eos_token_id != golden["eos_id"]:
        print(f"vocab {len(tok)} eos {tok.eos_token_id}")
        bad += 1
    for case in golden["cases"]:
        ids = tok.encode(case["text"], add_special_tokens=False)
        if ids != case["ids"]:
            print(f"MISMATCH {case['text']!r}\n  hf {ids}\n  golden {case['ids']}")
            bad += 1
    for script in golden["scripts"]:
        norm = punc(script["input"])
        ids = tok.encode(norm, add_special_tokens=False)
        if norm != script["normalized"] or ids != script["ids"]:
            print(f"MISMATCH script {script['input']!r}\n  norm {norm!r}\n  ids {ids}")
            bad += 1
    for tag in golden["tags"]:
        ids = tok.encode(tag["tag"], add_special_tokens=False)
        if ids != [tag["id"]]:
            print(f"TAG {tag['tag']} -> {ids}, expected [{tag['id']}]")
            bad += 1
    if bad:
        raise SystemExit(f"{bad} mismatches")
    print(f"ok {len(golden['cases'])} strings, {len(golden['tags'])} single-id tags, vocab {len(tok)}")


if __name__ == "__main__":
    main()
