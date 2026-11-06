#!/usr/bin/env python3
"""One-time weight lift: NeMo Parakeet-TDT .nemo archive -> HF-layout model.safetensors + config.json.

This only re-keys checkpoint tensors so pyre's Rust converter (convert/parakeet) can turn them into a
Burn record. No inference happens here; pyre runs Parakeet natively in Burn.
Usage: extract_nemo_parakeet.py --nemo X.nemo --out DIR [--vocab vocab.txt]
"""
import argparse, io, json, os, re, tarfile

import torch
from safetensors.torch import save_file

RENAMES = [
    (r"^encoder\.pre_encode\.conv\.(\d+)\.", r"encoder.subsampling.layers.\1."),
    (r"^encoder\.pre_encode\.out\.", "encoder.subsampling.linear."),
    (r"\.self_attn\.linear_q\.", ".self_attn.q_proj."),
    (r"\.self_attn\.linear_k\.", ".self_attn.k_proj."),
    (r"\.self_attn\.linear_v\.", ".self_attn.v_proj."),
    (r"\.self_attn\.linear_out\.", ".self_attn.o_proj."),
    (r"\.self_attn\.linear_pos\.", ".self_attn.relative_k_proj."),
    (r"\.self_attn\.pos_bias_u$", ".self_attn.bias_u"),
    (r"\.self_attn\.pos_bias_v$", ".self_attn.bias_v"),
    (r"\.conv\.batch_norm\.", ".conv.norm."),
    (r"^decoder\.prediction\.embed\.", "decoder.embedding."),
    (r"^decoder\.prediction\.dec_rnn\.lstm\.", "decoder.lstm."),
    (r"^joint\.enc\.", "encoder_projector."),
    (r"^joint\.pred\.", "decoder.decoder_projector."),
    (r"^joint\.joint_net\.2\.", "joint.head."),
]


def rename(k):
    for pat, rep in RENAMES:
        k = re.sub(pat, rep, k)
    return k


def scalar(yaml, key):
    m = re.search(r"^\s*" + key + r":\s*(\S+)", yaml, re.M)
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nemo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab")
    a = ap.parse_args()
    tar = tarfile.open(a.nemo)
    members = {os.path.basename(m.name): m for m in tar.getmembers() if m.isfile()}
    yaml = tar.extractfile(members["model_config.yaml"]).read().decode()
    sd = torch.load(io.BytesIO(tar.extractfile(members["model_weights.ckpt"]).read()), map_location="cpu")
    out = {}
    for k, v in sd.items():
        if k.startswith("preprocessor.") or k.endswith("num_batches_tracked"):
            continue
        out[rename(k)] = v.detach().to(torch.float32).contiguous()
    # NeMo conformer layers are bias-free; pyre's Burn module (HF layout) has bias slots, so give them zeros.
    for k in list(out):
        if re.search(r"\.(self_attn\.(q|k|v|o)_proj|feed_forward[12]\.linear[12]|conv\.(pointwise_conv[12]|depthwise_conv))\.weight$", k):
            b = k[: -len("weight")] + "bias"
            if b not in out:
                out[b] = torch.zeros(out[k].shape[0], dtype=torch.float32)
    vocab = int(scalar(yaml, "vocab_size")) + 1  # + blank
    m = re.search(r"durations:\s*\n((?:\s*-\s*\d+\s*\n)+)", yaml)
    durations = [int(x) for x in re.findall(r"\d+", m.group(1))] if m else [0, 1, 2, 3, 4]
    head = out["joint.head.weight"].shape[0]
    assert head == vocab + len(durations), (head, vocab, durations)
    cfg = {
        "architectures": ["ParakeetForTDT"],
        "vocab_size": vocab,
        "blank_token_id": vocab - 1,
        "pad_token_id": vocab - 1,
        "durations": durations,
        "encoder_config": {
            "hidden_size": int(scalar(yaml, "d_model")),
            "num_attention_heads": int(scalar(yaml, "n_heads")),
            "num_hidden_layers": int(scalar(yaml, "n_layers")),
            "num_mel_bins": int(scalar(yaml, "features")),
            "scale_input": scalar(yaml, "xscaling") == "true",
        },
    }
    os.makedirs(a.out, exist_ok=True)
    save_file(out, os.path.join(a.out, "model.safetensors"))
    with open(os.path.join(a.out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    # Token table for pyre's tokenizer loader: the SentencePiece pieces in id order, as a WordLevel tokenizer.json.
    vname = next(n for n in members if n.endswith("tokenizer.vocab"))
    pieces = [l.split("\t")[0] for l in tar.extractfile(members[vname]).read().decode("utf-8").splitlines() if l]
    tok = {"version": "1.0", "truncation": None, "padding": None, "added_tokens": [], "normalizer": None,
           "pre_tokenizer": None, "post_processor": None, "decoder": None,
           "model": {"type": "WordLevel", "vocab": {p: i for i, p in enumerate(pieces)}, "unk_token": pieces[0]}}
    with open(os.path.join(a.out, "tokenizer.json"), "w", encoding="utf-8") as f:
        json.dump(tok, f, ensure_ascii=False)    print(f"wrote {len(out)} tensors, config {cfg}", flush=True)


if __name__ == "__main__":
    main()
