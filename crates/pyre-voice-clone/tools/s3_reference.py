#!/usr/bin/env python3
"""Dev-time PyTorch reference for the Nano S3 speech tokenizer.

Not used by pyre. Needs torch. Writes a short formant (not a voice) and the
token ids `S3TokenizerV2` produces from `s3gen_meanflow.safetensors`.

    python crates/pyre-voice-clone/tools/s3_reference.py \
        --checkpoint /path/to/s3gen_meanflow.safetensors
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file


def formant(n: int = 7680, sr: int = 16000) -> np.ndarray:
    """Eight harmonics of 140 Hz. A test signal, not a speaker."""
    t = np.arange(n, dtype=np.float64) / sr
    wav = np.zeros(n, dtype=np.float64)
    for h in range(1, 9):
        wav += (0.15 / h) * np.sin(2 * math.pi * 140.0 * h * t)
    return wav.astype(np.float32)


def log_mel(wav: torch.Tensor, filters: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    stft = torch.stft(wav, 400, 160, window=window, return_complex=True, center=True)
    mag = stft[..., :-1].abs() ** 2
    mel = filters @ mag
    log_spec = torch.clamp(mel, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    return (log_spec + 4.0) / 4.0


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x [1, T, 20, 64], cos/sin [T, 64]
    d = x.shape[-1]
    half_l, half_r = x[..., : d // 2], x[..., d // 2 :]
    xr = torch.cat((-half_r, half_l), dim=-1)
    return x * cos[None, :, None, :] + xr * sin[None, :, None, :]


def build_rope(length: int = 2048, dim: int = 64, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    ang = torch.outer(torch.arange(length).float(), freqs)
    ang = torch.cat((ang, ang), dim=-1)
    return ang.cos(), ang.sin()


def encode(weights: dict, wav: np.ndarray, max_tokens: int = 375) -> list[int]:
    window = weights["tokenizer.window"]
    filters = weights["tokenizer._mel_filters"]
    mel = log_mel(torch.from_numpy(wav), filters, window)
    mel = mel[:, : max_tokens * 4]
    x = mel.unsqueeze(0)
    conv1 = torch.nn.Conv1d(128, 1280, 3, stride=2, padding=1)
    conv2 = torch.nn.Conv1d(1280, 1280, 3, stride=2, padding=1)
    conv1.weight.data.copy_(weights["tokenizer.encoder.conv1.weight"])
    conv1.bias.data.copy_(weights["tokenizer.encoder.conv1.bias"])
    conv2.weight.data.copy_(weights["tokenizer.encoder.conv2.weight"])
    conv2.bias.data.copy_(weights["tokenizer.encoder.conv2.bias"])
    x = F.gelu(conv1(x))
    x = F.gelu(conv2(x))
    x = x.permute(0, 2, 1)  # [1, T, 1280]
    tlen = x.shape[1]
    cos, sin = build_rope()
    cos, sin = cos[:tlen], sin[:tlen]
    scale = 64 ** -0.25
    for i in range(6):
        p = f"tokenizer.encoder.blocks.{i}"
        h = F.layer_norm(x, (1280,), weights[f"{p}.attn_ln.weight"], weights[f"{p}.attn_ln.bias"], 1e-5)
        q = F.linear(h, weights[f"{p}.attn.query.weight"], weights[f"{p}.attn.query.bias"])
        k = F.linear(h, weights[f"{p}.attn.key.weight"], None)
        v = F.linear(h, weights[f"{p}.attn.value.weight"], weights[f"{p}.attn.value.bias"])
        q = q.view(1, tlen, 20, 64)
        k = k.view(1, tlen, 20, 64)
        v = v.view(1, tlen, 20, 64)
        q, k = rope(q, cos, sin), rope(k, cos, sin)
        fsm = v.reshape(1, tlen, 1280)
        fsm_c = F.pad(fsm.transpose(1, 2), (15, 15))
        fsm_c = F.conv1d(fsm_c, weights[f"{p}.attn.fsmn_block.weight"], groups=1280)
        fsm = fsm + fsm_c.transpose(1, 2)
        q = q.permute(0, 2, 1, 3) * scale
        k = k.permute(0, 2, 3, 1) * scale
        vv = v.permute(0, 2, 1, 3)
        attn = torch.softmax(q @ k, dim=-1) @ vv
        attn = attn.permute(0, 2, 1, 3).reshape(1, tlen, 1280)
        attn = F.linear(attn, weights[f"{p}.attn.out.weight"], weights[f"{p}.attn.out.bias"])
        x = x + attn + fsm
        h = F.layer_norm(x, (1280,), weights[f"{p}.mlp_ln.weight"], weights[f"{p}.mlp_ln.bias"], 1e-5)
        h = F.gelu(F.linear(h, weights[f"{p}.mlp.0.weight"], weights[f"{p}.mlp.0.bias"]))
        x = x + F.linear(h, weights[f"{p}.mlp.2.weight"], weights[f"{p}.mlp.2.bias"])
    z = F.linear(
        x,
        weights["tokenizer.quantizer._codebook.project_down.weight"],
        weights["tokenizer.quantizer._codebook.project_down.bias"],
    )
    h = z.tanh() * 0.9990000128746033
    h = h.round() + 1
    powers = 3 ** torch.arange(8)
    ids = (h * powers).sum(-1).to(torch.int64)
    return ids.view(-1).tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()
    w = load_file(args.checkpoint)
    keep = {k: v for k, v in w.items() if k.startswith("tokenizer.")}
    wav = formant()
    with torch.no_grad():
        ids = encode(keep, wav, 375)
    print(f"tokens {len(ids)} head {ids[:12]} unique {len(set(ids))}")
    if any(i < 0 or i >= 6561 for i in ids):
        raise SystemExit("id outside 0..6560")
    root = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parents[1] / "testdata"
    root.mkdir(parents=True, exist_ok=True)
    (root / "s3_formant.f32").write_bytes(wav.tobytes())
    (root / "s3_formant_golden.json").write_text(json.dumps({
        "sample_rate": 16000,
        "samples": int(wav.shape[0]),
        "note": "Eight harmonics of 140 Hz. Not a voice and not a speaker-verification sample.",
        "max_tokens": 375,
        "tokens": ids,
    }))
    print(f"wrote {root / 's3_formant_golden.json'}")


if __name__ == "__main__":
    main()
