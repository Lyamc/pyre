#!/usr/bin/env python3
"""Dev-time reference for the Chatterbox voice encoder. Not used at runtime.

Compares librosa's mel and a NumPy LSTM against the Rust port. Requires
numpy, librosa, and safetensors, which are not pyre dependencies.

    python crates/pyre-voice-clone/tools/ve_reference.py \\
        --ve /path/to/ve.safetensors --out /tmp/ve_golden.json

The Rust test reads that JSON when PYRE_CHATTERBOX_VE_GOLDEN is set.
"""

from __future__ import annotations

import argparse
import json
import math

import librosa
import numpy as np
from safetensors import safe_open


SR = 16000
N_FFT = 400
HOP = 160
N_MELS = 40
PARTIAL = 160
RATE = 1.3
MIN_COV = 0.8


def synth_harmonic(f0: float, seconds: float) -> np.ndarray:
    n = int(round(seconds * SR))
    t = np.arange(n, dtype=np.float32) / SR
    s = np.zeros(n, dtype=np.float32)
    for k in range(1, 9):
        s += (1.0 / k) * np.sin(2.0 * math.pi * f0 * k * t)
    return s * 0.15


def partials(mel_tm: np.ndarray) -> np.ndarray:
    n_frames = mel_tm.shape[0]
    step = int(round((SR / RATE) / PARTIAL))
    span = max(n_frames - PARTIAL + step, 0)
    n_wins, rem = divmod(span, step)
    if n_wins == 0 or (rem + (PARTIAL - step)) / PARTIAL >= MIN_COV:
        n_wins += 1
    target = PARTIAL + step * (n_wins - 1)
    padded = np.zeros((target, N_MELS), dtype=np.float32)
    copy = min(n_frames, target)
    padded[:copy] = mel_tm[:copy]
    return np.stack([padded[i * step : i * step + PARTIAL] for i in range(n_wins)])


def lstm_embed(weights: dict, partials_btm: np.ndarray) -> np.ndarray:
    x = partials_btm.astype(np.float32)
    for layer in range(3):
        w_ih = weights[f"lstm.weight_ih_l{layer}"]
        w_hh = weights[f"lstm.weight_hh_l{layer}"]
        b_ih = weights[f"lstm.bias_ih_l{layer}"]
        b_hh = weights[f"lstm.bias_hh_l{layer}"]
        b, t, _ = x.shape
        h = np.zeros((b, w_hh.shape[1]), np.float32)
        c = np.zeros_like(h)
        outs = []
        for ti in range(t):
            gates = x[:, ti] @ w_ih.T + b_ih + h @ w_hh.T + b_hh
            hdim = h.shape[1]
            i, f, g, o = np.split(gates, 4, axis=1)
            i, f, o = 1 / (1 + np.exp(-i)), 1 / (1 + np.exp(-f)), 1 / (1 + np.exp(-o))
            g = np.tanh(g)
            c = f * c + i * g
            h = o * np.tanh(c)
            outs.append(h)
        x = np.stack(outs, axis=1)
    last = x[:, -1]
    raw = last @ weights["proj.weight"].T + weights["proj.bias"]
    raw = np.maximum(raw, 0)
    raw /= np.linalg.norm(raw, axis=1, keepdims=True).clip(1e-12)
    mean = raw.mean(axis=0)
    mean /= np.linalg.norm(mean).clip(1e-12)
    return mean.astype(np.float32)


def embed(weights: dict, wav: np.ndarray) -> np.ndarray:
    wav, _ = librosa.effects.trim(wav, top_db=20)
    mel = librosa.feature.melspectrogram(
        y=wav,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP,
        win_length=N_FFT,
        n_mels=N_MELS,
        fmin=0,
        fmax=8000,
        power=2.0,
        center=True,
        pad_mode="reflect",
        htk=False,
        norm="slaney",
    )
    return lstm_embed(weights, partials(mel.T.astype(np.float32)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ve", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    weights = {}
    with safe_open(args.ve, framework="numpy") as f:
        for k in f.keys():
            weights[k] = f.get_tensor(k)
    vowel = embed(weights, synth_harmonic(140.0, 3.0))
    other = embed(weights, synth_harmonic(280.0, 3.0))
    payload = {
        "vowel_140": vowel.tolist(),
        "vowel_280": other.tolist(),
        "cosine": float(np.dot(vowel, other)),
    }
    with open(args.out, "w") as fh:
        json.dump(payload, fh)
    print(f"wrote {args.out} cosine={payload['cosine']:.6f} norm={np.linalg.norm(vowel):.6f}")


if __name__ == "__main__":
    main()
