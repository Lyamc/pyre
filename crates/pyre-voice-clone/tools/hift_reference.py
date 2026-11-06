#!/usr/bin/env python3
"""Dev-only HiFT reference. Not imported by the server.

Builds the Nano HiFTGenerator (Resemble chatterbox s3gen/hifigan.py, Apache-2.0
via CosyVoice) and writes a short mel plus the waveform PyTorch emits.

The clip is the first second of the public DeepSpeech smoke file LDC93S1
("she had your dark suit in greasy wash water all year"), resampled to 24 kHz.
It is a vocoder reconstruction of that mel, not a cloned voice. The harmonic
source draws torch phase and noise; the saved excitation is what Rust must
decode. Rust's own seeded source is a separate repeatability check.

Usage:
  python3 tools/hift_reference.py --checkpoint /tmp/s3gen_meanflow.safetensors
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import wave
from pathlib import Path

import numpy as np
import torch


def load_py(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        raw = w.readframes(n)
    if sw != 2:
        raise SystemExit(f"expected pcm16, got width {sw}")
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr


def resample_linear(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    """Match librosa's default kaiser_fast only if librosa is present."""
    import librosa

    return librosa.resample(x, orig_sr=src, target_sr=dst).astype(np.float32)


def mel_24k(y: torch.Tensor) -> torch.Tensor:
    """CosyVoice / chatterbox mel_spectrogram defaults. y is [1, samples]."""
    import librosa

    n_fft, hop, win, n_mels, sr = 1920, 480, 1920, 80, 24000
    filt = torch.from_numpy(
        librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=0, fmax=8000)
    ).float()
    window = torch.hann_window(win)
    pad = (n_fft - hop) // 2
    y = torch.nn.functional.pad(y.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop,
        win_length=win,
        window=window,
        center=False,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    mag = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-9)
    mel = torch.matmul(filt, mag)
    return torch.log(torch.clamp(mel, min=1e-5))


def tiny_ops(out: Path) -> None:
    """Hand-checkable conv-transpose, Snake, and iSTFT vectors."""
    x = torch.tensor([[[1.0, -0.5, 0.25, 0.0]]], dtype=torch.float32)
    w = torch.tensor([[[1.0, 0.0, -1.0], [0.5, 0.5, 0.5]]], dtype=torch.float32)
    # weight for ConvTranspose1d is [in, out, k] = [1, 2, 3]
    y = torch.nn.functional.conv_transpose1d(x, w, bias=None, stride=2, padding=1)
    snake_a = torch.tensor([2.0])
    sx = torch.tensor([[[-0.4, 0.2, 1.5]]])
    alpha = snake_a.view(1, -1, 1)
    snake = sx + (1.0 / (alpha + 1e-9)) * torch.sin(sx * alpha).pow(2)
    # 16-point iSTFT, 3 frames, hop 4, periodic hann, center True.
    n_fft, hop = 16, 4
    window = torch.hann_window(n_fft, periodic=True)
    mag = torch.zeros(1, 9, 3)
    phase = torch.zeros(1, 9, 3)
    mag[0, 1, :] = 0.5
    phase[0, 1, 0] = 0.3
    phase[0, 1, 1] = -0.2
    phase[0, 1, 2] = 0.7
    real = mag * torch.cos(phase)
    imag = mag * torch.sin(phase)
    wav = torch.istft(torch.complex(real, imag), n_fft, hop, n_fft, window=window)
    payload = {
        "conv_t_in": x.numpy().reshape(-1).tolist(),
        "conv_t_w": w.numpy().reshape(-1).tolist(),
        "conv_t_out": y.numpy().reshape(-1).tolist(),
        "snake_in": sx.numpy().reshape(-1).tolist(),
        "snake_alpha": 2.0,
        "snake_out": snake.numpy().reshape(-1).tolist(),
        "istft_wav": wav.numpy().reshape(-1).tolist(),
        "hann16": window.numpy().tolist(),
    }
    (out / "hift_tiny.json").write_text(json.dumps(payload))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--wav", type=Path, default=Path("/tmp/voice-ref/ldc93s1.wav"))
    ap.add_argument("--seconds", type=float, default=1.0)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = root / "testdata"
    out.mkdir(exist_ok=True)
    src = Path("/tmp/chatterbox-src/src/chatterbox/models/s3gen")
    if not src.exists():
        sys.exit(f"missing upstream sources at {src}")
    hifigan = load_py("hifigan", src / "hifigan.py")
    f0m = load_py("f0_predictor", src / "f0_predictor.py")
    tiny_ops(out)

    pcm, sr = read_wav(args.wav)
    y = resample_linear(pcm, sr, 24000)
    n = int(args.seconds * 24000)
    y = y[:n]
    mel = mel_24k(torch.from_numpy(y).float()[None, :])
    print("mel", tuple(mel.shape), "wav_in", y.shape[0])

    gen = hifigan.HiFTGenerator(
        sampling_rate=24000,
        upsample_rates=[8, 5, 3],
        upsample_kernel_sizes=[16, 11, 7],
        source_resblock_kernel_sizes=[7, 7, 11],
        source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        f0_predictor=f0m.ConvRNNF0Predictor(),
    )
    from safetensors.torch import load_file

    raw = load_file(str(args.checkpoint))
    sd = {k[len("mel2wav.") :]: v for k, v in raw.items() if k.startswith("mel2wav.")}
    missing, unexpected = gen.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit(f"state dict missing={missing[:8]} unexpected={unexpected[:8]}")
    gen.eval()
    torch.manual_seed(0)
    with torch.no_grad():
        f0 = gen.f0_predictor(mel)
        s_up = gen.f0_upsamp(f0[:, None]).transpose(1, 2)
        source, _noise, _uv = gen.m_source(s_up)
        source = source.transpose(1, 2)
        wav = gen.decode(mel, source)
    wav_np = wav.squeeze(0).numpy().astype(np.float32)
    print(
        "f0",
        tuple(f0.shape),
        "source",
        tuple(source.shape),
        "wav",
        wav_np.shape,
        "peak",
        float(np.max(np.abs(wav_np))),
        "finite",
        bool(np.isfinite(wav_np).all()),
    )
    mel.numpy().astype(np.float32).tofile(out / "hift_mel.f32")
    f0.numpy().astype(np.float32).tofile(out / "hift_f0.f32")
    source.numpy().astype(np.float32).tofile(out / "hift_source.f32")
    wav_np.tofile(out / "hift_wav.f32")
    pcm16 = np.clip(wav_np, -1, 1)
    pcm16 = (pcm16 * 32767.0).astype(np.int16)
    with wave.open(str(out / "hift_audition.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(pcm16.tobytes())
    meta = {
        "sample_rate": 24000,
        "mel_frames": int(mel.shape[-1]),
        "mel_bins": 80,
        "f0_len": int(f0.shape[-1]),
        "source_len": int(source.shape[-1]),
        "wav_len": int(wav_np.shape[0]),
        "note": (
            "HiFT reconstruction of the first 1s of LDC93S1 at 24 kHz. "
            "Not a voice clone. Source excitation includes torch.manual_seed(0) "
            "phase and noise; decode(mel, source) is the deterministic check."
        ),
        "reference": "LDC93S1 DeepSpeech smoke test, first 1.0s, librosa resample to 24 kHz",
    }
    (out / "hift_golden.json").write_text(json.dumps(meta, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
