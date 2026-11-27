"""Benchmark Qwen3-ASR-0.6B with transcribe.cpp (CUDA/Vulkan/CPU).

Usage: python scripts/qwen_tcpp_bench.py samples/speech.wav [runs] [cuda|vulkan|cpu]
Model: handy-computer/Qwen3-ASR-0.6B-gguf Q8_0 from the HF cache.
"""

import os
import statistics
import subprocess
import sys
import tempfile
import time
import wave

import numpy as np
import transcribe_cpp


def read_16k(path):
    with wave.open(path, "rb") as w:
        rate = w.getframerate()
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    if rate == 16000:
        return raw.astype(np.float32) / 32768.0
    out = os.path.join(tempfile.gettempdir(), "qwen_bench_16k.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-ar", "16000", "-ac", "1", out],
        check=True,
    )
    with wave.open(out, "rb") as w:
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return raw.astype(np.float32) / 32768.0


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "samples/speech.wav"
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    kind = sys.argv[3] if len(sys.argv) > 3 else "cuda"
    model_path = os.environ.get("QWEN_GGUF")
    if not model_path:
        import glob
        model_path = glob.glob(
            os.path.expanduser(
                "~/.cache/huggingface/hub/models--handy-computer--Qwen3-ASR-0.6B-gguf/snapshots/*/Qwen3-ASR-0.6B-Q8_0.gguf"
            )
        )[0]
    device = next(d for d in transcribe_cpp.backends() if d.kind == kind)
    audio = read_16k(path)
    with transcribe_cpp.Model(model_path, device=device) as model:
        with model.session() as session:
            session.run(audio[:16000])  # warm-up
            times, text = [], ""
            for _ in range(runs):
                t = time.perf_counter()
                result = session.run(audio)
                times.append(time.perf_counter() - t)
                text = result.text
    print(f"backend={kind} median={statistics.median(times):.3f}s runs={[round(t,3) for t in times]}")
    print(f"text: {text}")


if __name__ == "__main__":
    main()
