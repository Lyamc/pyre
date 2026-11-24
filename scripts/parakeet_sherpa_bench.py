"""Benchmark sherpa-onnx NeMo Parakeet TDT v3 (int8) on a WAV file.

Usage: python scripts/parakeet_sherpa_bench.py samples/speech.wav [runs] [threads] [provider]
The model is downloaded from csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.
"""

import os
import statistics
import sys
import time
import wave

import numpy as np
import sherpa_onnx
from huggingface_hub import snapshot_download

MODEL_REPO = os.environ.get(
    "SHERPA_MODEL_REPO", "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
)
ENC = os.environ.get("SHERPA_ENCODER", "encoder.int8.onnx")
DEC = os.environ.get("SHERPA_DECODER", "decoder.int8.onnx")
JOIN = os.environ.get("SHERPA_JOINER", "joiner.int8.onnx")
TOKENS = os.environ.get("SHERPA_TOKENS", "tokens.txt")


def read_wav(path):
    with wave.open(path, "rb") as w:
        assert w.getnchannels() == 1, "mono expected"
        rate = w.getframerate()
        samples = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return rate, samples.astype(np.float32) / 32768.0


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "samples/speech.wav"
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    threads = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    provider = sys.argv[4] if len(sys.argv) > 4 else "cpu"
    root = os.environ.get("SHERPA_MODEL_DIR") or snapshot_download(
        MODEL_REPO, allow_patterns=["*.onnx", "*.onnx.data", TOKENS]
    )
    rec = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=f"{root}/{ENC}",
        decoder=f"{root}/{DEC}",
        joiner=f"{root}/{JOIN}",
        tokens=f"{root}/{TOKENS}",
        num_threads=threads,
        feature_dim=int(os.environ.get("SHERPA_FEATURE_DIM", "128")),
        model_type="nemo_transducer",
        provider=provider,
        decoding_method="greedy_search",
    )
    rate, audio = read_wav(path)

    def once():
        stream = rec.create_stream()
        stream.accept_waveform(rate, audio)
        rec.decode_stream(stream)
        return stream.result.text

    once()  # warm-up
    times = []
    text = ""
    for _ in range(runs):
        t = time.perf_counter()
        text = once()
        times.append(time.perf_counter() - t)
    print(f"threads={threads} provider={provider} median={statistics.median(times):.3f}s "
          f"runs={[round(t, 3) for t in times]}")
    print(f"text: {text}")


if __name__ == "__main__":
    main()
