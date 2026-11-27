"""Benchmark sherpa-onnx Nemotron 3.5 ASR Streaming 0.6B (int8, 560 ms chunks).

Usage: python scripts/nemotron_sherpa_bench.py samples/speech.wav [runs] [threads]
"""

import os
import statistics
import sys
import time
import wave

import numpy as np
import sherpa_onnx

MODEL = os.environ.get(
    "SHERPA_NEMOTRON_DIR",
    os.path.join(os.environ.get("TEMP", ""), "nem35-sherpa"),
)


def read_wav(path):
    with wave.open(path, "rb") as w:
        rate = w.getframerate()
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return rate, raw.astype(np.float32) / 32768.0


def recognize(rec, rate, audio):
    stream = rec.create_stream()
    stream.set_option("language", os.environ.get("SHERPA_LANG", "auto"))
    stream.accept_waveform(rate, audio)
    tail = np.zeros(int(0.5 * rate), dtype=np.float32)
    stream.accept_waveform(rate, tail)
    stream.input_finished()
    while rec.is_ready(stream):
        rec.decode_stream(stream)
    return rec.get_result(stream)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "samples/speech.wav"
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    threads = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    rate, audio = read_wav(path)
    rec = sherpa_onnx.OnlineRecognizer.from_transducer(
        encoder=f"{MODEL}/encoder.int8.onnx",
        decoder=f"{MODEL}/decoder.int8.onnx",
        joiner=f"{MODEL}/joiner.int8.onnx",
        tokens=f"{MODEL}/tokens.txt",
        num_threads=threads,
        feature_dim=int(os.environ.get("SHERPA_FEATURE_DIM", "80")),
        model_type="nemo_transducer",
        provider="cpu",
    )
    recognize(rec, rate, audio[:rate])  # warm-up
    times, text = [], ""
    for _ in range(runs):
        t = time.perf_counter()
        text = recognize(rec, rate, audio)
        times.append(time.perf_counter() - t)
    print(f"threads={threads} median={statistics.median(times):.3f}s runs={[round(t,3) for t in times]}")
    print(f"text: {text}")


if __name__ == "__main__":
    main()
