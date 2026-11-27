"""Tiny word-error-rate helper for STT comparisons (no external deps).

Usage: python scripts/stt_wer.py "reference text" "hypothesis text"
Normalizes case/punctuation and prints edits, ref words and WER.
"""

import re
import sys


def words(text: str):
    return re.findall(r"[a-z0-9]+", text.lower())


def wer(ref, hyp):
    r, h = words(ref), words(hyp)
    if not r:
        return 0.0, 0, 0
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i]
        for j, hw in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw)))
        prev = cur
    return prev[-1] / len(r), prev[-1], len(r)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    value, edits, total = wer(sys.argv[1], sys.argv[2])
    print(f"WER {value:.3%} ({edits}/{total})")
