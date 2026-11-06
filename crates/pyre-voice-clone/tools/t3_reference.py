#!/usr/bin/env python3
"""Dev-time NumPy reference for the Chatterbox-Nano T3 forward.

Not used by pyre. Needs numpy and safetensors. Torch is not required.

    # tiny fixture already committed under testdata/
    python crates/pyre-voice-clone/tools/t3_reference.py --write-tiny

    # real checkpoint (about 870 MB). Writes logits and greedy speech ids.
    python crates/pyre-voice-clone/tools/t3_reference.py \
        --checkpoint /path/to/t3_nano_v1.safetensors \
        --out /tmp/t3_real_golden.json

The Rust test loads the same file. `--write-tiny` regenerates the committed
fixture; do that only when the math below changes on purpose.

`--torch /path/to/t3_nano_v1.safetensors` reruns the first-step logits through
Hugging Face `GPT2Model` (optional; not imported otherwise) and prints the max
absolute difference against this NumPy graph.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file


def layer_norm(x, w, b, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * w + b


def gelu_new(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def split_heads(t, n_head):
    seq, dim = t.shape
    hd = dim // n_head
    return t.reshape(seq, n_head, hd).transpose(1, 0, 2)


def merge_heads(t):
    h, seq, hd = t.shape
    return t.transpose(1, 0, 2).reshape(seq, h * hd)


def block_forward(x, layer, n_head):
    residual = x
    n = layer_norm(x, layer["ln1_w"], layer["ln1_b"])
    qkv = n @ layer["c_attn_w"] + layer["c_attn_b"]
    d = x.shape[-1]
    q, k, v = qkv[:, :d], qkv[:, d : 2 * d], qkv[:, 2 * d :]
    q, k, v = split_heads(q, n_head), split_heads(k, n_head), split_heads(v, n_head)
    hd = q.shape[-1]
    scores = np.matmul(q, np.swapaxes(k, -1, -2)) / np.sqrt(hd)
    t, s = scores.shape[-2], scores.shape[-1]
    # Full sequence, no cache: position i attends to 0..=i.
    ban = np.triu(np.ones((t, s), dtype=bool), k=1)
    scores = np.where(ban, -np.inf, scores)
    m = np.max(scores, axis=-1, keepdims=True)
    exp = np.exp(scores - m)
    probs = exp / exp.sum(axis=-1, keepdims=True)
    ctx = merge_heads(np.matmul(probs, v))
    x = residual + (ctx @ layer["c_proj_w"] + layer["c_proj_b"])
    residual = x
    h = gelu_new(layer_norm(x, layer["ln2_w"], layer["ln2_b"]) @ layer["fc_w"] + layer["fc_b"])
    return residual + (h @ layer["proj_w"] + layer["proj_b"])


def gpt_hidden(embeds, weights, n_head):
    x = embeds + weights["wpe"][: embeds.shape[0]]
    for layer in weights["layers"]:
        x = block_forward(x, layer, n_head)
    return layer_norm(x, weights["ln_f_w"], weights["ln_f_b"])


def condition(weights, speaker, prompt):
    spkr = speaker.reshape(1, -1) @ weights["spkr_w"].T + weights["spkr_b"]
    if len(prompt) == 0:
        return spkr
    return np.concatenate([spkr, weights["speech_emb"][np.asarray(prompt, dtype=np.int64)]], axis=0)


def sequence_embeds(weights, speaker, prompt, text, speech):
    parts = [condition(weights, speaker, prompt), weights["text_emb"][np.asarray(text, dtype=np.int64)]]
    if len(speech):
        parts.append(weights["speech_emb"][np.asarray(speech, dtype=np.int64)])
    return np.concatenate(parts, axis=0)


def speech_logits(weights, n_head, speaker, prompt, text, speech):
    hidden = gpt_hidden(sequence_embeds(weights, speaker, prompt, text, speech), weights, n_head)
    last = hidden[-1]
    return last @ weights["speech_head_w"].T + weights["speech_head_b"]


def process_logits(logits, input_ids, temperature, top_k, top_p, repetition_penalty):
    scores = np.array(logits, dtype=np.float64)
    if temperature > 0 and temperature != 1.0:
        scores = scores / temperature
    if top_k > 0:
        k = min(top_k, scores.shape[-1])
        kth = np.partition(scores, -k)[-k]
        scores = np.where(scores < kth, -np.inf, scores)
    if top_p < 1.0:
        order = np.argsort(scores)  # ascending, matching HF TopPLogitsWarper
        ordered = scores[order]
        # softmax
        m = np.max(ordered)
        if np.isfinite(m):
            e = np.exp(ordered - m)
            e = np.where(np.isfinite(ordered), e, 0.0)
            probs = e / e.sum()
            cum = np.cumsum(probs)
            remove = cum <= (1.0 - top_p)
            remove[-1] = False  # min_tokens_to_keep = 1
            ban = np.zeros_like(remove)
            ban[order] = remove
            scores = np.where(ban, -np.inf, scores)
    if repetition_penalty != 1.0:
        for i in set(int(j) for j in input_ids):
            s = scores[i]
            scores[i] = s * repetition_penalty if s < 0 else s / repetition_penalty
    return scores


def greedy(weights, n_head, speaker, prompt, text, start, stop, max_gen_len, decode):
    speech = [start]
    generated = []
    raw_first = None
    processed_first = None
    for step in range(1 + max_gen_len):
        raw = speech_logits(weights, n_head, speaker, prompt, text, speech)
        ids = [start] if step == 0 else generated
        processed = process_logits(raw, ids, **decode)
        if step == 0:
            raw_first = raw
            processed_first = processed
        if not np.isfinite(processed).any():
            break
        nxt = int(np.argmax(processed))
        generated.append(nxt)
        speech.append(nxt)
        if step > 0 and nxt == stop:
            break
    if generated and generated[-1] == stop:
        generated.pop()
    return raw_first, processed_first, generated


def lcg(n, seed):
    x = seed & 0xFFFFFFFF
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        x = (1664525 * x + 1013904223) & 0xFFFFFFFF
        out[i] = (x / 2**32) - 0.5
    return out


def fill(shape, seed):
    return lcg(int(np.prod(shape)), seed).reshape(shape)


def tiny_weights():
    n_embd, n_head, n_layer = 8, 2, 2
    n_pos, text_v, speech_v, speaker = 32, 16, 16, 4
    seed = 20
    layers = []
    for i in range(n_layer):
        b = 1000 + i * 100
        layers.append(
            {
                "c_attn_w": fill((n_embd, 3 * n_embd), b),
                "c_attn_b": fill((3 * n_embd,), b + 1),
                "c_proj_w": fill((n_embd, n_embd), b + 2),
                "c_proj_b": fill((n_embd,), b + 3),
                "ln1_w": np.ones(n_embd, dtype=np.float32),
                "ln1_b": np.zeros(n_embd, dtype=np.float32),
                "ln2_w": np.ones(n_embd, dtype=np.float32),
                "ln2_b": np.zeros(n_embd, dtype=np.float32),
                "fc_w": fill((n_embd, 4 * n_embd), b + 4),
                "fc_b": fill((4 * n_embd,), b + 5),
                "proj_w": fill((4 * n_embd, n_embd), b + 6),
                "proj_b": fill((n_embd,), b + 7),
            }
        )
    w = {
        "spkr_w": fill((n_embd, speaker), seed),
        "spkr_b": fill((n_embd,), seed + 1),
        "text_emb": fill((text_v, n_embd), seed + 2),
        "speech_emb": fill((speech_v, n_embd), seed + 3),
        "speech_head_w": fill((speech_v, n_embd), seed + 4),
        "speech_head_b": fill((speech_v,), seed + 5),
        "wpe": fill((n_pos, n_embd), seed + 6),
        "ln_f_w": np.ones(n_embd, dtype=np.float32),
        "ln_f_b": np.zeros(n_embd, dtype=np.float32),
        "layers": layers,
        "n_head": n_head,
        "start": 14,
        "stop": 15,
    }
    return w


def pack_safetensors(w):
    tensors = {
        "cond_enc.spkr_enc.weight": w["spkr_w"],
        "cond_enc.spkr_enc.bias": w["spkr_b"],
        "text_emb.weight": w["text_emb"],
        "speech_emb.weight": w["speech_emb"],
        "speech_head.weight": w["speech_head_w"],
        "speech_head.bias": w["speech_head_b"],
        "tfmr.wpe.weight": w["wpe"],
        "tfmr.ln_f.weight": w["ln_f_w"],
        "tfmr.ln_f.bias": w["ln_f_b"],
    }
    for i, layer in enumerate(w["layers"]):
        p = f"tfmr.h.{i}"
        tensors[f"{p}.attn.c_attn.weight"] = layer["c_attn_w"]
        tensors[f"{p}.attn.c_attn.bias"] = layer["c_attn_b"]
        tensors[f"{p}.attn.c_proj.weight"] = layer["c_proj_w"]
        tensors[f"{p}.attn.c_proj.bias"] = layer["c_proj_b"]
        tensors[f"{p}.ln_1.weight"] = layer["ln1_w"]
        tensors[f"{p}.ln_1.bias"] = layer["ln1_b"]
        tensors[f"{p}.ln_2.weight"] = layer["ln2_w"]
        tensors[f"{p}.ln_2.bias"] = layer["ln2_b"]
        tensors[f"{p}.mlp.c_fc.weight"] = layer["fc_w"]
        tensors[f"{p}.mlp.c_fc.bias"] = layer["fc_b"]
        tensors[f"{p}.mlp.c_proj.weight"] = layer["proj_w"]
        tensors[f"{p}.mlp.c_proj.bias"] = layer["proj_b"]
    return tensors


def load_checkpoint(path, n_head, start, stop):
    raw = load_file(path)
    n_layer = 0
    while f"tfmr.h.{n_layer}.attn.c_attn.weight" in raw:
        n_layer += 1
    layers = []
    for i in range(n_layer):
        p = f"tfmr.h.{i}"
        layers.append(
            {
                "c_attn_w": raw[f"{p}.attn.c_attn.weight"],
                "c_attn_b": raw[f"{p}.attn.c_attn.bias"],
                "c_proj_w": raw[f"{p}.attn.c_proj.weight"],
                "c_proj_b": raw[f"{p}.attn.c_proj.bias"],
                "ln1_w": raw[f"{p}.ln_1.weight"],
                "ln1_b": raw[f"{p}.ln_1.bias"],
                "ln2_w": raw[f"{p}.ln_2.weight"],
                "ln2_b": raw[f"{p}.ln_2.bias"],
                "fc_w": raw[f"{p}.mlp.c_fc.weight"],
                "fc_b": raw[f"{p}.mlp.c_fc.bias"],
                "proj_w": raw[f"{p}.mlp.c_proj.weight"],
                "proj_b": raw[f"{p}.mlp.c_proj.bias"],
            }
        )
    return {
        "spkr_w": raw["cond_enc.spkr_enc.weight"],
        "spkr_b": raw["cond_enc.spkr_enc.bias"],
        "text_emb": raw["text_emb.weight"],
        "speech_emb": raw["speech_emb.weight"],
        "speech_head_w": raw["speech_head.weight"],
        "speech_head_b": raw["speech_head.bias"],
        "wpe": raw["tfmr.wpe.weight"],
        "ln_f_w": raw["tfmr.ln_f.weight"],
        "ln_f_b": raw["tfmr.ln_f.bias"],
        "layers": layers,
        "n_head": n_head,
        "start": start,
        "stop": stop,
    }


def golden_case(w, speaker, prompt, text, max_gen_len, decode):
    raw, processed, tokens = greedy(
        w, w["n_head"], np.asarray(speaker, dtype=np.float32), prompt, text, w["start"], w["stop"], max_gen_len, decode
    )
    return {
        "speaker": [float(v) for v in speaker],
        "prompt_speech": prompt,
        "text": text,
        "start_speech": w["start"],
        "stop_speech": w["stop"],
        "n_head": w["n_head"],
        "max_gen_len": max_gen_len,
        "decode": decode,
        "raw_logits": [float(v) for v in raw],
        "processed_logits": [float(v) if np.isfinite(v) else None for v in processed],
        "greedy_tokens": tokens,
    }


def torch_check(path: str) -> None:
    """Optional. Imports torch only when asked."""
    import torch
    from safetensors.torch import load_file
    from transformers import GPT2Config, GPT2Model

    w = load_checkpoint(path, 12, 6561, 6562)
    rng = np.random.default_rng(0)
    speaker = rng.standard_normal(256).astype(np.float32)
    speaker = speaker / np.linalg.norm(speaker)
    prompt = [10, 20, 30, 40, 50]
    text = [15496, 11, 995, 13]
    raw, _, _ = greedy(
        w, 12, speaker, prompt, text, 6561, 6562, 0,
        {"temperature": 1.0, "top_k": 0, "top_p": 1.0, "repetition_penalty": 1.0},
    )
    tensors = load_file(path)
    cfg = GPT2Config(
        activation_function="gelu_new", n_ctx=8196, n_embd=768, n_head=12, n_layer=12,
        n_positions=8196, vocab_size=50276, layer_norm_epsilon=1e-5,
        attn_pdrop=0.0, embd_pdrop=0.0, resid_pdrop=0.0,
    )
    tfmr = GPT2Model(cfg).eval()
    sd = {k[len("tfmr."):]: v for k, v in tensors.items() if k.startswith("tfmr.")}
    tfmr.load_state_dict(sd, strict=False)
    spkr = torch.nn.functional.linear(
        torch.from_numpy(speaker).view(1, -1),
        tensors["cond_enc.spkr_enc.weight"],
        tensors["cond_enc.spkr_enc.bias"],
    )
    emb = torch.cat([
        spkr,
        tensors["speech_emb.weight"][prompt],
        tensors["text_emb.weight"][text],
        tensors["speech_emb.weight"][[6561]],
    ], dim=0).unsqueeze(0)
    with torch.no_grad():
        h = tfmr(inputs_embeds=emb).last_hidden_state[0, -1]
        logits = torch.nn.functional.linear(h, tensors["speech_head.weight"], tensors["speech_head.bias"]).numpy()
    diff = float(np.max(np.abs(logits - np.asarray(raw, dtype=np.float32))))
    print(f"max abs vs HF GPT2Model {diff:.3e} argmax torch {int(logits.argmax())} numpy {int(np.argmax(raw))}")
    if diff > 1e-4:
        raise SystemExit("HF GPT2Model disagrees with the NumPy graph")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-tiny", action="store_true")
    ap.add_argument("--checkpoint")
    ap.add_argument("--out")
    ap.add_argument("--n-head", type=int, default=12)
    ap.add_argument("--torch", metavar="CHECKPOINT", help="compare first-step logits to HF GPT2Model")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    decode = {"temperature": 0.8, "top_k": 1000, "top_p": 0.95, "repetition_penalty": 1.2}
    if args.torch:
        torch_check(args.torch)
        return
    if args.write_tiny:
        w = tiny_weights()
        path = root / "testdata" / "t3_tiny.safetensors"
        save_file(pack_safetensors(w), str(path))
        speaker = [0.1, -0.2, 0.3, -0.4]
        case = golden_case(w, speaker, [1, 4, 7], [3, 9, 2, 11], 3, decode)
        out = root / "testdata" / "t3_tiny_golden.json"
        out.write_text(json.dumps(case))
        print(f"wrote {path} and {out} tokens={case['greedy_tokens']}")
        return
    if not args.checkpoint:
        ap.error("pass --write-tiny or --checkpoint")
    w = load_checkpoint(args.checkpoint, args.n_head, 6561, 6562)
    # Fixed, short context. Not a real voice; it checks the graph.
    rng = np.random.default_rng(0)
    speaker = rng.standard_normal(256).astype(np.float32)
    speaker = speaker / np.linalg.norm(speaker)
    prompt = [10, 20, 30, 40, 50]
    text = [15496, 11, 995, 13]  # "Hello, world." from the tokenizer golden
    case = golden_case(w, speaker.tolist(), prompt, text, 2, decode)
    text_out = json.dumps(case)
    if args.out:
        Path(args.out).write_text(text_out)
        print(f"wrote {args.out} tokens={case['greedy_tokens']} logits={len(case['raw_logits'])}")
    else:
        print(text_out[:200], "...", case["greedy_tokens"])


if __name__ == "__main__":
    main()
