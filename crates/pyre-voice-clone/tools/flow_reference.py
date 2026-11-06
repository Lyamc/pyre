#!/usr/bin/env python3
"""Dev-only mean-flow reference. Not used by the server.

Rebuilds CausalMaskedDiffWithXvec (meanflow=True) with torch.nn and the
Nano checkpoint. Two Euler steps, no CFG: distilled mean-flow does not apply
inference_cfg_rate. See chatterbox s3gen/flow_matching.py basic_euler.

Writes a short fixed-input golden (tokens, prompt mel, speaker, noise, mu,
mel) so the Rust port can be checked stage by stage.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_q = nn.Linear(512, 512)
        self.linear_k = nn.Linear(512, 512)
        self.linear_v = nn.Linear(512, 512)
        self.linear_out = nn.Linear(512, 512)
        self.linear_pos = nn.Linear(512, 512, bias=False)
        self.pos_bias_u = nn.Parameter(torch.empty(8, 64))
        self.pos_bias_v = nn.Parameter(torch.empty(8, 64))

    def rel_shift(self, x):
        zero = x.new_zeros(x.size(0), x.size(1), x.size(2), 1)
        x_padded = torch.cat([zero, x], dim=-1)
        x_padded = x_padded.view(x.size(0), x.size(1), x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)[:, :, :, : x.size(-1) // 2 + 1]
        return x

    def forward(self, x, pos_emb):
        b, t, _ = x.shape
        q = self.linear_q(x).view(b, t, 8, 64).transpose(1, 2)
        k = self.linear_k(x).view(b, t, 8, 64).transpose(1, 2)
        v = self.linear_v(x).view(b, t, 8, 64).transpose(1, 2)
        p = self.linear_pos(pos_emb).view(pos_emb.size(0), -1, 8, 64).transpose(1, 2)
        qu = (q.transpose(1, 2) + self.pos_bias_u).transpose(1, 2)
        qv = (q.transpose(1, 2) + self.pos_bias_v).transpose(1, 2)
        ac = torch.matmul(qu, k.transpose(-2, -1))
        bd = torch.matmul(qv, p.transpose(-2, -1))
        if ac.shape != bd.shape:
            bd = self.rel_shift(bd)
        scores = (ac + bd) / math.sqrt(64)
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, t, 512)
        return self.linear_out(ctx)


class ConfLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = RelAttn()
        self.feed_forward = nn.Module()
        self.feed_forward.w_1 = nn.Linear(512, 2048)
        self.feed_forward.w_2 = nn.Linear(2048, 512)
        self.norm_mha = nn.LayerNorm(512, eps=1e-12)
        self.norm_ff = nn.LayerNorm(512, eps=1e-12)

    def forward(self, x, pos_emb):
        h = self.norm_mha(x)
        x = x + self.self_attn(h, pos_emb)
        h = self.norm_ff(x)
        # Swish is SiLU. Eval dropout is the identity.
        return x + self.feed_forward.w_2(F.silu(self.feed_forward.w_1(h)))


def espnet_pos(length: int, d_model: int = 512) -> torch.Tensor:
    pe_positive = torch.zeros(length, d_model)
    pe_negative = torch.zeros(length, d_model)
    position = torch.arange(0, length, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model))
    pe_positive[:, 0::2] = torch.sin(position * div)
    pe_positive[:, 1::2] = torch.cos(position * div)
    pe_negative[:, 0::2] = torch.sin(-position * div)
    pe_negative[:, 1::2] = torch.cos(-position * div)
    pe = torch.cat([torch.flip(pe_positive, [0]).unsqueeze(0), pe_negative[1:].unsqueeze(0)], dim=1)
    return pe[:, pe.size(1) // 2 - length + 1 : pe.size(1) // 2 + length]


class Embed(nn.Module):
    def __init__(self):
        super().__init__()
        self.out = nn.Sequential(nn.Linear(512, 512), nn.LayerNorm(512, eps=1e-5))

    def forward(self, x):
        x = self.out(x) * math.sqrt(512)
        return x, espnet_pos(x.size(1))


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = Embed()
        self.pre_lookahead_layer = nn.Module()
        self.pre_lookahead_layer.conv1 = nn.Conv1d(512, 512, 4)
        self.pre_lookahead_layer.conv2 = nn.Conv1d(512, 512, 3)
        self.encoders = nn.ModuleList([ConfLayer() for _ in range(6)])
        self.up_layer = nn.Module()
        self.up_layer.conv = nn.Conv1d(512, 512, 5)
        self.up_embed = Embed()
        self.up_encoders = nn.ModuleList([ConfLayer() for _ in range(4)])
        self.after_norm = nn.LayerNorm(512, eps=1e-5)

    def lookahead(self, x):
        h = x.transpose(1, 2)
        h = F.pad(h, (0, 3))
        h = F.leaky_relu(self.pre_lookahead_layer.conv1(h))
        h = F.pad(h, (2, 0))
        h = self.pre_lookahead_layer.conv2(h)
        return h.transpose(1, 2) + x

    def upsample(self, x):
        h = x.transpose(1, 2)
        h = F.interpolate(h, scale_factor=2.0, mode="nearest")
        h = F.pad(h, (4, 0))
        h = self.up_layer.conv(h)
        return h.transpose(1, 2)

    def forward(self, x):
        x, pos = self.embed(x)
        x = self.lookahead(x)
        for layer in self.encoders:
            x = layer(x, pos)
        x = self.upsample(x)
        x, pos = self.up_embed(x)
        for layer in self.up_encoders:
            x = layer(x, pos)
        return self.after_norm(x)


class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(256, 512, bias=False)
        self.to_k = nn.Linear(256, 512, bias=False)
        self.to_v = nn.Linear(256, 512, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(512, 256)])

    def forward(self, x):
        b, t, _ = x.shape
        q = self.to_q(x).view(b, t, 8, 64).permute(0, 2, 1, 3)
        k = self.to_k(x).view(b, t, 8, 64).permute(0, 2, 1, 3)
        v = self.to_v(x).view(b, t, 8, 64).permute(0, 2, 1, 3)
        scale = 64 ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, v).permute(0, 2, 1, 3).contiguous().view(b, t, 512)
        return self.to_out[0](ctx)


class GEGLUish(nn.Module):
    """diffusers GELU: linear then erf gelu. Named proj to match the checkpoint."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(256, 1024)

    def forward(self, x):
        return F.gelu(self.proj(x))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(256, eps=1e-5)
        self.attn1 = Attn()
        self.norm3 = nn.LayerNorm(256, eps=1e-5)
        self.ff = nn.Module()
        self.ff.net = nn.ModuleList([GEGLUish(), nn.Identity(), nn.Linear(1024, 256)])

    def forward(self, x):
        x = x + self.attn1(self.norm1(x))
        h = self.ff.net[2](self.ff.net[0](self.norm3(x)))
        return x + h


class CausalConv(nn.Conv1d):
    def __init__(self, cin, cout, k):
        super().__init__(cin, cout, k, padding=0)
        self.left = k - 1

    def forward(self, x):
        return super().forward(F.pad(x, (self.left, 0)))


class ResBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.block1 = nn.Module()
        self.block1.block = nn.Sequential(
            CausalConv(cin, cout, 3), nn.Identity(), nn.LayerNorm(cout, eps=1e-5), nn.Identity(), nn.Mish()
        )
        self.block2 = nn.Module()
        self.block2.block = nn.Sequential(
            CausalConv(cout, cout, 3), nn.Identity(), nn.LayerNorm(cout, eps=1e-5), nn.Identity(), nn.Mish()
        )
        self.mlp = nn.Sequential(nn.Mish(), nn.Linear(1024, cout))
        self.res_conv = nn.Conv1d(cin, cout, 1)

    def _block(self, block, x):
        h = block[0](x)
        h = h.transpose(1, 2)
        h = block[2](h)
        h = h.transpose(1, 2)
        return block[4](h)

    def forward(self, x, temb):
        h = self._block(self.block1.block, x)
        h = h + self.mlp(temb).unsqueeze(-1)
        h = self._block(self.block2.block, h)
        return h + self.res_conv(x)


class Estimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_mlp = nn.Module()
        self.time_mlp.linear_1 = nn.Linear(320, 1024)
        self.time_mlp.linear_2 = nn.Linear(1024, 1024)
        self.time_embed_mixer = nn.Linear(2048, 1024, bias=False)
        self.down_blocks = nn.ModuleList(
            [nn.ModuleList([ResBlock(320, 256), nn.ModuleList([Block() for _ in range(4)]), CausalConv(256, 256, 3)])]
        )
        self.mid_blocks = nn.ModuleList(
            [nn.ModuleList([ResBlock(256, 256), nn.ModuleList([Block() for _ in range(4)])]) for _ in range(12)]
        )
        self.up_blocks = nn.ModuleList(
            [nn.ModuleList([ResBlock(512, 256), nn.ModuleList([Block() for _ in range(4)]), CausalConv(256, 256, 3)])]
        )
        self.final_block = nn.Module()
        self.final_block.block = nn.Sequential(
            CausalConv(256, 256, 3), nn.Identity(), nn.LayerNorm(256, eps=1e-5), nn.Identity(), nn.Mish()
        )
        self.final_proj = nn.Conv1d(256, 80, 1)

    def time(self, t, r):
        def one(v):
            half = 160
            freq = math.log(10000) / (half - 1)
            emb = torch.exp(torch.arange(half) * -freq)
            ang = 1000.0 * v * emb
            e = torch.cat([ang.sin(), ang.cos()], dim=-1)
            e = F.silu(self.time_mlp.linear_1(e))
            return self.time_mlp.linear_2(e)

        return self.time_embed_mixer(torch.cat([one(t), one(r)], dim=-1))

    def transformers(self, x, blocks):
        h = x.transpose(1, 2)
        for block in blocks:
            h = block(h)
        return h.transpose(1, 2)

    def forward(self, x, mu, spk, cond, t, r):
        temb = self.time(t.view(1), r.view(1))
        spk = spk.view(1, -1, 1).expand(-1, -1, x.size(-1))
        h = torch.cat([x, mu, spk, cond], dim=1)
        skips = []
        res, tr, down = self.down_blocks[0]
        h = res(h, temb)
        h = self.transformers(h, tr)
        skips.append(h)
        h = down(h)
        for res, tr in self.mid_blocks:
            h = res(h, temb)
            h = self.transformers(h, tr)
        res, tr, up = self.up_blocks[0]
        skip = skips.pop()
        h = torch.cat([h[:, :, : skip.size(-1)], skip], dim=1)
        h = res(h, temb)
        h = self.transformers(h, tr)
        h = up(h)
        h = self.final_block.block[0](h)
        h = self.final_block.block[2](h.transpose(1, 2)).transpose(1, 2)
        h = self.final_block.block[4](h)
        return self.final_proj(h)


class Flow(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_embedding = nn.Embedding(6561, 512)
        self.spk_embed_affine_layer = nn.Linear(192, 80)
        self.encoder = Encoder()
        self.encoder_proj = nn.Linear(512, 80)
        self.decoder = nn.Module()
        self.decoder.estimator = Estimator()


def load_flow(path: Path) -> Flow:
    from safetensors.torch import load_file

    raw = load_file(str(path))
    sd = {k[len("flow.") :]: v for k, v in raw.items() if k.startswith("flow.")}
    flow = Flow()
    missing, unexpected = flow.load_state_dict(sd, strict=False)
    bad = [k for k in missing if "num_batches" not in k]
    if bad or unexpected:
        raise SystemExit(f"missing {bad[:12]} unexpected {unexpected[:12]} nmiss {len(bad)} nunexp {len(unexpected)}")
    flow.eval()
    return flow


def run(flow: Flow, tokens, prompt_mel, speaker, noise_prompt, noise_gen):
    """tokens [n], prompt_mel [frames, 80], speaker [192], noises [80, frames]."""
    with torch.no_grad():
        spk = F.normalize(speaker.view(1, -1), dim=1)
        spk = flow.spk_embed_affine_layer(spk)
        emb = flow.input_embedding(tokens.view(1, -1))
        h = flow.encoder(emb)
        h = flow.encoder_proj(h)
        mu = h.transpose(1, 2)
        cond = torch.zeros_like(mu)
        cond[:, :, : prompt_mel.size(0)] = prompt_mel.transpose(0, 1)
        z = noise_prompt.clone()
        z[:, :, prompt_mel.size(0) :] = noise_gen
        t_span = torch.linspace(0, 1, 3)
        x = z
        dx0 = None
        for i, (t, r) in enumerate(zip(t_span[:-1], t_span[1:])):
            dx = flow.decoder.estimator(x, mu, spk, cond, t, r)
            if i == 0:
                dx0 = dx
            x = x + (r - t) * dx
        mel = x[:, :, prompt_mel.size(0) :]
    return mu, dx0, mel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1] / "testdata"
    flow = load_flow(args.checkpoint)
    g = torch.Generator().manual_seed(0)
    tokens = torch.tensor([10, 20, 30, 40, 50, 60], dtype=torch.long)
    prompt_mel = torch.randn(8, 80, generator=g)
    speaker = torch.randn(192, generator=g)
    noise_prompt = torch.randn(1, 80, 12, generator=g)
    noise_gen = torch.randn(1, 80, 4, generator=g)
    mu, dx0, mel = run(flow, tokens, prompt_mel, speaker, noise_prompt, noise_gen)
    print("mu", tuple(mu.shape), "dx", tuple(dx0.shape), "mel", tuple(mel.shape), "finite", bool(mel.isfinite().all()))
    def dump(name, t):
        if isinstance(t, torch.Tensor):
            t = t.detach().float().cpu().numpy()
        t.tofile(root / name)
    tokens.numpy().astype("int32").tofile(root / "flow_tokens.i32")
    dump("flow_prompt_mel.f32", prompt_mel)
    dump("flow_speaker.f32", speaker)
    dump("flow_noise_prompt.f32", noise_prompt)
    dump("flow_noise_gen.f32", noise_gen)
    dump("flow_mu.f32", mu)
    dump("flow_dx0.f32", dx0)
    dump("flow_mel.f32", mel)
    (root / "flow_golden.json").write_text(json.dumps({
        "tokens": 6,
        "prompt_mel_frames": 8,
        "mel_bins": 80,
        "gen_frames": 4,
        "note": "Fixed generator seed 0. Not a voice. Two Euler steps, no CFG. mu is [1,80,12] before the prompt crop.",
    }, indent=2))
    print("wrote", root)


if __name__ == "__main__":
    main()
