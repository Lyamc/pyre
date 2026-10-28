"""Tiny random-weight Qwen3.8-Flash-Next (transformers `qwen4_exp`) for verifying pyre's port.
make:  python qwen_next_ref.py make OUTDIR      -> config.json + model.safetensors (f32) + ref.json
check: python qwen_next_ref.py check OUTDIR pyre_dump.json
"""
import json, sys, os, torch
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM

IDS = [5, 17, 99, 3, 250, 7, 7, 42, 1, 300, 12, 64, 64, 64, 8, 77, 200, 31, 2, 9]

def make(out):
    torch.manual_seed(0)
    cfg = Qwen4ExpTextConfig(
        hidden_size=64, num_hidden_layers=8, layer_types=None, full_attention_interval=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32, partial_rotary_factor=0.25,
        linear_num_key_heads=2, linear_num_value_heads=6, linear_key_head_dim=16, linear_value_head_dim=16,
        num_experts=16, num_experts_per_tok=4, moe_intermediate_size=32, shared_expert_intermediate_size=32,
        hc_count=4, hc_lowrank=16, vocab_size=320, eos_token_id=2, bos_token_id=2, pad_token_id=None,
        ngram_size=3, heads_per_ngram=2, ngram_vocab_size_base=1000, make_ngram_vocab_size_divisible_by=8,
        ple_embed_dim=32, ple_conv_kernel_size=4, ple_layer_ids=[2, 6], output_gate_type="sigmoid",
        indexer_budget=2048, indexer_compress_ratio=4, indexer_head_dim=16, indexer_n_heads=2, indexer_kv_heads=1,
        max_position_embeddings=4096, rope_parameters={"rope_type": "default", "rope_theta": 10000000.0,
        "partial_rotary_factor": 0.25, "mrope_section": [2, 1, 1], "mrope_interleaved": True},
        tie_word_embeddings=False,
    )
    cfg._attn_implementation = "eager"
    m = Qwen4ExpForCausalLM(cfg).float().eval()
    with torch.no_grad():  # norms init at zero / ones: perturb everything so each weight matters
        for n, p in m.named_parameters():
            p.add_(torch.randn_like(p) * (0.2 if p.ndim == 1 else 0.02))
            if n.endswith("A_log"): p.copy_(torch.log(torch.empty_like(p).uniform_(0.5, 4)))
    m.save_pretrained(out, safe_serialization=True)
    d = json.load(open(os.path.join(out, "config.json")))
    with torch.no_grad():
        logits = m(torch.tensor([IDS])).logits[0]
        g = m.generate(torch.tensor([IDS]), max_new_tokens=8, do_sample=False)[0, len(IDS):].tolist()
    json.dump({"ids": IDS, "logits": logits.tolist(), "greedy": g}, open(os.path.join(out, "ref.json"), "w"))
    print("saved", out, "greedy", g, "keys", d.get("model_type"))

def check(out, dump):
    r = json.load(open(os.path.join(out, "ref.json"))); p = json.load(open(dump))
    a, b = torch.tensor(r["logits"]), torch.tensor(p["logits"])
    print("shape", tuple(a.shape), tuple(b.shape))
    d = (a - b).abs()
    print("max abs diff %.3e  mean %.3e  ref absmax %.3f" % (d.max(), d.mean(), a.abs().max()))
    print("argmax agree %d/%d" % ((a.argmax(-1) == b.argmax(-1)).sum(), a.shape[0]))
    print("greedy ref", r["greedy"], "pyre", p["greedy"][:len(r["greedy"])])

if __name__ == "__main__":
    make(sys.argv[2]) if sys.argv[1] == "make" else check(sys.argv[2], sys.argv[3])
