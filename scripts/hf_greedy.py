"""HF reference: greedy ids for the chat-templated prompt. Usage: hf_greedy.py <repo> [n]"""
import sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
repo = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
tok = AutoTokenizer.from_pretrained(repo)
m = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float32, attn_implementation="eager").eval()
text = tok.apply_chat_template([{"role": "user", "content": "Write a haiku about rust programming."}], tokenize=False, add_generation_prompt=True)
ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
print("prompt ids", ids.shape[1], ids[0, :8].tolist())
with torch.no_grad():
    out = m.generate(ids, max_new_tokens=n, do_sample=False)
print("hf greedy ids:", out[0, ids.shape[1]:].tolist())
print(repr(tok.decode(out[0, ids.shape[1]:])))
