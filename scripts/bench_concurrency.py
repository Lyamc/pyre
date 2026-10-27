"""Aggregate decode throughput at N concurrent requests. Usage: bench_concurrency.py port N[,N..] [max_tokens]"""
import sys, time, json, threading, urllib.request
port, ns = sys.argv[1], [int(x) for x in sys.argv[2].split(",")]; mt = int(sys.argv[3]) if len(sys.argv) > 3 else 128
prompts = ["Write a short story about a lighthouse.", "Explain how a CPU cache works.", "List ten fruits and describe each.", "Describe the water cycle in detail.",
           "Write a poem about autumn leaves.", "Explain TCP vs UDP.", "Give a recipe for pancakes.", "Summarize the history of Rome."]
def one(i, out):
    body = json.dumps({"model": "qwen", "messages": [{"role": "user", "content": prompts[i % 8]}], "max_tokens": mt, "temperature": 0, "ignore_eos": True}).encode()
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body, {"content-type": "application/json"}), timeout=600))
    out[i] = (r["usage"]["completion_tokens"], r["choices"][0]["message"]["content"])
ref = {}
for n in ns:
    out = {}; t = time.time()
    ts = [threading.Thread(target=one, args=(i, out)) for i in range(n)]; [x.start() for x in ts]; [x.join() for x in ts]
    dt = time.time() - t; tot = sum(v[0] for v in out.values())
    same = all(ref.get(i, out[i][1]) == out[i][1] for i in out)
    for i in out: ref.setdefault(i, out[i][1])
    print(f"N={n}: {tot} tokens in {dt:.2f}s = {tot/dt:.1f} tok/s aggregate ({tot/dt/n:.1f}/stream); outputs match N=1/first run: {same}", flush=True)
