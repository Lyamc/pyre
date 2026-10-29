# pyre: fetch only the MTP tensors of Qwen/Qwen3.8-Flash-Next (pinned revision) with HTTP range reads and write them as
# one safetensors file in the HF cache. Approach and pinned revision/SHA256 list from Strata's tools/mtp_fetch.py (MIT).
import json, struct, sys, os, hashlib, re, urllib.request, concurrent.futures
REV = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
BASE = "https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/%s/" % REV
OUT = os.path.expanduser(r"~\.cache\huggingface\hub\models--Qwen--Qwen3.8-Flash-Next\snapshots\%s" % REV)
src = open(r"C:\Build\ai\strata-src\tools\mtp_fetch.py", encoding="utf-8").read()
SHA = dict(re.findall(r'"(mtp\.[^"]+)":\s*\n?\s*"([0-9a-f]{64})"', src))
def get(url, a, b, tries=6):
    for t in range(tries):
        try:
            r = urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=%d-%d" % (a, b), "User-Agent": "pyre-mtp"}), timeout=120)
            if r.status != 206 or not (r.headers.get("Content-Range") or "").startswith("bytes %d-%d/" % (a, b)): raise IOError("range not honoured")
            d = r.read()
            if len(d) != b - a + 1: raise IOError("short")
            return d
        except Exception as e:
            if t == tries - 1: raise
            print("retry", e, file=sys.stderr)
idx = json.loads(urllib.request.urlopen(urllib.request.Request(BASE + "model.safetensors.index.json", headers={"User-Agent": "pyre-mtp"}), timeout=120).read())
names = sorted(k for k in idx["weight_map"] if k.startswith("mtp."))
print(len(names), "mtp tensors;", len(SHA), "pinned hashes")
shards = sorted(set(idx["weight_map"][k] for k in names))
meta = {}
for sh in shards:
    n = struct.unpack("<Q", get(BASE + sh, 0, 7))[0]; h = json.loads(get(BASE + sh, 8, 8 + n - 1))
    for k in names:
        if idx["weight_map"][k] == sh: meta[k] = (sh, 8 + n, h[k])
total = sum(m[2]["data_offsets"][1] - m[2]["data_offsets"][0] for m in meta.values()); print("total %.2f GB" % (total / 1e9))
if "--plan" in sys.argv: sys.exit(0)
os.makedirs(OUT, exist_ok=True)
hdr = {}; off = 0
for k in names:
    s, e = meta[k][2]["data_offsets"]; hdr[k] = {"dtype": meta[k][2]["dtype"], "shape": meta[k][2]["shape"], "data_offsets": [off, off + e - s]}; off += e - s
hdr["__metadata__"] = {"source": "Qwen/Qwen3.8-Flash-Next@" + REV + " mtp.* tensors only (range reads)"}
hb = json.dumps(hdr).encode(); hb += b" " * ((8 - len(hb) % 8) % 8)
tmp = os.path.join(OUT, "mtp.safetensors.part")
CH = 32 << 20
with open(tmp, "wb") as f, concurrent.futures.ThreadPoolExecutor(16) as ex:
    f.write(struct.pack("<Q", len(hb))); f.write(hb); done = 0
    for k in names:
        sh, dbase, m = meta[k]; s, e = m["data_offsets"]; a0 = dbase + s; last = dbase + e - 1
        spans = [(p, min(p + CH - 1, last)) for p in range(a0, last + 1, CH)]
        hsh = hashlib.sha256()
        for d in ex.map(lambda sp: get(BASE + sh, sp[0], sp[1]), spans):
            f.write(d); hsh.update(d); done += len(d)
        ok = SHA.get(k) == hsh.hexdigest()
        print("%-70s %s %.2f/%.2f GB" % (k, "ok" if ok else ("NOHASH" if k not in SHA else "BAD"), done / 1e9, total / 1e9), flush=True)
        if k in SHA and not ok: sys.exit("sha mismatch " + k)
os.replace(tmp, os.path.join(OUT, "mtp.safetensors")); print("wrote", os.path.join(OUT, "mtp.safetensors"))
