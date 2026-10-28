import numpy as np, csv, os, json
S = r"C:\Users\Lyam\.cache\huggingface\hub\models--Redstonexs--kagami-24k\snapshots\fbf04252c68c9cbf03c8b343e537e3cd7594c8a1"
tags=[r["name"] for r in csv.DictReader(open(os.path.join(S,"selected_tags.csv"),encoding="utf-8"))]
ref=np.load(r"C:\Temp\kagami\ref_probs.npy"); onx=np.load(r"C:\Temp\kagami\onnx_probs.npy")
py=np.fromfile(r"C:\Temp\kagami\pyre_ref_probs.bin",dtype="<f4").reshape(-1,24000)
files=[json.loads(l)["file"] for l in open(r"C:\Temp\kagami\pyre_ref.jsonl")]
print("n", len(py), ref.shape)
out=[]
for i in range(len(py)):
    d=np.abs(py[i]-ref[i]); do=np.abs(py[i]-onx[i])
    tr=set(np.argsort(-ref[i])[:10]); tp=set(np.argsort(-py[i])[:10])
    a=set(np.where(ref[i]>=0.37)[0]); b=set(np.where(py[i]>=0.37)[0])
    print(f"{os.path.basename(files[i]):24s} max|pyre-timm| {d.max():.4f} mean {d.mean():.2e} | max|pyre-onnx| {do.max():.4f} | top10 same set: {tr==tp} | tags>=0.37 timm {len(a)} pyre {len(b)} sym-diff {sorted(tags[k] for k in a^b)}")
