import os, numpy as np, torch, timm, onnxruntime as ort
S = r"C:\Users\Lyam\.cache\huggingface\hub\models--Redstonexs--kagami-24k\snapshots\fbf04252c68c9cbf03c8b343e537e3cd7594c8a1"
x = np.load(r"C:\Temp\kagami\ref_input.npy")
ck = torch.load(os.path.join(S, "weights.pt"), map_location="cpu", weights_only=False)
torch.set_num_threads(16)
m = timm.create_model("eva02_large_patch14_448", pretrained=False, num_classes=24000, ref_feat_shape=(32, 32)); m.load_state_dict(ck["model"]); m.eval()
with torch.no_grad(): p = torch.sigmoid(m(torch.from_numpy(x))).numpy()
sess = ort.InferenceSession(os.path.join(S, "onnx", "model_prob.onnx"), providers=["CPUExecutionProvider"])
po = np.concatenate([sess.run(None, {"input": x[i:i+1]})[0] for i in range(len(x))])
d = np.abs(p - po); print("timm(ref32) vs onnx: max", float(d.max()), "mean", float(d.mean()))
np.save(r"C:\Temp\kagami\ref_probs.npy", p); np.save(r"C:\Temp\kagami\onnx_probs.npy", po)
