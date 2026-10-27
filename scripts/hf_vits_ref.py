"""HF reference for VITS with noise off; compares to a pyre wav. Usage: hf_vits_ref.py "<text>" pyre.wav"""
import sys, torch, numpy as np, soundfile as sf
from transformers import VitsModel, AutoTokenizer
tok = AutoTokenizer.from_pretrained("facebook/mms-tts-eng"); m = VitsModel.from_pretrained("facebook/mms-tts-eng").eval()
m.noise_scale = 0.0; m.noise_scale_duration = 0.0
ids = tok(sys.argv[1], return_tensors="pt")
print("hf ids", ids.input_ids[0, :12].tolist(), ids.input_ids.shape[1])
with torch.no_grad(): w = m(**ids).waveform[0].numpy()
sf.write("samples/tts-hf-ref.wav", w, 16000)
p, _ = sf.read(sys.argv[2]); n = min(len(p), len(w))
print(f"hf {len(w)} samples, pyre {len(p)}; corr {np.corrcoef(w[:n], p[:n])[0,1]:.4f}; max|d| {np.abs(w[:n]-p[:n]).max():.4f}")
