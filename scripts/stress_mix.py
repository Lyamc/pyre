"""Stress: concurrent STT + chat against a running pyre server. Usage: stress_mix.py [port] [rounds]"""
import sys, time, threading, json, urllib.request, uuid
port = sys.argv[1] if len(sys.argv) > 1 else "8099"; rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 20
B = f"http://127.0.0.1:{port}"
files = ["samples/speech.wav", "samples/speech.webm", "samples/speech.ogg", "samples/speech-vorbis.ogg"]
res = {"chat": [], "stt": [], "err": []}
def chat(i):
    t = time.time()
    body = json.dumps({"model": "qwen", "messages": [{"role": "user", "content": f"Say hi #{i}"}], "max_tokens": 8}).encode()
    r = urllib.request.urlopen(urllib.request.Request(B + "/v1/chat/completions", body, {"content-type": "application/json"}), timeout=90)
    json.load(r); res["chat"].append(time.time() - t)
def stt(i):
    f = files[i % len(files)]; bnd = uuid.uuid4().hex; data = open(f, "rb").read()
    body = (f"--{bnd}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\ntiny_en\r\n--{bnd}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{f.split('/')[-1]}\"\r\n\r\n").encode() + data + f"\r\n--{bnd}--\r\n".encode()
    t = time.time()
    r = urllib.request.urlopen(urllib.request.Request(B + "/v1/audio/transcriptions", body, {"content-type": f"multipart/form-data; boundary={bnd}"}), timeout=180)
    json.load(r); res["stt"].append(time.time() - t)
def wrap(fn, i):
    try: fn(i)
    except Exception as e: res["err"].append(f"{fn.__name__}#{i}: {e!r}")
for r in range(rounds):
    ts = [threading.Thread(target=wrap, args=(stt, r * 4 + k)) for k in range(4)] + [threading.Thread(target=wrap, args=(chat, r * 4 + k)) for k in range(4)]
    [t.start() for t in ts]; [t.join() for t in ts]
    print(f"round {r}: chat n={len(res['chat'])} max={max(res['chat'] or [0]):.1f}s stt n={len(res['stt'])} max={max(res['stt'] or [0]):.1f}s errs={len(res['err'])}", flush=True)
print("errors:", res["err"][:10])
