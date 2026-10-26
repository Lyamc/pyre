from openai import OpenAI
import time
c = OpenAI(base_url="http://127.0.0.1:8099/v1", api_key="sk-test")
print("models:", [(m.id, getattr(m, "kind", None)) for m in c.models.list().data])
r = c.chat.completions.create(model="qwen", messages=[{"role":"user","content":"Name three primary colors, comma separated."}], max_tokens=30, temperature=0)
print("chat:", repr(r.choices[0].message.content), r.usage)
s = c.chat.completions.create(model="qwen", messages=[{"role":"user","content":"Count to five."}], max_tokens=30, stream=True, stream_options={"include_usage": True})
parts=[]; u=None
for ch in s:
    if ch.choices and ch.choices[0].delta.content: parts.append(ch.choices[0].delta.content)
    if ch.usage: u=ch.usage
print("stream:", repr("".join(parts)), u)
print("completion:", repr(c.completions.create(model="qwen", prompt="The capital of France is", max_tokens=8, temperature=0).choices[0].text))
e = c.embeddings.create(model="qwen", input=["hello", "world"])
print("embeddings:", len(e.data), len(e.data[0].embedding))
t=time.time()
tr = c.audio.transcriptions.create(model="whisper-1", file=open("samples/speech.wav","rb"))
print("transcription:", repr(tr.text), f"{time.time()-t:.2f}s")
v = c.audio.transcriptions.create(model="tiny_en", file=open("samples/speech.wav","rb"), response_format="verbose_json")
print("verbose segments:", len(v.segments), v.duration)
print("srt:", c.audio.transcriptions.create(model="tiny_en", file=open("samples/speech.wav","rb"), response_format="srt")[:60].replace("\n"," | "))
try:
    c.images.generate(model="x", prompt="a cat")
except Exception as ex: print("images:", type(ex).__name__, str(ex)[:120])
