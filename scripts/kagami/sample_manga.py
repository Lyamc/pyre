# Sample cover + ~10 spread pages per book from Z:\Manga (READ-ONLY: files opened 'rb' only, nothing written there).
import zipfile, os, sys, json, re
ROOT = r"Z:\Manga"; OUT = r"C:\Temp\kagami\pages"
BOOKS = ["Inside the Tentacle Cave", "JK Haru is a Sex Worker in Another World", "Call Girl in Another World",
 "Are You Okay with a Slightly Older Girlfriend (Uncensored)", "Parallel Paradise", "Kaifuku Jutsushi no Yarinaoshi", "Prison School",
 "Dragon Ball", "Fruits Basket Collector's Edition", "A Bride's Story", "Nodame Cantabile", "Dandadan"]
IMG = (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp")
def natkey(s): return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]
manifest = []
for b in BOOKS:
    d = os.path.join(ROOT, b)
    files = sorted([f for f in os.listdir(d) if f.lower().endswith((".cbz", ".zip", ".epub"))], key=natkey)
    if not files: print("skip (no zip-like volume)", b); continue
    vol = files[0]; src = os.path.join(d, vol)
    with open(src, "rb") as fh, zipfile.ZipFile(fh) as z:
        names = sorted([n for n in z.namelist() if n.lower().endswith(IMG) and not n.endswith("/")], key=natkey)
        if not names: print("skip (no images)", b); continue
        k = 10; rest = names[1:]
        pick = [names[0]] + [rest[int((i + 0.5) * len(rest) / k)] for i in range(min(k, len(rest)))]
        slug = re.sub(r"[^A-Za-z0-9]+", "_", b).strip("_")[:60]
        od = os.path.join(OUT, slug); os.makedirs(od, exist_ok=True)
        pages = []
        for i, n in enumerate(pick):
            ext = os.path.splitext(n)[1].lower()
            fn = f"p{i:02d}_{names.index(n):04d}{ext}"
            with open(os.path.join(od, fn), "wb") as w: w.write(z.read(n))
            pages.append({"file": fn, "index": names.index(n), "cover": i == 0, "entry": n})
        manifest.append({"book": b, "volume": vol, "slug": slug, "total_pages": len(names), "pages": pages})
        print(f"{b} | {vol} | {len(names)} pages -> {len(pages)} sampled")
json.dump(manifest, open(r"C:\Temp\kagami\manifest.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
