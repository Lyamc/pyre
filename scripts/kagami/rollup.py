# Per-book rollup of pyre tagger output (Kagami-24k) -> C:\Temp\kagami\results\<slug>.json + summary.json
#
# Additive union, the same rules as monga (crates/monga-core/src/content.rs, branch content-tags):
#  - a tag is on the book if ANY tagged page has it >= THR; the book keeps the max score and the pages that carry it;
#  - warning categories come from content_categories.txt (model tags -> category, per-category threshold, restricted/open);
#  - indexes (Purity, Explicit, Gore; formula version) from content_index.py + content_weights.txt, shared with monga;
#  - kid_friendly is derived: every page tagged AND within the kid limits (Purity >= 0.60, no sexual content, Gore <= 0.50,
#    no strong sexual violence / sexualised minors / sexual content). A partly tagged book is "unknown" until fully tagged.
# Page numbers in the output are 1-based ("page"), with the 0-based archive index alongside ("page_index").
import json, os, csv, collections, sys
import numpy as np
import content_index as ci
R = r"C:\Temp\kagami"; S = r"C:\Users\Lyam\.cache\huggingface\hub\models--Redstonexs--kagami-24k\snapshots\fbf04252c68c9cbf03c8b343e537e3cd7594c8a1"
HERE = os.path.dirname(os.path.abspath(__file__))
THR = 0.37; STORE_THR = 0.2
tags = [r["name"] for r in csv.DictReader(open(os.path.join(S, "selected_tags.csv"), encoding="utf-8"))]
index = {t: i for i, t in enumerate(tags)}

def load_categories(path=os.path.join(HERE, "content_categories.txt")):
    cats = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"): continue
        p = [x.strip() for x in line.split("|")]
        if len(p) < 5: continue
        cats.append({"id": p[0], "label": p[1], "threshold": float(p[2]), "restricted": p[3] == "restricted",
                     "tags": [t.strip() for t in p[4].split(",") if t.strip()]})
    return cats
CATS = load_categories()
missing = sorted({t for c in CATS for t in c["tags"] if t not in index})
if missing: print("WARNING: category tags not in the vocabulary:", missing, file=sys.stderr)

def union(page_probs, page_numbers, thr=THR):
    """page_probs: (n_pages, n_tags); returns [{tag, max_score, pages}] most widespread first."""
    out = []
    hit = page_probs >= thr
    for k in np.where(hit.any(0))[0]:
        pg = [int(page_numbers[i]) for i in np.where(hit[:, k])[0]]
        out.append({"tag": tags[k], "max_score": round(float(page_probs[:, k].max()), 4), "pages": pg})
    out.sort(key=lambda t: (-len(t["pages"]), -t["max_score"], t["tag"]))
    return out

def page_hits(probs):
    hits = []
    for c in CATS:
        best = max(((t, float(probs[index[t]])) for t in c["tags"] if t in index), key=lambda x: x[1], default=None)
        if best and best[1] >= c["threshold"]:
            hits.append({"category": c["id"], "tag": best[0], "score": round(best[1], 4)})
    return hits

def categories(page_probs, page_numbers):
    out = {}
    for row, pg in zip(page_probs, page_numbers):
        for h in page_hits(row):
            c = next(c for c in CATS if c["id"] == h["category"])
            v = out.setdefault(c["id"], {"id": c["id"], "label": c["label"], "restricted": c["restricted"], "max_score": 0.0, "pages": [], "tags": []})
            v["max_score"] = round(max(v["max_score"], h["score"]), 4); v["pages"].append(int(pg))
            for t in c["tags"]:
                if t in index and row[index[t]] >= c["threshold"] and t not in v["tags"]: v["tags"].append(t)
    order = [c["id"] for c in CATS]
    return sorted(out.values(), key=lambda v: order.index(v["id"]))

def rating(cats, idx, tagged, total):
    if not total or tagged < total: return "unknown"
    return "restricted" if ci.kid_over(idx, cats) else "kid_friendly"

def page_dict(probs):
    return {tags[k]: float(probs[k]) for k in np.where(probs >= STORE_THR)[0]}

if __name__ == "__main__":
    rows = [json.loads(l) for l in open(os.path.join(R, "results", "pages.jsonl"), encoding="utf-8")]
    P = np.fromfile(os.path.join(R, "manga_probs.bin"), dtype="<f4").reshape(len(rows), len(tags))
    man = {b["slug"]: b for b in json.load(open(os.path.join(R, "manifest.json"), encoding="utf-8"))}
    by = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by[os.path.basename(os.path.dirname(r["file"]))].append(i)
    summary = []
    for slug, idx in by.items():
        b = man[slug]; pages = {p["file"]: p for p in b["pages"]}
        info = [pages[os.path.basename(rows[i]["file"])] for i in idx]
        numbers = [p["index"] + 1 for p in info]
        p = P[idx]; n = len(idx)
        cats = categories(p, numbers)
        idx_ = ci.indexes([page_dict(P[i]) for i in idx])
        rt = rating(cats, idx_, n, b["total_pages"])
        indexes = {"version": idx_["version"], "purity": idx_["purity"], "explicit": idx_["explicit"], "gore": idx_["gore"],
                   "sexual_pages": [numbers[i] for i in idx_["sexual_pages"]]}
        book = {
            "schema": "pyre-tagger-book-rollup/3", "model": "Redstonexs/kagami-24k", "engine": "pyre (Burn, CUDA, f32)",
            "book": b["book"], "volume": b["volume"], "total_pages": b["total_pages"], "tagged_pages": n,
            "coverage": "complete" if n >= b["total_pages"] else f"partial {n}/{b['total_pages']}",
            "threshold": THR, "store_threshold": STORE_THR, "page_numbers": "1-based",
            "rating": rt, "kid_friendly": rt == "kid_friendly", "kid_limits_failed": ci.kid_over(idx_, cats),
            "indexes": indexes,
            "categories": cats,
            "union_tags": union(p, numbers),
            "pages": [{"page": pi["index"] + 1, "page_index": pi["index"], "file": pi["file"], "cover": pi["cover"],
                       "categories": page_hits(P[i]), "tags": rows[i]["tags"]} for i, pi in zip(idx, info)],
        }
        json.dump(book, open(os.path.join(R, "results", slug + ".json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        summary.append({"book": b["book"], "volume": b["volume"], "slug": slug, "tagged": n, "total": b["total_pages"], "rating": rt,
                        "kid_friendly": rt == "kid_friendly", "indexes": indexes,
                        "categories": {c["id"]: {"pages": c["pages"], "max": c["max_score"]} for c in cats},
                        "top": [t["tag"] for t in book["union_tags"][:12]]})
    json.dump(summary, open(os.path.join(R, "results", "summary.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    for s in summary:
        print(f"\n## {s['book']} ({s['volume']}) tagged {s['tagged']}/{s['total']} rating={s['rating']} kid_friendly={s['kid_friendly']}")
        i = s["indexes"]; print(f"  purity {i['purity']:.3f} explicit {'none' if i['explicit'] is None else format(i['explicit'], '.3f')} gore {i['gore']:.3f}")
        for k, v in s["categories"].items(): print(f"  {k}: max {v['max']} pages {v['pages']}")
        print("  top:", ", ".join(s["top"]))