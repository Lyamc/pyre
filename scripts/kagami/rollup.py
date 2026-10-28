# Per-book rollup of pyre tagger output (Kagami-24k) -> C:\Temp\kagami\results\<slug>.json + summary.json/.md
import json, os, csv, numpy as np, collections
R = r"C:\Temp\kagami"; S = r"C:\Users\Lyam\.cache\huggingface\hub\models--Redstonexs--kagami-24k\snapshots\fbf04252c68c9cbf03c8b343e537e3cd7594c8a1"
THR = 0.37; RTHR = 0.5
tags = [r["name"] for r in csv.DictReader(open(os.path.join(S, "selected_tags.csv"), encoding="utf-8"))]
rows = [json.loads(l) for l in open(os.path.join(R, "results", "pages.jsonl"), encoding="utf-8")]
P = np.fromfile(os.path.join(R, "manga_probs.bin"), dtype="<f4").reshape(len(rows), len(tags))
man = {b["slug"]: b for b in json.load(open(os.path.join(R, "manifest.json"), encoding="utf-8"))}
CENSOR = ["censored", "uncensored", "mosaic_censoring", "bar_censor", "convenient_censoring", "heart_censor", "novelty_censor", "blur_censor", "light_censor", "steam_censor", "censored_nipples", "pointless_censoring"]
FORMAT = ["monochrome", "greyscale", "comic", "speech_bubble", "english_text", "4koma", "cover", "cover_page", "manga_cover", "halftone", "screentones", "text_focus"]
CONTENT = ["nude", "completely_nude", "nipples", "sex", "penis", "pussy", "hetero", "yaoi", "yuri", "loli", "shota", "child", "blood", "guro", "injury"]
ORDER = ["general", "sensitive", "questionable", "explicit"]
by = collections.defaultdict(list)
for i, r in enumerate(rows):
    slug = os.path.basename(os.path.dirname(r["file"])); by[slug].append(i)
summary = []
for slug, idx in by.items():
    b = man[slug]; pages = {p["file"]: p for p in b["pages"]}
    p = P[idx]; n = len(idx)
    mx, mean, frac = p.max(0), p.mean(0), (p >= THR).mean(0)
    def stat(t):
        k = tags.index(t); return {"max": round(float(mx[k]), 4), "mean": round(float(mean[k]), 4), "frac_pages": round(float(frac[k]), 3)}
    labels = [rows[i]["rating"]["label"] for i in idx]
    rs = {lvl: [rows[i]["rating"]["scores"][lvl] for i in idx] for lvl in ORDER}
    cnt = collections.Counter(labels)
    worst = max(labels, key=ORDER.index)
    # weighted: share of pages at or above each level
    share = {lvl: round(sum(ORDER.index(l) >= ORDER.index(lvl) for l in labels) / n, 3) for lvl in ORDER}
    noncover = [rows[i]["rating"]["label"] for i in idx if not pages[os.path.basename(rows[i]["file"])]["cover"]]
    cover_i = [i for i in idx if pages[os.path.basename(rows[i]["file"])]["cover"]]
    top_general = sorted([k for k in np.where(mx >= THR)[0]], key=lambda k: (-frac[k], -mx[k]))
    book = {
        "schema": "pyre-tagger-book-rollup/1", "model": "Redstonexs/kagami-24k", "engine": "pyre (Burn, CUDA, f32)",
        "book": b["book"], "volume": b["volume"], "total_pages": b["total_pages"], "sampled_pages": n, "threshold": THR, "rating_threshold": RTHR,
        "rating": {
            "source": "derived (Kagami-24k has no rating tags; max over explicit/questionable/sensitive evidence tags)",
            "worst_page": worst, "page_counts": dict(cnt), "share_at_or_above": share,
            "max_scores": {lvl: round(max(v), 4) for lvl, v in rs.items()}, "mean_scores": {lvl: round(float(np.mean(v)), 4) for lvl, v in rs.items()},
            "cover": rows[cover_i[0]]["rating"]["label"] if cover_i else None,
        },
        # Suggested policy for a reader (e.g. monga): hide from kid profiles unless every sampled page is general/sensitive
        # with questionable < 0.5; treat a book as adult if >= 2 sampled pages (or >= 15%) are explicit.
        "filter": {
            "kid_safe": worst in ("general", "sensitive"),
            "adult": cnt.get("explicit", 0) >= 2 or share["explicit"] >= 0.15,
            "needs_review": worst in ("questionable", "explicit") and not (cnt.get("explicit", 0) >= 2 or share["explicit"] >= 0.15),
        },
        "censorship": {t: stat(t) for t in CENSOR if t in tags},
        "format": {t: stat(t) for t in FORMAT if t in tags},
        "content": {t: stat(t) for t in CONTENT if t in tags},
        "top_tags": [{"tag": tags[k], "max": round(float(mx[k]), 4), "mean": round(float(mean[k]), 4), "frac_pages": round(float(frac[k]), 3)} for k in top_general[:60]],
        "pages": [{"file": os.path.basename(rows[i]["file"]), "page_index": pages[os.path.basename(rows[i]["file"])]["index"], "cover": pages[os.path.basename(rows[i]["file"])]["cover"],
                   "rating": rows[i]["rating"], "watch": rows[i]["watch"], "tags": rows[i]["tags"]} for i in idx],
    }
    json.dump(book, open(os.path.join(R, "results", slug + ".json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    summary.append({"book": b["book"], "volume": b["volume"], "slug": slug, "sampled": n, "worst_page": worst, "page_counts": dict(cnt), "share_explicit": share["explicit"],
        "max_explicit": book["rating"]["max_scores"]["explicit"], "kid_safe": book["filter"]["kid_safe"], "adult": book["filter"]["adult"],
        "censor": {t: v["max"] for t, v in book["censorship"].items() if v["max"] >= 0.1}, "censor_frac": {t: v["frac_pages"] for t, v in book["censorship"].items() if v["frac_pages"] > 0},
        "format": {t: v["frac_pages"] for t, v in book["format"].items() if v["frac_pages"] > 0},
        "top": [t["tag"] for t in book["top_tags"][:12]]})
json.dump(summary, open(os.path.join(R, "results", "summary.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
for s in summary:
    print(f"\n## {s['book']} ({s['volume']}) n={s['sampled']} worst={s['worst_page']} counts={s['page_counts']} share_expl={s['share_explicit']} max_expl={s['max_explicit']} kid_safe={s['kid_safe']} adult={s['adult']}")
    print("  censor max:", s["censor"], "| censor frac:", s["censor_frac"])
    print("  format frac:", s["format"])
    print("  top:", ", ".join(s["top"]))
