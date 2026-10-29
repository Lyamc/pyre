"""Purity / Explicit / Gore indexes, formula v1 -- the same as monga
(crates/monga-core/src/content_index.rs, branch content-tags). Weights and parameters
come from content_weights.txt, a copy of monga's; keep the two files identical.

A page is a dict {tag: score}. See the Rust module docs for the formula."""
import math, os
HERE = os.path.dirname(os.path.abspath(__file__))

def load_weights(path=os.path.join(HERE, "content_weights.txt")):
    w = {k: {} for k in ("param", "purity", "explicit", "hint", "extreme", "gore")}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"): continue
        p = [x.strip() for x in line.split("|")]
        if len(p) != 3 or p[0] not in w: continue
        try: w[p[0]][p[1]] = float(p[2])
        except ValueError: pass
    return w
W = load_weights()

def param(name, default, w=W): return w["param"].get(name, default)

def conf(score, w=W):
    lo, hi = param("conf_low", 0.25, w), param("conf_high", 0.65, w)
    return min(1.0, max(0.0, (score - lo) / max(hi - lo, 1e-6)))

def combine(page, table, w=W):
    keep = 1.0
    for tag, s in page.items():
        if tag in table: keep *= 1.0 - table[tag] * conf(s, w)
    return min(1.0, max(0.0, 1.0 - keep))

def page_index(page, w=W):
    impurity = combine(page, w["purity"], w)
    sexual = combine(page, w["explicit"], w)
    explicit = None
    if sexual >= param("sexual_page", 0.3, w):
        hint = max([w["hint"][t] * conf(s, w) for t, s in page.items() if t in w["hint"]], default=0.0)
        base = sexual * (1.0 - min(1.0, max(0.0, hint)))
        explicit = 1.0 - (1.0 - base) * (1.0 - combine(page, w["extreme"], w))
    return {"purity": 1.0 - impurity, "explicit": explicit, "gore": combine(page, w["gore"], w)}

def roll(values, mean_k, worst_k, pow_, top):
    if not values: return 0.0
    mean = sum(values) / len(values)
    s = sorted(values, reverse=True); top = min(max(top, 1), len(s))
    worst = sum(s[:top]) / top
    spread = 1.0 - math.exp(-mean_k * mean)
    return 1.0 - (1.0 - spread) * (1.0 - min(1.0, max(0.0, worst_k * worst ** pow_)))

def indexes(pages, w=W):
    """pages: list of {tag: score}. Returns version, purity, explicit (None = no sexual content), gore, sexual page indices."""
    idx = [page_index(p, w) for p in pages]
    out = {"version": int(param("version", 1, w)), "pages": len(idx), "purity": None, "explicit": None, "gore": None,
           "sexual_pages": [i for i, p in enumerate(idx) if p["explicit"] is not None]}
    if not idx: return out
    top = int(param("worst_pages", 3, w))
    out["purity"] = 1.0 - roll([1.0 - p["purity"] for p in idx], param("purity_mean", 4.0, w), param("purity_worst", 0.35, w), param("purity_worst_pow", 6.0, w), top)
    out["gore"] = roll([p["gore"] for p in idx], param("gore_mean", 4.0, w), param("gore_worst", 0.5, w), param("gore_worst_pow", 2.0, w), top)
    sex = [p["explicit"] for p in idx if p["explicit"] is not None]
    if sex:
        raw = (param("explicit_mean", 0.55, w) * sum(sex) / len(sex) + param("explicit_share", 0.3, w) * math.sqrt(len(sex) / len(idx))
               + param("explicit_worst", 0.15, w) * max(sex))
        floor = param("explicit_floor", 0.15, w)
        out["explicit"] = min(1.0, max(0.0, (raw - floor) / (1.0 - floor)))
    return out

KID = {"min_purity": 0.6, "max_explicit": -1.0, "max_gore": 0.5, "block": ["sexual_violence", "child_sexual", "sexual"]}

def kid_over(idx, cats, limits=KID, w=W):
    """Reasons a book fails the kid limits (empty = kid friendly). cats: rollup categories with max_score and pages."""
    why = []
    if idx["purity"] is not None and idx["purity"] < limits["min_purity"]: why.append(f"Purity {idx['purity']:.2f} < {limits['min_purity']:.2f}")
    if idx["explicit"] is not None and (limits["max_explicit"] < 0 or idx["explicit"] > limits["max_explicit"]): why.append(f"Explicit {idx['explicit']:.2f}")
    if idx["gore"] is not None and idx["gore"] > limits["max_gore"]: why.append(f"Gore {idx['gore']:.2f} > {limits['max_gore']:.2f}")
    strong = param("block_score", 0.5, w)
    for c in cats:
        if c["id"] in limits["block"] and (c["max_score"] >= strong or len(c["pages"]) >= 2): why.append(c["label"])
    return why