#!/usr/bin/env python3
"""Offline VRAM expert-cache policy simulator for PYRE_QN_XTRACE traces (qwen-int, moe.md p1).
CPU only. usage: python tools/xsim.py trace_C.txt [trace_A.txt ...] [--cap N] [--sweep]
Model: cache contents are fixed during a verify window (refills happen at window start), a hit = (layer, expert)
resident and the layer exact-eligible; pairs = (token, expert) routings; rejected draft tokens still cost (they ran).
"""
import sys, collections, math

def parse(fn):
    hdr = None; size = {}; elig = {}; pre = []; wins = []; cur = None; seen = set(); real = [0, 0]
    pendR = []
    zmark = any(l.startswith('Z') for l in open(fn))
    for ln in open(fn):
        a = ln.split()
        if not a: continue
        k = a[0]
        if k == 'C': hdr = dict(nslot=int(a[1]), slot=int(a[2]), nl=int(a[3]), ne=int(a[4]), k=int(a[5]))
        elif k == 'S': size[int(a[1])] = int(a[2]); elig[int(a[1])] = a[3] == '1'
        elif k == 'Z': pre = []; wins = []; cur = None; real = [0, 0]   # prewarm reset (counts, calls, hits zeroed)
        elif k == 'P':
            if wins and not zmark: pre = []; wins = []; cur = None; real = [0, 0]   # old traces without Z: prewarm = P block + windows before the prompt's P block
            pre.append((int(a[1]), [int(x) for x in a[2:]])); cur = None
        elif k == 'B': cur = dict(t=int(a[1]), keep=int(a[1]), r=collections.defaultdict(list), b=True, R=pendR); pendR = []; wins.append(cur); seen = set()
        elif k == 'L':
            il, ti = int(a[1]), int(a[2]); bar = a.index('|'); on = [int(x) for x in a[3:bar]]; off = [int(x) for x in a[bar + 1:]]
            if cur is None or (il, ti) in seen:
                cur = dict(t=1, keep=1, r=collections.defaultdict(list), b=False, R=pendR); pendR = []; wins.append(cur); seen = set()
            seen.add((il, ti)); cur['r'][il].append((ti, on + off)); real[0] += len(on); real[1] += len(on) + len(off)
        elif k == 'K': cur['keep'] = int(a[2])
        elif k == 'R':
            if cur is None or cur.get('done'): pendR.append((int(a[1]), int(a[2])))   # refill before the next window (prefill-time fill)
            else: cur.setdefault('R', []).append((int(a[1]), int(a[2])))
    return hdr, size, elig, pre, wins, real

class Sim:
    """generic policy: score() ranks (layer, expert); refill every `every` windows (or replicate base's call counting)"""
    def __init__(s, tr, cap, pol, **kw):
        s.hdr, s.size, s.elig, s.pre, s.wins, _ = tr; s.cap = cap; s.pol = pol; s.kw = kw
        s.nl, s.ne = s.hdr['nl'], s.hdr['ne']
        s.cnt = collections.defaultdict(float); s.wcnt = collections.defaultdict(float); s.last = {}; s.res = set()
        s.hits = s.pairs = s.dmiss = s.dtot = s.swaps = s.swaps_late = 0; s.calls = 0; s.wi = 0
        s.evict_t = {}; s.reref = [0, 0]  # just-evicted experts needed again within 4 windows / evictions
        s.lay = collections.defaultdict(lambda: [0, 0]); s.win_cost = []
    def score(s, key):
        p = s.pol; c = s.cnt[key]
        if p in ('base', 'lfu', 'lfud'): return c
        if p == 'lfuw': return s.wcnt[key]                     # window-uses (bytes read) instead of token count
        if p == 'lru': return s.last.get(key, -1)
        if p == 'fr':                                           # decayed freq + recency bonus
            age = s.wi - s.last.get(key, -10**9); return c + s.kw.get('rb', 2.0) * math.exp(-age / s.kw.get('rt', 8.0))
        raise ValueError(p)
    def refill(s, first=False):
        cands = [(il, e) for il in range(s.nl) if s.elig.get(il) for e in range(s.ne) if s.cnt[(il, e)] > 0 or (s.pol == 'lru' and (il, e) in s.last)]
        cands.sort(key=lambda k: (-s.score(k), k))
        if s.kw.get('perlayer'):
            q = s.kw['perlayer']; want = set(); per = collections.Counter()
            for k in cands:
                if len(want) >= s.cap: break
                if per[k[0]] < q[k[0]]: want.add(k); per[k[0]] += 1
            for k in cands:
                if len(want) >= s.cap: break
                want.add(k)
        else: want = set(cands[:s.cap])
        maxn = 10**9 if first or not s.res else s.kw.get('maxnew', 64)
        gain = s.kw.get('gain', 0.0)
        ev = sorted((k for k in s.res if k not in want), key=lambda k: (s.score(k), k))  # pop from front = worst
        evi = 0; n = 0
        for k in cands:
            if n >= maxn: break
            if k not in want or k in s.res: continue
            if len(s.res) >= s.cap:
                if evi >= len(ev): break
                v = ev[evi]
                if gain and s.score(k) < s.score(v) + gain: break   # admission threshold (hysteresis)
                evi += 1; s.res.discard(v); s.evict_t[v] = s.wi; s.reref[1] += 1
            s.res.add(k); n += 1
        s.swaps += n
        if not first: s.swaps_late += n
    def refill_free(s, mx):
        """fill empty slots only (no eviction): the best-scored routed experts not resident, at most mx"""
        cands = [(il, e) for il in range(s.nl) if s.elig.get(il) for e in range(s.ne) if s.cnt[(il, e)] > 0 and (il, e) not in s.res]
        cands.sort(key=lambda k: (-s.score(k), k)); n = 0
        for k in cands[:min(mx, s.cap - len(s.res))]: s.res.add(k); n += 1
        s.swaps += n
    def window(s, w):
        t = w['t']; keep = w['keep']; pairs = hits = dm = dt = 0
        for il, lst in w['r'].items():
            used = collections.Counter()
            for ti, es in lst:
                for e in es: used[e] += 1
            on = s.elig.get(il) 
            for e, nt in used.items():
                key = (il, e); h = on and key in s.res
                pairs += nt; dt += 1
                if h: hits += nt
                else: dm += 1
                s.lay[il][0] += nt * h; s.lay[il][1] += nt
                if not h and key in s.evict_t and s.wi - s.evict_t[key] <= 4: s.reref[0] += 1; del s.evict_t[key]
        s.hits += hits; s.pairs += pairs; s.dmiss += dm; s.dtot += dt
        s.win_cost.append((t, pairs - hits, dm))
        # counts: accepted tokens only (base semantics: rejected drafts are rolled back)
        dec = s.kw.get('decay')
        if dec:
            for k in list(s.cnt): s.cnt[k] *= dec
            for k in list(s.wcnt): s.wcnt[k] *= dec
        for il, lst in w['r'].items():
            ws = set()
            for ti, es in lst:
                for e in es:
                    s.last[(il, e)] = s.wi
                    if ti < keep: s.cnt[(il, e)] += 1; ws.add(e)
            for e in ws: s.wcnt[(il, e)] += 1
        s.wi += 1
    def run(s):
        for il, es in s.pre:
            for e in es: s.cnt[(il, e)] += 1; s.wcnt[(il, e)] += 1.0 / 8; s.last[(il, e)] = -1
        every = s.kw.get('every'); tr = every == 'trace'
        if not tr: s.calls = 1; s.refill(first=True)
        for w in s.wins:
            if tr:
                for c, r in w.get('R', []): s.refill(first=(c == 1 or not s.res))
            elif every is None:   # base call counting: one call per token of the window, refill when calls % 16 == 0
                for _ in range(w['t']):
                    s.calls += 1
                    if s.calls % 16 == 0: s.refill()
            elif s.wi % every == 0 and s.wi > 0: s.refill()
            s.window(w)
            if every is None: s.calls -= w['t'] - w['keep']
            if s.kw.get('fillfree') and len(s.res) < s.cap: n0 = s.swaps; mf = s.kw.get('fillmax'); (s.refill(first=True) if not mf else s.refill_free(mf)); s.swaps_late += s.swaps - n0
        return s

def belady(tr, cap):
    """demand-paged Belady with bypass: after a window, a missed item is admitted iff its next use is sooner than the
    furthest next use in the cache (applied from the next window on, like a refill); unlimited swaps -> upper bound"""
    hdr, size, elig, pre, wins, _ = tr
    acc = collections.defaultdict(list); wl = []
    for i, w in enumerate(wins):
        u = collections.Counter()
        for il, lst in w['r'].items():
            if not elig.get(il): continue
            for ti, es in lst:
                for e in es: u[(il, e)] += 1
        wl.append(u)
        for k in u: acc[k].append(i)
    ptr = collections.defaultdict(int)
    def nxt(k, i):
        a = acc[k]; j = ptr[k]
        while j < len(a) and a[j] <= i: j += 1
        ptr[k] = j; return a[j] if j < len(a) else 10**9
    # start: best static fill from prefill counts is not oracle; Belady may also start from the oracle first set
    res = set(); hits = pairs = 0; tot_all = 0
    import heapq
    first = collections.Counter()
    for i, u in enumerate(wl):
        for k in u: first.setdefault(k, i) if False else None
    # initial residency: the cap items with the soonest first use (prefetch at load allowed, like the base's prefill fill)
    fu = sorted(acc, key=lambda k: acc[k][0])[:cap]; res = set(fu)
    for i, u in enumerate(wl):
        for k, n in u.items():
            pairs += n
            if k in res: hits += n
        miss = [k for k in u if k not in res]
        if miss:
            heap = [(-nxt(r, i), r) for r in res]; heapq.heapify(heap)
            for k in sorted(miss, key=lambda k: nxt(k, i)):
                nk = nxt(k, i)
                if nk >= 10**9: continue
                if len(res) < cap: res.add(k); continue
                while heap and heap[0][1] not in res: heapq.heappop(heap)
                if not heap or -heap[0][0] <= nk: break
                _, v = heapq.heappop(heap); res.discard(v); res.add(k)
    return hits, pairs

def static_oracle(tr, cap):
    hdr, size, elig, pre, wins, _ = tr; u = collections.Counter()
    for w in wins:
        for il, lst in w['r'].items():
            if not elig.get(il): continue
            for ti, es in lst:
                for e in es: u[(il, e)] += 1
    tot = sum(len(es) for w in wins for il, lst in w['r'].items() for ti, es in lst)
    return sum(n for _, n in u.most_common(cap)), tot

def reuse(tr):
    hdr, size, elig, pre, wins, _ = tr; last = {}; d = []; once = collections.Counter()
    for i, w in enumerate(wins):
        for il, lst in w['r'].items():
            for e in {e for ti, es in lst for e in es}:
                k = (il, e)
                if k in last: d.append(i - last[k])
                last[k] = i; once[k] += 1
    d.sort(); q = lambda p: d[int(p * (len(d) - 1))] if d else 0
    single = sum(1 for v in once.values() if v == 1)
    return dict(n=len(d), p25=q(.25), p50=q(.5), p75=q(.75), p90=q(.9), distinct=len(once), once=single)

SWAP_MS = float(next((a.split('=')[1] for a in sys.argv if a.startswith('--swap=')), 0.105))  # p3: fast repack 64 experts = 6.7 ms (was 10.7 = 0.167/expert)   # measured: a 64-expert refill = 10.7 ms on the critical path (repack 2.3 ms/expert over rayon + write); pool ms = 7.2 + 0.0274 * miss pairs (fit, 203 windows)
def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    caps = [int(a.split('=')[1]) for a in sys.argv[1:] if a.startswith('--cap=')]
    pols = [
        ('base replay (refills at the trace R points)', 'base', dict(every='trace')),
        ('base(LFU cum, refill/16 calls, 64 new)', 'base', {}),
        ('base + fill free slots every window', 'base', dict(fillfree=1)),
        ('base replay + fill free slots every window', 'base', dict(every='trace', fillfree=1)),
        ('LFU cum, refill every window, 64 new', 'lfu', dict(every=1)),
        ('LFU cum, every window, unlimited', 'lfu', dict(every=1, maxnew=10**9)),
        ('LFU decay .95/window, every win', 'lfud', dict(every=1, decay=.95)),
        ('LFU decay .9/window, every win', 'lfud', dict(every=1, decay=.9)),
        ('LFU decay .8/window, every win', 'lfud', dict(every=1, decay=.8)),
        ('LFU window-uses (bytes), every win', 'lfuw', dict(every=1)),
        ('LRU, every window, unlimited', 'lru', dict(every=1, maxnew=10**9)),
        ('LRU, every window, 64 new', 'lru', dict(every=1)),
        ('freq+recency rb2 rt8', 'fr', dict(every=1)),
        ('freq+recency rb4 rt4', 'fr', dict(every=1, rb=4, rt=4)),
        ('LFU cum + gain 1 hysteresis', 'lfu', dict(every=1, gain=1.0)),
        ('LFU cum + gain 2 hysteresis', 'lfu', dict(every=1, gain=2.0)),
        ('LFU dec .9 + gain 1', 'lfud', dict(every=1, decay=.9, gain=1.0)),
        ('LFU cum every win, 16 new', 'lfu', dict(every=1, maxnew=16)),
        ('LFU cum every win, 32 new', 'lfu', dict(every=1, maxnew=32)),
        ('LFU cum every 2 win, 64 new', 'lfu', dict(every=2)),
        ('LFU cum every win 32 new gain 2', 'lfu', dict(every=1, maxnew=32, gain=2.0)),
        ('LFU cum every win gain 3', 'lfu', dict(every=1, gain=3.0)),
        ('LFU dec .9 every win 32 new gain 2', 'lfud', dict(every=1, decay=.9, maxnew=32, gain=2.0)),
        ('base + free-slot fill only', 'base', dict(fillfree=1)),
        ('base + free fill max 32/win', 'base', dict(fillfree=1, fillmax=32)),
        ('base + free fill max 64/win', 'base', dict(fillfree=1, fillmax=64)),
        ('base + free fill max 128/win', 'base', dict(fillfree=1, fillmax=128)),
        ('base + free fill max 256/win', 'base', dict(fillfree=1, fillmax=256)),
    ]
    for fn in args:
        tr = parse(fn); hdr = tr[0]; real = tr[5]
        cl = caps or [hdr['nslot']]
        print(f"== {fn}: windows {len(tr[4])} prefill tokens {len(tr[3]) // max(1, hdr['nl'])} slots(trace) {hdr['nslot']} real hit {100 * real[0] / max(1, real[1]):.1f}% ({real[1]} pairs)")
        print("   reuse distance (windows) of (layer,expert):", reuse(tr))
        for cap in cl:
            h, p = belady(tr, cap); so, st = static_oracle(tr, cap)
            print(f"  cap {cap}: Belady(demand, bypass) {100 * h / p:.1f}%   static hindsight top-cap {100 * so / st:.1f}%")
            for nm, pol, kw in pols:
                s = Sim(tr, cap, pol, **kw).run()
                ms = [m for t, m, d in s.win_cost]; dm = [d for t, m, d in s.win_cost]
                est = (7.2 * len(ms) + 0.0274 * sum(ms) + SWAP_MS * s.swaps_late) / len(ms)
                print(f"   {nm:42s} est {est:5.2f} ms/win (swaps late {s.swaps_late:4d}) hit {100 * s.hits / s.pairs:5.1f}%  miss pairs/win {sum(ms) / len(ms):6.1f}  distinct miss/win {sum(dm) / len(dm):6.1f}  swaps {s.swaps:5d}  evict->reuse<=4w {s.reref[0]}/{s.reref[1]}")
            if '--layers' in sys.argv:
                s = Sim(tr, cap, 'base').run(); print('   base per-layer hit%:', ' '.join(f"{il}:{100 * a / max(1, b):.0f}" for il, (a, b) in sorted(s.lay.items())))
main()
