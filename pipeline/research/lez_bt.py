#!/usr/bin/env python3
"""Walk-forward for Liquidity Entry Zones on the point scale.

Inputs labeled "pips" are points (1/10 pip). Gold pip 0.1 → stop $5, target $13.
BTC auto pip is assumed to be $1 → point $0.10 → stop $50, target $130.
The ATR mode multiplies those distances by ATR / train-median ATR (fit on the first 60% only).
Structure mode keeps the point-scale sweep filters and puts the stop one point past the sweep wick.

Selection uses the train half only. The unseen last 40% decides alerts vs display.
Costs: Vantage round trip (gold $0.35, BTC 0.06%) plus slippage (gold $0.10, BTC 2 bps).
A trade whose friction exceeds 25% of the stop is not taken, matching the rest of this repo.
"""
import json, os, sys, time, urllib.request
from multiprocessing import Pool
_PIPE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PIPE)
os.environ["TRADING_LOAD_ONLY"] = "1"
import numpy as np, pandas as pd
import edge
import engine3 as E
sys.path.insert(0, os.path.join(_PIPE, "research", "trader_models"))
from tm import sim_trade, metr
CACHE = "/tmp/lez_cache.pkl"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lez_oos.json")
EDGE = [("XAUUSD","1d"),("XAUUSD","4h"),("XAUUSD","1h"),("BTCUSD","1d"),("BTCUSD","1h")]
MODES = ["point","atr","struct2","struct3"]
UA = {"User-Agent": "Mozilla/5.0"}

def get(url, timeout=40):
    req = urllib.request.Request(url, headers=UA)
    for i in range(5):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception:
            if i == 4: raise
            time.sleep(1.5 * (i + 1))

def _df(rows):
    df = pd.DataFrame(rows) if isinstance(rows, dict) else pd.DataFrame(rows, columns=["t","o","h","l","c","v"])
    df = df.drop_duplicates("t").sort_values("t")
    df = df.dropna(subset=["o","h","l","c"]).reset_index(drop=True)
    return df

def yahoo(sym, interval, range_=None, period1=None):
    if period1 is not None:
        q = f"period1={int(period1)}&period2={int(time.time())}&interval={interval}"
    else:
        q = f"interval={interval}&range={range_}"
    raw = json.loads(get(f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?{q}&includePrePost=true"))
    r = raw["chart"]["result"][0]
    q = r["indicators"]["quote"][0]
    ts = np.array(r["timestamp"], dtype="int64") * 1000
    df = _df({"t": ts, "o": q["open"], "h": q["high"], "l": q["low"], "c": q["close"], "v": np.nan_to_num(np.array(q["volume"], dtype=float), nan=0.0)})
    step = {"15m": 900_000, "60m": 3_600_000, "1d": 86_400_000}[interval]
    now = time.time() * 1000
    return df[df.t + step <= now].reset_index(drop=True)

def to_4h(df):
    idx = pd.to_datetime(df.t, unit="ms", utc=True)
    g = df.set_index(idx).resample("4h", label="left", closed="left").agg(
        {"t": "first", "o": "first", "h": "max", "l": "min", "c": "last", "v": "sum"}).dropna(subset=["o","h","l","c"])
    ts = g.index.astype("int64")
    if len(ts) and int(ts.max()) > 10**14: ts = ts // 10**6
    now = time.time() * 1000
    g = g[(np.asarray(ts) + 4 * 3_600_000) <= now]
    return g.reset_index(drop=True)

def binance_hist(symbol, interval, start_ms):
    step = {"15m": 900_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}[interval]
    rows = []; cur = int(start_ms); now = int(time.time() * 1000)
    while cur < now:
        url = f"https://data-api.binance.vision/api/v3/klines?symbol={symbol}&interval={interval}&startTime={cur}&limit=1000"
        data = json.loads(get(url))
        if not data: break
        for r in data:
            rows.append((int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])))
        nxt = int(data[-1][0]) + step
        if nxt <= cur or len(data) < 1000: break
        cur = nxt
    df = _df(rows)
    return df[df.t + step <= now].reset_index(drop=True)

def load_raw():
    if os.path.exists(CACHE):
        raw = pd.read_pickle(CACHE)
        print("cache", {k: len(v) for k, v in raw.items()}, flush=True)
    else:
        raw = {}
    d1 = raw.get(("XAUUSD","1d"))
    step = None if d1 is None or len(d1) < 3 else float(d1.t.diff().median())
    if d1 is None or len(d1) < 1000 or (step and step > 2 * 86_400_000):
        print("download GC=F daily", flush=True)
        raw[("XAUUSD","1d")] = yahoo("GC=F", "1d", period1=946684800)  # 2000-01-01; range=max comes back monthly
        pd.to_pickle(raw, CACHE)
    if ("XAUUSD","1h") in raw and ("BTCUSD","15m") in raw:
        return raw
    print("download GC=F", flush=True)
    h1 = yahoo("GC=F", "60m", "730d")
    raw[("XAUUSD","1h")] = h1
    raw[("XAUUSD","4h")] = to_4h(h1)
    raw[("XAUUSD","15m")] = yahoo("GC=F", "15m", "60d")
    print("download BTC", flush=True)
    raw[("BTCUSD","1d")] = binance_hist("BTCUSDT", "1d", 1502928000000)       # 2017-08-17
    raw[("BTCUSD","4h")] = binance_hist("BTCUSDT", "4h", 1546300800000)       # 2019-01-01
    raw[("BTCUSD","1h")] = binance_hist("BTCUSDT", "1h", 1546300800000)
    raw[("BTCUSD","15m")] = binance_hist("BTCUSDT", "15m", 1609459200000)     # 2021-01-01
    print("download PAXG", flush=True)
    raw[("PAXG","1h")] = binance_hist("PAXGUSDT", "1h", 1598918400000)        # 2020-09-01
    raw[("PAXG","15m")] = binance_hist("PAXGUSDT", "15m", 1622505600000)      # 2021-06-01
    for k, df in raw.items():
        print(" got", k, len(df), flush=True)
    pd.to_pickle(raw, CACHE)
    return raw

def friction(sym, px, book):
    spread, slip = edge.lez_friction_parts(sym, px)
    if book == "zero": return 0.0
    if book == "cost": return spread
    return spread + slip

def book_of(trades, stage, book):
    rs = []
    skip = mtm = 0
    ratios = []
    for t in trades:
        if t["stage"] != stage or t.get("gross") is None or not t.get("risk"): continue
        fr = friction(t["sym"], t["e"], "real" if book == "forced" else book)
        ratios.append(fr / t["risk"] if t["risk"] else None)
        net = t["gross"] - fr / t["risk"]
        survives = fr <= 0.25 * t["risk"]
        if book != "forced" and not survives:
            skip += 1
            continue
        if t.get("how") == "mtm": mtm += 1
        rs.append(net)
    m = metr(rs)
    m["skip"] = skip
    m["mtm"] = mtm
    good = [x for x in ratios if x is not None]
    m["med_fr"] = None if not good else round(float(np.median(good)), 4)
    return m

def lez_trades(df, sym, tf, mode, cut):
    med = None
    if mode == "atr":
        A = edge._atr(df).values
        sl = A[60:cut]
        sl = sl[np.isfinite(sl) & (sl > 0)]
        med = float(np.median(sl)) if len(sl) else None
    pack = edge.lez_run(df, sym, tf, mode=mode, med=med)
    A = edge._atr(df).values
    atr_med = float(np.nanmedian(A[60:cut])) if cut > 60 else None
    out = []
    for s in pack["signals"]:
        out.append({"sym": sym, "tf": tf, "mode": mode, "i": s["i"], "d": s["d"], "e": s.get("e"),
                    "risk": s.get("risk"), "gross": s.get("gross"), "how": s.get("how"), "j": s.get("j"),
                    "stage": "test" if s["i"] >= cut else "train"})
    return out, pack, atr_med, med

def swing_lists(df):
    sh, sl = E.pivots_confirmed(df.h.values, df.l.values, 3)
    n = len(df)
    highs = [(i, float(sh[i])) for i in range(n) if sh[i] == sh[i]]
    lows = [(i, float(sl[i])) for i in range(n) if sl[i] == sl[i]]
    return highs, lows

def swing_dist(entry, side, a, highs, lows, i):
    if not (a == a) or a <= 0: return None
    if side > 0:
        cands = [px for j, px in lows if i - 120 <= j <= i and px < entry]
        dist = (entry - (cands[-1] - 0.15 * a)) if cands else 1.2 * a
    else:
        cands = [px for j, px in highs if i - 120 <= j <= i and px > entry]
        dist = ((cands[-1] + 0.15 * a) - entry) if cands else 1.2 * a
    if dist < 0.8 * a: dist = 1.2 * a
    if dist > 3 * a: dist = 2.0 * a
    if dist <= 0: return None
    return dist

def collect_system(df, sym, dirs):
    """Current alert path: next open, swing stop, breakeven at +1R, 4×ATR trail, 48 bars. Gross R, friction applied later."""
    o, h, l, c = df.o.values, df.h.values, df.l.values, df.c.values
    A = E.atr(df).values
    highs, lows = swing_lists(df)
    n = len(df); rows = []; i = 0
    while i < n - 50:
        d = int(dirs[i])
        if d == 0 or not (A[i] == A[i]):
            i += 1; continue
        e = float(o[i + 1])
        dist = swing_dist(e, d, A[i], highs, lows, i)
        if dist is None:
            i += 1; continue
        stop = e - d * dist
        r = sim_trade(o, h, l, c, i + 1, d, e, stop, None, i + 1 + 47, be=1.0, trail=4.0, A=A[i], cost=0.0)
        if r is None: break
        rows.append({"sym": sym, "i": i, "d": d, "e": e, "risk": float(dist), "gross": float(r[0]), "how": "ch", "j": int(r[1]),
                     "stage": None})
        i = r[1] + 1
    return rows

def fit_weights(raw, sym, tf, cut):
    df = raw[(sym, tf)]; lo = max(E.PRM["window"], 80)
    if cut - lo < 30: return None
    stride = max(1, (cut - lo) // E.PRM["samples"])
    idx = list(range(lo, cut, stride))[:E.PRM["samples"]]
    rec = []
    for i in idx:
        v, c, rk = E.votes_at(raw, sym, tf, i)
        rec.append((v, c, rk, E.outcome(df, i, 1), E.outcome(df, i, -1)))
    nag = len(rec[0][0]); wts = []
    for k in range(nag):
        rs = [(r[3] if r[0][k] > 0 else r[4]) for r in rec if r[0][k] != 0]
        wts.append(0.0 if k == E.NEWS_IDX else E.weight_from(E.stats(rs)))
    return wts

RAW = FILT = WTS = CATL = NAMES = None

def _score(job):
    sym, tf, idxs = job
    fv, ftd, fhb, bL, bS, cdir = FILT[(sym, tf)]
    wts = WTS[(sym, tf)]
    out = []
    for i in idxs:
        v, c, rk = E.votes_at(RAW, sym, tf, i)
        s = E.consensus(v, c, rk, wts, CATL, NAMES)
        d = 1 if s >= E.PRM["buy_th"] else -1 if s <= E.PRM["sell_th"] else 0
        if not d: continue
        if not (fv[i] and ftd[i] == d and (fhb is None or fhb[i] == d) and (cdir == 0 or cdir == d)): continue
        if (d > 0 and bL[i]) or (d < 0 and bS[i]): continue
        out.append((i, int(d)))
    return sym, tf, out

def system_dirs(raw, jobs_meta):
    """jobs_meta: list of (sym, tf, cut). Returns {(sym,tf): dirs ndarray}."""
    global RAW, FILT, WTS, CATL, NAMES
    RAW = raw
    E.RATES = {}
    CATL = [E.W["categories"][nm] for nm in E.W["agents"]]
    NAMES = E.W["agents"]
    FILT = {}; WTS = {}; spans = {}
    HTF = {"1h": "4h", "4h": "1d", "1d": None}
    for sym, tf, cut in jobs_meta:
        t0 = time.time()
        w = fit_weights(raw, sym, tf, cut)
        if w is None or len(w) != len(NAMES):
            print("skip weights", sym, tf, flush=True); continue
        WTS[(sym, tf)] = w
        df = raw[(sym, tf)]
        hdf = raw.get((sym, HTF[tf])) if HTF[tf] else None
        fv, ftd, fhb = edge.filters(df, hdf)
        ltf_tf = edge.LTF.get(tf)
        ltf = raw.get((sym, ltf_tf)) if ltf_tf else None
        masks = edge.protect_masks(df, tf, sym, ltf, ltf_tf)
        cdir, _ = edge.carry_dir(sym, E.RATES)
        FILT[(sym, tf)] = (fv, ftd, None if hdf is None else fhb, masks["blockLong"], masks["blockShort"], cdir)
        lo = max(E.PRM["window"], 80); hi = len(df) - 50
        spans[(sym, tf)] = (lo, hi, len(df))
        print("fit", sym, tf, "secs", round(time.time() - t0, 1), "bars", hi - lo, flush=True)
    jobs = []
    for (sym, tf), (lo, hi, n) in spans.items():
        step = 2500
        for a in range(lo, hi, step):
            jobs.append((sym, tf, list(range(a, min(hi, a + step)))))
    try:
        import multiprocessing as mp
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass
    t0 = time.time()
    with Pool(8) as pool:
        parts = pool.map(_score, jobs)
    print("scored", len(jobs), "chunks in", round(time.time() - t0, 1), "s", flush=True)
    dirs = {}
    for (sym, tf), (lo, hi, n) in spans.items():
        arr = np.zeros(n, int)
        dirs[(sym, tf)] = arr
    for sym, tf, rows in parts:
        arr = dirs[(sym, tf)]
        for i, d in rows:
            arr[i] = d
    return dirs

def tag_stage(rows, cut):
    for r in rows:
        r["stage"] = "test" if r["i"] >= cut else "train"
    return rows

def apply_window(dirs, L, S, window):
    out = np.zeros_like(dirs)
    n = len(dirs)
    for i in range(n):
        d = int(dirs[i])
        if d == 0: continue
        ok = False
        for k in range(max(0, i - window), i + 1):
            if d > 0 and L[k]: ok = True
            elif d < 0 and S[k]: ok = True
        if ok: out[i] = d
    return out

def pack_mode(rows):
    return {st: {bk: book_of(rows, st, bk) for bk in ("zero", "cost", "real", "forced")} for st in ("train", "test")}

def main():
    raw = load_raw()
    series = {}
    lez_rows = {}
    atr_info = {}
    ref = [("XAUUSD","15m"),("BTCUSD","4h"),("BTCUSD","15m"),("PAXG","1h"),("PAXG","15m")]
    combos = EDGE + ref
    for sym, tf in combos:
        df = raw[(sym, tf)]
        cut = int(len(df) * E.PRM["fit_frac"])
        for mode in MODES:
            rows, pack, atr_med, med = lez_trades(df, sym, tf, mode, cut)
            lez_rows[(sym, tf, mode)] = rows
            if mode == "point":
                series[(sym, tf)] = pack
                sw, rng, sl, tp = edge.lez_bracket(sym)
                atr_info[(sym, tf)] = {"bars": len(df), "cut": cut,
                                       "atr_med_train": None if atr_med is None or atr_med != atr_med else round(float(atr_med), 4),
                                       "sl": sl, "tp": tp, "sl_over_atr": None if not atr_med else round(sl / float(atr_med), 3)}
            print(sym, tf, mode, "signals", len(rows), "test", sum(r["stage"] == "test" for r in rows), flush=True)
    print("baseline", flush=True)
    meta = []
    for sym, tf in EDGE:
        meta.append((sym, tf, int(len(raw[(sym, tf)]) * E.PRM["fit_frac"])))
    dirs = system_dirs(raw, meta)
    base_rows = {}
    for sym, tf, cut in meta:
        if (sym, tf) not in dirs: continue
        rows = tag_stage(collect_system(raw[(sym, tf)], sym, dirs[(sym, tf)]), cut)
        base_rows[(sym, tf)] = rows
        print("base", sym, tf, "trades", len(rows), "test", sum(r["stage"] == "test" for r in rows), flush=True)
    # Train selection. Point is the primary reading; ATR and wick stops are the other pre-specified variants.
    def pooled(mode, stage, book):
        rs = []
        for sym, tf in EDGE:
            rs += [r for r in [] ]
        bag = []
        for sym, tf in EDGE:
            for t in lez_rows[(sym, tf, mode)]:
                if t["stage"] != stage or t.get("gross") is None or not t.get("risk"): continue
                fr = friction(sym, t["e"], book if book != "forced" else "real")
                if book != "forced" and fr > 0.25 * t["risk"]: continue
                bag.append(t["gross"] - fr / t["risk"])
        return metr(bag)
    chosen = "point"; best = None
    train_table = {}
    for mode in MODES:
        m = pooled(mode, "train", "real")
        train_table[mode] = m
        if m.get("n", 0) >= 30 and m.get("exp") is not None and (best is None or m["exp"] > best[0]):
            best = (m["exp"], mode)
    if best: chosen = best[1]
    print("TRAIN pick", chosen, train_table, flush=True)
    # Filter window on train, using the chosen mode's signals as a pattern (not its bracket).
    def base_pool(stage, book, rows_map):
        bag = []
        for sym, tf in EDGE:
            for t in rows_map.get((sym, tf), []):
                if t["stage"] != stage or t.get("gross") is None or not t.get("risk"): continue
                fr = friction(sym, t["e"], "real" if book == "forced" else book)
                if book != "forced" and fr > 0.25 * t["risk"]: continue
                bag.append(t["gross"] - fr / t["risk"])
        return metr(bag)
    base_train = base_pool("train", "real", base_rows)
    filt_rows = {}
    win_pick = 0; win_best = None
    for window in (0, 2):
        alt = {}
        for sym, tf, cut in meta:
            if (sym, tf) not in dirs: continue
            pack = series[(sym, tf)] if chosen == "point" else None
            if chosen != "point":
                # rebuild sides from the already computed signals
                n = len(raw[(sym, tf)])
                L = np.zeros(n, bool); S = np.zeros(n, bool)
                for t in lez_rows[(sym, tf, chosen)]:
                    if t["d"] > 0: L[t["i"]] = True
                    elif t["d"] < 0: S[t["i"]] = True
            else:
                L, S = series[(sym, tf)]["L"], series[(sym, tf)]["S"]
            alt[(sym, tf)] = tag_stage(collect_system(raw[(sym, tf)], sym, apply_window(dirs[(sym, tf)], L, S, window)), cut)
        m = base_pool("train", "real", alt)
        filt_rows[window] = alt
        print("filter window", window, "train", m, flush=True)
        if m.get("n", 0) >= 20 and m.get("exp") is not None and base_train.get("exp") is not None and m["exp"] > base_train["exp"]:
            if win_best is None or m["exp"] > win_best[0]:
                win_best = (m["exp"], window)
    if win_best: win_pick = win_best[1]
    # Unseen half. Do not switch the precommitted mode or window.
    def beats(base, new, min_n):
        if not new or new.get("n", 0) < min_n or not base or not base.get("n"): return False
        if new.get("exp") is None or base.get("exp") is None: return False
        if not (new["exp"] > base["exp"] and new["exp"] > 0): return False
        if base.get("pf") is not None and new.get("pf") is not None and new["pf"] < base["pf"]: return False
        return True
    base_test = base_pool("test", "real", base_rows)
    solo_test = pooled(chosen, "test", "real")
    filt_test_map = filt_rows.get(win_pick, {})
    filt_test = base_pool("test", "real", filt_test_map)
    # Per-combo guard for the filter: a combo with 8+ current trades must not lose expectancy or disappear.
    filt_ok = beats(base_test, filt_test, 20)
    if filt_ok:
        for sym, tf in EDGE:
            b = book_of(base_rows.get((sym, tf), []), "test", "real")
            n = book_of(filt_test_map.get((sym, tf), []), "test", "real")
            if b.get("n", 0) >= 8 and (n.get("n", 0) == 0 or (n.get("exp") is not None and b.get("exp") is not None and n["exp"] < b["exp"])):
                filt_ok = False
    solo_ok = beats(base_test, solo_test, 30)
    if solo_ok:
        pos = 0
        for sym, tf in EDGE:
            m = book_of(lez_rows[(sym, tf, chosen)], "test", "real")
            if m.get("n", 0) >= 5 and m.get("exp", -1) > 0: pos += 1
        if pos < 3: solo_ok = False
    if filt_ok: integrate = "filter"
    elif solo_ok: integrate = "standalone"
    else: integrate = "display"
    ship_mode = chosen if integrate != "display" else "point"
    ship_window = win_pick if integrate != "display" else 0
    print("DECISION", integrate, "mode", ship_mode, "window", ship_window, "solo", solo_ok, solo_test, "filt", filt_ok, filt_test, "base", base_test, flush=True)
    def slim(rows, mode=None):
        use = rows if mode is None else lez_rows[rows] if False else rows
        return pack_mode(use)
    per = []
    for sym, tf in combos:
        role = "edge" if (sym, tf) in EDGE else "reference"
        rec = {"sym": sym, "tf": tf, "role": role, "info": atr_info.get((sym, tf)),
               "point": pack_mode(lez_rows[(sym, tf, "point")]),
               "atr": pack_mode(lez_rows[(sym, tf, "atr")]),
               "struct2": pack_mode(lez_rows[(sym, tf, "struct2")]),
               "struct3": pack_mode(lez_rows[(sym, tf, "struct3")])}
        if (sym, tf) in base_rows:
            rec["current"] = pack_mode(base_rows[(sym, tf)])
            if (sym, tf) in filt_rows.get(win_pick, {}):
                rec["filtered"] = pack_mode(filt_rows[win_pick][(sym, tf)])
        per.append(rec)
    # Spread survival at representative prices, plus the empirical skip rate of the point model.
    spread = {
        "gold": {"pip": 0.1, "point": 0.01, "sl": 5.0, "tp": 13.0, "spread": 0.35, "slip": 0.10,
                 "spread_over_sl": 0.07, "real_over_sl": 0.09, "survives_25pct": True,
                 "note": "The $5 stop clears a $0.35 gold round trip on every timeframe. Slippage of $0.10 makes the haircut 0.09R."},
        "btc": {"pip": 1.0, "point": 0.1, "assumption": "Auto pip = $1 (dollar pip for a four-to-six-digit quote). Point = pip/10 = $0.10. Not the 0.01 mintick.",
                "sl": 50.0, "tp": 130.0, "spread_rate": 0.0006, "slip_rate": 0.0002,
                "survives_below_price": round(0.25 * 50 / 0.0008, 0),
                "at": []},
    }
    for px in (10000, 15625, 30000, 60000, 100000):
        fr = (0.0006 + 0.0002) * px
        spread["btc"]["at"].append({"price": px, "friction": round(fr, 2), "over_sl": round(fr / 50, 3), "survives_25pct": fr <= 0.25 * 50})
    doc = {
        "note": "Point scale: indicator inputs / 10, then × the pip. Gold pip 0.1. BTC auto pip assumed $1. "
                "Walk-forward is the app's 60/40 split. Tuning (which pre-specified variant, and a 0-bar vs 2-bar filter) is chosen on the train half only. "
                "real = Vantage round trip + slippage, skipping trades where that friction exceeds 25% of the stop. "
                "zero is the TradingView commission-0 slippage-0 book. forced pays the friction even when it exceeds the stop.",
        "integrate": integrate, "ship_mode": ship_mode, "ship_window": ship_window,
        "chosen_on_train": chosen, "filter_window_on_train": win_pick,
        "train_real": train_table, "train_base_real": base_train,
        "test_base_real": base_test, "test_solo_real": solo_test, "test_filter_real": filt_test,
        "solo_ok": solo_ok, "filter_ok": filt_ok,
        "spread": spread, "per_combo": per,
    }
    json.dump(doc, open(OUT, "w"), indent=1)
    print("wrote", OUT, flush=True)

if __name__ == "__main__":
    main()
