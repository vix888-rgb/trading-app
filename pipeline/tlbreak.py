"""15m trendline break for gold (XAUUSD): DISPLAY-ONLY module (added 28 Nov 2026).

Nothing here feeds the agents, weights, journal, gated signals or phone alerts. It reads the closed 15m gold bars the scan already
saved (RAW_PKL), finds trendline-break entries, replays the exit model on the same bars and writes:
  * data.json -> "tl15"  (live + recent signals for the dedicated page in the app)
  * state/tl15_log.json  (forward trade log from the first scan that ran this module, for the nightly review)
A failure here is caught by run.sh and never stops the scan.

Rule (backtest of 5-27 Nov 2026, 15m, 34 trades, +0.45R/trade, PF 2.69, spread $0.30; small sample, not a proven edge):
  swing = 3-bar fractal (confirmed 3 bars later). Falling line through the last two swing highs (2nd lower, >= 8 bars apart, last one < 150 bars old):
  a 15m close above the line after a close at/below it -> BUY at the next candle open. Rising line through the last two swing lows (2nd higher): close below -> SELL.
  SL = line value at the break -/+ 1.5 x ATR(14). Skip if risk > 3.5 x ATR, widen to 0.7 x ATR if tighter.
  Exit: 1/3 at 1R, 1/3 at 2R, 1/3 at 3R, SL to entry after TP1, flat at the end of the Belgrade day.
  Entries 01:00-20:30 Belgrade, max 3 trades per day, one position at a time.
"""
import os, sys, json, pickle, time
import numpy as np, pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
_PIPE = os.path.dirname(os.path.abspath(__file__))
if _PIPE not in sys.path: sys.path.insert(0, _PIPE)

TZ = ZoneInfo("Europe/Belgrade")
SYM, TF, STEP_MS = "XAUUSD", "15m", 900_000
COST = 0.30            # spread, price units ($) per trade
LOT = 0.02; OZ = LOT * 100   # 0.02 lot = 2 oz -> $2 per $1 move ($0.20 per pip)
SL_ATR, MIN_ATR, MAX_ATR = 1.5, 0.7, 3.5
WIN = (1.0, 20.5); MAXTR = 3
W = 3; MIN_GAP = 8; MAX_AGE = 150
NAME = "15m trendline break (gold)"
RULE = ("Falling line through the last two swing highs (or rising line through the last two swing lows). A 15m close through the line, after a close on the other side, "
        "enters at the next candle open. SL 1.5×ATR(14) beyond the line. TP 1R / 2R / 3R (1/3 each), SL to entry after TP1, flat at day end. "
        "Entries 01:00–20:30 Belgrade, max 3 a day. Backtest 5–27 Nov (34 trades): +0.45R, PF 2.69, 71% wins, spread $0.30 included. Small sample, display only.")
BT_REF = {"n": 34, "win": 71, "avgR": 0.45, "pf": 2.69, "span": "5–26 Nov 2026, 15m", "note": "backtest, not live"}

def frame(df):
    """Closed 15m bars with Belgrade-time helpers. df has t (ms UTC), o,h,l,c."""
    d = df[["t", "o", "h", "l", "c"]].copy().reset_index(drop=True)
    d = d.dropna().reset_index(drop=True)
    idx = pd.to_datetime(d.t.astype("int64"), unit="ms", utc=True).dt.tz_convert(TZ)
    d["day"] = idx.dt.strftime("%Y-%m-%d").values
    d["hr"] = (idx.dt.hour + idx.dt.minute / 60).values
    d["ts"] = idx
    return d

def wilder(x, p=14):
    n = len(x); out = np.full(n, np.nan)
    if n < p: return out
    out[p - 1] = x[:p].mean()
    for i in range(p, n): out[i] = (out[i - 1] * (p - 1) + x[i]) / p
    return out

def atr14(d):
    h, l, c = d.h.values, d.l.values, d.c.values
    pc = np.r_[c[0], c[:-1]]
    return wilder(np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc))))

def signals(d):
    """Causal: a signal at bar i is known at the close of bar i."""
    h, l, c = d.h.values, d.l.values, d.c.values; A = atr14(d); n = len(d)
    swh, swl, out = [], [], []
    for i in range(n):
        j = i - W
        if j >= W:
            if h[j] == h[j - W:j + W + 1].max() and h[j] > h[j - W:j].max(): swh.append((j, h[j]))
            if l[j] == l[j - W:j + W + 1].min() and l[j] < l[j - W:j].min(): swl.append((j, l[j]))
        if i < 2 or not np.isfinite(A[i]): continue
        if len(swh) >= 2:
            (a, pa), (b, pb) = swh[-2], swh[-1]
            if pb < pa and b - a >= MIN_GAP and i - b < MAX_AGE:
                f = lambda x: pa + (pb - pa) * (x - a) / (b - a)
                if c[i] > f(i) and c[i - 1] <= f(i - 1):
                    out.append(dict(i=i, side=1, line=f(i), sl=f(i) - SL_ATR * A[i]))
        if len(swl) >= 2:
            (a, pa), (b, pb) = swl[-2], swl[-1]
            if pb > pa and b - a >= MIN_GAP and i - b < MAX_AGE:
                f = lambda x: pa + (pb - pa) * (x - a) / (b - a)
                if c[i] < f(i) and c[i - 1] >= f(i - 1):
                    out.append(dict(i=i, side=-1, line=f(i), sl=f(i) + SL_ATR * A[i]))
    return out, A

def current_lines(d):
    """Falling / rising trendline as it stands on the last closed bar (level to watch)."""
    h, l = d.h.values, d.l.values; n = len(d); swh, swl = [], []
    for i in range(n):
        j = i - W
        if j >= W:
            if h[j] == h[j - W:j + W + 1].max() and h[j] > h[j - W:j].max(): swh.append((j, h[j]))
            if l[j] == l[j - W:j + W + 1].min() and l[j] < l[j - W:j].min(): swl.append((j, l[j]))
    res = {}; i = n - 1
    for nm, sw, ok in (("falling", swh, lambda pa, pb: pb < pa), ("rising", swl, lambda pa, pb: pb > pa)):
        if len(sw) >= 2:
            (a, pa), (b, pb) = sw[-2], sw[-1]
            if ok(pa, pb) and b - a >= MIN_GAP and i - b < MAX_AGE:
                res[nm] = {"now": pa + (pb - pa) * (i - a) / (b - a), "next": pa + (pb - pa) * (i + 1 - a) / (b - a)}
    return res

def simulate(d, E, s, sl, j):
    """Market entry at price E on bar j (bar j is processed fully, SL before TP in a bar). Returns dict."""
    h, l, c, day = d.h.values, d.l.values, d.c.values, d.day.values; n = len(d)
    R0 = abs(E - sl); T = [E + s * R0 * k for k in (1, 2, 3)]
    stop = sl; real = 0.0; rem = 1.0; hit = 0
    for b in range(j, n):
        if day[b] != day[j]:
            return dict(R=real + rem * s * (c[b - 1] - E) / R0, xi=b - 1, hits=hit, kind="eod")
        if (l[b] <= stop) if s > 0 else (h[b] >= stop):
            return dict(R=real + rem * s * (stop - E) / R0, xi=b, hits=hit, kind="sl" if hit == 0 else "be")
        for k in range(hit, 3):
            if (h[b] >= T[k]) if s > 0 else (l[b] <= T[k]):
                real += (1 / 3) * (k + 1); rem -= 1 / 3; hit = k + 1
                if k == 0: stop = E
            else: break
        if hit == 3: return dict(R=real, xi=b, hits=3, kind="tp3")
    return dict(R=real + rem * s * (c[n - 1] - E) / R0, xi=n - 1, hits=hit, kind="open")

def run(d):
    """All signals with their outcome. Same filters and order as the backtest (one position at a time, 3 a day, window, risk)."""
    sigs, A = signals(d); n = len(d)
    o, day, hr = d.o.values, d.day.values, d.hr.values
    res = []; last_exit = -1; cnt = {}
    for sg in sorted(sigs, key=lambda x: x["i"]):
        i, s = sg["i"], sg["side"]
        base = dict(bar_t=int(d.t.iloc[i]), side="BUY" if s > 0 else "SELL", line=float(sg["line"]), atr=float(A[i]))
        if i <= last_exit:
            res.append({**base, "status": "skipped", "why": "already in a trade"}); continue
        pending = i + 1 >= n
        if pending:
            nxt = d.ts.iloc[i] + pd.Timedelta(milliseconds=STEP_MS)
            dd = nxt.strftime("%Y-%m-%d"); hh = nxt.hour + nxt.minute / 60
        else:
            dd, hh = day[i + 1], hr[i + 1]
        if cnt.get(day[i], 0) >= MAXTR:
            res.append({**base, "status": "skipped", "why": "3 trades already today"}); continue
        if dd != day[i] or not (WIN[0] <= hh < WIN[1]):
            res.append({**base, "status": "skipped", "why": "outside 01:00–20:30"}); continue
        E = float(d.c.iloc[i]) if pending else float(o[i + 1]); sl = float(sg["sl"]); a = float(A[i])
        if (s > 0 and sl >= E) or (s < 0 and sl <= E):
            res.append({**base, "status": "skipped", "why": "stop on the wrong side of entry"}); continue
        risk = abs(E - sl)
        if risk > MAX_ATR * a:
            res.append({**base, "status": "skipped", "why": "risk above 3.5×ATR"}); continue
        if risk < MIN_ATR * a: sl = E - s * MIN_ATR * a; risk = MIN_ATR * a
        tp = [E + s * risk * k for k in (1, 2, 3)]
        row = {**base, "entry": E, "approx": pending, "sl": sl, "tp": tp, "risk": risk, "riskUsd": risk * OZ, "lot": LOT}
        if pending:
            row.update(status="pending", hits=0, R=None, usd=None, exit_t=None); res.append(row)
            last_exit = n - 1; cnt[day[i]] = cnt.get(day[i], 0) + 1; continue
        j = i + 1; r = simulate(d, E, s, sl, j); net = r["R"] - COST / risk
        status = {"sl": "sl", "tp3": "tp3", "be": "tp1be" if r["hits"] == 1 else "tp2be", "eod": "eod", "open": "active"}[r["kind"]]
        row.update(status=status, hits=r["hits"], R=float(net), usd=float(net * risk * OZ), entry_t=int(d.t.iloc[j]),
                   exit_t=None if r["kind"] == "open" else int(d.t.iloc[r["xi"]]))
        res.append(row); last_exit = r["xi"]; cnt[day[i]] = cnt.get(day[i], 0) + 1
    return res

def fmt_t(ms):
    return datetime.fromtimestamp(ms / 1000, TZ).strftime("%a %d %b %H:%M")

def stats(rows):
    rs = [r["R"] for r in rows if r.get("R") is not None and r["status"] not in ("active", "pending")]
    if not rs: return {"n": 0}
    w = sum(x for x in rs if x > 0); lo = -sum(x for x in rs if x < 0)
    usd = sum(r["usd"] for r in rows if r.get("usd") is not None and r["status"] not in ("active", "pending"))
    return {"n": len(rs), "win": round(100 * sum(x > 0 for x in rs) / len(rs)), "avgR": round(float(np.mean(rs)), 2),
            "pf": round(w / lo, 2) if lo > 0 else None, "sumR": round(float(sum(rs)), 2), "usd": round(float(usd), 1)}

def public_row(r):
    o = {k: r.get(k) for k in ("bar_t", "side", "status", "why", "hits")}
    o["t"] = fmt_t(r["bar_t"] + STEP_MS)           # signal candle closes here; entry is the next open
    for k in ("entry", "sl", "risk", "riskUsd", "R", "usd"):
        if r.get(k) is not None: o[k] = round(float(r[k]), 2)
    if r.get("tp"): o["tp"] = [round(float(x), 2) for x in r["tp"]]
    if r.get("approx"): o["approx"] = True
    if r.get("exit_t"): o["x"] = fmt_t(r["exit_t"] + STEP_MS)
    return o

def update_log(path, rows, now_ms):
    log = {"v": 1, "since": now_ms, "since_txt": fmt_t(now_ms), "note": "forward log of the display-only 15m trendline break (gold); R and $ at 0.02 lot include a $0.30 spread", "trades": {}}
    try:
        if os.path.exists(path): log = {**log, **json.load(open(path))}
    except Exception: pass
    tr = log["trades"]
    for r in rows:
        if r["status"] in ("skipped",) or r["bar_t"] < log["since"] - STEP_MS: continue   # forward only: signal candle must be at/after the first scan
        key = str(r["bar_t"]); old = tr.get(key)
        if old and old.get("status") not in ("pending", "active"): continue            # closed trades are final
        row = public_row(r); row["entry_t"] = r.get("entry_t"); row["exit_t"] = r.get("exit_t")
        tr[key] = row
    log["trades"] = dict(sorted(tr.items(), key=lambda kv: int(kv[0])))
    log["stats"] = stats(list(log["trades"].values()))
    log["updated"] = datetime.now(TZ).isoformat(timespec="minutes")
    tmp = path + ".tmp"; json.dump(log, open(tmp, "w"), separators=(",", ":")); os.replace(tmp, path)
    return log

def panel(df, state_dir=None, now_ms=None):
    now_ms = int(time.time() * 1000) if now_ms is None else now_ms
    d = frame(df)
    if len(d) < 60: raise ValueError(f"only {len(d)} closed 15m bars")
    rows = run(d)
    recent = [r for r in rows if r["bar_t"] >= d.t.iloc[-1] - 3 * 86_400_000]   # about the last 3 days
    pub = [public_row(r) for r in recent if r["status"] != "skipped"][::-1][:20]
    nskip = sum(r["status"] == "skipped" for r in recent)
    lines = current_lines(d)
    log = update_log(os.path.join(state_dir, "tl15_log.json"), rows, now_ms) if state_dir else None
    fwd = log["stats"] if log else {"n": 0}
    last = float(d.c.iloc[-1]); a = float(atr14(d)[-1])
    return {"v": 1, "name": NAME, "rule": RULE, "sym": SYM, "tf": TF, "lot": LOT, "oz": OZ, "displayOnly": True, "bt": BT_REF,
            "asof": fmt_t(int(d.t.iloc[-1]) + STEP_MS), "last": round(last, 2), "atr": round(a, 2),
            "lines": {k: {kk: round(float(vv), 2) for kk, vv in v.items()} for k, v in lines.items()},
            "sigs": pub, "skipped": nskip, "fwd": fwd, "since": log["since_txt"] if log else None,
            "n_replay": len(rows)}

def main():
    from paths import RAW_PKL, STATE, DATA_JSON
    raw = pickle.load(open(RAW_PKL, "rb"))[2]
    df = raw.get((SYM, TF))
    if df is None: raise ValueError("no XAUUSD 15m frame in the scan data")
    os.makedirs(STATE, exist_ok=True)
    p = panel(df, STATE)
    d = json.load(open(DATA_JSON)); d["tl15"] = p
    tmp = DATA_JSON + ".tmp"; json.dump(d, open(tmp, "w"), separators=(",", ":"), ensure_ascii=False); os.replace(tmp, DATA_JSON)
    print("tl15 ok", len(p["sigs"]), "signals shown; forward log", p["fwd"].get("n", 0), "closed")

if __name__ == "__main__":
    main()
