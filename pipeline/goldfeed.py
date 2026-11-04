"""Spot gold (XAU/USD) for the scan, from Twelve Data, with a Yahoo futures fallback.

Credit budget (Basic plan: 1 credit per ``/time_series`` symbol, 8/minute, 800/day).
``outputsize`` does not change the cost. A scan every 10 minutes is 144 scans/day:

- 1min every scan (the desk still has a 1-minute view): 144
- 5min every scan: 144
- 1h roughly once an hour, on the first scan after that bar has been closed
  for 60 seconds, plus the 30-minute refresh. The close fetch resets the
  refresh timer, so the steady state is still two scans an hour: 48

That is 336 credits/day, under the 800/day plan limit. Fetching 5min, 1min,
and 1h on every scan is 432/day. History lives in
``pipeline/state/gold_feed.json`` so later scans request only the new tail.

Bar boundaries match the candles the app already used:

- 1m, 5m, and 1h are Twelve Data bar opens with ``timezone=UTC``. Forex
  defaults to Australia/Sydney, so UTC is set on purpose. Those stamps land
  on the same minute / 5-minute / hour grid as Yahoo ``1m`` / ``5m`` / ``60m``.
- 15m is resampled from 5m on UTC quarter-hours (:00, :15, :30, :45), which
  is where Yahoo ``15m`` closed bars sat.
- 4h is resampled from 1h the same way ``engine.yahoo`` resampled GC=F:
  pandas ``4h`` on a UTC index (origin start of the UTC day, left-closed),
  so buckets are 00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC. A bucket is
  kept only after its window has ended and the last hour is in the cache
  (open time equals the bucket end minus 1h). The same last-hour rule applies
  to 1D. If that hour never trades (Friday, a holiday, or a maintenance gap),
  the bucket is kept once the 1h fetch is at least 10 minutes past the end.
- 1D is resampled from 1h onto America/New_York midnights. Yahoo GC=F daily
  stamps were midnight New York: 04:00 UTC in EDT (27 Oct 2026) and 05:00 UTC
  in EST (3 Nov 2026). The bar closes at the next New York midnight, including
  the 23h/25h daylight-saving days. The last hour of that bar opens at 23:00
  New York.

The API key is read from ``TWELVEDATA_API_KEY`` and is never written to state,
the snapshot, or an error string. If the key is missing, Twelve Data errors,
or it rate-limits, gold falls back to Yahoo ``GC=F`` minus the last measured
futures-minus-spot offset (close of GC=F minus close of XAU/USD on shared hours).
"""
import json, math, os, time, urllib.error, urllib.parse, urllib.request
from datetime import datetime, timezone

import pandas as pd

from paths import STATE

STATE_NAME = "gold_feed.json"
SYMBOL = "XAU/USD"
ALIGN = (
    "15m from 5m on UTC quarter-hours (:00/:15/:30/:45); "
    "4h from 1h on UTC 00/04/08/12/16/20; "
    "1D from 1h on America/New_York midnight (Yahoo GC=F daily stamp)"
)
SUCCESS_LABEL = "Twelve Data XAU/USD spot"
# Closed-bar steps. Daily uses the next New York midnight instead of a fixed 24h.
STEP = {"1min": 60_000, "5min": 300_000, "1h": 3_600_000, "15m": 900_000, "4h": 14_400_000}
TF_STEP = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}
PULL = ("5min", "1min", "1h")  # 5min first so a rate limit spends one credit, then stops
H1_EVERY = 30 * 60
H1_SLACK = 90
MIN_BARS = 150
CAP = 5000
SEED = 5000
OVERLAP = 4
NY = "America/New_York"


def credits_per_day():
    """Steady-state credits: 1min and 5min every 10 minutes, 1h twice an hour.

    1h is fetched right after each close and again on the 30-minute refresh.
    The close fetch restarts that timer, so the two triggers share two scans
    an hour (``86400 // H1_EVERY``). 5min + 1min + 1h on every scan is
    ``3 * 144`` = 432, still under the 800/day plan.
    """
    scans = 24 * 60 * 60 // 600
    hourly = 24 * 60 * 60 // H1_EVERY
    return scans + scans + hourly


def state_path(path=None):
    return path or os.path.join(STATE, STATE_NAME)


def _empty_state():
    return {
        "v": 1, "offset": None, "offsetAt": None, "offsetN": 0,
        "paxgOff": None, "paxgAt": None, "fetchedAt": {},
        "bars": {"1min": [], "5min": [], "1h": []},
        "calls": {}, "vendor": "", "label": "", "detail": "",
        "ok": False, "reason": "", "scale": "futures", "align": ALIGN,
        "creditsPerDay": credits_per_day(), "journalSpot": False, "journalShift": None,
        "tfErr": {},
    }


def _load(path=None):
    path = state_path(path)
    if not os.path.exists(path):
        return _empty_state()
    try:
        doc = json.load(open(path))
    except Exception:
        return _empty_state()
    base = _empty_state()
    if isinstance(doc, dict):
        base.update(doc)
    if not isinstance(base.get("bars"), dict):
        base["bars"] = {"1min": [], "5min": [], "1h": []}
    return base


def _save(state, path=None):
    path = state_path(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(state, fh, separators=(",", ":"))
    os.replace(tmp, path)


def _safe(msg, key):
    text = str(msg or "")
    if key:
        text = text.replace(str(key), "")
    return text.strip()[:160]


class TwelveError(Exception):
    def __init__(self, code, message):
        self.code = int(code or 0)
        super().__init__(_safe(message, ""))


def _reason(err, key):
    msg = _safe(err, key)
    code = getattr(err, "code", 0) or 0
    low = msg.lower()
    if code == 429 or "credit" in low or "too many" in low or "rate limit" in low:
        return "rate limit"
    return msg or "error"


def safe_reason(err):
    key = (os.environ.get("TWELVEDATA_API_KEY") or "").strip()
    return _reason(err, key)


def _http(url, timeout=25):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return getattr(resp, "status", 200), resp.read()
    except urllib.error.HTTPError as err:
        try:
            body = err.read()
        except Exception:
            body = b""
        return err.code, body
    except Exception:
        raise TwelveError(0, "request failed")


def _parse(values):
    rows = []
    for item in values or []:
        stamp = item.get("datetime")
        if not stamp:
            continue
        ts = pd.Timestamp(stamp)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        try:
            o, h, l, c = (float(item["open"]), float(item["high"]), float(item["low"]), float(item["close"]))
        except (KeyError, TypeError, ValueError):
            continue
        if not all(math.isfinite(x) for x in (o, h, l, c)):
            continue
        raw_v = item.get("volume")
        try:
            vol = 0.0 if raw_v in (None, "") else float(raw_v)
        except (TypeError, ValueError):
            vol = 0.0
        if not math.isfinite(vol):
            vol = 0.0
        rows.append((int(ts.timestamp() * 1000), o, h, l, c, vol))
    df = pd.DataFrame(rows, columns=["t", "o", "h", "l", "c", "v"])
    if len(df) == 0:
        return df
    return df.drop_duplicates("t", keep="last").sort_values("t").reset_index(drop=True)


def fetch_twelvedata(interval, outputsize, key):
    query = urllib.parse.urlencode({
        "symbol": SYMBOL, "interval": interval, "outputsize": str(int(outputsize)),
        "timezone": "UTC", "order": "ASC", "apikey": key,
    })
    status, body = _http("https://api.twelvedata.com/time_series?" + query)
    try:
        doc = json.loads(body.decode("utf-8", "replace"))
    except Exception:
        raise TwelveError(status or 0, "unreadable response")
    if not isinstance(doc, dict) or doc.get("status") == "error" or "values" not in doc:
        code = int((doc.get("code") if isinstance(doc, dict) else None) or status or 0)
        msg = _safe(doc.get("message") if isinstance(doc, dict) else "", key)
        raise TwelveError(code, msg or "twelve data error")
    return _parse(doc["values"])


def fetch_paxg():
    status, body = _http("https://data-api.binance.vision/api/v3/ticker/price?symbol=PAXGUSDT", timeout=15)
    if status and int(status) >= 400:
        raise TwelveError(status, "paxg http")
    doc = json.loads(body.decode("utf-8", "replace"))
    return float(doc["price"])


def _blank():
    return pd.DataFrame(columns=["t", "o", "h", "l", "c", "v"])


def _frame(rows):
    df = pd.DataFrame(rows or [], columns=["t", "o", "h", "l", "c", "v"])
    if len(df) == 0:
        return _blank()
    return df.drop_duplicates("t", keep="last").sort_values("t").reset_index(drop=True)


def _pack(df):
    if df is None or len(df) == 0:
        return []
    out = []
    for t, o, h, l, c, v in df[["t", "o", "h", "l", "c", "v"]].itertuples(index=False):
        out.append([int(t), round(float(o), 5), round(float(h), 5), round(float(l), 5), round(float(c), 5), round(float(v), 5)])
    return out


def drop_forming(df, step_ms, now_ms):
    if df is None or len(df) == 0:
        return _blank()
    return df[df.t + step_ms <= now_ms].reset_index(drop=True)


def merge_bars(old, new, cap=CAP):
    parts = [x for x in (old, new) if x is not None and len(x)]
    if not parts:
        return _blank()
    df = pd.concat(parts, ignore_index=True)
    return df.drop_duplicates("t", keep="last").sort_values("t").tail(cap).reset_index(drop=True)


def outputsize_for(last_t_ms, step_ms, now_ms, n_bars=None):
    # A short cache asks for a full seed. A warm cache asks only for the new tail.
    if last_t_ms is None or (n_bars is not None and n_bars < MIN_BARS):
        return SEED
    bars = int(max(0, now_ms - last_t_ms) / step_ms) + OVERLAP
    return max(5, min(SEED, bars))


def h1_due(fetched_at, now, n_bars, last_t_ms):
    if n_bars < MIN_BARS or not fetched_at:
        return True
    if now - float(fetched_at) >= H1_EVERY - H1_SLACK:
        return True
    if last_t_ms and now * 1000 - (int(last_t_ms) + STEP["1h"]) > 2 * STEP["1h"]:
        return True
    # The bar after the cached one has been closed for at least 60s. The
    # 30-minute refresh often misses that scan, which is how a 4h/1D close
    # was scored without its last hour.
    if last_t_ms and now * 1000 >= int(last_t_ms) + 2 * STEP["1h"] + 60_000:
        return True
    return False


def _epoch_ms(index):
    utc = index.tz_convert("UTC") if getattr(index, "tz", None) is not None else index.tz_localize("UTC")
    return (utc.as_unit("ns").asi8 // 1_000_000).astype("int64")


def resample_ohlc(df, rule, step_ms=None, now_ms=None, tz=None, complete=None):
    """Aggregate to ``rule``. ``t`` is the bucket open. The open bucket is dropped.

    ``tz`` converts the index before resampling so daily bars follow
    America/New_York midnights. Intraday rules stay on UTC (``tz`` is None),
    which matches the previous pandas ``resample('4h')`` call.

    ``complete``, when set, is called with each bucket end in milliseconds.
    A finished bucket is kept only when it also returns true. Boundaries are
    unchanged: 4h still ends on the UTC step, and 1D still ends at the next
    New York midnight (``pd.offsets.Day(1)``).
    """
    if df is None or len(df) == 0:
        return _blank()
    now_ms = time.time() * 1000 if now_ms is None else now_ms
    idx = pd.to_datetime(df.t.astype("int64"), unit="ms", utc=True)
    if tz:
        idx = idx.dt.tz_convert(tz)
    base = df[["o", "h", "l", "c", "v"]].copy()
    base.index = idx
    base = base.sort_index()
    base = base[~base.index.duplicated(keep="last")]
    agg = base.resample(rule, origin="start_day", label="left", closed="left").agg(
        o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"), v=("v", "sum")
    ).dropna(subset=["o", "c"])
    if len(agg) == 0:
        return _blank()
    starts = agg.index
    start_ms = _epoch_ms(starts)
    if step_ms:
        end_ms = start_ms + int(step_ms)
    else:
        ends = starts + pd.offsets.Day(1)
        end_ms = _epoch_ms(ends)
    keep = end_ms <= now_ms
    if complete is not None:
        keep = [bool(flag) and bool(complete(int(end))) for flag, end in zip(keep, end_ms)]
    pos = [i for i, flag in enumerate(list(keep)) if flag]
    kept_ms = start_ms[pos]
    agg = agg.iloc[pos]
    if len(agg) == 0:
        return _blank()
    out = agg.reset_index(drop=True)
    out.insert(0, "t", kept_ms)
    return out.reset_index(drop=True)


def shift_ohlc(df, offset):
    if df is None or offset is None:
        return df
    out = df.copy()
    for col in ("o", "h", "l", "c"):
        out[col] = out[col] - float(offset)
    return out.reset_index(drop=True)


def _offset_on(spot, futures, shift_ms):
    left = spot[["t", "c"]].copy()
    right = futures[["t", "c"]].copy()
    left["bucket"] = (left.t.astype("int64") // STEP["1h"]) * STEP["1h"]
    right["bucket"] = ((right.t.astype("int64") + shift_ms) // STEP["1h"]) * STEP["1h"]
    joined = left.merge(right, on="bucket", suffixes=("_s", "_f"))
    if len(joined) < 8:
        return None, 0
    med = float((joined.c_f - joined.c_s).tail(48).median())
    if not math.isfinite(med) or abs(med) > 80:
        return None, 0
    return round(med, 4), int(len(joined))


def measure_offset(spot, futures):
    """Median of GC=F close minus XAU/USD close on shared hours.

    Twelve Data stamps are bar opens. If a feed is labeled on the close instead,
    the same hours are tried one hour either way and the alignment with the
    most overlap wins. Exact overlap is preferred when the counts tie.
    """
    if spot is None or futures is None or len(spot) < 8 or len(futures) < 8:
        return None
    best = None
    for shift in (0, -STEP["1h"], STEP["1h"]):
        med, n = _offset_on(spot, futures, shift)
        if med is None:
            continue
        if best is None or n > best[1]:
            best = (med, n)
    return None if best is None else best[0]


def _iso(now):
    return datetime.fromtimestamp(now, timezone.utc).isoformat(timespec="seconds")


def _cached(state, interval):
    return _frame((state.get("bars") or {}).get(interval))


def _pull(state, interval, key, now, now_ms, fetch_td, calls):
    prev = _cached(state, interval)
    last = None if len(prev) == 0 else int(prev.t.iloc[-1])
    size = outputsize_for(last, STEP[interval], now_ms, len(prev))
    calls.append(interval)
    fresh = drop_forming(fetch_td(interval, size, key), STEP[interval], now_ms)
    merged = merge_bars(prev, fresh)
    state.setdefault("bars", {})[interval] = _pack(merged)
    state.setdefault("fetchedAt", {})[interval] = now
    return merged


def _bucket_ready(h1, fetched_at):
    """True when a finished 4h/1D bucket has its last hour, or the gap is confirmed.

    The last hour opens at ``bucket_end - 1h``. A Friday or holiday bar never
    prints that hour; a 1h fetch at least 10 minutes after the bucket end
    shows the gap is real rather than a late cache.
    """
    opens = set()
    if h1 is not None and len(h1):
        opens = {int(t) for t in h1.t}
    fetched_ms = None if fetched_at is None else float(fetched_at) * 1000.0

    def ready(end_ms):
        end_ms = int(end_ms)
        if end_ms - STEP["1h"] in opens:
            return True
        return fetched_ms is not None and fetched_ms >= end_ms + 600_000

    return ready


def _build(state, now_ms):
    m1 = drop_forming(_cached(state, "1min"), STEP["1min"], now_ms)
    m5 = drop_forming(_cached(state, "5min"), STEP["5min"], now_ms)
    h1 = drop_forming(_cached(state, "1h"), STEP["1h"], now_ms)
    ready = _bucket_ready(h1, (state.get("fetchedAt") or {}).get("1h"))
    frames = {
        "5m": m5,
        "15m": resample_ohlc(m5, "15min", step_ms=STEP["15m"], now_ms=now_ms),
        "1h": h1,
        "4h": resample_ohlc(h1, "4h", step_ms=STEP["4h"], now_ms=now_ms, complete=ready),
        "1d": resample_ohlc(h1, "1D", now_ms=now_ms, tz=NY, complete=ready),
    }
    terr = {}
    if len(m1) >= MIN_BARS:
        frames["1m"] = m1
    else:
        terr["1m"] = f"only {len(m1)} candles"
    for tf, df in list(frames.items()):
        if len(df) < MIN_BARS:
            terr[tf] = f"only {len(df)} candles"
    return frames, terr


def _remember_paxg(state, frames, fetch_px, now):
    if not fetch_px:
        return
    try:
        px = float(fetch_px())
    except Exception:
        return
    ref = None
    for tf in ("1m", "5m", "1h"):
        df = frames.get(tf)
        if df is not None and len(df):
            ref = float(df.c.iloc[-1])
            break
    if ref is None or not math.isfinite(px) or not math.isfinite(ref):
        return
    state["paxgOff"] = round(ref - px, 4)
    state["paxgAt"] = _iso(now)


def _fallback_label(reason, offset):
    why = reason or "unavailable"
    if offset is None:
        return f"Yahoo Finance GC=F futures (Twelve Data: {why}; no spot offset yet)"
    return f"Yahoo Finance GC=F minus {float(offset):.2f} futures-spot offset (Twelve Data: {why})"


def _finish_fallback(state, yahoo_fn, now, reason, calls, fetch_px, path):
    offset = state.get("offset")
    scale = "spot" if offset is not None else "futures"
    frames, terr = {}, {}
    for tf in ("1m", "5m", "15m", "1h", "4h", "1d"):
        try:
            df = yahoo_fn("GC=F", tf)
            frames[tf] = shift_ohlc(df, offset) if offset is not None else df
        except Exception as err:
            terr[tf] = _safe(err, "")[:80] or "yahoo failed"
    if scale == "spot":
        _remember_paxg(state, frames, fetch_px, now)
    else:
        state["paxgOff"] = None
    label = _fallback_label(reason, offset)
    state.update({
        "ok": False, "vendor": "yahoo", "scale": scale, "reason": reason, "label": label,
        "detail": label, "align": ALIGN, "creditsPerDay": credits_per_day(),
        "calls": {name: calls.count(name) for name in PULL}, "tfErr": terr,
    })
    _save(state, path)
    print("gold", label, "calls", state["calls"], flush=True)
    return frames, terr


def _stamp_success(state, calls, terr, path):
    state.update({
        "ok": True, "vendor": "twelvedata", "scale": "spot", "reason": "",
        "label": SUCCESS_LABEL, "detail": SUCCESS_LABEL, "align": ALIGN,
        "creditsPerDay": credits_per_day(),
        "calls": {name: calls.count(name) for name in PULL}, "tfErr": terr,
    })
    _save(state, path)
    print("gold", SUCCESS_LABEL, "calls", state["calls"], flush=True)


def load_xau(yahoo_fn, now=None, fetch_td=None, fetch_px=None, state_path=None, key=None):
    """Return ``(frames_by_tf, tf_errors)`` and write the source label into state.

    ``frames`` use the same columns as ``engine.yahoo`` (``t`` in ms, closed bars).
    """
    path = state_path
    state = _load(path)
    now = time.time() if now is None else float(now)
    now_ms = now * 1000
    calls = []
    if key is None:
        key = os.environ.get("TWELVEDATA_API_KEY") or ""
    key = str(key).strip()
    fetch_td = fetch_td or fetch_twelvedata
    if fetch_px is None:
        fetch_px = fetch_paxg
    if not key:
        return _finish_fallback(state, yahoo_fn, now, "missing key", calls, fetch_px, path)
    try:
        _pull(state, "5min", key, now, now_ms, fetch_td, calls)
    except TwelveError as err:
        _save(state, path)
        return _finish_fallback(state, yahoo_fn, now, _reason(err, key), calls, fetch_px, path)
    prior_err = {}
    try:
        _pull(state, "1min", key, now, now_ms, fetch_td, calls)
    except TwelveError as err:
        prior_err["1m"] = _reason(err, key)
    h1 = _cached(state, "1h")
    last_h = None if len(h1) == 0 else int(h1.t.iloc[-1])
    if h1_due((state.get("fetchedAt") or {}).get("1h"), now, len(h1), last_h):
        try:
            _pull(state, "1h", key, now, now_ms, fetch_td, calls)
        except TwelveError as err:
            if len(drop_forming(_cached(state, "1h"), STEP["1h"], now_ms)) < MIN_BARS:
                _save(state, path)
                return _finish_fallback(state, yahoo_fn, now, _reason(err, key), calls, fetch_px, path)
            state.setdefault("tfErr", {})["1h"] = _reason(err, key)
    if len(drop_forming(_cached(state, "5min"), STEP["5min"], now_ms)) < MIN_BARS or len(drop_forming(_cached(state, "1h"), STEP["1h"], now_ms)) < MIN_BARS:
        return _finish_fallback(state, yahoo_fn, now, "short history", calls, fetch_px, path)
    frames, terr = _build(state, now_ms)
    for tf, msg in prior_err.items():
        if tf not in frames:
            terr[tf] = msg
    try:
        fut = yahoo_fn("GC=F", "1h")
        off = measure_offset(frames.get("1h"), fut)
        if off is not None:
            state["offset"] = off
            state["offsetAt"] = _iso(now)
            state["offsetN"] = int(min(48, len(frames.get("1h") or [])))
    except Exception:
        pass
    _remember_paxg(state, frames, fetch_px, now)
    _stamp_success(state, calls, terr, path)
    return frames, terr


def public_meta(path=None):
    state = _load(path)
    label = state.get("label") or "Yahoo Finance GC=F (COMEX gold futures, proxy for spot XAU/USD)"
    return {
        "ok": bool(state.get("ok")),
        "vendor": state.get("vendor") or "",
        "label": label,
        "detail": state.get("detail") or label,
        "offset": state.get("offset"),
        "paxgOff": state.get("paxgOff") if state.get("scale") == "spot" else None,
        "align": ALIGN,
        "creditsPerDay": credits_per_day(),
        "reason": state.get("reason") or "",
        "scale": state.get("scale") or "futures",
        "calls": state.get("calls") or {},
    }


def feed_label(key, src, code, path=None):
    if key == "XAUUSD":
        return public_meta(path).get("label") or "Yahoo Finance GC=F (COMEX gold futures, proxy for spot XAU/USD)"
    if src == "binance":
        return "Binance public API " + code
    return "Yahoo Finance " + code + (" (COMEX gold futures, proxy for spot XAU/USD)" if code == "GC=F" else "")


def rebase_open_journal(path, state_path=None):
    """Shift open XAUUSD prices onto the spot scale once, by the measured futures-spot offset.

    Stops and targets are distances. Subtracting the same offset from the levels and from
    the candles keeps the trade geometry. Entries logged after the switch are already on spot.
    """
    state = _load(state_path)
    if state.get("journalSpot"):
        return 0
    if state.get("scale") != "spot" or state.get("offset") is None:
        return 0
    off = float(state["offset"])
    n = 0
    if path and os.path.exists(path):
        journal = json.load(open(path))
        for entry in journal:
            if entry.get("sym") != "XAUUSD" or entry.get("status") != "open" or entry.get("pxBasis") == "spot":
                continue
            for field in ("entry", "stop", "inv", "trail_stop"):
                value = entry.get(field)
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                entry[field] = round(float(value) - off, 4)
            if isinstance(entry.get("tp"), list):
                entry["tp"] = [None if x is None else round(float(x) - off, 4) for x in entry["tp"]]
            entry["pxBasis"] = "spot"
            entry["pxShift"] = off
            n += 1
        if n:
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(journal, fh, indent=1)
            os.replace(tmp, path)
    state["journalSpot"] = True
    state["journalShift"] = off
    _save(state, state_path)
    return n
