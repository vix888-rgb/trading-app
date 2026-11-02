#!/usr/bin/env python3
"""Shift published prices onto fresh Vantage quotes from the ntfy price topic.

TradingView posts either one JSON bar per symbol or one combined message,
{"q": {"XAUUSD": ..., ...}, "tf": "5", "t": ...}, to NTFY_PRICE_URL. This step
runs after data.json is built. For each symbol, the newest quote is kept.
A quote more than 15 minutes old is dropped:

    offset = vantage close − freshest feed last close

That offset is added to the symbol's current price and to every signal level
(entry, zone, stop, invalidation, TP1/2/3, and S/R plus the other price levels
on the same scale). The snapshot chart bounds move with them so the candles
stay aligned. Lot size is then recomputed from the shifted entry and stop with
the existing $20 risk rule. A uniform shift keeps the stop distance, so the
lot matches the Vantage quote.

If the topic is down, empty, stale, or unreadable, the symbol keeps its
existing feed. This step never fails the scan for that reason.
"""
import json, math, os, re, sys, time, urllib.request
from datetime import datetime, timezone

_PIPE = os.path.dirname(os.path.abspath(__file__))
if _PIPE not in sys.path:
    sys.path.insert(0, _PIPE)

from alerts import lot_for_20
from paths import DATA_JSON

# Price topic only. Do not read or write the phone-alert topic.
PRICE_URL = os.environ.get(
    "NTFY_PRICE_URL",
    "https://ntfy.sh/octa-prices-8223d362/json?poll=1&since=30m",
)
TIMEOUT = float(os.environ.get("NTFY_PRICE_TIMEOUT", "15"))
MAX_AGE_SEC = 15 * 60
FUTURE_SKEW_SEC = 120
TFS = ["1m", "5m", "15m", "1h", "4h", "1d"]
KNOWN = {"XAUUSD", "BTCUSD", "EURUSD", "USDJPY", "USDCHF", "EURCHF"}


def canon_symbol(raw):
    """Map VANTAGE:XAUUSD, XAUUSD+, and XAUUSD.r onto an app symbol id."""
    if raw is None:
        return None
    s = str(raw).strip().upper()
    if not s:
        return None
    if ":" in s:
        s = s.rsplit(":", 1)[-1].strip()
    s = re.split(r"[^A-Z0-9]", s, maxsplit=1)[0]
    return s if s in KNOWN else None


def symbol_of(body):
    if not isinstance(body, dict):
        return None
    for key in ("sym", "symbol", "tv", "ticker"):
        if body.get(key) in (None, ""):
            continue
        hit = canon_symbol(body.get(key))
        if hit:
            return hit
    return None


def as_float(v):
    if isinstance(v, bool) or v is None or v == "":
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return x


def as_price(v):
    x = as_float(v)
    if x is None or x <= 0:
        return None
    return x


def price_of(body):
    if not isinstance(body, dict):
        return None
    for key in ("c", "bid", "close"):
        if key not in body:
            continue
        px = as_price(body.get(key))
        if px is not None:
            return round(px, 8)
    return None


def as_unix(v):
    x = as_float(v)
    if x is None or x <= 0:
        return None
    if x > 1e11:
        x /= 1000.0
    return x


def quote_time(body, hint):
    for candidate in (body.get("t") if isinstance(body, dict) else None, hint):
        ts = as_unix(candidate)
        if ts is not None:
            return ts
    return None


def iso_utc(ts):
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def iter_payloads(text):
    raw = (text or "").strip()
    if not raw:
        return
    items = []
    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            items = parsed
        elif isinstance(parsed, dict):
            items = [parsed]
    else:
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    for ev in items:
        if not isinstance(ev, dict):
            continue
        if "message" in ev or ev.get("event") or ev.get("topic"):
            if ev.get("event") not in (None, "message"):
                continue
            msg = ev.get("message")
            hint = ev.get("time")
            if isinstance(msg, dict):
                yield msg, hint
                continue
            if isinstance(msg, str) and msg.strip():
                try:
                    body = json.loads(msg)
                except json.JSONDecodeError:
                    continue
                if isinstance(body, dict):
                    yield body, hint
            continue
        yield ev, None


def each_quote(body, hint):
    """Yield (symbol, price, unix time) from one message.

    A dict ``q`` is the combined feed: each known symbol with a positive price
    shares the message time. JSON null and unknown symbols are skipped. Any
    other object is the old one-symbol bar (``sym`` plus ``c``, ``bid``, or
    ``close``).
    """
    if not isinstance(body, dict):
        return
    ts = quote_time(body, hint)
    if ts is None:
        return
    book = body.get("q")
    if isinstance(book, dict):
        for raw, val in book.items():
            sym = canon_symbol(raw)
            px = as_price(val)
            if sym and px is not None:
                yield sym, px, ts
        return
    sym = symbol_of(body)
    px = price_of(body)
    if sym and px is not None:
        yield sym, px, ts


def latest_quotes(text, now=None):
    """Newest usable quote per symbol. Quotes older than 15 minutes are dropped."""
    now = time.time() if now is None else float(now)
    best = {}
    for body, hint in iter_payloads(text):
        for sym, px, ts in each_quote(body, hint):
            age = now - ts
            if age < -FUTURE_SKEW_SEC or age > MAX_AGE_SEC:
                continue
            prev = best.get(sym)
            if prev is None or ts >= prev[0]:
                best[sym] = (ts, px)
    return {sym: {"time": ts, "price": px} for sym, (ts, px) in best.items()}


def fetch_text(url):
    req = urllib.request.Request(url, headers={"User-Agent": "trading-signal-desk/vantage", "Cache-Control": "no-cache"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return resp.read().decode("utf-8", "replace")


def decimals(prec, sample):
    try:
        return int(prec) + 1
    except (TypeError, ValueError):
        sample = abs(float(sample))
        p = 2 if sample >= 100 else 4 if sample >= 10 else 5 if sample >= 0.5 else 6
        return p + 1


def add_px(value, offset, places):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    if not math.isfinite(float(value)):
        return value
    return round(float(value) + offset, places)


def reference_close(doc, sym):
    """Freshest feed last close. Ties go to the finer timeframe."""
    best = None
    det = doc.get("det") or {}
    for i, tf in enumerate(TFS):
        row = det.get(f"{sym}|{tf}")
        if not isinstance(row, dict):
            continue
        last = as_float(row.get("last"))
        if last is None:
            continue
        age = as_float(row.get("age"))
        if age is None:
            age = 1e12
        cand = (age, i, last, tf)
        if best is None or cand[0] < best[0] or (cand[0] == best[0] and cand[1] < best[1]):
            best = cand
    if best is None:
        return None
    return {"age": best[0], "tf": best[3], "last": best[2]}


def shift_pl(pl, offset, places):
    if not isinstance(pl, list) or len(pl) < 8:
        return
    pl[1] = add_px(pl[1], offset, places)
    if isinstance(pl[2], list):
        pl[2] = [add_px(x, offset, places) for x in pl[2]]
    pl[4] = add_px(pl[4], offset, places)
    pl[6] = add_px(pl[6], offset, places)
    if isinstance(pl[7], list):
        pl[7] = [add_px(x, offset, places) for x in pl[7]]


def shift_lv(lv, offset, places):
    if not isinstance(lv, dict):
        return
    for key, value in list(lv.items()):
        lv[key] = add_px(value, offset, places)


def shift_ch(ch, offset, places):
    if not isinstance(ch, list) or len(ch) < 2:
        return
    ch[0] = add_px(ch[0], offset, places)
    ch[1] = add_px(ch[1], offset, places)


def shift_symbol(doc, sym, offset):
    for row in doc.get("scan") or []:
        if not isinstance(row, list) or len(row) < 11 or row[0] != sym or row[2] is None:
            continue
        if not isinstance(row[3], (int, float)) or isinstance(row[3], bool):
            continue
        row[3] = add_px(row[3], offset, decimals(row[10] if len(row) > 10 else None, row[3]))
    det = doc.get("det") or {}
    for tf in TFS:
        row = det.get(f"{sym}|{tf}")
        if not isinstance(row, dict) or not isinstance(row.get("last"), (int, float)):
            continue
        places = decimals(row.get("p"), row["last"])
        row["last"] = add_px(row["last"], offset, places)
        shift_pl(row.get("pl"), offset, places)
        shift_lv(row.get("lv"), offset, places)
        shift_ch(row.get("ch"), offset, places)


def feed_name(meta, sym):
    info = (meta or {}).get(sym)
    if isinstance(info, list) and len(info) > 2 and info[2]:
        return info[2]
    return "feed"


def blank_px(feed):
    return {"price_source": feed, "vantage_price": None, "vantage_time": None, "offset": None}


def apply_quotes(doc, quotes):
    """Attach per-symbol price fields and shift levels. Returns the px map."""
    meta = doc.get("meta") or {}
    px = {}
    for sym in meta:
        q = (quotes or {}).get(sym)
        ref = reference_close(doc, sym) if q else None
        if not q or not ref:
            px[sym] = blank_px(feed_name(meta, sym))
            continue
        offset = round(float(q["price"]) - float(ref["last"]), 8)
        px[sym] = {
            "price_source": "Vantage",
            "vantage_price": q["price"],
            "vantage_time": iso_utc(q["time"]),
            "offset": offset,
        }
        shift_symbol(doc, sym, offset)
        det = (doc.get("det") or {}).get(f"{sym}|1h") or (doc.get("det") or {}).get(f"{sym}|{ref['tf']}")
        pl = (det or {}).get("pl") or []
        if len(pl) > 4:
            size = lot_for_20(sym, pl[1], pl[4], doc.get("rates") or {})
            print(f"vantage: {sym} offset {offset} on {ref['tf']} close {ref['last']} -> {q['price']} · $20 lot {size}")
        else:
            print(f"vantage: {sym} offset {offset} on {ref['tf']} close {ref['last']} -> {q['price']}")
    doc["px"] = px
    return px


def load_quotes(now=None):
    try:
        text = fetch_text(PRICE_URL)
    except Exception as exc:
        print(f"vantage: ntfy unavailable, keeping existing feeds ({exc})")
        return {}
    try:
        return latest_quotes(text, now=now)
    except Exception as exc:
        print(f"vantage: could not read ntfy prices, keeping existing feeds ({exc})")
        return {}


def main():
    with open(DATA_JSON, encoding="utf-8") as fh:
        doc = json.load(fh)
    quotes = load_quotes()
    px = apply_quotes(doc, quotes)
    raw = json.dumps(doc, separators=(",", ":"), ensure_ascii=False)
    with open(DATA_JSON, "w", encoding="utf-8") as fh:
        fh.write(raw)
    used = [sym for sym, row in px.items() if row.get("price_source") == "Vantage"]
    print(f"vantage: {len(used)} shifted, {len(px) - len(used)} on the existing feed, {len(quotes)} fresh quotes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
