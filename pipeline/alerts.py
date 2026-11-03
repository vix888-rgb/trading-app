#!/usr/bin/env python3
"""Alert when a gated 1h / 4h / 1D signal turns BUY or SELL, and decide whether to commit.

Compares scan[i][11] in the new data.json with the previous snapshot. 5m and 15m
(and 1m) never alert. An unchanged BUY or SELL does not alert again.
"""
import argparse, json, math, os, subprocess, sys
from datetime import datetime
from zoneinfo import ZoneInfo

TZ = ZoneInfo("Europe/Belgrade")
ALERT_TFS = {"1h", "4h", "1d"}
TF_LABEL = {"1h": "1h", "4h": "4h", "1d": "1D"}
NTFY_URL = os.environ.get("NTFY_URL", "https://ntfy.sh/octa-signals-a6368a35")
CLICK = "https://vix888-rgb.github.io/trading-app/"
# Rebuild (Nov 2026): phone alerts only for combos with a tested edge (set ALERT_EDGE_ONLY=0 to alert every 1h/4h/1D combo),
# and never twice for the same symbol / timeframe / bar / side (state/alerts_sent.json), even if a signal flips WAIT and back.
EDGE_ONLY = os.environ.get("ALERT_EDGE_ONLY", "1") != "0"
SENT_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state", "alerts_sent.json")
BIN = {"BTCUSD", "ETHUSD", "SOLUSD", "XRPUSD"}


def gated_map(doc):
    out = {}
    for row in doc.get("scan") or []:
        if not isinstance(row, list) or len(row) < 12 or row[2] is None:
            continue
        out[(row[0], str(row[1]).lower())] = row[11]
    return out


def parse_gen(text):
    return datetime.strptime(text, "%a %d %b %Y, %H:%M").replace(tzinfo=TZ)


def should_commit(changed, prev):
    if changed:
        return True
    try:
        age = (datetime.now(TZ) - parse_gen(prev.get("gen") or "")).total_seconds()
    except Exception:
        return True
    return age >= 3600


def alert_key(doc, sym, tf, sig):
    det = (doc.get("det") or {}).get(f"{sym}|{tf}") or {}
    return f"{sym}|{tf}|{det.get('lt')}|{sig}"


def load_sent(path):
    try:
        return json.load(open(path))
    except Exception:
        return []


def new_signals(prev, new, sent=None, edge_only=EDGE_ONLY):
    old, cur = gated_map(prev), gated_map(new)
    sent = set(sent or [])
    found = []
    for (sym, tf), sig in sorted(cur.items()):
        if tf not in ALERT_TFS or sig not in ("BUY", "SELL"):
            continue
        if old.get((sym, tf)) == sig:
            continue
        det = (new.get("det") or {}).get(f"{sym}|{tf}") or {}
        if edge_only and not det.get("edge"):
            print(f"skip {sym} {tf} {sig}: no tested edge (ALERT_EDGE_ONLY=1)")
            continue
        if alert_key(new, sym, tf, sig) in sent:
            print(f"skip {sym} {tf} {sig}: already alerted for this bar")
            continue
        found.append((sym, tf, sig, old.get((sym, tf))))
    return found


def spec(sym, rates):
    if sym == "XAUUSD":
        return 100.0, "lots", 0.01
    if sym in BIN:
        step = 0.0001 if sym == "BTCUSD" else 0.001 if sym == "ETHUSD" else 0.01
        return 1.0, sym[:3], step
    if sym == "USDJPY":
        rate = (rates or {}).get("USDJPY")
        if not rate:
            return None
        return 1e5 / float(rate), "lots", 0.01
    if sym in ("USDCHF", "EURCHF"):
        rate = (rates or {}).get("USDCHF")
        if not rate:
            return None
        return 1e5 / float(rate), "lots", 0.01
    return 1e5, "lots", 0.01


def fmt_px(x, p):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{float(x):,.{int(p)}f}"


def lot_for_20(sym, entry, stop, rates):
    info = spec(sym, rates)
    if not info or entry is None or stop is None:
        return "n/a"
    per, unit, step = info
    dist = abs(float(entry) - float(stop))
    if dist <= 0 or per <= 0:
        return "n/a"
    raw = 20.0 / (dist * per)
    size = math.floor(raw / step + 1e-9) * step
    if size <= 0:
        return f"below minimum {step:g} {unit}"
    decimals = 4 if step < 0.001 else 3 if step < 0.01 else 2
    return f"{size:.{decimals}f} {unit}"


def message_for(doc, sym, tf, sig):
    det = (doc.get("det") or {}).get(f"{sym}|{tf}") or {}
    pl = det.get("pl") or []
    p = det.get("p", 2)
    entry = pl[1] if len(pl) > 4 else None
    stop = pl[4] if len(pl) > 4 else None
    tps = pl[7] if len(pl) > 7 and isinstance(pl[7], list) else []
    while len(tps) < 3:
        tps.append(None)
    edge = bool(det.get("edge"))
    edge_txt = "tested edge: yes" if edge else "tested edge: no — UNTESTED EDGE (no proven edge after costs)"
    size = lot_for_20(sym, entry, stop, doc.get("rates") or {})
    body = (
        f"Entry {fmt_px(entry, p)}, SL {fmt_px(stop, p)}, "
        f"TP1/TP2/TP3 {fmt_px(tps[0], p)}/{fmt_px(tps[1], p)}/{fmt_px(tps[2], p)}, "
        f"lot size for $20 risk: {size}, {edge_txt}"
    )
    if len(pl) > 3 and str(pl[3]).startswith("market (tested plan"):
        body += ", market entry; SL to breakeven at +1R, then 4×ATR trail"
    title = f"{sym} {TF_LABEL.get(tf, tf)} {sig}"
    return title, body


def send(title, body, url):
    proc = subprocess.run(
        [
            "curl", "-fsS", "--retry", "3", "--retry-delay", "2", "--max-time", "30",
            "-H", f"Title: {title}",
            "-H", "Tags: chart_with_upwards_trend",
            "-H", f"Click: {CLICK}",
            "-d", body,
            url,
        ],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"ntfy failed for {title}: {err}")
    return (proc.stdout or "").strip()


def report_sources(snap_path):
    if not snap_path or not os.path.exists(snap_path):
        print("DATA SOURCES: snap3 not found, no source report")
        return []
    doc = json.load(open(snap_path))
    fails = []
    for row in doc.get("priceStatus") or []:
        if isinstance(row, list) and len(row) >= 5 and not row[2]:
            fails.append(f"price {row[0]} {row[1]}: {row[4]}")
        elif isinstance(row, dict) and not row.get("ok"):
            fails.append(f"price {row.get('sym')} {row.get('tf')}: {row.get('err')}")
    for row in doc.get("newsStatus") or []:
        if isinstance(row, dict) and not row.get("ok"):
            fails.append(f"news {row.get('source')}: {row.get('err')}")
    if fails:
        print(f"DATA SOURCE FAILURES ({len(fails)})")
        for item in fails:
            print(" -", item)
    else:
        print("DATA SOURCES: all reported ok")
    return fails


def write_output(commit, changed, alerts, failed):
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(f"commit={'true' if commit else 'false'}\n")
        fh.write(f"signals_changed={'true' if changed else 'false'}\n")
        fh.write(f"alerts={alerts}\n")
        fh.write(f"alert_failed={'true' if failed else 'false'}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prev", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--snap")
    ap.add_argument("--send", action="store_true")
    ap.add_argument("--url", default=NTFY_URL)
    ap.add_argument("--sent", default=SENT_DEFAULT)
    args = ap.parse_args()
    prev = json.load(open(args.prev))
    new = json.load(open(args.new))
    changed_map = gated_map(prev) != gated_map(new)
    commit = should_commit(changed_map, prev)
    sent = load_sent(args.sent)
    alerts = new_signals(prev, new, sent)
    print(f"signals_changed={changed_map} commit={commit} new_buy_sell={len(alerts)} gen={new.get('gen')}")
    report_sources(args.snap)
    failed = 0
    for sym, tf, sig, was in alerts:
        title, body = message_for(new, sym, tf, sig)
        print(f"ALERT {title} (was {was})")
        print(body)
        if args.send:
            try:
                send(title, body, args.url)
                print(f"sent {title}")
                sent.append(alert_key(new, sym, tf, sig))
            except Exception as exc:
                failed += 1
                print(f"SEND FAILED {title}: {exc}", file=sys.stderr)
    if args.send:
        try:
            json.dump(sent[-500:], open(args.sent, "w"), indent=0)
        except Exception as exc:
            print(f"could not save {args.sent}: {exc}", file=sys.stderr)
    write_output(commit, changed_map, len(alerts), failed > 0)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
