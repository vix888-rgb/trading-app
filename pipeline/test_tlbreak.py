"""Trendline break (display-only module): causality, exit model, 5m log isolation from 15m and from the alert logic."""
import os, sys, json, tempfile, unittest
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tlbreak as T

def series(n=420, seed=3, start="2026-11-24 00:00"):
    """Synthetic 15m gold random walk with waves, UTC ms timestamps."""
    rng = np.random.default_rng(seed); t0 = int(pd.Timestamp(start, tz="Europe/Belgrade").timestamp() * 1000)
    x = 3600 + np.cumsum(rng.normal(0, 1.2, n) + 3 * np.sin(np.arange(n) / 9.0))
    o = np.r_[x[0], x[:-1]]; c = x
    h = np.maximum(o, c) + rng.random(n) * 1.5; l = np.minimum(o, c) - rng.random(n) * 1.5
    return pd.DataFrame({"t": t0 + np.arange(n) * 900_000, "o": o, "h": h, "l": l, "c": c, "v": 0.0})

class TL(unittest.TestCase):
    def test_signals_are_causal(self):
        df = series(); full, _ = T.signals(T.frame(df))
        self.assertGreater(len(full), 3)
        for cut in (200, 260, 330):
            part, _ = T.signals(T.frame(df.iloc[:cut]))
            ref = [(s["i"], s["side"], round(s["sl"], 6)) for s in full if s["i"] < cut]
            self.assertEqual([(s["i"], s["side"], round(s["sl"], 6)) for s in part], ref)
    def test_exit_model(self):
        # BUY at 100, SL 90 (R=10): TP1 110, TP2 120, TP3 130
        n = 6; base = int(pd.Timestamp("2026-11-26 10:00", tz="Europe/Belgrade").timestamp() * 1000)
        def mk(hs, ls, cs): return T.frame(pd.DataFrame({"t": base + np.arange(n) * 900_000, "o": cs, "h": hs, "l": ls, "c": cs}))
        d = mk([101, 131, 131, 131, 131, 131], [99, 99, 99, 99, 99, 99], [100] * 6)       # all targets in one bar
        r = T.simulate(d, 100.0, 1, 90.0, 1); self.assertEqual(r["kind"], "tp3"); self.assertAlmostEqual(r["R"], 2.0)
        d = mk([101, 111, 111, 111, 111, 111], [99, 99, 99, 99, 99, 99], [100] * 6)       # TP1 only, then SL at entry? low 99 < 100
        r = T.simulate(d, 100.0, 1, 90.0, 1); self.assertEqual(r["kind"], "be"); self.assertAlmostEqual(r["R"], 1 / 3)
        d = mk([101, 101, 101, 101, 101, 101], [99, 89, 99, 99, 99, 99], [100] * 6)       # SL first
        r = T.simulate(d, 100.0, 1, 90.0, 1); self.assertEqual(r["kind"], "sl"); self.assertAlmostEqual(r["R"], -1.0)
        d = mk([101] * 6, [99] * 6, [100] * 6)
        r = T.simulate(d, 100.0, -1, 110.0, 1); self.assertEqual(r["kind"], "open")      # nothing hit, still open at the last bar
    def test_panel_and_log(self):
        df = series(n=900); d = tempfile.mkdtemp()
        p = T.panel(df, d, now_ms=int(df.t.iloc[300]))
        json.dumps(p); self.assertTrue(p["displayOnly"]); self.assertEqual(p["lot"], 0.02)
        for r in p["sigs"]:
            if r["status"] != "skipped":
                self.assertAlmostEqual(r["riskUsd"], r["risk"] * 2.0, delta=0.02)
                self.assertEqual(len(r["tp"]), 3)
        log = json.load(open(os.path.join(d, "tl15_log.json")))
        self.assertTrue(all(int(k) >= int(df.t.iloc[300]) - T.STEP_MS for k in log["trades"]))
        before = dict(log["trades"]); T.panel(df, d, now_ms=int(df.t.iloc[301]))      # second scan keeps `since` and closed trades
        log2 = json.load(open(os.path.join(d, "tl15_log.json")))
        self.assertEqual(log2["since"], log["since"])
        for k, v in before.items():
            if v["status"] not in ("pending", "active"): self.assertEqual(log2["trades"][k], v)
    def test_isolated_from_alerts(self):
        import alerts
        doc = {"scan": [["XAUUSD", "1h", 10, 1, 0, 0, 0, 0, [], "x", 2, "BUY", True, ""]], "tl15": {"sigs": [{"side": "SELL"}]}, "tl5": {"sigs": [{"side": "BUY"}]}}
        self.assertEqual(alerts.gated_map(doc), {("XAUUSD", "1h"): "BUY"})
        src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tlbreak.py")).read()
        self.assertNotIn("weights", src.split('"""')[2])                                # code never touches weights
    def series5(self, n=420, seed=3, start="2026-11-24 00:00"):
        """Same price path as the 15m fixture, stamped every 5 minutes."""
        df = series(n=n, seed=seed, start=start).copy()
        df["t"] = int(df.t.iloc[0]) + np.arange(len(df)) * T.TF_STEP["5m"]
        return df
    def test_5m_signals_are_causal(self):
        self.assertEqual(T.TF_STEP["5m"], 300_000)
        self.assertEqual((T.W, T.MIN_GAP, T.MAX_AGE, T.SL_ATR, T.MAX_ATR, T.WIN, T.MAXTR, T.LOT), (3, 8, 150, 1.5, 3.5, (1.0, 20.5), 3, 0.02))
        df = self.series5(); full, _ = T.signals(T.frame(df))
        self.assertGreater(len(full), 3)
        for cut in (200, 260, 330):
            part, _ = T.signals(T.frame(df.iloc[:cut]))
            ref = [(s["i"], s["side"], round(s["sl"], 6)) for s in full if s["i"] < cut]
            self.assertEqual([(s["i"], s["side"], round(s["sl"], 6)) for s in part], ref)
        p = T.panel(df, now_ms=int(df.t.iloc[-1]))
        self.assertEqual(p["tf"], "15m")  # default stays 15m when the caller does not pass tf
        p5 = T.panel(df, tf="5m", now_ms=int(df.t.iloc[-1]))
        self.assertEqual(p5["tf"], "5m"); self.assertEqual(p5["asof"], T.fmt_t(int(df.t.iloc[-1]) + 300_000))
        self.assertEqual(p5["bt"]["note"], "backtest: 5m +0.13R/trade vs 15m +0.45R")
        self.assertTrue(p5["displayOnly"]); self.assertEqual(p5["lot"], 0.02)
    def test_5m_log_is_isolated(self):
        df15 = series(n=900); df5 = self.series5(n=900); d = tempfile.mkdtemp()
        now15 = int(df15.t.iloc[300]); now5 = int(df5.t.iloc[400])
        p15 = T.panel(df15, d, now_ms=now15, tf="15m")
        p5 = T.panel(df5, d, now_ms=now5, tf="5m")
        self.assertEqual(p15["tf"], "15m"); self.assertEqual(p5["tf"], "5m")
        self.assertTrue(os.path.exists(os.path.join(d, "tl15_log.json")))
        self.assertTrue(os.path.exists(os.path.join(d, "tl5_log.json")))
        log15 = json.load(open(os.path.join(d, "tl15_log.json")))
        log5 = json.load(open(os.path.join(d, "tl5_log.json")))
        self.assertNotEqual(log15["note"], log5["note"])
        self.assertIn("5m", log5["note"]); self.assertIn("15m", log15["note"])
        self.assertTrue(all(int(k) >= now5 - T.TF_STEP["5m"] for k in log5["trades"]))
        snap15 = json.dumps(log15, sort_keys=True); snap5 = json.dumps(log5, sort_keys=True)
        T.panel(df15, d, now_ms=int(df15.t.iloc[301]), tf="15m")   # 15m rescan must not touch the 5m log
        self.assertEqual(json.dumps(json.load(open(os.path.join(d, "tl5_log.json"))), sort_keys=True), snap5)
        before = dict(log5["trades"]); T.panel(df5, d, now_ms=int(df5.t.iloc[401]), tf="5m")
        log5b = json.load(open(os.path.join(d, "tl5_log.json")))
        self.assertEqual(log5b["since"], log5["since"])
        for k, v in before.items():
            if v["status"] not in ("pending", "active"): self.assertEqual(log5b["trades"][k], v)
        log15b = json.load(open(os.path.join(d, "tl15_log.json")))
        self.assertEqual(json.dumps(log15b, sort_keys=True), snap15)   # 5m rescan must not touch the 15m log
        alone = tempfile.mkdtemp(); T.panel(df5, alone, now_ms=now5, tf="5m")
        self.assertTrue(os.path.exists(os.path.join(alone, "tl5_log.json")))
        self.assertFalse(os.path.exists(os.path.join(alone, "tl15_log.json")))
        root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
        html = open(os.path.join(root, "index.html")).read(); sw = open(os.path.join(root, "sw.js")).read()
        self.assertIn('#tl5', html); self.assertIn("backtest: 5m +0.13R/trade vs 15m +0.45R", html)
        self.assertIn("signal-desk-v19", sw)
        self.assertIn("Njoftimet: vetëm ari", html)
        self.assertIn('"?sym="', html)
        self.assertIn('data-tlsym="XAUUSD"', html)
        self.assertIn('data-tlsym="BTCUSD"', html)
        self.assertLess(html.index('data-tlsym="XAUUSD"'), html.index('data-tlsym="BTCUSD"'))

def swing_frame(start="2026-11-26 10:00", step=900_000, n=40, cross=False, close_at=None):
    """Two lower highs, then either a flat approach or one close up through the falling line."""
    t0 = int(pd.Timestamp(start, tz="Europe/Belgrade").timestamp() * 1000)
    c = np.full(n, 100.0); h = np.full(n, 100.4); l = np.full(n, 99.6); o = c.copy()
    h[5], h[16] = 102.0, 101.0
    if cross:
        c[21], c[22] = 100.40, 100.70
        h[22] = 100.80
        o[22] = c[21]; o[23] = c[22]
    if close_at is not None:
        c[-1] = close_at
        h[-1] = max(h[-1], close_at + 0.15)
        l[-1] = min(l[-1], close_at - 0.15)
        o[-1] = close_at
    return pd.DataFrame({"t": t0 + np.arange(n) * step, "o": o, "h": h, "l": l, "c": c, "v": 0.0})

class TLAlerts(unittest.TestCase):
    def test_pre_alert_proximity(self):
        base = swing_frame()
        d = T.frame(base)
        watch = [x for x in T.watch_lines(d) if x["side"] == "BUY"]
        self.assertEqual(len(watch), 1)
        line, atr = watch[0]["line"], watch[0]["atr"]
        self.assertGreater(atr, 0)
        near = swing_frame(close_at=line - 0.4 * atr)
        got = T.pre_candidates(T.frame(near), "15m")
        self.assertEqual(len(got), 1)
        body = got[0]["body"]
        self.assertTrue(body.startswith("TL15 gati BUY XAUUSD — vija ~"))
        self.assertIn("çmimi ", body)
        self.assertIn("Prit mbylljen mbi ", body)
        self.assertIn("SL ", body)
        self.assertNotIn("skip", body)
        self.assertEqual(got[0]["priority"], "default")
        far = swing_frame(close_at=line - 0.6 * atr)
        self.assertEqual(T.pre_candidates(T.frame(far), "15m"), [])
        above = swing_frame(close_at=line + 0.2 * atr)   # already through the line: not a pre-alert
        self.assertEqual(T.pre_candidates(T.frame(above), "15m"), [])
        self.assertTrue(T.near_line(100, 100.4, 1))
        self.assertFalse(T.near_line(100, 100.6, 1))
        skip = T.pre_message("15m", "BUY", 100, 99.5, 50, True)
        self.assertTrue(skip.endswith("skip"))
        self.assertIn("SL 50.00", skip)
        self.assertTrue(T.risk_skip(99.5, 50, 10))
        self.assertFalse(T.risk_skip(99.5, 98, 10))
        five = T.pre_candidates(T.frame(swing_frame(step=300_000, close_at=line - 0.2 * atr)), "5m")
        self.assertEqual(len(five), 1)
        self.assertTrue(five[0]["body"].startswith("TL5 (TEST) gati BUY XAUUSD"))
        self.assertEqual(five[0]["priority"], "min")

    def test_entry_alert_and_session_window(self):
        inside = swing_frame(cross=True)
        d = T.frame(inside)
        rows = T.run(d, T.TF_STEP["15m"])
        self.assertTrue(any(r["status"] != "skipped" and r["side"] == "BUY" for r in rows))
        got = T.entry_candidates(d, "15m")
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]["priority"], "high")
        self.assertRegex(got[0]["body"], r"^TL15 ENTRY BUY XAUUSD @ \d+\.\d{2}, SL \d+\.\d{2}, TP1 1R / TP2 2R / TP3 3R, 0.02 lot, BE pas TP1\.$")
        outside = swing_frame(start="2026-11-26 18:00", cross=True)
        od = T.frame(outside)
        orows = T.run(od, T.TF_STEP["15m"])
        self.assertTrue(any(r.get("why") == "outside 01:00–20:30" for r in orows))
        self.assertEqual(T.entry_candidates(od, "15m"), [])
        five = T.entry_candidates(T.frame(swing_frame(step=300_000, cross=True)), "5m")
        self.assertEqual(len(five), 1)
        self.assertTrue(five[0]["body"].startswith("TL5 (TEST) ENTRY BUY XAUUSD @ "))
        self.assertEqual(five[0]["priority"], "low")
        self.assertLess({"min": 1, "low": 2, "default": 3, "high": 4}[five[0]["priority"]],
                        {"min": 1, "low": 2, "default": 3, "high": 4}[got[0]["priority"]])

    def test_dedup_and_seed_no_storm(self):
        df = swing_frame(cross=True)
        state = tempfile.mkdtemp()
        box = []
        first = T.dispatch({"15m": df}, state, send_fn=box.append)
        self.assertEqual(first, [])
        self.assertEqual(box, [])
        sent = json.load(open(os.path.join(state, "tl_alerts_sent.json")))
        self.assertGreater(len(sent["ids"]), 0)
        self.assertTrue(any("|entry|" in x for x in sent["ids"]))
        again = T.dispatch({"15m": df}, state, send_fn=box.append)
        self.assertEqual(again, [])
        self.assertEqual(box, [])
        # An initialized empty book sends the live entry once, then never again.
        state2 = tempfile.mkdtemp()
        json.dump({"v": 1, "ids": []}, open(os.path.join(state2, "tl_alerts_sent.json"), "w"))
        live = T.dispatch({"15m": df, "5m": swing_frame(step=300_000, cross=True)}, state2, send_fn=box.append)
        self.assertEqual(len(live), 2)
        self.assertEqual({x["tf"] for x in live}, {"15m", "5m"})
        self.assertEqual(T.dispatch({"15m": df, "5m": swing_frame(step=300_000, cross=True)}, state2, send_fn=box.append), [])
        self.assertEqual(len(box), 2)
        ids = json.load(open(os.path.join(state2, "tl_alerts_sent.json")))["ids"]
        self.assertEqual(len(ids), len(set(ids)))
        # Historical random-walk signals are seeded, not pushed.
        storm = tempfile.mkdtemp()
        pushed = []
        none = T.dispatch({"15m": series(n=420)}, storm, send_fn=pushed.append)
        self.assertEqual(none, [])
        self.assertEqual(pushed, [])
        self.assertGreater(len(json.load(open(os.path.join(storm, "tl_alerts_sent.json")))["ids"]), 0)
        self.assertFalse(os.path.exists(os.path.join(storm, "alerts_sent.json")))

    def test_nongold_is_display_only_and_never_alerted(self):
        df = series(n=420)
        state = tempfile.mkdtemp()
        p = T.panel(df, state, now_ms=int(df.t.iloc[300]), tf="15m", sym="BTCUSD")
        self.assertEqual(p["sym"], "BTCUSD")
        self.assertFalse(p["phone"])
        self.assertIsNone(p["lot"])
        self.assertNotIn("oz", p)
        self.assertTrue(p["displayOnly"])
        self.assertEqual(p["p"], 2)
        for r in p["sigs"]:
            self.assertNotIn("riskUsd", r)
            self.assertNotIn("usd", r)
            self.assertEqual(len(r["tp"]), 3)
            self.assertIn("R", r)
        self.assertTrue(os.path.exists(os.path.join(state, "tl15_BTCUSD_log.json")))
        self.assertFalse(os.path.exists(os.path.join(state, "tl15_log.json")))
        eur = T.panel(df, now_ms=int(df.t.iloc[-1]), tf="5m", sym="EURUSD")
        self.assertEqual(eur["p"], 5)
        self.assertFalse(eur["phone"])
        jpy = T.panel(df, now_ms=int(df.t.iloc[-1]), tf="15m", sym="USDJPY")
        self.assertEqual(jpy["p"], 3)
        # Missing frames are skipped. Gold keeps its own log. Nested books do not alert BTC or EUR.
        books = {"BTCUSD": {"15m": df}, "EURUSD": {}, "USDCHF": {"5m": df}, "XAUUSD": {"15m": series(n=200, seed=3)}}
        gold, by, failed = T.build(books, state, now_ms=int(df.t.iloc[300]))
        self.assertIn("XAUUSD", by["15m"])
        self.assertIn("BTCUSD", by["15m"])
        self.assertIn("USDCHF", by["5m"])
        self.assertNotIn("EURUSD", by["15m"])
        self.assertNotIn("EURUSD", by["5m"])
        self.assertEqual(gold["15m"]["sym"], "XAUUSD")
        self.assertEqual(gold["15m"]["lot"], 0.02)
        self.assertTrue(os.path.exists(os.path.join(state, "tl15_log.json")))
        box = []
        sent_path = os.path.join(state, "tl_alerts_sent.json")
        json.dump({"v": 1, "ids": []}, open(sent_path, "w"))
        only_fx = {"BTCUSD": {"15m": df, "5m": df}, "EURUSD": {"15m": df}, "USDJPY": {"5m": df}, "USDCHF": {"15m": df}, "EURCHF": {"5m": df}}
        self.assertEqual(T.collect_alerts(only_fx), [])
        self.assertEqual(T.dispatch(only_fx, state, send_fn=box.append), [])
        self.assertEqual(box, [])
        mixed = dict(only_fx)
        mixed["XAUUSD"] = {"15m": swing_frame(cross=True), "5m": swing_frame(step=300_000, cross=True)}
        live = T.dispatch(mixed, state, send_fn=box.append)
        self.assertGreater(len(live), 0)
        self.assertTrue(all("XAUUSD" in it["body"] and it["body"].find("BTC") < 0 and "EUR" not in it["body"] and "JPY" not in it["body"] and "CHF" not in it["body"] for it in live))
        self.assertTrue(all(it["tf"] in ("15m", "5m") for it in live))
if __name__ == "__main__": unittest.main()
