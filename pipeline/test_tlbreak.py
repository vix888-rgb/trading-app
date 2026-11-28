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
        self.assertIn("signal-desk-v17", sw)
if __name__ == "__main__": unittest.main()
