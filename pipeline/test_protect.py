"""Range coil, sweep-then-breakout, and the USD news window. No network."""
import json, os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import edge
from vantage import shift_flt

def frame(rows):
    t0 = 1_700_000_000_000
    step = 900_000
    rec = []
    for i, (h, l, c) in enumerate(rows):
        rec.append((t0 + i * step, c, h, l, c))
    return pd.DataFrame(rec, columns=["t", "o", "h", "l", "c"])

class RangeTest(unittest.TestCase):
    def test_coil_is_tight_and_breakout_is_not(self):
        rows = []
        for i in range(40):
            rows.append((101.0, 100.0, 100.4 + (i % 3) * 0.05))
        df = frame(rows)
        tight, hi, lo = edge.range_series(df, "15m", "XAUUSD")
        self.assertTrue(bool(tight[-2]))
        self.assertLess(lo[-2], 100.4)
        self.assertGreater(hi[-2], 100.4)
        df.loc[df.index[-1], "c"] = 130.0
        df.loc[df.index[-1], "h"] = 130.0
        tight, _, _ = edge.range_series(df, "15m", "XAUUSD")
        self.assertFalse(bool(tight[-1]))

    def test_wide_trend_is_not_a_coil(self):
        rows = [(100 + i, 99 + i, 99.5 + i) for i in range(40)]
        tight, _, _ = edge.range_series(frame(rows), "15m", "XAUUSD")
        self.assertFalse(bool(tight[-1]))

class SweepTest(unittest.TestCase):
    def test_downside_sweep_then_upside_break_blocks_shorts(self):
        rows = [(101.0, 100.0, 100.5)] * 24
        rows.append((100.6, 99.0, 100.4))  # wick through the low, close back inside
        rows.append((102.5, 100.4, 102.2))  # close through the prior high
        df = frame(rows)
        bL, bS, lo, hi = edge.sweep_series(df, "15m", "XAUUSD")
        self.assertTrue(bool(bS[-1]))
        self.assertFalse(bool(bL[-1]))
        self.assertLessEqual(float(lo[-1]), 100.0)
        self.assertGreater(float(hi[-1]), 100.5)
        gate = {"range": {"on": False}, "sweep": {"blockLong": False, "blockShort": True}, "news": {"on": False}}
        old = edge.SWEEP_ALERTS
        try:
            edge.SWEEP_ALERTS = False
            self.assertEqual(edge.apply_protect("SELL", gate), ("SELL", ""))
            edge.SWEEP_ALERTS = True
            self.assertEqual(edge.apply_protect("SELL", gate), ("WAIT", "sweep"))
            self.assertEqual(edge.apply_protect("BUY", gate)[0], "BUY")
        finally:
            edge.SWEEP_ALERTS = old

class NewsTest(unittest.TestCase):
    def test_window_is_direction_neutral(self):
        ev = [{"title": "CPI", "at": "2026-11-02T13:30:00+00:00", "ms": 1_000_000}]
        self.assertEqual(edge.news_hit(1_000_000, ev)["title"], "CPI")
        self.assertEqual(edge.news_hit(1_000_000 + 31 * 60 * 1000, ev), None)
        self.assertIsNone(edge.news_hit(1_000_000, []))

class ResearchFlagTest(unittest.TestCase):
    def test_alert_flags_match_the_unseen_pass(self):
        with open(os.path.join(os.path.dirname(__file__), "research", "protect_oos.json")) as fh:
            doc = json.load(fh)
        live = set(doc["live_for_alerts"])
        self.assertEqual(edge.RANGE_ALERTS, "range" in live)
        self.assertEqual(edge.SWEEP_ALERTS, "sweep" in live)
        self.assertFalse(edge.NEWS_ALERTS)
        self.assertTrue(doc["nov2"]["flagged_04h_sell_bar"])
        self.assertIn("Mon 02 Nov 14:15", doc["nov2"]["m15_short_blocks_starting"])

class ShiftTest(unittest.TestCase):
    def test_vantage_offset_moves_breakout_levels(self):
        flt = {"range": {"on": True, "lo": 3900.0, "hi": 3924.0}, "sweep": {"on": True, "lo": 3907.8, "hi": 3916.4}}
        shift_flt(flt, -12.5, 2)
        self.assertEqual(flt["range"]["lo"], 3887.5)
        self.assertEqual(flt["range"]["hi"], 3911.5)
        self.assertEqual(flt["sweep"]["lo"], 3895.3)
        self.assertEqual(flt["sweep"]["hi"], 3903.9)

if __name__ == "__main__":
    unittest.main()
