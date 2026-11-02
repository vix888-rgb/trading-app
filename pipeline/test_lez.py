"""Liquidity Entry Zones on the point scale. No network."""
import json, os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import edge
from vantage import shift_st

def frame(o, h, l, c):
    n = len(c)
    t = 1_700_000_000_000 + np.arange(n) * 3_600_000
    return pd.DataFrame({"t": t, "o": o, "h": h, "l": l, "c": c, "v": np.ones(n)})

def rising(n=90):
    c = 2300 + np.arange(n) * 0.05
    o = c - 0.15
    h = c + 0.35
    l = c - 0.40
    return o, h, l, c

class BracketTest(unittest.TestCase):
    def test_point_scale_gold_and_btc(self):
        self.assertEqual(edge.lez_bracket("XAUUSD"), (0.30, 0.20, 5.0, 13.0))
        self.assertEqual(edge.lez_bracket("BTCUSD"), (3.0, 2.0, 50.0, 130.0))
        self.assertAlmostEqual(13.0 / 5.0, 2.6)

    def test_spread_gate(self):
        # Gold $5 stop vs $0.35 spread + $0.10 slip survives the 25% rule and shaves 0.09R.
        self.assertAlmostEqual(edge.lez_net_r(2.6, 5.0, 0.45), 2.6 - 0.09)
        self.assertAlmostEqual(edge.lez_net_r(-1.0, 5.0, 0.45), -1.09)
        # BTC $50 stop vs a 0.06% round trip at 100k ($60) plus 2 bps ($20) does not.
        self.assertIsNone(edge.lez_net_r(2.6, 50.0, 80.0))
        self.assertIsNone(edge.lez_net_r(2.6, 50.0, 60.0))
        # At $10k the 0.06% round trip is $6, which still fits under a $50 stop.
        self.assertAlmostEqual(edge.lez_net_r(2.6, 50.0, 8.0), 2.6 - 0.16)

    def test_flags_match_the_unseen_pass(self):
        with open(os.path.join(os.path.dirname(__file__), "research", "lez_oos.json")) as fh:
            doc = json.load(fh)
        self.assertEqual(doc["integrate"], "display")
        self.assertEqual(edge.LEZ_ALERTS, doc["integrate"] == "standalone")
        self.assertEqual(edge.LEZ_FILTER, doc["integrate"] == "filter")
        self.assertEqual(edge.LEZ_MODE, doc["ship_mode"])
        self.assertEqual(edge.LEZ_WINDOW, doc["ship_window"])
        self.assertFalse(edge.LEZ_ALERTS)
        self.assertFalse(edge.LEZ_FILTER)
        self.assertEqual(edge.apply_lez("SELL", 1, "XAUUSD", "1h"), ("SELL", ""))


class SignalTest(unittest.TestCase):
    def test_long_confirms_on_the_sweep_and_pays_2_6r(self):
        o, h, l, c = rising()
        o[70], h[70], l[70], c[70] = 2303.2, 2303.6, 2301.0, 2303.3
        level = l[70]
        o[80], c[80], h[80] = 2303.6, 2304.4, 2304.8
        l[80] = level - 0.6
        # Climb $13 from the next open without touching the $5 stop.
        entry = 2304.4
        o[81] = entry
        for j in range(81, 88):
            o[j] = entry + (j - 81) * 2.5
            l[j] = o[j] - 0.2
            h[j] = o[j] + 2.6
            c[j] = o[j] + 1.0
        pack = edge.lez_run(frame(o, h, l, c), "XAUUSD", "1h", mode="point")
        sigs = [s for s in pack["signals"] if s["i"] == 80]
        self.assertTrue(sigs, [s["i"] for s in pack["signals"]])
        s = sigs[0]
        self.assertEqual(s["d"], 1)
        self.assertEqual(s["how"], "tp")
        self.assertAlmostEqual(s["gross"], 2.6, places=6)
        self.assertFalse(bool(pack["L"][79]))
        # Pivot is not known before its right side closes, so the level cannot fire early.
        early = edge.lez_run(frame(o, h, l, c).iloc[:76].reset_index(drop=True), "XAUUSD", "1h")
        self.assertFalse(any(s["i"] >= 70 for s in early["signals"]))

    def test_no_signal_without_a_reclaim_or_above_the_ema(self):
        o, h, l, c = rising()
        o[70], h[70], l[70], c[70] = 2303.2, 2303.6, 2301.0, 2303.3
        level = l[70]
        o[80], h[80], l[80], c[80] = level - 0.2, level + 0.2, level - 0.8, level - 0.1
        pack = edge.lez_run(frame(o, h, l, c), "XAUUSD", "1h")
        self.assertFalse(any(s["i"] == 80 for s in pack["signals"]))
        o, h, l, c = rising()
        o[70], h[70], l[70], c[70] = 2303.2, 2303.6, 2301.0, 2303.3
        level = l[70]
        # Close back inside and through the midline, but the whole candle is far under the EMA.
        o[80], c[80], h[80], l[80] = level + 0.2, level + 0.8, level + 1.0, level - 0.6
        pack = edge.lez_run(frame(o, h, l, c), "XAUUSD", "1h")
        self.assertFalse(any(s["i"] == 80 for s in pack["signals"]))

    def test_confirmation_is_the_sweep_bar_or_the_next_two_only(self):
        o, h, l, c = rising()
        o[70], h[70], l[70], c[70] = 2303.2, 2303.6, 2301.0, 2303.3
        level = l[70]
        # Reclaim with a bearish body, so the sweep bar itself is not the confirmation.
        o[80], c[80], h[80], l[80] = level + 2.0, level + 0.5, level + 2.2, level - 0.8
        mid = (h[80] + l[80]) / 2
        # Bar +1 stays under the midline. Bar +2 closes through it and is bullish, above the EMA.
        o[81], h[81], l[81], c[81] = 2304.0, 2304.3, 2303.5, mid - 0.05
        o[82], h[82], l[82], c[82] = 2304.0, 2305.0, 2303.8, 2304.8
        pack = edge.lez_run(frame(o, h, l, c), "XAUUSD", "1h")
        self.assertEqual([s["i"] for s in pack["signals"] if s["d"] == 1 and s["i"] >= 80], [82])
        # Three bars later is outside the window.
        o[82], h[82], l[82], c[82] = 2304.0, 2304.2, 2303.6, mid - 0.1
        o[83], h[83], l[83], c[83] = 2304.0, 2305.0, 2303.8, 2304.8
        pack = edge.lez_run(frame(o, h, l, c), "XAUUSD", "1h")
        self.assertFalse(any(s["i"] >= 80 for s in pack["signals"]))

    def test_vantage_offset_moves_the_strategy_bracket(self):
        st = [[1, 0, 4, 50, 0.2, 1.4, 0, 2300.0, 2295.0, 2313.0, 1]]
        shift_st(st, -12.5, 2)
        self.assertEqual(st[0][7:], [2287.5, 2282.5, 2300.5, 1])
        shift_st([[1, 0, 0, None, None, None, 0]], -1, 2)


if __name__ == "__main__":
    unittest.main()
