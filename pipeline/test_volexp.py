"""Gold 4h volatility-expansion gate (weekly tune-up 14 Nov 2026)."""
import os, sys, unittest
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import edge
def bars(rng_list):
    c = 100 + np.cumsum(np.zeros(len(rng_list)))
    r = np.array(rng_list, float)
    return pd.DataFrame({"o": c, "h": c + r / 2, "l": c - r / 2, "c": c, "v": 0.0})
class VolExp(unittest.TestCase):
    def test_ok(self):
        self.assertTrue(edge.volexp_ok(bars([1.0] * 30 + [3.0] * 5)))    # expansion
        self.assertFalse(edge.volexp_ok(bars([3.0] * 30 + [1.0] * 5)))   # contraction
        self.assertFalse(edge.volexp_ok(bars([1.0] * 10)))               # too short
    def test_gate(self):
        self.assertEqual(edge.apply_volexp("BUY", "XAUUSD", "4h", False), ("WAIT", "volexp"))
        self.assertEqual(edge.apply_volexp("SELL", "XAUUSD", "4h", False), ("WAIT", "volexp"))
        self.assertEqual(edge.apply_volexp("SELL", "XAUUSD", "4h", True), ("SELL", ""))
        self.assertEqual(edge.apply_volexp("BUY", "XAUUSD", "1h", False), ("BUY", ""))    # gold 1h not gated
        self.assertEqual(edge.apply_volexp("BUY", "BTCUSD", "4h", False), ("BUY", ""))    # BTC not gated
        self.assertEqual(edge.apply_volexp("WAIT", "XAUUSD", "4h", False), ("WAIT", ""))
        self.assertEqual(edge.VOLEXP_GATE, ("XAUUSD|4h",))
if __name__ == "__main__": unittest.main()
