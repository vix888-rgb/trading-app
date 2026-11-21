"""Weekly tune-up 21 Nov 2026 (round 5): USDCHF 1D leaves the phone-alert list; gold/BTC alert combos unchanged."""
import os, sys, json, tempfile, unittest, importlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OLD = ["XAUUSD|1d", "XAUUSD|4h", "XAUUSD|1h", "BTCUSD|1d", "USDJPY|1d", "USDCHF|1d", "EURUSD|15m"]
class Round5(unittest.TestCase):
    def load(self, W):
        d = tempfile.mkdtemp(); json.dump(W, open(os.path.join(d, "weights.json"), "w"))
        os.environ["TRADING_STATE"] = d; os.environ.pop("TRADING_REFIT", None)
        import paths; importlib.reload(paths)
        import engine3; return importlib.reload(engine3)
    def test_stored_round4_is_migrated(self):
        e = self.load({"params": {"params_round": 4, "tested_edge": OLD}})
        self.assertNotIn("USDCHF|1d", e.PRM["tested_edge"])
        self.assertEqual(e.PRM["params_round"], 5)
        for k in ("XAUUSD|1d", "XAUUSD|4h", "XAUUSD|1h", "BTCUSD|1d", "USDJPY|1d"):
            self.assertIn(k, e.PRM["tested_edge"])
        self.assertNotIn("BTCUSD|1h", e.PRM["tested_edge"])
    def test_fresh_state(self):
        e = self.load({})
        self.assertNotIn("USDCHF|1d", e.PRM["tested_edge"])
    def test_round5_respected(self):
        keep = ["XAUUSD|1d", "XAUUSD|4h"]
        e = self.load({"params": {"params_round": 5, "tested_edge": keep}})
        self.assertEqual(e.PRM["tested_edge"], keep)
if __name__ == "__main__": unittest.main()
