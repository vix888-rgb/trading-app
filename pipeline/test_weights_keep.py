"""Weekly tune-up 14 Nov 2026: a failed download must not drop a combo from weights.json, and a missing combo must not force a full refit."""
import os, sys, json, tempfile, unittest, importlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
class Keep(unittest.TestCase):
    def load(self, W):
        d = tempfile.mkdtemp(); json.dump(W, open(os.path.join(d, "weights.json"), "w"))
        os.environ["TRADING_STATE"] = d; os.environ.pop("TRADING_REFIT", None)
        import paths; importlib.reload(paths)
        import engine3; return importlib.reload(engine3)
    def test_keep_stored(self):
        e = self.load({})
        bc = {"A|1h": {"weights": {"x": 1}}, "B|4h": {"weights": {"x": 2}}}
        out = e.keep_stored({"A|1h": {"weights": {"x": 9}}}, bc)
        self.assertEqual(out["A|1h"]["weights"]["x"], 9)      # fresh value wins
        self.assertEqual(out["B|4h"]["weights"]["x"], 2)      # failed fetch keeps stored
        self.assertEqual(e.keep_stored({}, None), {})
    def test_missing_no_full_refit(self):
        from datetime import datetime
        W = {"refit_at": datetime.now().astimezone().isoformat(timespec="minutes"), "by_combo": {"A|1h": {}}}
        e = self.load(W)
        self.assertFalse(e.refit_due([("A", "1h"), ("B", "4h")]))
        self.assertEqual(e.missing_combos([("A", "1h"), ("B", "4h")]), [("B", "4h")])
    def test_env_overrides(self):
        e = self.load({"by_combo": {}})
        os.environ["TRADING_REFIT"] = "1"; self.assertTrue(e.refit_due([("A", "1h")]))
        os.environ["TRADING_REFIT"] = "0"; self.assertFalse(e.refit_due([("A", "1h")]))
        os.environ.pop("TRADING_REFIT")
        self.assertTrue(e.refit_due([("A", "1h")]))            # no refit_at -> refit
if __name__ == "__main__": unittest.main()
