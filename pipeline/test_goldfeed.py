"""Spot gold feed: bar alignment, credit budget, and Yahoo fallback. No network."""
import json, os, sys, tempfile, unittest
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import goldfeed

NOW = pd.Timestamp("2026-11-03 22:10:00", tz="UTC")
NOW_S = float(NOW.timestamp())
KEY = "supersecretkey123"


def series(step_ms, n, last_open_ms, price=3900.0):
    start = int(last_open_ms) - (n - 1) * step_ms
    t = start + np.arange(n) * step_ms
    c = price + np.arange(n) * 0.01
    return pd.DataFrame({"t": t.astype("int64"), "o": c, "h": c + 0.4, "l": c - 0.4, "c": c + 0.2, "v": np.zeros(n)})


def ms(ts):
    return int(pd.Timestamp(ts).timestamp() * 1000)


class GoldFeed(unittest.TestCase):
    def test_credit_budget(self):
        self.assertEqual(goldfeed.credits_per_day(), 336)
        self.assertLess(goldfeed.credits_per_day(), 500)

    def test_outputsize_is_small_once_history_is_cached(self):
        step = goldfeed.STEP["5min"]
        last = ms("2026-11-03 22:00:00+00:00")
        now = ms("2026-11-03 22:10:00+00:00")
        self.assertEqual(goldfeed.outputsize_for(None, step, now), 5000)
        self.assertEqual(goldfeed.outputsize_for(last, step, now, n_bars=10), 5000)
        warm = goldfeed.outputsize_for(last, step, now, n_bars=500)
        self.assertLessEqual(warm, 30)
        self.assertGreaterEqual(warm, 5)

    def test_hourly_is_not_fetched_every_scan(self):
        now = NOW_S
        last = ms("2026-11-03 21:00:00+00:00")
        self.assertFalse(goldfeed.h1_due(now - 600, now, 500, last))
        self.assertTrue(goldfeed.h1_due(now - 1800, now, 500, last))
        self.assertTrue(goldfeed.h1_due(None, now, 500, last))
        self.assertTrue(goldfeed.h1_due(now - 600, now, 10, last))

    def test_15m_and_4h_boundaries_match_the_old_clock(self):
        # 5m bars 21:00 through 21:25 UTC. now=21:30 closes the 21:00 and 21:15 buckets.
        m5 = series(300_000, 6, ms("2026-11-03 21:25:00+00:00"), 100)
        now = ms("2026-11-03 21:30:00+00:00")
        m15 = goldfeed.resample_ohlc(m5, "15min", step_ms=900_000, now_ms=now)
        self.assertEqual(list(m15.t), [ms("2026-11-03 21:00:00+00:00"), ms("2026-11-03 21:15:00+00:00")])
        self.assertAlmostEqual(float(m15.o.iloc[0]), float(m5.o.iloc[0]))
        self.assertAlmostEqual(float(m15.c.iloc[0]), float(m5.c.iloc[2]))
        self.assertAlmostEqual(float(m15.h.iloc[0]), float(m5.h.iloc[:3].max()))
        self.assertAlmostEqual(float(m15.l.iloc[0]), float(m5.l.iloc[:3].min()))
        # 4h: 16:00-19:00 is closed at 22:10; 20:00-22:00 is still forming.
        hours = series(3_600_000, 7, ms("2026-11-03 22:00:00+00:00"), 200)
        h4 = goldfeed.resample_ohlc(hours, "4h", step_ms=14_400_000, now_ms=ms(NOW))
        self.assertEqual(list(h4.t), [ms("2026-11-03 16:00:00+00:00")])
        self.assertTrue((h4.t % 14_400_000 == 0).all())

    def test_daily_uses_new_york_midnight_including_the_dst_fallback(self):
        # Hourly bars from 31 Oct 00:00 UTC through 3 Nov 12:00 UTC.
        start = ms("2026-10-31 00:00:00+00:00")
        end = ms("2026-11-03 12:00:00+00:00")
        n = (end - start) // 3_600_000 + 1
        hours = series(3_600_000, n, end, 300)
        now = ms("2026-11-04 12:00:00+00:00")
        daily = goldfeed.resample_ohlc(hours, "1D", now_ms=now, tz="America/New_York")
        stamps = [pd.Timestamp(int(t), unit="ms", tz="UTC").tz_convert("America/New_York") for t in daily.t]
        self.assertTrue(all(s.hour == 0 and s.minute == 0 for s in stamps))
        # 1 Nov 2026 is the US fall-back. That daily bar opens 04:00 UTC and the next opens 05:00 UTC.
        opens = {s.strftime("%Y-%m-%d"): int(t) for s, t in zip(stamps, daily.t)}
        self.assertEqual(opens["2026-11-01"], ms("2026-11-01 04:00:00+00:00"))
        self.assertEqual(opens["2026-11-02"], ms("2026-11-02 05:00:00+00:00"))
        # 04:00 UTC on 2 Nov is still 1 Nov in New York, so it belongs to the 1 Nov bar.
        bar = hours[hours.t == ms("2026-11-02 04:00:00+00:00")].iloc[0]
        nov1 = daily[daily.t == opens["2026-11-01"]].iloc[0]
        self.assertAlmostEqual(float(nov1.c), float(bar.c))

    def test_missing_key_uses_yahoo_minus_the_saved_offset(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "gold_feed.json")
            json.dump({"v": 1, "offset": 10, "bars": {}, "fetchedAt": {}, "scale": "futures"}, open(path, "w"))
            seen = []

            def yahoo(sym, tf):
                seen.append((sym, tf))
                df = series(goldfeed.TF_STEP[tf], 180, ms(NOW) - goldfeed.TF_STEP[tf], 3940)
                return df

            def boom(*_a, **_k):
                raise AssertionError("Twelve Data was called without a key")

            frames, _terr = goldfeed.load_xau(yahoo, now=NOW_S, fetch_td=boom, fetch_px=lambda: 3931, state_path=path, key="")
            self.assertAlmostEqual(float(frames["5m"].c.iloc[-1]), 3940 + 179 * 0.01 + 0.2 - 10)
            self.assertAlmostEqual(float(frames["1h"].c.iloc[-1]), 3940 + 179 * 0.01 + 0.2 - 10)
            self.assertNotIn("twelvedata", json.dumps(goldfeed.public_meta(path)).lower())
            meta = goldfeed.public_meta(path)
            self.assertEqual(meta["vendor"], "yahoo")
            self.assertEqual(meta["scale"], "spot")
            self.assertIn("minus 10.00", meta["label"])
            self.assertIn("missing key", meta["label"])
            self.assertTrue(any(tf == "1h" for _s, tf in seen))
            raw = open(path).read()
            self.assertNotIn("apikey", raw)

    def test_rate_limit_falls_back_and_does_not_store_the_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "gold_feed.json")
            calls = []

            def fetch_td(interval, outputsize, key):
                calls.append((interval, outputsize, key))
                raise goldfeed.TwelveError(429, "API credits exhausted for " + key)

            def yahoo(sym, tf):
                return series(goldfeed.TF_STEP[tf], 180, ms(NOW) - goldfeed.TF_STEP[tf], 3940)

            goldfeed.load_xau(yahoo, now=NOW_S, fetch_td=fetch_td, fetch_px=lambda: 1, state_path=path, key=KEY)
            self.assertEqual([c[0] for c in calls], ["5min"])
            blob = open(path).read() + json.dumps(goldfeed.public_meta(path))
            self.assertNotIn(KEY, blob)
            self.assertIn("rate limit", goldfeed.public_meta(path)["label"])
            self.assertEqual(goldfeed.public_meta(path)["vendor"], "yahoo")

    def test_spot_path_resamples_and_measures_the_offset(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "gold_feed.json")
            last_5 = ms("2026-11-03 22:05:00+00:00")
            last_1 = ms("2026-11-03 22:09:00+00:00")
            last_h = ms("2026-11-03 21:00:00+00:00")
            book = {
                "5min": series(300_000, 600, last_5, 3900),
                "1min": series(60_000, 200, last_1, 3910),
                "1h": series(3_600_000, 4000, last_h, 3800),
            }
            sizes = []

            def fetch_td(interval, outputsize, key):
                self.assertEqual(key, KEY)
                sizes.append((interval, outputsize))
                return book[interval]

            def yahoo(sym, tf):
                self.assertEqual((sym, tf), ("GC=F", "1h"))
                fut = book["1h"].copy()
                for col in ("o", "h", "l", "c"):
                    fut[col] = fut[col] + 9.25
                return fut

            frames, terr = goldfeed.load_xau(
                yahoo, now=NOW_S, fetch_td=fetch_td, fetch_px=lambda: 3900, state_path=path, key=KEY,
            )
            self.assertEqual([s[0] for s in sizes], ["5min", "1min", "1h"])
            self.assertTrue(all(s[1] == 5000 for s in sizes))
            self.assertGreaterEqual(len(frames["15m"]), 150)
            self.assertGreaterEqual(len(frames["4h"]), 150)
            self.assertGreaterEqual(len(frames["1d"]), 150)
            self.assertTrue((frames["15m"].t % 900_000 == 0).all())
            self.assertTrue((frames["4h"].t % 14_400_000 == 0).all())
            ny = pd.to_datetime(frames["1d"].t, unit="ms", utc=True).dt.tz_convert("America/New_York")
            self.assertTrue(((ny.dt.hour == 0) & (ny.dt.minute == 0)).all())
            meta = goldfeed.public_meta(path)
            self.assertEqual(meta["vendor"], "twelvedata")
            self.assertEqual(meta["label"], goldfeed.SUCCESS_LABEL)
            self.assertEqual(meta["scale"], "spot")
            self.assertAlmostEqual(meta["offset"], 9.25)
            self.assertNotIn(KEY, json.dumps(meta) + open(path).read())
            self.assertNotIn("1m", terr)
            # Same minute: 1h is cached, so the next scan only refreshes 1min and 5min, with a small outputsize.
            sizes.clear()
            goldfeed.load_xau(yahoo, now=NOW_S, fetch_td=fetch_td, fetch_px=lambda: 3900, state_path=path, key=KEY)
            self.assertEqual([s[0] for s in sizes], ["5min", "1min"])
            self.assertTrue(all(s[1] <= 30 for s in sizes))

    def test_rebase_shifts_open_gold_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "gold_feed.json")
            journal = os.path.join(tmp, "journal.json")
            json.dump({
                "v": 1, "offset": 8.5, "scale": "spot", "bars": {}, "fetchedAt": {}, "journalSpot": False,
            }, open(path, "w"))
            json.dump([
                {"sym": "XAUUSD", "status": "open", "entry": 3940.2, "stop": 3950.2, "tp": [3930.2, 3920.2], "inv": 3960.2},
                {"sym": "XAUUSD", "status": "closed", "entry": 3900, "stop": 3910, "tp": [3890]},
                {"sym": "BTCUSD", "status": "open", "entry": 100, "stop": 90, "tp": [120]},
            ], open(journal, "w"))
            self.assertEqual(goldfeed.rebase_open_journal(journal, path), 1)
            rows = json.load(open(journal))
            self.assertAlmostEqual(rows[0]["entry"], 3931.7)
            self.assertAlmostEqual(rows[0]["stop"], 3941.7)
            self.assertAlmostEqual(rows[0]["tp"][0], 3921.7)
            self.assertEqual(rows[0]["pxBasis"], "spot")
            self.assertEqual(rows[1]["entry"], 3900)
            self.assertEqual(rows[2]["entry"], 100)
            self.assertEqual(goldfeed.rebase_open_journal(journal, path), 0)
            self.assertEqual(json.load(open(journal))[0]["entry"], rows[0]["entry"])

    def test_offset_follows_a_one_hour_stamp_mismatch(self):
        spot = series(3_600_000, 30, ms("2026-11-03 21:00:00+00:00"), 3900)
        fut = spot.copy()
        fut["t"] = fut["t"] + 3_600_000
        for col in ("o", "h", "l", "c"):
            fut[col] = fut[col] + 8
        self.assertAlmostEqual(goldfeed.measure_offset(spot, fut), 8)

    def test_error_text_strips_the_key(self):
        self.assertEqual(goldfeed._reason(goldfeed.TwelveError(400, "bad " + KEY), KEY), "bad")
        self.assertNotIn(KEY, goldfeed._reason(goldfeed.TwelveError(400, "bad " + KEY), KEY))


if __name__ == "__main__":
    unittest.main()
