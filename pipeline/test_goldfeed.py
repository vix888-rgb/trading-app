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
        # 1h is once an hour right after the close, plus the 30-minute refresh.
        # Those share two scans an hour, so the steady state stays 336.
        self.assertEqual(goldfeed.credits_per_day(), 336)
        self.assertLess(goldfeed.credits_per_day(), 800)
        # 5min + 1min + 1h on every 10-minute scan.
        self.assertLessEqual(3 * (24 * 60 * 60 // 600), 432)

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

    def test_hourly_fetches_when_the_next_bar_has_closed(self):
        # 06:03 Europe/Belgrade is 05:03 UTC. The cached bar is 03:00 UTC, so
        # the 04:00 UTC bar closed at 05:00. Nine minutes since the last fetch
        # is not enough for the 30-minute refresh; the closed bar still is.
        now = float(pd.Timestamp("2026-11-04 06:03:00", tz="Europe/Belgrade").timestamp())
        last = ms("2026-11-04 03:00:00+00:00")
        self.assertEqual(pd.Timestamp(now, unit="s", tz="UTC").strftime("%H:%M"), "05:03")
        self.assertTrue(goldfeed.h1_due(now - 9 * 60, now, 500, last))

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
                "1h": series(3_600_000, 5000, last_h, 3800),
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


def _state(hours, fetched_at):
    return {"bars": {"1min": [], "5min": [], "1h": goldfeed._pack(hours)}, "fetchedAt": {"1h": fetched_at}}


def _hours(start, end, price=4000.0):
    n = (int(end) - int(start)) // 3_600_000 + 1
    return series(3_600_000, n, end, price)


def _opens(df):
    return {int(t) for t in df.t}


class ClosedBuckets(unittest.TestCase):
    def test_incomplete_4h_and_daily_are_not_emitted(self):
        # 05:00-09:00 Belgrade is 04:00-08:00 UTC. Bars stop on the 3rd hour
        # (06:00 UTC / 07:00 Belgrade). The fetch is still before the bucket end.
        hours = _hours(ms("2026-11-03 00:00:00+00:00"), ms("2026-11-04 06:00:00+00:00"))
        fetched = float(pd.Timestamp("2026-11-04 07:50:00", tz="UTC").timestamp())
        now = float(pd.Timestamp("2026-11-04 08:03:00", tz="UTC").timestamp())
        frames, _terr = goldfeed._build(_state(hours, fetched), now * 1000)
        opens = _opens(frames["4h"])
        self.assertNotIn(ms("2026-11-04 04:00:00+00:00"), opens)
        self.assertIn(ms("2026-11-04 00:00:00+00:00"), opens)

        # Tue 3 Nov, scored at 06:03 Belgrade, without the 23:00 New York hour
        # (04:00 UTC on 4 Nov). The cached tail is the 22:00 NY / 04:00 Belgrade bar.
        hours = _hours(ms("2026-11-02 00:00:00+00:00"), ms("2026-11-04 03:00:00+00:00"), 3900)
        fetched = float(pd.Timestamp("2026-11-04 04:50:00", tz="UTC").timestamp())
        now = float(pd.Timestamp("2026-11-04 06:03:00", tz="Europe/Belgrade").timestamp())
        frames, _terr = goldfeed._build(_state(hours, fetched), now * 1000)
        days = _opens(frames["1d"])
        self.assertNotIn(ms("2026-11-03 05:00:00+00:00"), days)
        self.assertIn(ms("2026-11-02 05:00:00+00:00"), days)

    def test_buckets_emit_once_the_last_hour_is_present(self):
        hours = _hours(ms("2026-11-03 00:00:00+00:00"), ms("2026-11-04 07:00:00+00:00"))
        fetched = float(pd.Timestamp("2026-11-04 07:50:00", tz="UTC").timestamp())
        now = float(pd.Timestamp("2026-11-04 08:03:00", tz="UTC").timestamp())
        frames, _terr = goldfeed._build(_state(hours, fetched), now * 1000)
        h4 = frames["4h"]
        row = h4[h4.t == ms("2026-11-04 04:00:00+00:00")].iloc[0]
        last_hour = hours[hours.t == ms("2026-11-04 07:00:00+00:00")].iloc[0]
        self.assertAlmostEqual(float(row.c), float(last_hour.c))

        hours = _hours(ms("2026-11-02 00:00:00+00:00"), ms("2026-11-04 04:00:00+00:00"), 3900)
        fetched = float(pd.Timestamp("2026-11-04 04:50:00", tz="UTC").timestamp())
        now = float(pd.Timestamp("2026-11-04 06:03:00", tz="Europe/Belgrade").timestamp())
        frames, _terr = goldfeed._build(_state(hours, fetched), now * 1000)
        day = frames["1d"]
        row = day[day.t == ms("2026-11-03 05:00:00+00:00")].iloc[0]
        last_hour = hours[hours.t == ms("2026-11-04 04:00:00+00:00")].iloc[0]
        self.assertAlmostEqual(float(row.c), float(last_hour.c))
        ny = pd.Timestamp(int(last_hour.t), unit="ms", tz="UTC").tz_convert("America/New_York")
        self.assertEqual((ny.hour, ny.minute), (23, 0))

    def test_friday_daily_emits_once_the_fetch_is_past_the_gap(self):
        # Friday 6 Nov 2026. The last trading hour is 16:00 New York (21:00 UTC).
        # The daily bucket still ends at Saturday 00:00 NY, which has no 23:00 bar.
        hours = _hours(ms("2026-11-06 05:00:00+00:00"), ms("2026-11-06 21:00:00+00:00"))
        bucket_end = ms("2026-11-07 05:00:00+00:00")
        now = (bucket_end + 600_000) / 1000.0
        early = (bucket_end + 600_000 - 60_000) / 1000.0
        held, _terr = goldfeed._build(_state(hours, early), now * 1000)
        self.assertNotIn(ms("2026-11-06 05:00:00+00:00"), _opens(held["1d"]))
        frames, _terr = goldfeed._build(_state(hours, now), now * 1000)
        self.assertIn(ms("2026-11-06 05:00:00+00:00"), _opens(frames["1d"]))
        day = frames["1d"]
        row = day[day.t == ms("2026-11-06 05:00:00+00:00")].iloc[0]
        last_hour = hours[hours.t == ms("2026-11-06 21:00:00+00:00")].iloc[0]
        self.assertAlmostEqual(float(row.c), float(last_hour.c))


class WeekendBars(unittest.TestCase):
    """Gold weekend-closure bars (Fri 17:00 ET–Sun 18:00 ET) must not reach indicators."""

    def _weekend_hours(self):
        # Thursday 12 Nov 13:00 ET (18:00 UTC) through Monday 16 Nov 12:00 ET (17:00 UTC).
        # November 2026 is EST (UTC-5). Last Friday hour is 16:00 ET = 21:00 UTC.
        start = ms("2026-11-12 18:00:00+00:00")
        end = ms("2026-11-16 17:00:00+00:00")
        hours = _hours(start, end, 3900)
        t = pd.to_datetime(hours.t, unit="ms", utc=True)
        ny = t.dt.tz_convert("America/New_York")
        closed = ((ny.dt.dayofweek == 4) & (ny.dt.hour >= 17)) | (ny.dt.dayofweek == 5) | ((ny.dt.dayofweek == 6) & (ny.dt.hour < 18))
        # Twelve Data style: tiny synthetic range on closed hours, not always a perfect doji.
        hours.loc[closed, ["o", "h", "l", "c"]] = 3910.0
        hours.loc[closed, "h"] = 3910.26
        hours.loc[closed, "l"] = 3910.00
        hours.loc[closed, "c"] = 3910.10
        hours.loc[closed, "v"] = 0.0
        return hours, closed

    def test_session_window(self):
        self.assertTrue(goldfeed.session_closed_at(ms("2026-11-13 22:00:00+00:00")))  # Fri 17:00 ET
        self.assertFalse(goldfeed.session_closed_at(ms("2026-11-13 21:00:00+00:00")))  # Fri 16:00 ET last hour
        self.assertTrue(goldfeed.session_closed_at(ms("2026-11-14 05:00:00+00:00")))  # Sat
        self.assertTrue(goldfeed.session_closed_at(ms("2026-11-15 22:00:00+00:00")))  # Sun 17:00 ET
        self.assertFalse(goldfeed.session_closed_at(ms("2026-11-15 23:00:00+00:00")))  # Sun 18:00 ET reopen

    def test_weekend_hours_are_dropped_before_4h_and_daily(self):
        hours, closed = self._weekend_hours()
        now = ms("2026-11-16 18:10:00+00:00")
        fetched = now / 1000.0
        frames, _terr = goldfeed._build(_state(hours, fetched), now)
        kept = _opens(frames["1h"])
        self.assertIn(ms("2026-11-13 21:00:00+00:00"), kept)  # Fri 16:00 ET
        self.assertNotIn(ms("2026-11-13 22:00:00+00:00"), kept)  # Fri 17:00 ET
        self.assertNotIn(ms("2026-11-14 12:00:00+00:00"), kept)  # Saturday
        self.assertNotIn(ms("2026-11-15 22:00:00+00:00"), kept)  # Sun 17:00 ET
        self.assertIn(ms("2026-11-15 23:00:00+00:00"), kept)  # Sun 18:00 ET
        self.assertEqual(len(frames["1h"]), int((~closed).sum()))

        days = _opens(frames["1d"])
        self.assertIn(ms("2026-11-13 05:00:00+00:00"), days)  # Friday
        self.assertNotIn(ms("2026-11-14 05:00:00+00:00"), days)  # Saturday, all hours dropped
        self.assertIn(ms("2026-11-15 05:00:00+00:00"), days)  # Sunday evening session
        fri = frames["1d"][frames["1d"].t == ms("2026-11-13 05:00:00+00:00")].iloc[0]
        last_fri = hours[hours.t == ms("2026-11-13 21:00:00+00:00")].iloc[0]
        self.assertAlmostEqual(float(fri.c), float(last_fri.c))
        sun = frames["1d"][frames["1d"].t == ms("2026-11-15 05:00:00+00:00")].iloc[0]
        first_sun = hours[hours.t == ms("2026-11-15 23:00:00+00:00")].iloc[0]
        last_sun = hours[hours.t == ms("2026-11-16 04:00:00+00:00")].iloc[0]
        self.assertAlmostEqual(float(sun.o), float(first_sun.o))
        self.assertAlmostEqual(float(sun.c), float(last_sun.c))

        # Friday 20:00 UTC 4h bucket keeps only the 20:00 and 21:00 hours (16:00 ET still open).
        h4 = frames["4h"]
        row = h4[h4.t == ms("2026-11-13 20:00:00+00:00")].iloc[0]
        self.assertAlmostEqual(float(row.c), float(last_fri.c))
        self.assertNotIn(ms("2026-11-14 00:00:00+00:00"), _opens(h4))  # Saturday 4h gone

    def test_merge_does_not_cache_weekend_or_zero_range_bars(self):
        hours, closed = self._weekend_hours()
        midweek = series(3_600_000, 4, ms("2026-11-12 16:00:00+00:00"), 3880)
        # A mid-week holiday print: flat, volume 0, should also be dropped.
        holiday = pd.DataFrame({
            "t": [ms("2026-11-11 15:00:00+00:00")],
            "o": [4000.0], "h": [4000.0], "l": [4000.0], "c": [4000.0], "v": [0.0],
        })
        merged = goldfeed.merge_bars(midweek, pd.concat([hours, holiday], ignore_index=True))
        opens = _opens(merged)
        self.assertNotIn(ms("2026-11-13 22:00:00+00:00"), opens)
        self.assertNotIn(ms("2026-11-14 12:00:00+00:00"), opens)
        self.assertNotIn(ms("2026-11-11 15:00:00+00:00"), opens)
        self.assertIn(ms("2026-11-13 21:00:00+00:00"), opens)
        self.assertIn(ms("2026-11-12 16:00:00+00:00"), opens)
        # A mid-week bar with volume 0 but a real range is kept (the feed stores v=0 on every bar).
        self.assertIn(ms("2026-11-12 16:00:00+00:00"), opens)
        row = merged[merged.t == ms("2026-11-12 16:00:00+00:00")].iloc[0]
        self.assertGreater(float(row.h - row.l), 0)

    def test_load_xau_cache_and_frames_skip_weekend_bars(self):
        hours, closed = self._weekend_hours()
        # Need >= MIN_BARS of 5m and 1h. Pad with earlier mid-week hours.
        pad_h = series(3_600_000, 200, ms("2026-11-12 17:00:00+00:00"), 3800)
        h1 = goldfeed.merge_bars(pad_h, hours)  # already drops weekend from the padded merge...
        # Rebuild an unfiltered book for the fetch stub, and let load_xau drop them.
        raw_h = pd.concat([pad_h, hours], ignore_index=True).drop_duplicates("t", keep="last").sort_values("t").reset_index(drop=True)
        last_5 = ms("2026-11-16 17:55:00+00:00")
        m5 = series(300_000, 600, last_5, 3900)
        t5 = pd.to_datetime(m5.t, unit="ms", utc=True).dt.tz_convert("America/New_York")
        closed5 = ((t5.dt.dayofweek == 4) & (t5.dt.hour >= 17)) | (t5.dt.dayofweek == 5) | ((t5.dt.dayofweek == 6) & (t5.dt.hour < 18))
        m5.loc[closed5, ["o", "h", "l", "c"]] = 3910.0
        m5.loc[closed5, "h"] = 3910.26
        m5.loc[closed5, "l"] = 3910.00
        m5.loc[closed5, "v"] = 0.0
        last_1 = ms("2026-11-16 17:59:00+00:00")
        m1 = series(60_000, 200, last_1, 3910)
        book = {"5min": m5, "1min": m1, "1h": raw_h}
        now = ms("2026-11-16 18:10:00+00:00") / 1000.0
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "gold_feed.json")

            def fetch_td(interval, outputsize, key):
                self.assertEqual(key, KEY)
                return book[interval]

            def yahoo(sym, tf):
                self.assertEqual((sym, tf), ("GC=F", "1h"))
                fut = raw_h.copy()
                for col in ("o", "h", "l", "c"):
                    fut[col] = fut[col] + 9.25
                return fut

            frames, _terr = goldfeed.load_xau(
                yahoo, now=now, fetch_td=fetch_td, fetch_px=lambda: 3900, state_path=path, key=KEY,
            )
            self.assertNotIn(ms("2026-11-13 22:00:00+00:00"), _opens(frames["1h"]))
            self.assertIn(ms("2026-11-13 21:00:00+00:00"), _opens(frames["1h"]))
            self.assertIn(ms("2026-11-15 23:00:00+00:00"), _opens(frames["1h"]))
            cached = goldfeed._frame(json.load(open(path))["bars"]["1h"])
            self.assertNotIn(ms("2026-11-13 22:00:00+00:00"), _opens(cached))
            self.assertNotIn(KEY, open(path).read() + json.dumps(goldfeed.public_meta(path)))

    def test_zero_range_midweek_bar_is_dropped(self):
        hours = _hours(ms("2026-11-10 00:00:00+00:00"), ms("2026-11-10 20:00:00+00:00"), 4000)
        hours.loc[hours.t == ms("2026-11-10 15:00:00+00:00"), ["o", "h", "l", "c"]] = 4001.0
        cleaned = goldfeed.drop_closed_bars(hours, goldfeed.STEP["1h"])
        self.assertNotIn(ms("2026-11-10 15:00:00+00:00"), _opens(cleaned))
        self.assertIn(ms("2026-11-10 14:00:00+00:00"), _opens(cleaned))
        self.assertEqual(len(cleaned), len(hours) - 1)



if __name__ == "__main__":
    unittest.main()
