#!/usr/bin/env python3
"""Parse mocked Vantage price messages. Does not call ntfy."""
import json, os, subprocess, sys, unittest

_PIPE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_PIPE, ".."))
if _PIPE not in sys.path:
    sys.path.insert(0, _PIPE)

from vantage import MAX_AGE_SEC, FUTURE_SKEW_SEC, latest_quotes

NOW = 1_700_000_300
T_OLD = NOW - 600
T_NEW = NOW - 120
T_STALE = NOW - (MAX_AGE_SEC + 1)
T_EDGE = NOW - MAX_AGE_SEC
COMBINED = {
    "q": {
        "XAUUSD.r": 3910.47,
        "BTCUSD": None,
        "EURUSD": "1.08521",
        "USDJPY": 151.234,
        "USDCHF": 0.88123,
        "EURCHF": 0.94567,
        "ETHUSD": 2500,
    },
    "tf": "5",
    "t": T_NEW * 1000,
}


def env(message, t=None, event="message"):
    row = {"id": "x", "event": event, "topic": "octa-prices-8223d362", "message": message}
    if t is not None:
        row["time"] = t
    return row


class LatestQuotesTest(unittest.TestCase):
    def test_old_and_combined_keep_newest(self):
        lines = [
            "this is not json",
            "",
            "   ",
            json.dumps(env('{"sym":"XAUUSD+","c":3900.12,"tf":"1","t":%d}' % (T_OLD * 1000), t=T_OLD)),
            json.dumps(env('{"sym":"VANTAGE:BTCUSD","bid":97000.5,"t":%d}' % T_OLD, t=T_OLD)),
            json.dumps(env(json.dumps(COMBINED), t=T_NEW)),
            json.dumps(env("not-json", t=T_NEW)),
            json.dumps(env("", t=T_NEW)),
            json.dumps(env('{"sym":"USDJPY","c":151.5,"t":%d}' % (T_NEW + 30), t=T_NEW + 30)),
            json.dumps(env('{"sym":"XAUUSD","c":3800,"t":%d}' % (T_NEW - 10), t=T_NEW - 10)),
            json.dumps({"event": "keepalive", "message": '{"sym":"EURUSD","c":9,"t":%d}' % T_NEW}),
        ]
        got = latest_quotes("\n".join(lines), now=NOW)
        self.assertEqual(got["XAUUSD"]["price"], 3910.47)
        self.assertEqual(got["XAUUSD"]["time"], T_NEW)
        self.assertEqual(got["BTCUSD"], {"time": T_OLD, "price": 97000.5})
        self.assertEqual(got["EURUSD"]["price"], 1.08521)
        self.assertEqual(got["USDJPY"], {"time": T_NEW + 30, "price": 151.5})
        self.assertEqual(got["USDCHF"]["price"], 0.88123)
        self.assertEqual(got["EURCHF"]["price"], 0.94567)
        self.assertNotIn("ETHUSD", got)

    def test_array_seconds_and_age_window(self):
        fresh = {"sym": "EURCHF", "close": 0.94, "t": T_EDGE}
        stale = {"sym": "USDCHF", "c": 0.8, "t": T_STALE}
        future = {"sym": "EURUSD", "c": 1.1, "t": NOW + FUTURE_SKEW_SEC + 5}
        bare = {"q": {"USDJPY": 150.5, "BTCUSD": 0, "XAUUSD": -1}, "tf": "5", "t": T_NEW}
        text = json.dumps([env(json.dumps(fresh), t=T_EDGE), env(json.dumps(stale)), env(json.dumps(future)), bare])
        got = latest_quotes(text, now=NOW)
        self.assertEqual(set(got), {"EURCHF", "USDJPY"})
        self.assertEqual(got["EURCHF"]["time"], T_EDGE)
        self.assertEqual(got["USDJPY"]["price"], 150.5)

    def test_empty_and_tie_go_to_the_later_message(self):
        self.assertEqual(latest_quotes("", now=NOW), {})
        self.assertEqual(latest_quotes("[]", now=NOW), {})
        self.assertEqual(latest_quotes("{", now=NOW), {})
        a = json.dumps({"sym": "XAUUSD", "c": 1, "t": T_NEW})
        b = json.dumps({"sym": "XAUUSD", "c": 2, "t": T_NEW})
        got = latest_quotes(a + "\n" + b, now=NOW)
        self.assertEqual(got["XAUUSD"]["price"], 2)

    def test_missing_bar_time_uses_envelope_time(self):
        row = env('{"sym":"EURUSD","c":1.2}', t=T_NEW)
        got = latest_quotes(json.dumps(row), now=NOW)
        self.assertEqual(got["EURUSD"], {"time": T_NEW, "price": 1.2})


JS = r"""
const fs = require("fs");
const vm = require("vm");
const html = fs.readFileSync(process.argv[2], "utf8");
const sw = fs.readFileSync(process.argv[3], "utf8");
const pine = fs.readFileSync(process.argv[4], "utf8");
if (!sw.includes('const CACHE = "signal-desk-v8"')) throw new Error("cache is not v8");
for (const sym of ["VANTAGE:XAUUSD", "VANTAGE:BTCUSD", "VANTAGE:EURUSD", "VANTAGE:USDJPY", "VANTAGE:USDCHF", "VANTAGE:EURCHF"]) {
  if (!pine.includes('request.security("' + sym + '", timeframe.period, close)')) throw new Error("missing " + sym);
}
if (!pine.includes("alert(msg, alert.freq_once_per_bar_close)")) throw new Error("alert call missing");
if (!pine.includes('"#.##"') || !pine.includes('"#.###"') || !pine.includes('"#.#####"')) throw new Error("precision missing");
if (!pine.includes('"null"')) throw new Error("null missing");
const ms = html.match(/const VANTAGE_MS=[^;]+;/);
const ids = html.match(/const VANTAGE_IDS=\[[^\]]+\];/);
if (!ms || !ids) throw new Error("price constants missing");
const start = html.indexOf("function canonTicker");
const end = html.indexOf("async function primePrices");
if (start < 0 || end < start) throw new Error("parser slice missing");
const sandbox = {calls: [], vTicks: {}, S: {sym: "EURUSD"}, console};
sandbox.paintLive = function () { sandbox.calls.push(sandbox.S.sym); };
vm.createContext(sandbox);
vm.runInContext(ids[0] + "\n" + ms[0] + "\nif (VANTAGE_MS !== 12 * 60 * 1000) throw new Error('VANTAGE_MS ' + VANTAGE_MS);\n" + html.slice(start, end), sandbox);

function reset() { for (const k of Object.keys(sandbox.vTicks)) delete sandbox.vTicks[k]; sandbox.calls.length = 0; }
function dump() {
  const out = {};
  for (const k of Object.keys(sandbox.vTicks)) out[k] = {px: sandbox.vTicks[k].px, t: sandbox.vTicks[k].at.getTime() / 1000};
  return out;
}
function eq(got, want, label) {
  const a = JSON.stringify(got), b = JSON.stringify(want);
  if (a !== b) throw new Error(label + "\n got " + a + "\n want " + b);
}

const T_OLD = 1700000000;
const T_NEW = 1700000300;
reset();
const blob = [
  "not json",
  "",
  "   ",
  JSON.stringify({event: "message", time: T_OLD, message: JSON.stringify({sym: "XAUUSD+", c: 3900.12, t: T_OLD * 1000})}),
  JSON.stringify({event: "message", message: "not-json"}),
  JSON.stringify({event: "message", message: ""}),
  JSON.stringify({event: "message", message: "   "}),
  JSON.stringify({event: "keepalive", message: JSON.stringify({sym: "EURUSD", c: 9, t: T_NEW * 1000})}),
  JSON.stringify({event: "message", time: T_NEW, message: JSON.stringify({q: {"XAUUSD.r": 3910.47, BTCUSD: null, EURUSD: "1.08521", USDJPY: 151.234, USDCHF: 0.88123, EURCHF: 0.94567, ETHUSD: 2500}, tf: "5", t: T_NEW * 1000})}),
].join("\n");
sandbox.ingestStreamText(blob);
eq(dump(), {
  XAUUSD: {px: 3910.47, t: T_NEW},
  EURUSD: {px: 1.08521, t: T_NEW},
  USDJPY: {px: 151.234, t: T_NEW},
  USDCHF: {px: 0.88123, t: T_NEW},
  EURCHF: {px: 0.94567, t: T_NEW},
}, "combined over older single");
if (!sandbox.calls.includes("EURUSD")) throw new Error("paintLive was not called for the open symbol");

sandbox.ingestPrice({event: "message", message: JSON.stringify({sym: "USDJPY", c: 151.5, t: (T_NEW + 30) * 1000})});
sandbox.ingestPrice({event: "message", message: JSON.stringify({sym: "XAUUSD", bid: 3800, t: (T_NEW - 10) * 1000})});
eq(dump().USDJPY, {px: 151.5, t: T_NEW + 30}, "newer single wins");
eq(dump().XAUUSD, {px: 3910.47, t: T_NEW}, "older single loses");

reset();
sandbox.ingestStreamText(JSON.stringify([
  {event: "message", message: {sym: "VANTAGE:BTCUSD", close: 97000.5, t: T_OLD}},
  {q: {EURUSD: 1.2, BTCUSD: 0, XAUUSD: null}, tf: "5", t: T_NEW},
]));
eq(dump(), {BTCUSD: {px: 97000.5, t: T_OLD}, EURUSD: {px: 1.2, t: T_NEW}}, "array of both formats");

reset();
sandbox.ingestPrice({event: "message", time: T_NEW, message: JSON.stringify({sym: "USDCHF", c: 0.9})});
eq(dump().USDCHF, {px: 0.9, t: T_NEW}, "envelope time");

reset();
sandbox.ingestPrice({event: "message", message: JSON.stringify({sym: "XAUUSD", c: 1, t: T_NEW})});
sandbox.ingestPrice({event: "message", message: JSON.stringify({sym: "XAUUSD", c: 2, t: T_NEW})});
eq(dump().XAUUSD.px, 2, "tie keeps the later message");

reset();
sandbox.vTicks.XAUUSD = {px: 10, at: new Date(Date.now() - 11 * 60 * 1000)};
if (!sandbox.vantageFresh("XAUUSD")) throw new Error("11 min quote should show");
sandbox.vTicks.XAUUSD = {px: 10, at: new Date(Date.now() - 12 * 60 * 1000)};
if (!sandbox.vantageFresh("XAUUSD")) throw new Error("12 min quote should still show");
sandbox.vTicks.XAUUSD = {px: 10, at: new Date(Date.now() - (12 * 60 * 1000 + 1))};
if (sandbox.vantageFresh("XAUUSD")) throw new Error("quote older than 12 min should hide");
sandbox.ingestStreamText("");
sandbox.ingestStreamText("   ");
console.log("js ok");
"""


class PageParserTest(unittest.TestCase):
    def test_page_parser_against_index(self):
        script = os.path.join(_PIPE, "_price_parse_test.js")
        with open(script, "w", encoding="utf-8") as fh:
            fh.write(JS)
        try:
            subprocess.run(
                ["node", script, os.path.join(ROOT, "index.html"), os.path.join(ROOT, "sw.js"), os.path.join(ROOT, "tradingview", "vantage-price-feed.pine")],
                check=True,
            )
        finally:
            os.remove(script)


if __name__ == "__main__":
    unittest.main()
