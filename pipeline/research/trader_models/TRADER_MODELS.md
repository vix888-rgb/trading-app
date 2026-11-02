# Trader-model rebuild research (2 Nov 2026)

Goal: test what five of today's most followed traders actually teach, coded as mechanical rules, against the app's round-2 system on unseen data.
Everything below is a *mechanical reading* of discretionary methods. Rules marked (inferred) are my choices where the source is a judgement call.

## The five traders and the rules used

| Trader | Model (1 line) | Sources |
|---|---|---|
| **PB Blake** (PB Trading, NQ, NY open) | 4 steps: HTF bias from Daily/4H/1H(/15m) FVGs respected vs disrespected + draw on liquidity (next swing high/low) → valid key level (unmitigated 5m/15m+ FVG, CISD, rejection block) → wait for the highest-TF IFVG (1–5m) of the manipulation leg to get a body close → enter on that close, stop at the leg's swing, TP at DOL, 1:1–1:3, BE at an intermediate swing. Golden hour 9:30–11:00 ET, ≤2 trades. | Video transcript "My NEW UPDATED Trading Strategy for 2026" https://www.youtube.com/watch?v=9O6JU5_xTd8 (saved here as source_pbblake_2026_transcript.txt); PB Theory liquidity https://www.youtube.com/watch?v=IfQ-38IED04 ; indicator that codes his 4 steps (score ≥2, 9:30–11:00, max 2) https://www.tradingview.com/script/h179SZH8-EZ-PB-Blake-v1-0/ ; profile https://www.marketframework.com/traders/pb-trading |
| **Fabio Valentini** ("the Italian guy") | Auction-market theory: Direction → Location → Aggression. IVB = first 30 min of NY (9:30–10:00) range; a 5m close outside 10:00–14:45 → trade the break, stop other side, 1R, flat 15:00; VWAP side as bias. Mean-reversion model: failed break of prior-day value area → back to POC. Order-flow aggression (footprint/CVD) is **not** codable from OHLC. | https://www.tradezella.com/strategies/auction-market-strategy ; https://www.tradingview.com/script/QbOnCVCT-IVB-Opening-Range-Protocol-long-only-breakout-IDFX/ ; https://blog.pickmytrade.trade/fabio-valentini-pro-scalper-nasdaq-scalping-strategy/ ; https://www.chartacademy.com/instructors/fabio-valentini ; X: https://x.com/SiliconCharts/status/2087072059541069985 |
| **ICT** (Michael Huddleston) | Silver Bullet: 10:00–11:00 NY (also 03–04, 14–15): mark prior-day/session H/L as draw on liquidity, wait for sweep + market-structure shift, enter the first FVG in that hour (limit at 50%/CE), stop beyond FVG/sweep, target the pool, ≥15 pips framework on FX. | https://edgeofict.com/learn/mentorship-2023/29-silver-bullet-time-based-trading-model-may-15 ; https://michaeljhuddleston.org/notes/ict-silver-bullet-strategy-capture-10-40-pips-in-just-60-minutes/ ; https://propbooktrading.com/learn/concepts/silver-bullet |
| **TJR** | Sweep of Asia/London/prior-day H/L at 9:30–9:50 → 5m BOS (body close) → retrace into 5m FVG or 50% of the leg → entry before 10:30, stop beyond the swing, target the opposite draw. | Rule-by-rule with timestamps: https://telonicstrading.com/research-tjr.html ; https://backtestify.com/strategies/tjr-sweep-mss-fvg |
| **Tom Hougaard** | School Run: after the open, take the 2nd 15-minute bar's high/low; buy-stop above / sell-stop below (few points offset), stop at the other side, 1:1 or hold a runner; "Anti" = fade the first break. Never add to losers. | https://benniewel.org/pdfs/The%20School%20Run.pdf ; https://pdfcoffee.com/trader-tom-school-run-strategy-december-2022-pdf-free.html ; Best Loser Wins https://www.tradertom.com/wp-content/uploads/2021/03/BEST-LOSER-WINS-1.pdf |

Why Fabio Valentini for "the Italian guy": he is the Italian scalper most often named next to PB Blake in the same NQ / prop-firm circle (Robbins World Cup top-3, DeepCharts co-founder, "#1 scalper" interviews; X posts in Oct 2026 still circulate his IVB model). Runner-up Italians: Lorenzo Corrado (174K YT, live gold & NQ with volume profile/VWAP — his weekly POC/VAH/VAL idea is covered by the AMT mean-reversion test), Andrea Cimi (237K, order flow, Fabio's DeepCharts partner), Saix (TikTok 700K, gold signals Telegram, no published rules or track record, not testable). Alessio Rastani and Marco Mazzei (your suggestions) did not come up in the 2026 popularity sources I checked, so I did not pick them (not verified further).

## Data and protocol
- Intraday: Binance 5m 2023-01 → 2026-09 (BTCUSDT; PAXGUSDT as gold proxy, 5m return corr 0.96 with GC=F; EURUSDT as a *weak* EURUSD proxy, corr 0.57). Dukascopy was rate-limited (HTTP 429/503) from this box. Plus native Yahoo 5m (60 days) for all 6 symbols as a sanity check. USDJPY/USDCHF/EURCHF have only the 60-day intraday check.
- Sessions in New York time (zoneinfo America/New_York, DST handled); London open via Europe/London.
- Costs = Vantage-like round trip from the earlier rounds (gold $0.35, BTC 0.06%, EURUSD 0.8 pip, USDJPY 1.0, USDCHF 1.3, EURCHF 1.8). Trades where cost > 25% of the stop are skipped (not executable at those spreads).
- Conservative fills: stop checked before target inside a bar; limit orders fill only if touched; entries on bar close use the next bar's open.
- Train = first half by time, TEST = unseen second half. Parameters were fixed from the sources before running; ~25 variants tried in total, so one good-looking result can be luck.
- HTF tests (1h/4h/1D) use the identical series and identical unseen half as round 2 (/tmp/long.pkl, split from /tmp/votes).

## Results (unseen half). Baseline = round-2 app, focus set: n 248, win 24.2%, +0.319R, PF 1.74, maxDD 11.2R
See summ_all.txt and hybrid.txt. No trader model beat the baseline; none was positive in both halves with t > 2.
Survivor: ICT/TJR sweep → MSS → FVG on 1h/4h/1D with the app's exit: train +0.33R (151), test +0.13R (141, PF 1.31, t 1.09). Added as a display-only strategy.
PB's FVG-respect bias as a filter on the baseline: validation +1.34R (31) but test +0.26R (42) < +0.32 → rejected.
Baseline + sweep model (combos chosen on train): test +0.264R/trade, +86.5R vs +79.0R total, P(no gain)=0.44 → not adopted for alerts.

Reproduce: python3 run_intraday.py && python3 htf.py && python3 hybrid.py && python3 summ.py (paths point at /workspace/rebuild and /tmp/long.pkl, /tmp/votes, /tmp/exp7_focus.pkl on the Grok box).
