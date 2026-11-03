# Strategy rebuild round 4 (3 Nov 2026)
Harness: round-2 walk-forward (`/workspace/trading/research/bt2.py` + `exp2.py`, votes `/tmp/votes`, prep `/tmp/prep2_all.pkl`, history `/tmp/long.pkl` on the Grok box).
Weights refit on the train half only; VAL = 3 expanding folds inside the train half; TEST = unseen 2nd half. Costs: gold $0.35/oz, BTC 0.06%, FX 0.8–1.8 pip.
Entry next bar open, stop 1.5×ATR, BE at +1R, 4×ATR chandelier, 48 bars (= baseline "AFTER: carry + trail 4").
Run order: base.py (reproduces +0.296R / PF 1.64 / 291 trades) → cand.py (round A, 13 arms) → cand2.py (round B, unions) → final.py (scoped composites, ship rule, bootstrap) → agents.py (agent stability pruning).
Ship rule (same as the nightly review): pooled exp and PF not lower on VAL and TEST, no combo with ≥8 base trades loses expectancy, ≥60% of base trades kept.
Results: before_after.md (shipped combination), candidates.md (every arm), *_results.json.
