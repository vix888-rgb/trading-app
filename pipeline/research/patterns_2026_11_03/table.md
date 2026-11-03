| Arm | TEST R/trade | PF | win % | trades | max DD R | VAL R/trade | verdict |
|---|---|---|---|---|---|---|---|
| **Baseline alert book (round 4)** | +0.427 | 2.06 | 26.8 | 205 | 10.9 | +0.661 | — |
| Wolfe standalone (gold+BTC 1h/4h/1D, harness window) | +0.683 | 3.48 | 50.0 | 12 | 2.1 | 0.59 | fail: too few trades / below baseline |
| Wolfe standalone+pattern stop (gold+BTC 1h/4h/1D, harness window) | +0.236 | 1.91 | 38.5 | 13 | 2.1 | 0.281 | fail: too few trades / below baseline |
| Wolfe standalone, full history (TRAIN instead of VAL) | +0.272 | 1.59 | 30.4 | 23 | 5.3 | 0.443 | fail |
| Wolfe standalone + pattern stop, full history (TRAIN instead of VAL) | +0.053 | 1.15 | 33.3 | 24 | 3.0 | 0.45 | fail |
| Wolfe as confirm filter (alert book) | +0.177 | 1.38 | 27.3 | 33 | 6.8 | +0.486 | fail: val:XAUUSD 1d, val:XAUUSD 1h, val:BTCUSD 1d, test:XAUUSD 1d, test:XAUUSD 4h, tes |
| Wolfe as veto filter (alert book) | +0.427 | 2.06 | 26.8 | 205 | 10.9 | +0.661 | no change (no vetoes hit) |
| Harmonic standalone (gold+BTC 1h/4h/1D, harness window) | +0.368 | 1.93 | 25.0 | 8 | 1.1 | 0.174 | fail: too few trades / below baseline |
| Harmonic standalone+pattern stop (gold+BTC 1h/4h/1D, harness window) | -0.005 | 0.99 | 33.3 | 9 | 2.1 | -0.079 | fail: too few trades / below baseline |
| Harmonic standalone, full history (TRAIN instead of VAL) | +0.106 | 1.26 | 19.0 | 21 | 3.3 | -0.244 | fail |
| Harmonic standalone + pattern stop, full history (TRAIN instead of VAL) | +0.213 | 1.45 | 40.9 | 22 | 2.8 | -0.363 | fail |
| Harmonic as confirm filter (alert book) | +0.214 | 1.48 | 28.1 | 32 | 6.8 | +0.326 | fail: val:XAUUSD 1d, val:XAUUSD 4h, val:XAUUSD 1h, val:BTCUSD 1d, test:XAUUSD 1d, test |
| Harmonic as veto filter (alert book) | +0.422 | 2.04 | 26.8 | 205 | 10.9 | +0.674 | fail: test:BTCUSD 1d |
| AB=CD standalone (gold+BTC 1h/4h/1D, harness window) | +0.075 | 1.14 | 15.2 | 33 | 9.0 | 0.231 | fail: too few trades / below baseline |
| AB=CD standalone+pattern stop (gold+BTC 1h/4h/1D, harness window) | -0.130 | 0.77 | 12.5 | 32 | 8.4 | 0.17 | fail: too few trades / below baseline |
| AB=CD standalone, full history (TRAIN instead of VAL) | +0.047 | 1.09 | 17.7 | 62 | 12.8 | 0.457 | fail |
| AB=CD standalone + pattern stop, full history (TRAIN instead of VAL) | +0.045 | 1.08 | 20.0 | 60 | 10.5 | 0.225 | fail |
| AB=CD as confirm filter (alert book) | +0.116 | 1.24 | 27.8 | 36 | 6.8 | +0.267 | fail: val:XAUUSD 1d, val:XAUUSD 4h, val:XAUUSD 1h, val:BTCUSD 1d, test:XAUUSD 1d, test |
| AB=CD as veto filter (alert book) | +0.425 | 2.05 | 26.3 | 205 | 10.9 | +0.674 | fail: test:XAUUSD 4h |
| Wyckoff standalone (gold+BTC 1h/4h/1D, harness window) | -0.327 | 0.37 | 11.1 | 27 | 10.3 | 0.749 | fail: too few trades / below baseline |
| Wyckoff standalone+pattern stop (gold+BTC 1h/4h/1D, harness window) | -0.265 | 0.49 | 22.2 | 27 | 8.9 | 0.77 | fail: too few trades / below baseline |
| Wyckoff standalone, full history (TRAIN instead of VAL) | +0.080 | 1.15 | 19.1 | 47 | 6.3 | -0.007 | fail |
| Wyckoff standalone + pattern stop, full history (TRAIN instead of VAL) | +0.159 | 1.29 | 27.7 | 47 | 5.3 | 0.083 | fail |
| Wyckoff as confirm filter (alert book) | +0.071 | 1.15 | 23.1 | 39 | 6.8 | +0.450 | fail: val:XAUUSD 1d, val:XAUUSD 4h, val:XAUUSD 1h, test:XAUUSD 1d, test:XAUUSD 4h, tes |
| Wyckoff as veto filter (alert book) | +0.430 | 2.07 | 26.6 | 203 | 10.9 | +0.652 | fail: val:XAUUSD 1d, test:XAUUSD 1h |

Sensitivity (pivot width 2 instead of 4, full history, ATR exit): Wolfe TEST +0.254 (n51, PF 1.45), TRAIN +0.277; Harmonic TEST -0.093 (n33, PF 0.83), TRAIN -0.254; AB=CD TEST +0.206 (n101, PF 1.38), TRAIN +0.029; Wyckoff TEST -0.011 (n58, PF 0.97), TRAIN +0.237
