import pandas as pd,numpy as np,sys
sys.path.insert(0,'.'); from tm import metr
T=pd.read_pickle("intraday_trades.pkl")
def f(m): return f"n={m['n']:4d} win={m.get('win','-'):>5} exp={m.get('exp','-'):>7} pf={m.get('pf','-')} dd={m.get('dd','-')} t={m.get('t','-')}" if m['n'] else "n=0"
print("=== LONG HISTORY (Binance 5m 2023-01..2026-09; gold=PAXG proxy, EURUSD=EURUSDT weak proxy), train | TEST (unseen 2nd half)")
L=T[T.set.str.contains("2023")]
for mdl in L.model.unique():
    g=L[L.model==mdl]
    print(f"\n{mdl}\n   ALL3  train {f(metr(g[g.stage=='train'].R))} | TEST {f(metr(g[g.stage=='test'].R))}")
    for s in ["BTCUSD","XAUUSD","EURUSD"]:
        x=g[g.sym==s]; print(f"   {s} train {f(metr(x[x.stage=='train'].R))} | TEST {f(metr(x[x.stage=='test'].R))}")
print("\n=== NATIVE 60-day 5m (Yahoo FX/gold, Binance BTC), all 6 symbols pooled, whole 60d (too short to split meaningfully)")
S=T[T.set.str.contains("60d")]
for mdl in S.model.unique():
    g=S[S.model==mdl]; print(f"{mdl:58s} {f(metr(g.R))}  by sym: "+" ".join(f"{s}:{metr(g[g.sym==s].R).get('exp','-')}({len(g[g.sym==s])})" for s in ["XAUUSD","BTCUSD","EURUSD","USDJPY","USDCHF","EURCHF"]))
H=pd.read_pickle("htf_trades.pkl")
print("\n=== HTF (same series & unseen half as round-2 baseline)")
for mdl in H.model.unique():
    g=H[H.model==mdl]
    for lab,x in (("focus",g[g.focus]),("all 18",g)):
        print(f"{mdl:52s} {lab:6s} train {f(metr(x[x.stage=='train'].R))} | TEST {f(metr(x[x.stage=='test'].R))}")
    x=g[g.stage=="test"]
    print("     TEST by combo:", "; ".join(f"{s}{t}:{metr(y.R)['exp']}({len(y)})" for (s,t),y in x.groupby(["sym","tf"])))
