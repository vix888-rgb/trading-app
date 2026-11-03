# Reproduce baseline (round-2 AFTER: carry + trail 4) on all 30 combos via the existing harness.
import sys
src=open("/workspace/trading/research/exp2.py").read().replace('if __name__=="__main__":','if False:')
src=src.replace('PP="/tmp/prep2.pkl"','FOCUS=[(s,t) for s in SYMS for t in TFL]\nPP="/tmp/prep2_all.pkl"')
exec(src)
VARIANTS["AFTER"]=(carryf,{**B,"trail":4.0})
if __name__=="__main__":
    T=run_all(["AFTER"]); T.to_pickle("/workspace/strategy-rebuild/research/base_all.pkl")
    EDGE=[('XAUUSD','1d'),('XAUUSD','4h'),('XAUUSD','1h'),('BTCUSD','1d'),('BTCUSD','1h'),('USDJPY','1d'),('USDCHF','1d')]
    E=T[[(s,t) in EDGE for s,t in zip(T.sym,T.tf)]]
    for st in ("val","test"):
        x=E[E.stage==st]; print(st,metr(list(zip(x.i,x.R))))
    for (s,t),g in T[T.stage=="test"].groupby(["sym","tf"]): print(s,t,metr(list(zip(g.i,g.R))))
