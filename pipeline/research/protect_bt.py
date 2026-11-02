#!/usr/bin/env python3
"""Out-of-sample pass for the pre-breakout filters on the edge combos.

Same walk-forward as the scan: agent weights are fit on the first 60% with the
barrier outcome (TP 1.5×ATR vs SL 1×ATR inside 12 bars). The unseen last 40%
is traded the way the journal trades: next-bar open, swing stop clamped like
the live plan, breakeven at +1R then a 4×ATR trail, 48-bar time exit, and the
research round-trip cost. Quality filters (volatility, 60-bar trend, higher-TF
bias, carry) stay on in every arm. One position at a time.
"""
import os, sys, time
from datetime import datetime
from zoneinfo import ZoneInfo
_PIPE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PIPE)
os.environ["TRADING_LOAD_ONLY"] = "1"
import numpy as np, pandas as pd
import engine3 as E
import edge
sys.path.insert(0, os.path.join(_PIPE, "research", "trader_models"))
from tm import cost_of, sim_trade, metr
TZ = ZoneInfo("Europe/Belgrade")
EDGE = [("XAUUSD","1d"),("XAUUSD","4h"),("XAUUSD","1h"),("BTCUSD","1d"),("BTCUSD","1h"),("USDJPY","1d"),("USDCHF","1d")]
HTF = {"1h":"4h","4h":"1d","1d":None}
NEED = {"XAUUSD":("yahoo","GC=F"),"BTCUSD":("binance","BTCUSDT"),"USDJPY":("yahoo","JPY=X"),"USDCHF":("yahoo","CHF=X")}
TFS = ["5m","15m","1h","4h","1d"]

def fetch():
    raw = {}
    for sym,(src,code) in NEED.items():
        for tf in TFS:
            df = E.binance(code, tf) if src=="binance" else E.yahoo(code, tf)
            raw[(sym,tf)] = df.reset_index(drop=True)
            print("got", sym, tf, len(df), flush=True)
    return raw

def swing_dist(entry, side, a, highs, lows, i):
    if not (a==a) or a<=0: return None
    if side>0:
        cands=[px for j,px in lows if i-120<=j<=i and px<entry]
        dist=(entry-(cands[-1]-0.15*a)) if cands else 1.2*a
    else:
        cands=[px for j,px in highs if i-120<=j<=i and px>entry]
        dist=((cands[-1]+0.15*a)-entry) if cands else 1.2*a
    if dist<0.8*a: dist=1.2*a
    if dist>3*a: dist=2.0*a
    if dist<=0: return None
    return dist

def fit_weights(raw, key, tf, idx):
    df=raw[(key,tf)]; rec=[]
    for i in idx:
        v,c,rk=E.votes_at(raw,key,tf,i)
        rec.append((v,c,rk,E.outcome(df,i,1),E.outcome(df,i,-1)))
    nag=len(rec[0][0]); wts=[]
    for k in range(nag):
        rs=[(r[3] if r[0][k]>0 else r[4]) for r in rec if r[0][k]!=0]
        w=0.0 if k==E.NEWS_IDX else E.weight_from(E.stats(rs))
        wts.append(w)
    return wts

def trades_for(df, sym, dirs, A, highs, lows):
    o,h,l,c=df.o.values,df.h.values,df.l.values,df.c.values
    n=len(df); rs=[]; i=0
    while i<n-2:
        d=int(dirs[i])
        if d==0 or not (A[i]==A[i]): i+=1; continue
        e=o[i+1]; dist=swing_dist(e,d,A[i],highs,lows,i)
        if dist is None: i+=1; continue
        stop=e-d*dist
        r=sim_trade(o,h,l,c,i+1,d,e,stop,None,i+1+47,be=1.0,trail=4.0,A=A[i],cost=cost_of(sym,e))
        if r is None: break
        rs.append(r[0]); i=r[1]+1
    return rs

def one(raw, sym, tf):
    t0=time.time(); df=raw[(sym,tf)]; n=len(df)
    lo=min(E.PRM["window"], max(80, n//5)); cut=int(n*E.PRM["fit_frac"]); hi=n-E.PRM["trail_max_bars"]-2
    if hi-cut<20 or cut-lo<30: return {"sym":sym,"tf":tf,"err":"short history","n":n}
    stride=max(1,(cut-lo)//E.PRM["samples"]); train=list(range(lo,cut,stride))[:E.PRM["samples"]]
    wts=fit_weights(raw,sym,tf,train)
    hdf=raw.get((sym,HTF[tf])) if HTF[tf] else None
    fv,ftd,fhb=edge.filters(df,hdf)
    cdir,_=edge.carry_dir(sym,E.RATES) if E.PRM["carry_filter"] else (0,None)
    ltf_tf=edge.LTF.get(tf); masks=edge.protect_masks(df,tf,sym,raw.get((sym,ltf_tf)) if ltf_tf else None, ltf_tf)
    catl=None; dirs=np.zeros(n,int)
    # names/cats from one full pass at the last train bar
    agents=None
    for i in range(cut, hi):
        v,c,rk=E.votes_at(raw,sym,tf,i)
        if catl is None:
            # regime categories follow the stored agent order
            catl=[E.W["categories"][nm] for nm in E.W["agents"]]
        s=E.consensus(v,c,rk,wts,catl,E.W["agents"])
        d=1 if s>=E.PRM["buy_th"] else -1 if s<=E.PRM["sell_th"] else 0
        if d and fv[i] and ftd[i]==d and (hdf is None or fhb[i]==d) and (cdir==0 or cdir==d):
            dirs[i]=d
    sh,sl=E.pivots_confirmed(df.h.values,df.l.values,3)
    highs=[(i,float(sh[i])) for i in range(n) if sh[i]==sh[i]]
    lows=[(i,float(sl[i])) for i in range(n) if sl[i]==sl[i]]
    A=E.atr(df).values
    base=dirs.copy()
    rng=base.copy(); rng[masks["range"]]=0
    sw=base.copy(); sw[(sw>0)&masks["blockLong"]]=0; sw[(sw<0)&masks["blockShort"]]=0
    both=base.copy(); both[masks["range"]]=0; both[(both>0)&masks["blockLong"]]=0; both[(both<0)&masks["blockShort"]]=0
    pack={}
    for name, d in (("base",base),("range",rng),("sweep",sw),("both",both)):
        rs=trades_for(df,sym,d,A,highs,lows)
        pack[name]=metr(rs)
        pack[name+"_R"]=[round(float(x),4) for x in rs]
    out={"sym":sym,"tf":tf,"bars":int(hi-cut),"secs":round(time.time()-t0,1),
         "base":pack["base"],"range":pack["range"],"sweep":pack["sweep"],"both":pack["both"],
         "base_R":pack["base_R"],"range_R":pack["range_R"],"sweep_R":pack["sweep_R"],"both_R":pack["both_R"],
         "range_blocked":int(((base!=0)&(rng==0)).sum()),
         "sweep_blocked":int(((base!=0)&(sw==0)).sum())}
    print(sym, tf, "done", out["secs"], "s base", out["base"].get("n"), out["base"].get("exp"), out["base"].get("pf"), flush=True)
    return out

def not_worse(base, new):
    """True when expectancy and profit factor are not lower. An empty book is not an improvement."""
    if not new or new.get("n",0)==0 or not base or base.get("n",0)==0: return False
    if new["exp"]<base["exp"]-1e-9: return False
    if base.get("pf") is None: return True
    if new.get("pf") is None: return True
    return new["pf"]+1e-9>=base["pf"]

def combo_ok(rows, arm):
    """Pooled edge book must not lose expectancy or PF, and no combo with 8+ baseline trades may lose expectancy."""
    base=metr([x for r in rows for x in r.get("base_R") or []])
    new=metr([x for r in rows for x in r.get(arm+"_R") or []])
    if not not_worse(base, new): return False, base, new
    for r in rows:
        b,n=r.get("base") or {}, r.get(arm) or {}
        if b.get("n",0)>=8 and n.get("n",0) and n["exp"]<b["exp"]-1e-9: return False, base, new
        if b.get("n",0)>=8 and n.get("n",0)==0: return False, base, new
    return True, base, new

RAW=None
def run_pair(pair):
    return one(RAW, pair[0], pair[1])

def nov2(raw):
    """The 2 Nov 2026 XAUUSD 1h sell: coil on the 15m box, then the sweep that broke the other side."""
    h1=raw[("XAUUSD","1h")]; m15=raw[("XAUUSD","15m")]
    masks=edge.protect_masks(h1,"1h","XAUUSD",m15,"15m")
    bL,bS,_,_=edge.sweep_series(m15,"15m","XAUUSD")
    rows=[]
    for i,t in enumerate(h1.t.values):
        dt=datetime.fromtimestamp(int(t)/1000, TZ)
        if dt.month!=11 or dt.day!=2 or dt.year!=2026: continue
        rows.append({"bar":dt.strftime("%a %d %b %H:%M"),"close":round(float(h1.c.iloc[i]),2),
                     "range":bool(masks["range"][i]),"lo":None if masks["lo"][i]!=masks["lo"][i] else round(float(masks["lo"][i]),2),
                     "hi":None if masks["hi"][i]!=masks["hi"][i] else round(float(masks["hi"][i]),2),
                     "src":str(masks["src"][i]),"blockShort":bool(masks["blockShort"][i])})
    sweeps=[]
    for i in range(len(m15)):
        if not (bS[i] and not (i and bS[i-1])): continue
        dt=datetime.fromtimestamp(int(m15.t.iloc[i])/1000, TZ)
        if dt.month==11 and dt.day==2 and dt.year==2026:
            sweeps.append(dt.strftime("%a %d %b %H:%M"))
    flagged=any(r["range"] and r["bar"].endswith("04:00") for r in rows)
    return {"flagged_04h_sell_bar":flagged,"bars":rows,"m15_short_blocks_starting":sweeps}

def main():
    import json
    global RAW
    E.RATES=edge.policy_rates(os.path.join(E.STATE,"rates.json"))
    raw=fetch(); RAW=raw
    case=nov2(raw)
    print("NOV2 flagged", case["flagged_04h_sell_bar"], "sweeps", case["m15_short_blocks_starting"], flush=True)
    try:
        import multiprocessing as mp
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass
    from multiprocessing import Pool
    with Pool(4) as P: rows=P.map(run_pair, EDGE)
    summary={"base":metr([x for r in rows for x in r.get("base_R") or []])}
    decision={}
    for arm in ("range","sweep","both"):
        ok, base, new = combo_ok(rows, arm)
        summary[arm]=new
        decision[arm]=ok
    # Ship the widest set that does not reduce the edge book. Range is preferred over sweep
    # because it is the gate that catches the 2 Nov gold short while price was still inside the coil.
    live=[]
    if decision.get("both"): live=["range","sweep"]
    elif decision.get("range"): live=["range"]
    elif decision.get("sweep"): live=["sweep"]
    per=[]
    for r in rows:
        per.append({"sym":r["sym"],"tf":r["tf"],"bars":r.get("bars"),"secs":r.get("secs"),
                    "base":r.get("base"),"range":r.get("range"),"sweep":r.get("sweep"),"both":r.get("both"),
                    "range_blocked":r.get("range_blocked"),"sweep_blocked":r.get("sweep_blocked"),"err":r.get("err")})
    doc={"note":"Unseen last 40% after a 60% barrier weight fit. Exit is the live chandelier (BE at +1R, 4×ATR trail, 48 bars) after costs. News has no multi-month calendar in the feed, so it is not in this pass and stays display-only.",
         "live_for_alerts":live,"decision":decision,"pooled":summary,"per_combo":per,"nov2":case,
         "news":"display-only (Forex Factory week feed has no history long enough to score the edge book)"}
    path=os.path.join(os.path.dirname(os.path.abspath(__file__)),"protect_oos.json")
    json.dump(doc, open(path,"w"), indent=1)
    print(json.dumps({"live":live,"pooled":summary,"decision":decision}, default=str), flush=True)
    return doc

if __name__=="__main__":
    main()
