# Trader-model research library (Nov 2026). Causal signals, conservative fills (stop before target inside a bar), costs in R.
import numpy as np, pandas as pd, pickle
from zoneinfo import ZoneInfo
NY=ZoneInfo("America/New_York"); LDN=ZoneInfo("Europe/London")
COST={"XAUUSD":("abs",0.35),"BTCUSD":("pct",0.0006),"EURUSD":("abs",0.00008),"USDJPY":("abs",0.010),"USDCHF":("abs",0.00013),"EURCHF":("abs",0.00018)}
PIP={"XAUUSD":0.1,"BTCUSD":None,"EURUSD":1e-4,"USDJPY":0.01,"USDCHF":1e-4,"EURCHF":1e-4}
MAXCOST=0.25
def cost_of(sym,px): k,v=COST[sym]; return v if k=="abs" else v*px
def atr(df,n=14):
    pc=df.c.shift(); tr=pd.concat([df.h-df.l,(df.h-pc).abs(),(df.l-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False).mean().values
def et_cols(df,tz=NY):
    ix=pd.to_datetime(df.t,unit="ms",utc=True).dt.tz_convert(tz)
    return (ix.dt.hour*60+ix.dt.minute).values, ix.dt.date.values, ix.dt.weekday.values
def resample(df5,rule):
    ix=pd.to_datetime(df5.t,unit="ms",utc=True)
    g=df5.set_index(ix).resample(rule,label="left",closed="left").agg({"t":"first","o":"first","h":"max","l":"min","c":"last","v":"sum"}).dropna()
    g["t"]=g.index.as_unit("ms").asi8.astype("int64"); return g.reset_index(drop=True)
def fractals(h,l,w=2):
    """swing highs/lows; value known at bar i+w (confirmation index returned)."""
    n=len(h); SH=[]; SL=[]
    for i in range(w,n-w):
        if h[i]==max(h[i-w:i+w+1]) and h[i]>max(h[i-w:i]): SH.append((i+w,i,h[i]))
        if l[i]==min(l[i-w:i+w+1]) and l[i]<min(l[i-w:i]): SL.append((i+w,i,l[i]))
    return SH,SL
def fvg_bias_states(df):
    """PB Blake step 1 (respect vs disrespect of FVGs), state at each bar close: +1 last event bullish (bull FVG respected or bear FVG disrespected), -1 bearish, 0 none."""
    o,h,l,c=df.o.values,df.h.values,df.l.values,df.c.values; n=len(df); st=np.zeros(n); cur=0; act=[]  # (dir,bot,top,born)
    for i in range(n):
        new=[]
        for g in act:
            d,b,t,born=g
            if i<=born: new.append(g); continue
            if d>0:
                if c[i]<b: cur=-1; continue            # bullish gap disrespected (closed through)
                if l[i]<=t and c[i]>t: cur=1; new.append(g); continue  # traded into, closed back above = respected
            else:
                if c[i]>t: cur=1; continue
                if h[i]>=b and c[i]<b: cur=-1; new.append(g); continue
            new.append(g)
        act=new[-30:]
        if i>=2:
            if l[i]>h[i-2]: act.append((1,h[i-2],l[i],i))
            if h[i]<l[i-2]: act.append((-1,h[i],l[i-2],i))
        st[i]=cur
    return st
def map_htf(df,htf,vals,htf_ms):
    """value of an HTF series known at each LTF bar close (HTF bar must be closed)."""
    av=(htf.t.values+htf_ms).astype("int64"); close=df.t.values+ (df.t.diff().median())
    j=np.searchsorted(av,close,"right")-1
    return np.where(j>=0,np.asarray(vals)[np.clip(j,0,None)],0)
def sim_trade(o,h,l,c,j0,d,e,stop,tp,flat,be=None,trail=None,A=None,cost=0.0):
    """fill at bar j0 at price e. Returns (R,exit_idx). be: R-multiple to move stop to entry. trail: ATR multiple (after be)."""
    risk=abs(e-stop)
    if risk<=0 or cost>MAXCOST*risk: return None   # untradeable: spread/commission would eat >25% of the stop
    st=stop; armed=False; ext=e
    for j in range(j0,min(flat,len(c)-1)+1):
        if (d>0 and l[j]<=st) or (d<0 and h[j]>=st): return (d*(st-e)/risk-cost/risk,j)
        if tp is not None and ((d>0 and h[j]>=tp) or (d<0 and l[j]<=tp)): return (d*(tp-e)/risk-cost/risk,j)
        ext=max(ext,h[j]) if d>0 else min(ext,l[j])
        if be is not None and (ext-e)*d>=be*risk: armed=True
        if armed: st=max(st,e) if d>0 else min(st,e)
        if armed and trail is not None: t_=ext-d*trail*A; st=max(st,t_) if d>0 else min(st,t_)
    j=min(flat,len(c)-1)
    if j>=len(c)-1 and flat>len(c)-1: return None
    return (d*(c[j]-e)/risk-cost/risk,j)
def fill_limit(o,h,l,start,end,d,px):
    """first bar in [start,end] touching limit px -> (j,fill). A bar that gaps through fills at its open."""
    for j in range(start,min(end,len(h)-1)+1):
        if (d>0 and l[j]<=px) or (d<0 and h[j]>=px): return (j,min(o[j],px) if d>0 else max(o[j],px))
    return None
def metr(R):
    R=np.asarray(R,float)
    if len(R)==0: return {"n":0}
    gw=R[R>0].sum(); gl=-R[R<0].sum(); eq=np.cumsum(R); dd=float((np.maximum.accumulate(np.r_[0,eq])[1:]-eq).max())
    t=float(R.mean()/R.std(ddof=1)*np.sqrt(len(R))) if len(R)>2 and R.std()>0 else None
    return {"n":int(len(R)),"win":round(100*float((R>0).mean()),1),"exp":round(float(R.mean()),3),"pf":round(float(gw/gl),2) if gl>0 else None,"dd":round(dd,1),"tot":round(float(R.sum()),1),"t":None if t is None else round(t,2)}
