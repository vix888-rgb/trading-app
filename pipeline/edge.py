# edge.py - evidence-based additions (Nov 2026 research pass). All series are causal (value at bar i uses bars <= i).
import numpy as np, pandas as pd
from zoneinfo import ZoneInfo
NY=ZoneInfo("America/New_York")
def _ema(s,n): return s.ewm(span=n,adjust=False).mean()
def _atr(df,n=14):
    pc=df.c.shift(); tr=pd.concat([df.h-df.l,(df.h-pc).abs(),(df.l-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False).mean()
def _rsi(c,n):
    d=c.diff(); up=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean(); dn=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean()
    return (100-100/(1+up/dn.replace(0,np.nan))).where(dn>0,100.0)
TSM_LB=(20,60,120)
def tsmom(c,lbs=TSM_LB):
    """Time-series momentum (Moskowitz-Ooi-Pedersen style, multi-lookback): mean sign of return over each lookback, in [-1,1]."""
    return sum(np.sign(c-c.shift(L)).fillna(0) for L in lbs)/len(lbs)
def atr_pct(df,n=250):
    """Percentile rank (0..1) of ATR14/price within the last n bars (volatility regime)."""
    a=_atr(df)/df.c
    return a.rolling(n,min_periods=60).rank(pct=True)
def htf_bias(df,htf):
    """Higher-TF EMA20 vs EMA50 sign, known only after the HTF bar closes."""
    if htf is None or len(htf)<60: return np.zeros(len(df))
    hb=pd.DataFrame({"t":(htf.t+htf.t.diff().median()).astype("int64").values,"b":np.sign(_ema(htf.c,20)-_ema(htf.c,50)).values}).sort_values("t")
    m=pd.merge_asof(pd.DataFrame({"t":df.t.values.astype("int64")}),hb,on="t")
    return m.b.fillna(0).values
def filters(df,htf):
    """Quality filters: vol regime OK (ATR pct 0.20-0.95), TSMOM(60) direction, HTF bias. Returns (volok, trend_dir, htf_dir)."""
    ap=atr_pct(df).values; volok=(ap>=0.20)&(ap<=0.95)
    td=np.sign(df.c-df.c.shift(60)).fillna(0).values
    return volok,td,htf_bias(df,htf)
# ---------- new strategies (vectorised) ----------
KEEP_NEW=[0,1]  # kept after Nov-2026 walk-forward: TSMOM, Keltner (positive OOS on 1d). RSI(2), gold NY ORB, Supertrend+MTF did not hold up OOS after costs -> not shown
NEW_STRATS=["TSMOM multi-lookback trend","Keltner volatility breakout","Connors RSI(2) + 200 MA","Supertrend flip + MTF filter","Gold NY opening-range breakout"]
NEW_RULES=["Returns over 20, 60 and 120 bars all turn positive (all three agree) → long; all negative → short; ATR percentile 20–95%",
 "Close crosses above EMA20 + 2×ATR20 (Keltner upper) with TSMOM(60) up → long (mirror short)",
 "Close above SMA200 and RSI(2) < 10 → long; below SMA200 and RSI(2) > 90 → short; exit on close back through SMA5",
 "Supertrend(10,3) flips up while higher-TF EMA20>EMA50 and 60-bar return > 0 → long (mirror short)",
 "Range of 08:30–09:00 New York (14:30/15:30 Belgrade); first close beyond it before 11:30 NY → trade the break (5m/15m only)"]
def supertrend_dir(df,n=10,m=3):
    a=_atr(df,n).values; hl=((df.h+df.l)/2).values; ub=hl+m*a; lb=hl-m*a; c=df.c.values
    fu=ub.copy(); fl=lb.copy(); d=np.ones(len(c))
    for i in range(1,len(c)):
        fu[i]=ub[i] if (ub[i]<fu[i-1] or c[i-1]>fu[i-1]) else fu[i-1]
        fl[i]=lb[i] if (lb[i]>fl[i-1] or c[i-1]<fl[i-1]) else fl[i-1]
        d[i]=1 if c[i]>fu[i-1] else (-1 if c[i]<fl[i-1] else d[i-1])
    return d
def new_strat_signals(df,tf,htf):
    c=df.c; n=len(df); L=[np.zeros(n,bool) for _ in NEW_STRATS]; S=[np.zeros(n,bool) for _ in NEW_STRATS]
    ts=tsmom(c); ap=atr_pct(df); vok=((ap>=0.2)&(ap<=0.95)).values; td=np.sign(c-c.shift(60)).fillna(0).values
    L[0]=((ts>=1)&(ts.shift()<1)).values&vok; S[0]=((ts<=-1)&(ts.shift()>-1)).values&vok
    a20=_atr(df,20); e20=_ema(c,20); ku=e20+2*a20; kl=e20-2*a20
    L[1]=((c>ku)&(c.shift()<=ku.shift())).values&(td>0); S[1]=((c<kl)&(c.shift()>=kl.shift())).values&(td<0)
    r2=_rsi(c,2); s200=c.rolling(200).mean()
    L[2]=((c>s200)&(r2<10)).values; S[2]=((c<s200)&(r2>90)).values
    sd=supertrend_dir(df); hb=htf_bias(df,htf); flipU=np.r_[False,(sd[1:]>0)&(sd[:-1]<0)]; flipD=np.r_[False,(sd[1:]<0)&(sd[:-1]>0)]
    L[3]=flipU&(hb>0)&(td>0); S[3]=flipD&(hb<0)&(td<0)
    if tf in ("5m","15m"):
        ix=pd.to_datetime(df.t,unit="ms",utc=True).dt.tz_convert(NY); mins=(ix.dt.hour*60+ix.dt.minute).values; day=ix.dt.date.values
        tmp=pd.DataFrame({"d":day,"m":mins,"h":df.h.values,"l":df.l.values})
        orr=tmp[(tmp.m>=510)&(tmp.m<540)].groupby("d").agg(oh=("h","max"),ol=("l","min"),k=("h","size"))
        tmp=tmp.join(orr,on="d"); done=set(); cv=c.values; need=6 if tf=="5m" else 2
        for i in range(n):
            if 540<=mins[i]<690 and tmp.k.iloc[i]==tmp.k.iloc[i] and tmp.k.iloc[i]>=need and day[i] not in done:
                if cv[i]>tmp.oh.iloc[i]: L[4][i]=True; done.add(day[i])
                elif cv[i]<tmp.ol.iloc[i]: S[4][i]=True; done.add(day[i])
    exits={2:((c>c.rolling(5).mean()).values,(c<c.rolling(5).mean()).values)}  # native exits (RSI2)
    return L,S,exits
# ---------- new agent (vote at last bar) ----------
def tsmom_agent(df):
    ts=tsmom(df.c); x=float(ts.iloc[-1]); ap=float(atr_pct(df).iloc[-1]) if len(df)>60 else float("nan")
    vote="bullish" if x>=0.99 else "bearish" if x<=-0.99 else "neutral"
    conf=60 if vote!="neutral" else 20
    parts=", ".join(f"{L}b {'up' if df.c.iloc[-1]>df.c.iloc[-1-L] else 'down'}" for L in TSM_LB if len(df)>L)
    return {"agent":"Time-series momentum (20/60/120)","vote":vote,"conf":conf,"reason":f"Returns over {parts}; "+("all agree" if vote!="neutral" else "lookbacks disagree")+(f"; ATR percentile {ap*100:.0f}%" if ap==ap else "")}
# ---------- Round 2 (Nov 2026): carry filter + turn-of-month ----------
import json as _json, os as _os, time as _time, urllib.request as _ur
FRED={"US":"DFF","EU":"ECBDFR","JP":"IRSTCI01JPM156N","CH":"IR3TIB01CHM156N"}
CARRY_PAIRS={"USDJPY":("US","JP"),"USDCHF":("US","CH"),"EURUSD":("EU","US"),"EURCHF":("EU","CH")}
def policy_rates(cache):
    """Latest policy / short rates from FRED (free CSV), cached for 24h in `cache`; returns {cc:(rate,date)}."""
    old=_json.load(open(cache)) if _os.path.exists(cache) else {}
    if old and _time.time()-old.get("_ts",0)<86400: return old
    out={"_ts":_time.time()}
    for cc,sid in FRED.items():
        try:
            txt=_ur.urlopen(_ur.Request(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd=2024-01-01",headers={"User-Agent":"curl/8.5.0"}),timeout=20).read().decode()
            rows=[r.split(",") for r in txt.strip().splitlines()[1:] if r.split(",")[1] not in ("",".")]
            out[cc]=[float(rows[-1][1]),rows[-1][0]]
        except Exception:
            if cc in old: out[cc]=old[cc]
    _json.dump(out,open(cache,"w")); return out
def carry_dir(sym,rates,th=0.25):
    """+1 = only longs have positive carry, -1 = only shorts, 0 = no carry filter (differential within ±0.25% or unknown)."""
    if sym not in CARRY_PAIRS: return 0,None
    a,b=CARRY_PAIRS[sym]
    if a not in rates or b not in rates: return 0,None
    d=rates[a][0]-rates[b][0]; return (1 if d>th else -1 if d<-th else 0),round(d,2)
TOM_NAME="Turn-of-month long (1D)"
TOM_RULE="Buy the close of the month's last trading day, exit 3 days later. Out-of-sample positive on gold (train +0.24R, test +0.32R per trade) and BTC in train only; negative on FX pairs in both halves"
def tom_signal(df,tf):
    n=len(df); L=np.zeros(n,bool)
    if tf!="1d" or n<3: return L
    dt=pd.to_datetime(df.t,unit="ms",utc=True); wk=(dt.dt.weekday>=5).any()
    nxt=dt+(pd.Timedelta(days=1) if wk else pd.offsets.BDay(1))
    return (nxt.dt.month!=dt.dt.month).values
# ---------- Rebuild (Nov 2026): trader-model research, see research/TRADER_MODELS.md ----------
SWEEP_NAME="ICT/TJR sweep → MSS → FVG retrace"
SWEEP_RULE=("Price runs the 20-bar low (high) and closes back inside, then within 10 bars a candle closes beyond the last swing high (low) = market-structure shift, "
            "leaving a fair value gap in the move → buy (sell) limit at the gap edge, stop beyond the swept extreme (0.5–3×ATR). Out-of-sample on 1h/4h/1D, 6 symbols: "
            "+0.13R/trade after costs (PF 1.31, 141 trades, not significant); display only, no alerts")
def _fractal_last(h,l,w=2):
    n=len(h); lh=np.full(n,np.nan); ll=np.full(n,np.nan)
    for i in range(w,n-w):
        if h[i]==h[i-w:i+w+1].max() and h[i]>h[i-w:i].max(): lh[i+w]=h[i]
        if l[i]==l[i-w:i+w+1].min() and l[i]<l[i-w:i].min(): ll[i+w]=l[i]
    return pd.Series(lh).ffill().values,pd.Series(ll).ffill().values
def sweep_mss_fvg_signals(df,lb=20):
    """Causal: signal is flagged on the MSS bar k (known at its close). Entry is a limit at the FVG edge (plan level), not market."""
    o,h,l,c=df.o.values,df.h.values,df.l.values,df.c.values; n=len(df); L=np.zeros(n,bool); S=np.zeros(n,bool)
    if n<lb+10: return L,S
    A=_atr(df).values; lastH,lastL=_fractal_last(h,l)
    hh=pd.Series(h).rolling(lb).max().shift(1).values; ll_=pd.Series(l).rolling(lb).min().shift(1).values
    i=lb+5
    while i<n:
        d=1 if (l[i]<ll_[i] and c[i]>ll_[i]) else -1 if (h[i]>hh[i] and c[i]<hh[i]) else 0
        if d==0: i+=1; continue
        ext=l[i] if d>0 else h[i]; ref=lastH[i] if d>0 else lastL[i]; hit=None
        for k in range(i+1,min(n,i+10)):
            ext=min(ext,l[k]) if d>0 else max(ext,h[k])
            if ref==ref and ((d>0 and c[k]>ref) or (d<0 and c[k]<ref)):
                gap=any((l[q]>h[q-2]) if d>0 else (h[q]<l[q-2]) for q in range(i+2,k+1))
                if gap:
                    risk=abs(c[k]-ext)
                    if 0.5*A[k]<=risk<=3.5*A[k]: hit=k
                break
        if hit is None: i+=1; continue
        (L if d>0 else S)[hit]=True; i=hit+1
    return L,S
