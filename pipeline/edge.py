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
# ---------- Nov 2026: pre-breakout protections (range coil, sweep-then-break, USD news window) ----------
# Alert flags follow research/protect_oos.json (unseen last 40%, chandelier + costs).
# Range lowered XAUUSD 1d (1.161R → 0.83R, PF 4.18 → 3.48) and BTCUSD 1h, so it is display-only.
# Sweep left every edge combo's expectancy and profit factor unchanged, so it may block alerts.
# The USD high-impact window has no multi-month calendar in the week feed, so it stays display-only.
RANGE_ALERTS=False
SWEEP_ALERTS=True
NEWS_ALERTS=False
LTF={"5m":"1m","15m":"5m","1h":"15m","4h":"1h","1d":"4h"}
STEP={"1m":60_000,"5m":300_000,"15m":900_000,"1h":3_600_000,"4h":14_400_000,"1d":86_400_000}
# Prior-N box. ATR caps sit near the 10th–15th percentile of width/ATR (measured on the feeds, Nov 2026).
# Percent caps are per symbol class. Gold 15m is 0.75% so the 2 Nov coil (~0.35–0.72% on 15m, 2.6×ATR at the 04:00 signal) qualifies.
RANGE_N={"1m":48,"5m":36,"15m":16,"1h":12,"4h":10,"1d":10}
RANGE_ATR={"1m":2.6,"5m":3.0,"15m":2.80,"1h":2.55,"4h":2.20,"1d":2.15}
_PCT={"XAU":{"1m":0.0030,"5m":0.0045,"15m":0.0075,"1h":0.0100,"4h":0.020,"1d":0.035},
      "BTC":{"1m":0.0050,"5m":0.0080,"15m":0.0120,"1h":0.0150,"4h":0.030,"1d":0.080},
      "FX":{"1m":0.0008,"5m":0.0012,"15m":0.0020,"1h":0.0030,"4h":0.007,"1d":0.017}}
# Sweep of a tight box, then a close through the other side. 15m uses 8 bars (~2h): the same clock window as the 20-bar
# pool on 5m, which is what caught the 2 Nov 14:00 sweep of 3907.8 and the 14:15 break of 3916.4. 5m/1h/4h/1d keep the ICT 20-bar pool.
SWEEP_LB={"1m":20,"5m":20,"15m":8,"1h":20,"4h":20,"1d":20}
SWEEP_BRK={"1m":10,"5m":10,"15m":6,"1h":10,"4h":6,"1d":5}
SWEEP_HOLD={"1m":30,"5m":12,"15m":8,"1h":6,"4h":4,"1d":3}
NEWS_WINDOW_MS=30*60*1000
FF_CAL="https://nfs.faireconomy.media/ff_calendar_thisweek.json"
def _klass(sym):
    if sym=="XAUUSD": return "XAU"
    if sym in ("BTCUSD","ETHUSD","SOLUSD","XRPUSD"): return "BTC"
    return "FX"
def _pct_cap(sym,tf):
    return _PCT[_klass(sym)].get(tf,0.02)
def range_series(df,tf,sym):
    """Causal tight-range mask. Box is the prior N bars (current bar excluded), so a close through the box is a breakout, not a coil."""
    n=len(df); z=np.zeros(n,bool); hi=np.full(n,np.nan); lo=np.full(n,np.nan)
    N=RANGE_N.get(tf); cap=RANGE_ATR.get(tf)
    if df is None or N is None or n<=N+2: return z,hi,lo
    rh=df.h.rolling(N).max().shift(1).values; rl=df.l.rolling(N).min().shift(1).values
    A=_atr(df).values; c=df.c.values; w=rh-rl
    pct=_pct_cap(sym,tf)
    inside=(c>rl)&(c<rh)&np.isfinite(w)&(w>0)&np.isfinite(A)&(A>0)
    tight=inside&((w<=cap*A)|(w<=pct*c))
    return tight,rh,rl
def sweep_series(df,tf,sym):
    """ICT-style sweep (wick through the pool, close back inside) of a tight box, then a close through the opposite side.
    Returns block_long, block_short, and the swept box (forward-filled while the block is active)."""
    n=len(df); bL=np.zeros(n,bool); bS=np.zeros(n,bool); lo=np.full(n,np.nan); hi=np.full(n,np.nan)
    lb=SWEEP_LB.get(tf,20); brk=SWEEP_BRK.get(tf,10); hold=SWEEP_HOLD.get(tf,6); cap=RANGE_ATR.get(tf,3.0)
    if n<lb+5: return bL,bS,lo,hi
    h,l,c=df.h.values,df.l.values,df.c.values; A=_atr(df).values
    hh=pd.Series(h).rolling(lb).max().shift(1).values; ll=pd.Series(l).rolling(lb).min().shift(1).values
    i=lb+1
    while i<n:
        if not (hh[i]==hh[i] and ll[i]==ll[i] and A[i]==A[i] and A[i]>0): i+=1; continue
        width=hh[i]-ll[i]
        if width<=0 or width>cap*A[i]: i+=1; continue
        down=l[i]<ll[i] and c[i]>ll[i]; up=h[i]>hh[i] and c[i]<hh[i]
        if not down and not up: i+=1; continue
        side=-1 if down and (not up or (ll[i]-l[i])>=(h[i]-hh[i])) else 1  # -1 swept the low (bullish reversal candidate)
        rh,rl=float(hh[i]),float(ll[i]); hit=None
        for k in range(i+1,min(n,i+1+brk)):
            if side<0 and c[k]>rh: hit=k; break
            if side>0 and c[k]<rl: hit=k; break
        if hit is None: i+=1; continue
        sl=slice(hit,min(n,hit+hold))
        if side<0: bS[sl]=True  # broke up after a downside sweep → block shorts
        else: bL[sl]=True
        hi[sl]=rh; lo[sl]=rl
        i=hit+1
    return bL,bS,lo,hi
def _align_idx(df,other,step,ostep):
    src=(other.t.values+ostep).astype("int64"); dst=(df.t.values+step).astype("int64")
    return np.searchsorted(src,dst,side="right")-1
def protect_masks(df,tf,sym,ltf=None,ltf_tf=None):
    """Per-bar masks on the signal timeframe. A lower-timeframe coil or sweep counts once that lower bar has closed."""
    n=len(df)
    rt,rh,rl=range_series(df,tf,sym); src=np.where(rt,tf,"")
    bL,bS,slo,shi=sweep_series(df,tf,sym)
    if ltf is not None and ltf_tf in STEP and tf in STEP and len(ltf)>10:
        idx=_align_idx(df,ltf,STEP[tf],STEP[ltf_tf]); ok=idx>=0
        lt,lrh,lrl=range_series(ltf,ltf_tf,sym)
        use=ok&lt[np.clip(idx,0,None)]
        # Prefer the tighter box when both timeframes are inside a coil.
        both=rt&use
        lhi=lrh[np.clip(idx,0,None)]; llo=lrl[np.clip(idx,0,None)]
        take_l=use&(~rt|((lhi-llo)<(rh-rl)))
        rt=rt|use
        rh=np.where(take_l,lhi,rh); rl=np.where(take_l,llo,rl)
        src=np.where(take_l,ltf_tf,np.where(rt,tf,src))
        lbL,lbS,llo2,lhi2=sweep_series(ltf,ltf_tf,sym)
        jb=np.clip(idx,0,None)
        addL=ok&lbL[jb]; addS=ok&lbS[jb]
        slo=np.where(addL|addS,np.where(np.isfinite(llo2[jb]),llo2[jb],slo),slo)
        shi=np.where(addL|addS,np.where(np.isfinite(lhi2[jb]),lhi2[jb],shi),shi)
        bL=bL|addL; bS=bS|addS
    return {"range":rt,"hi":rh,"lo":rl,"src":src,"blockLong":bL,"blockShort":bS,"sHi":shi,"sLo":slo}
def _box(on,hi,lo,src,alert):
    if not on or not (hi==hi and lo==lo): return {"on":False,"alert":bool(alert)}
    return {"on":True,"lo":round(float(lo),8),"hi":round(float(hi),8),"src":str(src or ""),"alert":bool(alert)}
def protect_last(df,tf,sym,ltf=None,ltf_tf=None,events=None,close_ms=None):
    """Last closed bar. `events` is the USD high-impact list from load_usd_events. News never picks a direction."""
    m=protect_masks(df,tf,sym,ltf,ltf_tf); i=len(df)-1
    news={"on":False,"alert":bool(NEWS_ALERTS)}
    if events and close_ms:
        hit=news_hit(close_ms,events)
        if hit: news={"on":True,"title":hit["title"],"at":hit["at"],"alert":bool(NEWS_ALERTS)}
    sw_on=bool(m["blockLong"][i] or m["blockShort"][i])
    sweep={"on":sw_on,"blockLong":bool(m["blockLong"][i]),"blockShort":bool(m["blockShort"][i]),
           "dir":(1 if m["blockShort"][i] else -1 if m["blockLong"][i] else 0),"alert":bool(SWEEP_ALERTS)}
    if sw_on and m["sLo"][i]==m["sLo"][i]:
        sweep["lo"]=round(float(m["sLo"][i]),8); sweep["hi"]=round(float(m["sHi"][i]),8)
    return {"range":_box(bool(m["range"][i]),m["hi"][i],m["lo"][i],m["src"][i],RANGE_ALERTS),"sweep":sweep,"news":news}
def apply_protect(sig,gate):
    """Downgrade BUY/SELL to WAIT only for filters that are armed for alerts. Returns (sig, why)."""
    if sig not in ("BUY","SELL"): return sig,""
    if gate["range"]["on"] and RANGE_ALERTS: return "WAIT","range"
    if sig=="BUY" and gate["sweep"].get("blockLong") and SWEEP_ALERTS: return "WAIT","sweep"
    if sig=="SELL" and gate["sweep"].get("blockShort") and SWEEP_ALERTS: return "WAIT","sweep"
    if gate["news"]["on"] and NEWS_ALERTS: return "WAIT","news"
    return sig,""
def load_usd_events(cache,now=None):
    """High-impact USD events from the Forex Factory week feed. Cached 6h. Failure keeps the previous cache, else none.
    This feed is the calendar the scan can actually read; it does not vote on direction."""
    now=now or _time.time(); old={}
    try:
        if _os.path.exists(cache): old=_json.load(open(cache))
    except Exception: old={}
    events=old.get("events") or []
    if old.get("_ts") and now-old["_ts"]<6*3600: return events
    try:
        raw=_ur.urlopen(_ur.Request(FF_CAL,headers={"User-Agent":"Mozilla/5.0"}),timeout=15).read()
        rows=_json.loads(raw.decode())
        out=[]
        for e in rows:
            if str(e.get("country","")).upper()!="USD": continue
            if str(e.get("impact","")).lower()!="high": continue
            ts=pd.Timestamp(e.get("date"))
            if ts.tzinfo is None: ts=ts.tz_localize("UTC")
            out.append({"title":str(e.get("title") or "USD")[:80],"at":ts.isoformat(),"ms":int(ts.timestamp()*1000)})
        _json.dump({"_ts":now,"events":out},open(cache,"w"))
        return out
    except Exception:
        return events
def news_hit(close_ms,events,window=NEWS_WINDOW_MS):
    """Title of a high-impact USD event within ±window of this bar's close, else None."""
    if not events or close_ms is None: return None
    best=None
    for e in events:
        if abs(int(e["ms"])-int(close_ms))<=window:
            if best is None or abs(int(e["ms"])-int(close_ms))<abs(int(best["ms"])-int(close_ms)): best=e
    return best
# ---------- Liquidity Entry Zones (point scale, Nov 2026) ----------
# The indicator labels its inputs "pips", but those numbers are points (1/10 of a pip).
# Gold pip = 0.1, so a point is 0.01: sweep 30 points = $0.30, range 20 = $0.20,
# stop 500 = $5, target 1300 = $13 (R:R 2.6).
# BTC auto pip is assumed to be $1 (a dollar pip, the usual auto/index pip once price
# is in the thousands — not the 0.01 mintick). A point is then $0.10: sweep $3,
# minimum range $2, stop $50, target $130. A mintick point ($0.01, stop $5) would
# sit inside the BTC spread on every bar; that reading is not used.
LEZ_NAME="Liquidity Entry Zones"
LEZ_PIP={"XAUUSD":0.1,"PAXG":0.1,"BTCUSD":1.0}
LEZ_INPUT=(("sweep",30.0),("range",20.0),("sl",500.0),("tp",1300.0))
LEZ_MODE="point"          # desk + live signals. Research may score other modes; this is the shipped reading.
LEZ_FILTER=False          # confirmation gate on the current alert. Set from the unseen test.
LEZ_ALERTS=False          # standalone alerts. Display-only while this stays false.
LEZ_WINDOW=0              # extra bars before the signal bar that still count as confirmation
LEZ_EDGE=("XAUUSD|1d","XAUUSD|4h","XAUUSD|1h","BTCUSD|1d","BTCUSD|1h")
LEZ_RULE=("Inputs are points (1/10 pip). Gold pip 0.1 → sweep $0.30, minimum candle $0.20, stop $5, target $13. "
          "BTC auto pip assumed $1 → point $0.10 → sweep $3, minimum candle $2, stop $50, target $130. "
          "EMA50: long only above, short only below. Pivots length 5, last 20 highs and 20 lows. "
          "Sweep trades past a stored level by the minimum and closes back inside; the sweep candle needs wick ≥35% and body ≤65%. "
          "Confirm on that bar or within the next 2: bullish (bearish) and a close through the sweep candle's midline. "
          "Next-bar open, 10-bar cooldown, one position. Desk stats subtract the Vantage round trip plus slippage. "
          "Unseen test: the point bracket and a wick-stop variant both trail the current system after costs, and the $50 BTC stop sits inside the spread. Display only, no alerts.")
def lez_point_size(sym):
    return LEZ_PIP.get(sym,0.0001)/10.0
def lez_bracket(sym):
    """(sweep, min range, stop, target) in price units for the point reading."""
    pt=lez_point_size(sym)
    return tuple(v*pt for _,v in LEZ_INPUT)
def lez_friction_parts(sym,px):
    """Vantage-like round trip (spread + commission) and extra adverse slippage, in price."""
    px=float(px)
    if sym in ("XAUUSD","PAXG"): return 0.35,0.10
    if sym=="BTCUSD": return 0.0006*px,0.0002*px
    return 0.0,0.0
def lez_net_r(gross,risk,friction):
    """R after a cash friction. None when friction would eat more than 25% of the stop (not taken)."""
    if gross is None or risk is None or risk<=0: return None
    if friction>0.25*risk: return None
    return float(gross)-float(friction)/float(risk)
def lez_basis(sym,mode):
    sw,rng,sl,tp=lez_bracket(sym)
    if mode=="point": return f"point bracket from the fill, stop {sl:g}, target {tp:g} (inputs are points = 1/10 pip)"
    if mode=="atr": return "ATR-scaled point bracket (distances × ATR / median ATR), 50:130 point ratio"
    if mode=="struct2": return "stop 1 point beyond the sweep wick, target 2R"
    if mode=="struct3": return "stop 1 point beyond the sweep wick, target 3R"
    return mode
def _lez_dists(sym,mode,atr_i,med):
    sw,rng,sl,tp=lez_bracket(sym)
    if mode=="atr":
        if med is None or med<=0 or not (atr_i==atr_i) or atr_i<=0: return None
        k=float(atr_i)/float(med)
        return sw*k,rng*k,sl*k,tp*k
    return sw,rng,sl,tp
def _lez_quality(o,h,l,c,i,d,min_range):
    rng=h[i]-l[i]
    if rng<min_range or rng<=0: return False
    if abs(c[i]-o[i])/rng>0.65+1e-12: return False
    wick=(min(o[i],c[i])-l[i]) if d>0 else (h[i]-max(o[i],c[i]))
    return wick/rng>=0.35-1e-12
def _lez_confirm(o,c,ema,i,d,mid):
    if not (ema[i]==ema[i]): return False
    if d>0: return c[i]>o[i] and c[i]>mid and c[i]>ema[i]
    return c[i]<o[i] and c[i]<mid and c[i]<ema[i]
def _lez_walk(o,h,l,c,j0,d,e,stop,tp):
    """Stop before target. A gap through the stop on the entry bar fills at the open. Returns (gross R, exit index, how)."""
    n=len(c); risk=abs(e-stop)
    if risk<=0: return None
    for j in range(j0,n):
        hit_sl=(d>0 and l[j]<=stop) or (d<0 and h[j]>=stop)
        if hit_sl:
            px=o[j] if (j==j0 and ((d>0 and o[j]<=stop) or (d<0 and o[j]>=stop))) else stop
            return (d*(px-e)/risk,j,"sl")
        hit_tp=tp is not None and ((d>0 and h[j]>=tp) or (d<0 and l[j]<=tp))
        if hit_tp:
            px=o[j] if (j==j0 and ((d>0 and o[j]>=tp) or (d<0 and o[j]<=tp))) else tp
            return (d*(px-e)/risk,j,"tp")
    j=n-1
    if j<j0: return None
    return (d*(c[j]-e)/risk,j,"mtm")
def lez_run(df,sym,tf,mode=None,med=None):
    """Causal Liquidity Entry Zones. Signal on the confirmation close; fill is the next open. No lookahead."""
    mode=mode or LEZ_MODE
    o,h,l,c=df.o.values.astype(float),df.h.values.astype(float),df.l.values.astype(float),df.c.values.astype(float)
    n=len(c); L=np.zeros(n,bool); S=np.zeros(n,bool)
    ema=pd.Series(c).ewm(span=50,adjust=False).mean().values
    A=_atr(df).values
    if mode=="atr" and med is None:
        ok=A[60:]
        ok=ok[np.isfinite(ok)&(ok>0)]
        med=float(np.median(ok)) if len(ok) else None
    highs=[]; lows=[]; signals=[]; plans=[]
    pending=None; last=-10**9; active_until=-1
    point=lez_point_size(sym)
    for i in range(n):
        p=i-5
        if p>=5:
            if h[p]==h[p-5:p+6].max() and h[p]>h[p-5:p].max():
                highs.append(float(h[p]))
                if len(highs)>20: highs.pop(0)
            if l[p]==l[p-5:p+6].min() and l[p]<l[p-5:p].min():
                lows.append(float(l[p]))
                if len(lows)>20: lows.pop(0)
        if i<=active_until:
            pending=None; continue
        if i<60 or i-last<10: continue
        fired=False
        if pending is not None and i<=pending["exp"] and _lez_confirm(o,c,ema,i,pending["d"],pending["mid"]):
            _lez_take(i,pending,o,h,l,c,L,S,signals,plans,sym,mode,point)
            last=i; active_until=_lez_block(signals); pending=None; fired=True
        elif pending is not None and i>=pending["exp"]:
            pending=None
        if fired or pending is not None: continue
        dist=_lez_dists(sym,mode,A[i],med)
        if dist is None or not (A[i]==A[i]) or A[i]<=0: continue
        sw,rng,sl,tp=dist
        if sw<=0 or rng<=0: continue
        cands=[]
        for lv in lows:
            if l[i]<=lv-sw and c[i]>lv: cands.append((lv-l[i],1,lv))
        for lv in highs:
            if h[i]>=lv+sw and c[i]<lv: cands.append((h[i]-lv,-1,lv))
        if not cands: continue
        cands.sort(key=lambda z: -z[0])
        # equal penetration on both sides is ambiguous
        if len(cands)>1 and abs(cands[0][0]-cands[1][0])<=1e-12 and cands[0][1]!=cands[1][1]: continue
        taken=None
        for pen,d,lv in cands:
            if not _lez_quality(o,h,l,c,i,d,rng): continue
            taken=(d,lv); break
        if taken is None: continue
        d,lv=taken
        if d>0: lows[:]=[x for x in lows if not (l[i]<=x-sw)]
        else: highs[:]=[x for x in highs if not (h[i]>=x+sw)]
        rec={"d":d,"mid":float((h[i]+l[i])/2),"exp":i+2,"extreme":float(l[i] if d>0 else h[i]),"sl":float(sl),"tp":float(tp),"sweep":i}
        if _lez_confirm(o,c,ema,i,d,rec["mid"]):
            _lez_take(i,rec,o,h,l,c,L,S,signals,plans,sym,mode,point)
            last=i; active_until=_lez_block(signals)
        else:
            pending=rec
    return {"L":L,"S":S,"signals":signals,"plans":plans,"mode":mode,"med":med}
def _lez_block(signals):
    if not signals: return -1
    j=signals[-1].get("j")
    return j if j is not None else -1
def _lez_take(i,rec,o,h,l,c,L,S,signals,plans,sym,mode,point):
    d=rec["d"]; (L if d>0 else S)[i]=True
    n=len(c); basis=lez_basis(sym,mode if not str(mode).startswith("struct") else mode)
    struct=str(mode).startswith("struct")
    rr=2.0 if mode=="struct2" else 3.0 if mode=="struct3" else None
    if i+1>=n:
        e=float(c[i]); stop,tp_px,risk=_lez_prices(e,d,rec,struct,rr,point)
        plans.append({"i":i,"d":d,"entry":e,"stop":stop,"tp":tp_px,"active":True,"how":None,"basis":basis})
        signals.append({"i":i,"d":d,"e":None,"risk":risk,"gross":None,"j":None,"how":None})
        return
    e=float(o[i+1]); stop,tp_px,risk=_lez_prices(e,d,rec,struct,rr,point)
    if risk is None or risk<=0 or (stop-e)*d>=0:
        plans.append({"i":i,"d":d,"entry":e,"stop":stop,"tp":tp_px,"active":False,"how":None,"basis":basis})
        signals.append({"i":i,"d":d,"e":e,"risk":None,"gross":None,"j":None,"how":"invalid"})
        return
    walked=_lez_walk(o,h,l,c,i+1,d,e,stop,tp_px)
    if walked is None:
        signals.append({"i":i,"d":d,"e":e,"risk":risk,"gross":None,"j":None,"how":None}); return
    gross,j,how=walked
    active=how=="mtm"
    plans.append({"i":i,"d":d,"entry":e,"stop":float(stop),"tp":float(tp_px),"active":active,"how":how,"basis":basis})
    signals.append({"i":i,"d":d,"e":e,"risk":float(risk),"gross":float(gross),"j":int(j),"how":how})
def _lez_prices(e,d,rec,struct,rr,point):
    if struct:
        stop=float(rec["extreme"]-d*point)
        risk=(e-stop)*d
        if risk<=0: return stop,e,None
        return stop,float(e+d*rr*risk),float(risk)
    sl,tp=rec["sl"],rec["tp"]
    return float(e-d*sl),float(e+d*tp),float(sl)
_LEZ_CACHE={}
def lez_cached(df,sym,tf,mode=None,med=None):
    mode=mode or LEZ_MODE
    key=(id(df),sym,tf,mode,None if med is None else round(float(med),8))
    hit=_LEZ_CACHE.get(key)
    if hit is None:
        hit=lez_run(df,sym,tf,mode=mode,med=med); _LEZ_CACHE[key]=hit
    return hit
def lez_sides(df,sym,tf):
    p=lez_cached(df,sym,tf); return p["L"],p["S"]
def lez_recent_dir(pack,i,window=None):
    if window is None: window=LEZ_WINDOW
    L,S=pack["L"],pack["S"]; d=0
    for k in range(max(0,i-int(window)),i+1):
        if L[k]: d=1
        elif S[k]: d=-1
    return d
def apply_lez(sig,d,sym,tf):
    """Standalone replaces the alert on the gold/BTC edge. Filter downgrades a side LEZ did not confirm."""
    if not (LEZ_FILTER or LEZ_ALERTS): return sig,""
    if f"{sym}|{tf}" not in LEZ_EDGE: return sig,""
    if LEZ_ALERTS:
        if d>0: return "BUY",""
        if d<0: return "SELL",""
        return "WAIT","lez"
    if sig=="BUY" and d!=1: return "WAIT","lez"
    if sig=="SELL" and d!=-1: return "WAIT","lez"
    return sig,""
def lez_trade_plan(pack,sym):
    """Consensus-plan shape for an alert sourced from the latest LEZ setup."""
    if not pack or not pack.get("plans"): return None
    pl=pack["plans"][-1]; e,s,t,d=pl["entry"],pl["stop"],pl["tp"],pl["d"]
    if e is None or s is None or t is None: return None
    risk=abs(e-s)
    tps=[e+d*risk,e+d*2*risk,t] if risk>0 else [t,t,t]
    return {"side":"long" if d>0 else "short","entry":float(e),"zone":[float(e),float(e)],
            "zoneWhy":"market (liquidity entry zone)","stop":float(s),"stopBasis":pl.get("basis") or lez_basis(sym,pack.get("mode") or LEZ_MODE),
            "inv":float(s),"tp":[float(x) for x in tps],"rr":[1,2,2.6],"dist":float(risk)}
def lez_desk_bt(df,sym,tf):
    """Loaded-bar stats with the Vantage round trip plus slippage. The plan is shown even when the stop is inside the spread."""
    pack=lez_cached(df,sym,tf); rs=[]; skipped=0
    for s in pack["signals"]:
        if s.get("gross") is None or not s.get("risk"): continue
        spread,slip=lez_friction_parts(sym,s["e"])
        net=lez_net_r(s["gross"],s["risk"],spread+slip)
        if net is None: skipped+=1; continue
        rs.append(net)
    if not rs: bt={"n":0}
    else:
        arr=np.array(rs,float); gw=arr[arr>0].sum(); gl=-arr[arr<0].sum()
        bt={"n":int(len(arr)),"win":int(round(100*float((arr>0).mean()))),"avgR":round(float(arr.mean()),2),
            "totR":round(float(arr.sum()),1),"pf":round(float(gw/gl),2) if gl>0 else None}
    bt["skip"]=skipped
    pl=pack["plans"][-1] if pack["plans"] else None
    bt["lez"]=None if pl is None else {"entry":float(pl["entry"]),"stop":float(pl["stop"]),"tp":float(pl["tp"]),"active":bool(pl.get("active"))}
    return bt

# ---------- Strategy rebuild round 4 (3 Nov 2026) ----------
# Research: pipeline/research/rebuild_2026_11_03/ (CHANGES.md, cand*.py, final_results.json). Harness = round-2 walk-forward (weights refit on the
# train half only, VAL = 3 folds inside train, TEST = unseen 2nd half, Vantage-like costs).
# Shipped: (1) Donchian-20 breakout confirmation gate on XAUUSD 1h alerts (BTC 1h display), (2) London+NY session gate on BTC 1h (display),
# (3) two display-only strategy rows, (4) indicator panel. Rejected and NOT used: structure/swing stops, 2xATR stop, Supertrend exit, EMA200/stack,
# Supertrend agree, RSI divergence, ADX>=20, AVWAP-1h, agent stability pruning, unions with the new agents (see research log).
BRK_NAME="Donchian-20 breakout + EMA200 + HTF (round 4)"
BRK_RULE=("Close breaks the prior 20-bar high (low) while above (below) EMA200 and the higher-TF EMA20/50 bias agrees → long (short). "
          "Unseen test (gold/BTC 1h/4h/1D + FX dailies): +0.315R/trade, PF 1.70, 249 trades; train folds +0.516R. Display only.")
SQZ_NAME="TTM squeeze fire + EMA200 (round 4)"
SQZ_RULE=("Bollinger(20,2) leaves Keltner(20,1.5×ATR) after a squeeze; momentum (LazyBear linreg) > 0 and close > EMA200 → long (mirror short). "
          "Unseen test +0.50R but train folds only +0.36R and weak on 1h; display only.")
BRK_GATE=("XAUUSD|1h","BTCUSD|1h")   # consensus 1h BUY/SELL must have a fresh Donchian-20 breakout the same way (last BRK_WINDOW bars)
BRK_WINDOW=10
SESSION_GATE=("BTCUSD|1h",)         # 1h signals only when the bar closes 07:00-19:59 London time (London + New York)
SESSION_HOURS=(7,20)
LDN=ZoneInfo("Europe/London")
def donchian_signals(df,htf,n=20):
    """Causal: prior-n-bar channel (shift 1), EMA200 side and HTF EMA20/50 bias known at the bar's close."""
    c=df.c; dh=df.h.rolling(n).max().shift(1); dl=df.l.rolling(n).min().shift(1); e200=_ema(c,200)
    hb=htf_bias(df,htf) if htf is not None else np.zeros(len(df))
    hok_l=(hb>0) if htf is not None else np.ones(len(df),bool); hok_s=(hb<0) if htf is not None else np.ones(len(df),bool)
    L=((c>dh)&(c>e200)).values&hok_l; S=((c<dl)&(c<e200)).values&hok_s
    return L,S,dh.values,dl.values,e200.values
def ttm_squeeze(df):
    c=df.c; sma=c.rolling(20).mean(); sd=c.rolling(20).std(ddof=0); a20=_atr(df,20)
    on=(((sma+2*sd)<(sma+1.5*a20))&((sma-2*sd)>(sma-1.5*a20))).values
    fire=np.r_[False,on[:-1]&~on[1:]]
    d=c-((df.h.rolling(20).max()+df.l.rolling(20).min())/2+sma)/2
    x=np.arange(20); xm=x.mean(); den=((x-xm)**2).sum()
    mom=d.rolling(20).apply(lambda y: y.mean()+((x-xm)*(y-y.mean())).sum()/den*(19-xm),raw=True).values
    return on,fire,mom
def squeeze_signals(df):
    on,fire,mom=ttm_squeeze(df); e200=_ema(df.c,200).values; c=df.c.values
    return fire&(mom>0)&(c>e200), fire&(mom<0)&(c<e200)
def recent_brk(L,S,i,window=BRK_WINDOW):
    """(long_ok, short_ok): a Donchian breakout that way happened in bars i-window+1..i (same window as the research test)."""
    a=max(0,i-int(window)+1); return bool(np.asarray(L[a:i+1]).any()), bool(np.asarray(S[a:i+1]).any())
def session_ok(t_open_ms,step_ms,hours=SESSION_HOURS):
    h=pd.Timestamp(int(t_open_ms)+int(step_ms),unit="ms",tz="UTC").tz_convert(LDN).hour
    return hours[0]<=h<hours[1]
def apply_round4(sig,key,tf,brk,sess_ok):
    """Downgrade BUY/SELL to WAIT: 'breakout' when the Donchian confirmation is missing, 'session' outside London/NY (BTC 1h)."""
    if sig not in ("BUY","SELL"): return sig,""
    k=f"{key}|{tf}"; d=1 if sig=="BUY" else -1
    if k in BRK_GATE and not (brk[0] if d>0 else brk[1]): return "WAIT","breakout"
    if k in SESSION_GATE and not sess_ok: return "WAIT","session"
    return sig,""
def indicator_panel(df,htf,tf,key,step_ms):
    """Values shown in the app (display): EMA20/50/200, Supertrend(10,3), Donchian-20, TTM squeeze, session, gate state."""
    c=df.c; i=len(df)-1; e20=_ema(c,20).iloc[-1]; e50=_ema(c,50).iloc[-1]; e200=_ema(c,200).iloc[-1]
    st=supertrend_dir(df)[-1]; L,S,dh,dl,_=donchian_signals(df,htf); on,fire,mom=ttm_squeeze(df)
    bl,bs=recent_brk(L,S,i); bd=1 if bl and not bs else -1 if bs and not bl else (2 if bl and bs else 0); last=float(c.iloc[-1])
    stack=1 if e20>e50>e200 else -1 if e20<e50<e200 else 0
    return {"e200":round(float(e200),8),"side200":1 if last>e200 else -1,"stack":stack,"st":int(st),
            "dh":None if dh[-1]!=dh[-1] else round(float(dh[-1]),8),"dl":None if dl[-1]!=dl[-1] else round(float(dl[-1]),8),"brk":int(bd),
            "sqz":"on" if on[-1] else ("fired" if fire[-1] else "off"),"mom":None if mom[-1]!=mom[-1] else (1 if mom[-1]>0 else -1),
            "sess":bool(session_ok(df.t.iloc[-1],step_ms)),"gate":f"{key}|{tf}" in BRK_GATE,"sgate":f"{key}|{tf}" in SESSION_GATE}
