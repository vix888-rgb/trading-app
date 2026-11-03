import os, sys
_PIPE = os.path.dirname(os.path.abspath(__file__))
if _PIPE not in sys.path: sys.path.insert(0, _PIPE)
exec(compile(open(os.path.join(_PIPE, "engine.py"), encoding="utf-8").read(), os.path.join(_PIPE, "engine.py"), "exec"))
import edge
CAT={"EMA trend (20/50/200)":"T","MACD (12/26/9)":"T","RSI (14)":"M","Stochastic (14/3/3)":"M","Bollinger Bands (20/2)":"M","ADX trend strength (14)":"T","Ichimoku cloud":"T","Supertrend (10/3)":"T","Volume / OBV":"T","VWAP (rolling 50 bars)":"T","Support / resistance":"M","Candlestick patterns":"N","RSI divergence":"M","Multi-timeframe confirmation":"T","Volatility (ATR 14)":"N","News sentiment":"N"}
def tz_index(df): return pd.to_datetime(df.t,unit="ms",utc=True).dt.tz_convert(TZ)
def pivots_confirmed(h,l,w=3):
    n=len(h); sh=np.full(n,np.nan); sl=np.full(n,np.nan)  # value known at i (confirmed w bars later)
    for i in range(w,n-w):
        if h[i]==h[i-w:i+w+1].max(): sh[i+w]=h[i]
        if l[i]==l[i-w:i+w+1].min(): sl[i+w]=l[i]
    return sh,sl
def structure(df,w=3):
    h,l,c=df.h.values,df.l.values,df.c.values; sh,sl=pivots_confirmed(h,l,w)
    lastH=np.nan; lastL=np.nan; trend=0; tr=np.zeros(len(c)); ev=[]; choch=np.zeros(len(c))
    for i in range(len(c)):
        if not np.isnan(sh[i]): lastH=sh[i]
        if not np.isnan(sl[i]): lastL=sl[i]
        if not np.isnan(lastH) and c[i]>lastH:
            ev.append((i,"CHoCH" if trend==-1 else "BOS",1,lastH)); 
            if trend==-1: choch[i]=1
            trend=1; lastH=np.nan
        elif not np.isnan(lastL) and c[i]<lastL:
            ev.append((i,"CHoCH" if trend==1 else "BOS",-1,lastL))
            if trend==1: choch[i]=-1
            trend=-1; lastL=np.nan
        tr[i]=trend
    return tr,ev,choch,sh,sl
def hma(s,n):
    def wma(x,m): w=np.arange(1,m+1); return x.rolling(m).apply(lambda a: np.dot(a,w)/w.sum(),raw=True)
    return wma(2*wma(s,n//2)-wma(s,n),int(np.sqrt(n)))
def linreg_last(y):
    x=np.arange(len(y)); m,b=np.polyfit(x,y,1); return m*(len(y)-1)+b
def squeeze_series(df):
    c=df.c; mb=c.rolling(20).mean(); sd=c.rolling(20).std(); a=atr(df,20)
    kcu=ema(c,20)+1.5*a; kcl=ema(c,20)-1.5*a; on=((mb+2*sd)<kcu)&((mb-2*sd)>kcl)
    mid=((df.h.rolling(20).max()+df.l.rolling(20).min())/2+mb)/2
    mom=(c-mid).rolling(20).apply(linreg_last,raw=True)
    return on,mom

def advanced(df,p,d1,last,a,tr,ev,choch,sh,sl):
    out=[]; c=df.c; h=df.h; l=df.l; hasvol=df.v.tail(50).sum()>0; n=len(df)
    # SMC structure
    recent=[e for e in ev if e[0]>=n-30]; le=ev[-1] if ev else None
    if le:
        vote="bullish" if tr[-1]>0 else "bearish"; conf=55+(20 if le[1]=="CHoCH" and le[0]>=n-10 else 0)
        out.append(V("SMC market structure (BOS/CHoCH)",vote,conf,f"Last event {le[1]} {'up' if le[2]>0 else 'down'} through {fmt(le[3],p)}, {n-1-le[0]} bars ago; structure {'bullish' if tr[-1]>0 else 'bearish'}"))
    else: out.append(V("SMC market structure (BOS/CHoCH)","neutral",20,"No confirmed structure break"))
    # Order blocks: for last bull and last bear break, last opposite candle before impulse
    obs=[]
    for (i,kind,dirn,lvl) in ev[-12:]:
        j0=max(0,i-12)
        for j in range(i-1,j0,-1):
            if (dirn>0 and c.iloc[j]<df.o.iloc[j]) or (dirn<0 and c.iloc[j]>df.o.iloc[j]):
                lo,hi=l.iloc[j],h.iloc[j]
                after=c.iloc[i:]; mitig=(after<lo).any() if dirn>0 else (after>hi).any()
                if not mitig: obs.append((dirn,lo,hi,i))
                break
    bull=[o for o in obs if o[0]>0 and o[2]<=last+0.3*a]; bear=[o for o in obs if o[0]<0 and o[1]>=last-0.3*a]
    nb=max(bull,key=lambda o:o[2]) if bull else None; ns=min(bear,key=lambda o:o[1]) if bear else None
    db=(last-nb[2])/a if nb else 99; ds=(ns[1]-last)/a if ns else 99
    if nb and db<1.0 and db<=ds: out.append(V("Order blocks","bullish",60 if db<=0.3 else 45,f"At/near bullish OB {fmt(nb[1],p)}–{fmt(nb[2],p)} ({max(db,0):.1f} ATR)"))
    elif ns and ds<1.0: out.append(V("Order blocks","bearish",60 if ds<=0.3 else 45,f"At/near bearish OB {fmt(ns[1],p)}–{fmt(ns[2],p)} ({max(ds,0):.1f} ATR)"))
    else: out.append(V("Order blocks","neutral",20,("Nearest bull OB "+(f"{fmt(nb[1],p)}–{fmt(nb[2],p)}" if nb else "none")+" · bear OB "+(f"{fmt(ns[1],p)}–{fmt(ns[2],p)}" if ns else "none"))))
    # FVG
    fv=[]
    for i in range(max(2,n-60),n):
        if l.iloc[i]>h.iloc[i-2]: lo,hi=h.iloc[i-2],l.iloc[i]; filled=(l.iloc[i+1:]<=lo).any(); fv.append((1,lo,hi,i,filled))
        if h.iloc[i]<l.iloc[i-2]: lo,hi=h.iloc[i],l.iloc[i-2]; filled=(h.iloc[i+1:]>=hi).any(); fv.append((-1,lo,hi,i,filled))
    op=[f for f in fv if not f[4]]
    inside=[f for f in op if f[1]-0.2*a<=last<=f[2]+0.2*a]
    if inside:
        f=inside[-1]; out.append(V("Fair value gaps","bullish" if f[0]>0 else "bearish",50,f"Price retesting unfilled {'bullish' if f[0]>0 else 'bearish'} FVG {fmt(f[1],p)}–{fmt(f[2],p)}"))
    else:
        ub=sum(1 for f in op if f[0]>0); us=sum(1 for f in op if f[0]<0)
        out.append(V("Fair value gaps","bullish" if ub>us+1 else "bearish" if us>ub+1 else "neutral",30,f"{ub} unfilled bullish / {us} bearish FVGs in last 60 bars"))
    # Liquidity sweep
    sw=None
    for k in range(1,6):
        i=n-k; ph=h.iloc[i-20:i].max(); pl=l.iloc[i-20:i].min()
        if h.iloc[i]>ph and c.iloc[i]<ph: sw=("bearish",f"Swept buy-side liquidity above {fmt(ph,p)} and closed back below, {k-1} bars ago"); break
        if l.iloc[i]<pl and c.iloc[i]>pl: sw=("bullish",f"Swept sell-side liquidity below {fmt(pl,p)} and closed back above, {k-1} bars ago"); break
    out.append(V("Liquidity sweeps",sw[0],60,sw[1]) if sw else V("Liquidity sweeps","neutral",15,f"No sweep in last 5 bars; liquidity rests at {fmt(h.tail(20).max(),p)} / {fmt(l.tail(20).min(),p)}"))
    # Volume profile
    seg=df.tail(200); bins=np.linspace(seg.l.min(),seg.h.max(),41); tp=((seg.h+seg.l+seg.c)/3).values
    wts=seg.v.values if hasvol else np.ones(len(seg)); hist,_=np.histogram(tp,bins=bins,weights=wts)
    pi=int(hist.argmax()); poc=(bins[pi]+bins[pi+1])/2; order=np.argsort(-hist); cum=0; tot=hist.sum(); sel=[]
    for k in order:
        sel.append(k); cum+=hist[k]
        if cum>=0.7*tot: break
    vah=bins[max(sel)+1]; val=bins[min(sel)]
    kind="Volume profile" if hasvol else "Time-at-price profile (no volume)"
    if last>vah: v_,cf,why="bullish",50,"accepted above value area"
    elif last<val: v_,cf,why="bearish",50,"accepted below value area"
    else: v_,cf,why=("bullish" if last<poc else "bearish"),25,f"inside value, pull toward POC"
    out.append(V("Volume profile / POC",v_,cf,f"{kind} 200 bars: POC {fmt(poc,p)}, VA {fmt(val,p)}–{fmt(vah,p)}; {why}"))
    # Anchored VWAP from most recent 100-bar extreme
    s100=df.tail(100); ih=s100.h.idxmax(); il=s100.l.idxmin(); anc=max(ih,il); seg=df.loc[anc:]
    tp2=(seg.h+seg.l+seg.c)/3; w2=seg.v if hasvol else pd.Series(1.0,index=seg.index)
    av=float((tp2*w2).sum()/w2.sum()) if w2.sum()>0 else float(tp2.mean()); dist=(last-av)/a
    out.append(V("Anchored VWAP","bullish" if dist>0.2 else "bearish" if dist<-0.2 else "neutral",clamp(35+abs(dist)*10,0,70),
        f"Anchored at {'swing high' if anc==ih else 'swing low'} {n-1-anc} bars ago: {fmt(av,p)}{'' if hasvol else ' (time-weighted, no volume)'}; price {abs(dist):.1f} ATR {'above' if dist>0 else 'below'}"))
    # Fibonacci
    hi_=s100.h.max(); lo_=s100.l.min(); up=il<ih; rng=hi_-lo_
    lv={k:(hi_-k*rng if up else lo_+k*rng) for k in (0.382,0.5,0.618,0.786)}
    ext={k:(lo_+k*rng if up else hi_-k*rng) for k in (1.272,1.618)}
    gz_lo,gz_hi=sorted([lv[0.5],lv[0.618]])
    conflu=[nm for nm,x in (("POC",poc),("AVWAP",av)) if gz_lo-0.3*a<=x<=gz_hi+0.3*a]
    if gz_lo-0.3*a<=last<=gz_hi+0.3*a:
        out.append(V("Fibonacci confluence","bullish" if up else "bearish",50+15*len(conflu),f"In 50–61.8% golden zone of {'up' if up else 'down'}-leg {fmt(gz_lo,p)}–{fmt(gz_hi,p)}"+(f"; confluence with {', '.join(conflu)}" if conflu else "")))
    else:
        out.append(V("Fibonacci confluence","neutral",20,f"{'Up' if up else 'Down'}-leg {fmt(lo_,p)}→{fmt(hi_,p)}; golden zone {fmt(gz_lo,p)}–{fmt(gz_hi,p)}; 1.272 ext {fmt(ext[1.272],p)}"))
    # Heikin Ashi
    hac=(df.o+df.h+df.l+df.c)/4; hao=hac.copy(); hao.iloc[0]=(df.o.iloc[0]+df.c.iloc[0])/2
    for i in range(1,n): hao.iloc[i]=(hao.iloc[i-1]+hac.iloc[i-1])/2
    col=np.sign(hac-hao).tail(5).values; lowwick=(np.minimum(hao,hac)-np.minimum(df.l,np.minimum(hao,hac))).iloc[-1]
    st=int(col[-1]); run=1
    for x in col[-2::-1]:
        if x==st: run+=1
        else: break
    out.append(V("Heikin Ashi trend","bullish" if st>0 else "bearish",30+run*8+(10 if (st>0 and lowwick<1e-12) else 0),f"{run} consecutive {'green' if st>0 else 'red'} HA candles"+(" with no lower wick (strong)" if st>0 and lowwick<1e-12 else "")))
    # TTM squeeze
    on,mom=squeeze_series(df); m0,m1=mom.iloc[-1],mom.iloc[-2]; fired=on.iloc[-2] and not on.iloc[-1]
    bars_on=int(on.tail(20)[::-1].cumprod().sum()) if on.iloc[-1] else 0
    out.append(V("TTM Squeeze (BB in Keltner)","bullish" if m0>0 else "bearish",(65 if fired else 30 if on.iloc[-1] else 45)+(5 if (m0>m1)==(m0>0) else -10),
        ("Squeeze just FIRED; " if fired else f"Squeeze ON {bars_on} bars (coiling); " if on.iloc[-1] else "No squeeze; ")+f"momentum {'positive' if m0>0 else 'negative'} and {'rising' if m0>m1 else 'falling'}"))
    # Hull
    H=hma(c,55); s0=H.iloc[-1]-H.iloc[-3]
    out.append(V("Hull MA (55)","bullish" if s0>0 else "bearish",45+(10 if (last>H.iloc[-1])==(s0>0) else 0),f"HMA {fmt(H.iloc[-1],p)} sloping {'up' if s0>0 else 'down'}; price {'above' if last>H.iloc[-1] else 'below'}"))
    # W%R & CCI
    wr=-100*(h.rolling(14).max()-c)/(h.rolling(14).max()-l.rolling(14).min()); tpp=(h+l+c)/3; cci=(tpp-tpp.rolling(20).mean())/(0.015*tpp.rolling(20).apply(lambda x: np.abs(x-x.mean()).mean(),raw=True))
    w0,c0=wr.iloc[-1],cci.iloc[-1]
    if w0<-80 and c0<-100: v_,cf,why="bullish",55,"both oversold"
    elif w0>-20 and c0>100: v_,cf,why="bearish",55,"both overbought"
    elif c0>100: v_,cf,why="bullish",40,"CCI momentum >100"
    elif c0<-100: v_,cf,why="bearish",40,"CCI momentum <−100"
    else: v_,cf,why="neutral",25,"no extreme"
    out.append(V("Williams %R / CCI",v_,cf,f"%R {w0:.0f}, CCI {c0:.0f}: {why}"))
    # Pivot points (prior day classic)
    if d1 is not None and len(d1)>2:
        pd_=d1.iloc[-2]; P=(pd_.h+pd_.l+pd_.c)/3; R1=2*P-pd_.l; S1=2*P-pd_.h; R2=P+(pd_.h-pd_.l); S2=P-(pd_.h-pd_.l)
        piv={"P":P,"R1":R1,"S1":S1,"R2":R2,"S2":S2}
        out.append(V("Daily pivot points","bullish" if last>P else "bearish",35+min(25,abs(last-P)/a*8),f"P {fmt(P,p)} · R1 {fmt(R1,p)} · S1 {fmt(S1,p)}; price {'above' if last>P else 'below'} pivot"))
    else: piv={}; out.append(V("Daily pivot points","neutral",0,"No daily data"))
    cat={"SMC market structure (BOS/CHoCH)":"T","Order blocks":"N","Fair value gaps":"N","Liquidity sweeps":"N","Volume profile / POC":"M","Anchored VWAP":"T","Fibonacci confluence":"N","Heikin Ashi trend":"T","TTM Squeeze (BB in Keltner)":"N","Hull MA (55)":"T","Williams %R / CCI":"M","Daily pivot points":"N"}
    lv_out={"poc":poc,"vah":vah,"val":val,"avwap":av,"fib":lv,"ext":ext,"piv":piv,"obBull":nb,"obBear":ns,"fvg":op[-1] if op else None}
    return out,cat,lv_out

def regime(df):
    X,_,_=adx(df); c=df.c; den_=c.diff().abs().tail(20).sum(); er=abs(c.iloc[-1]-c.iloc[-21])/den_ if den_>0 else 0.0
    x=X.iloc[-1]; e20,e50=ema(c,20).iloc[-1],ema(c,50).iloc[-1]
    if x>=25 and er>=0.3: return ("Trending up" if e20>e50 else "Trending down"),"trend",x,er
    if x<20 and er<0.25: return "Ranging","range",x,er
    return "Transitional","mixed",x,er

# ---------- strategies + backtest ----------
STRATS=["Trend-follow + ADX filter","Turtle 20-bar breakout","Range mean reversion","London breakout","ICT/SMC sweep + CHoCH","MTF top-down pullback","Wyckoff spring / upthrust","Squeeze momentum breakout"]+[edge.NEW_STRATS[k] for k in edge.KEEP_NEW]+[edge.TOM_NAME,edge.SWEEP_NAME,edge.LEZ_NAME,edge.BRK_NAME,edge.SQZ_NAME]
RULES=["EMA20>EMA50, ADX>25 with +DI>−DI, close reclaims EMA20 → long (mirror short)",
 "Close breaks prior 20-bar high → long; below 20-bar low → short (Donchian, Turtle System 1)",
 "ADX<20 regime, close outside Bollinger band with RSI<30 / >70 → fade back toward the mean",
 "Asia range 01:00–08:00 Belgrade; first close beyond it 09:00–12:00 → trade the break (≤1h only)",
 "Sell-side sweep of 20-bar low, then bullish CHoCH within 10 bars → long (mirror short)",
 "Higher-TF EMA20>EMA50 bias; RSI dips under 40 then reclaims it with price above EMA50 → long",
 "Range (ADX<25): wick below 30-bar low, close back inside (volume >1.5× if available) → long",
 "Squeeze on ≥5 bars then releases; momentum positive & rising → long (mirror short)"]+[edge.NEW_RULES[k] for k in edge.KEEP_NEW]+[edge.TOM_RULE,edge.SWEEP_RULE,edge.LEZ_RULE,edge.BRK_RULE,edge.SQZ_RULE]
def strat_signals(df,tf,htf,sym=None):
    c=df.c; n=len(df); e20,e50=ema(c,20),ema(c,50); X,PDI,NDI=adx(df); R=rsi(c)
    L=[np.zeros(n,bool) for _ in range(8)]; S=[np.zeros(n,bool) for _ in range(8)]
    L[0]=((e20>e50)&(X>25)&(PDI>NDI)&(c>e20)&(c.shift()<=e20.shift())).values
    S[0]=((e20<e50)&(X>25)&(PDI<NDI)&(c<e20)&(c.shift()>=e20.shift())).values
    hh=df.h.rolling(20).max().shift(); ll=df.l.rolling(20).min().shift()
    L[1]=((c>hh)&(c.shift()<=hh.shift())).values; S[1]=((c<ll)&(c.shift()>=ll.shift())).values
    mb=c.rolling(20).mean(); sd=c.rolling(20).std()
    L[2]=((X<20)&(c<mb-2*sd)&(R<30)).values; S[2]=((X<20)&(c>mb+2*sd)&(R>70)).values
    if tf in ("5m","15m","1h"):
        ix=tz_index(df); hr=ix.dt.hour.values; day=ix.dt.date.values
        tmp=pd.DataFrame({"d":day,"hr":hr,"h":df.h.values,"l":df.l.values,"c":c.values})
        asia=tmp[(tmp.hr>=1)&(tmp.hr<8)].groupby("d").agg(ah=("h","max"),al=("l","min"))
        tmp=tmp.join(asia,on="d"); done=set()
        for i in range(n):
            if 9<=hr[i]<12 and not np.isnan(tmp.ah.iloc[i]) and day[i] not in done:
                if c.iloc[i]>tmp.ah.iloc[i]: L[3][i]=True; done.add(day[i])
                elif c.iloc[i]<tmp.al.iloc[i]: S[3][i]=True; done.add(day[i])
    tr,ev,choch,_,_=structure(df)
    l20=df.l.rolling(20).min().shift(); h20=df.h.rolling(20).max().shift()
    swL=((df.l<l20)&(c>l20)).values; swH=((df.h>h20)&(c<h20)).values
    for i in range(n):
        if choch[i]>0 and swL[max(0,i-10):i+1].any(): L[4][i]=True
        if choch[i]<0 and swH[max(0,i-10):i+1].any(): S[4][i]=True
    if htf is not None:
        hb=pd.DataFrame({"t":htf.t.values,"b":np.sign(ema(htf.c,20)-ema(htf.c,50)).values})
        hb["t"]=(hb.t+htf.t.diff().median()).astype("int64")  # bias known only after HTF bar closes
        m=pd.merge_asof(pd.DataFrame({"t":df.t.values.astype("int64")}),hb.sort_values("t"),on="t")
        b=m.b.fillna(0).values
        L[5]=(b>0)&(R>40).values&(R.shift()<=40).values&(c>e50).values
        S[5]=(b<0)&(R<60).values&(R.shift()>=60).values&(c<e50).values
    l30=df.l.rolling(30).min().shift(); h30=df.h.rolling(30).max().shift(); hv=df.v.tail(50).sum()>0
    vok=(df.v>1.5*df.v.rolling(20).mean()).values if hv else np.ones(n,bool)
    L[6]=((X<25)&(df.l<l30)&(c>l30)).values&vok; S[6]=((X<25)&(df.h>h30)&(c<h30)).values&vok
    on,mom=squeeze_series(df); onr=on.rolling(5).sum().shift()
    L[7]=((onr>=5)&(~on)&(mom>0)&(mom>mom.shift())).values; S[7]=((onr>=5)&(~on)&(mom<0)&(mom<mom.shift())).values
    Ln,Sn,_=edge.new_strat_signals(df,tf,htf)  # Nov 2026 additions kept after walk-forward test
    for k in edge.KEEP_NEW: L.append(Ln[k]); S.append(Sn[k])
    L.append(edge.tom_signal(df,tf)); S.append(np.zeros(n,bool))  # round 2: turn-of-month
    Lw,Sw=edge.sweep_mss_fvg_signals(df); L.append(Lw); S.append(Sw)  # rebuild Nov 2026: ICT/TJR/PB sweep -> MSS -> FVG (display only)
    if sym:
        Lz,Sz=edge.lez_sides(df,sym,tf)
    else:
        Lz=np.zeros(n,bool); Sz=np.zeros(n,bool)
    L.append(Lz); S.append(Sz)  # Liquidity Entry Zones (point scale). Alerts only if the unseen test earned them.
    Lb,Sb,_,_,_=edge.donchian_signals(df,htf); L.append(Lb); S.append(Sb)  # round 4 (3 Nov 2026): display only
    Lq,Sq=edge.squeeze_signals(df); L.append(Lq); S.append(Sq)            # round 4 (3 Nov 2026): display only
    return L,S
def backtest(df,Lg,Sg,stopA=1.5,rr=2.0,maxbars=48):
    A=atr(df).values; o,h,l,c=df.o.values,df.h.values,df.l.values,df.c.values; n=len(df); res=[]; i=60
    while i<n-1:
        d=1 if Lg[i] else -1 if Sg[i] else 0
        if d==0 or np.isnan(A[i]): i+=1; continue
        e=o[i+1]; risk=stopA*A[i]; sp=e-d*risk; tp=e+d*rr*risk; r=None
        for j in range(i+1,min(n,i+1+maxbars)):
            if (d>0 and l[j]<=sp) or (d<0 and h[j]>=sp): r=-1.0; break
            if (d>0 and h[j]>=tp) or (d<0 and l[j]<=tp): r=rr; break
        if r is None:
            if j<n-1 or j-i>=maxbars: r=d*(c[j]-e)/risk
            else: break  # still open at end
        res.append(r); i=j+1
    if not res: return {"n":0}
    rs=np.array(res); gw=rs[rs>0].sum(); gl=-rs[rs<0].sum()
    return {"n":len(rs),"win":round(100*(rs>0).mean()),"avgR":round(rs.mean(),2),"totR":round(rs.sum(),1),"pf":round(gw/gl,2) if gl>0 else None}

def plan(df,p,score,a,last,trd,sh,sl,adv):
    side=1 if score>=0 else -1
    shv=[x for x in sh[-120:] if not np.isnan(x)]; slv=[x for x in sl[-120:] if not np.isnan(x)]
    # entry zone
    zone=None; why="market (current price)"
    ob=adv["obBull"] if side>0 else adv["obBear"]; f=adv["fvg"]
    if ob is not None and abs(last-(ob[2] if side>0 else ob[1]))<=1.5*a: zone=(ob[1],ob[2]); why="order block retest"
    elif f is not None and f[0]==side and abs(last-(f[2] if side>0 else f[1]))<=1.5*a: zone=(f[1],f[2]); why="fair value gap fill"
    if zone is None: zone=(last-0.25*a,last) if side>0 else (last,last+0.25*a)
    entry=min(last,zone[1]) if side>0 else max(last,zone[0])
    entry=(zone[0]+zone[1])/2 if why!="market (current price)" else last
    # invalidation: last swing beyond entry
    if side>0:
        cands=[x for x in slv if x<entry]; inv=cands[-1] if cands else df.l.tail(30).min()
        stop=inv-0.15*a; dist=entry-stop
    else:
        cands=[x for x in shv if x>entry]; inv=cands[-1] if cands else df.h.tail(30).max()
        stop=inv+0.15*a; dist=stop-entry
    basis="beyond swing structure"
    if dist<0.8*a: dist=1.2*a; basis="1.2×ATR (swing too close)"
    if dist>3*a: dist=2*a; basis="2×ATR (swing too far; invalidation shown separately)"
    stop=entry-side*dist
    tps=[entry+side*k*dist for k in (1,2,3)]
    return {"side":"long" if side>0 else "short","entry":entry,"zone":list(zone),"zoneWhy":why,"stop":stop,"stopBasis":basis,"inv":inv,"tp":tps,"rr":[1,2,3],"dist":dist}

WT={"trend":{"T":1.5,"M":0.6,"N":1.0},"range":{"T":0.6,"M":1.5,"N":1.0},"mixed":{"T":1.0,"M":1.0,"N":1.0}}
def R_(x,p):
    if x is None: return None
    try:
        if isinstance(x,float) and math.isnan(x): return None
    except Exception: pass
    return round(float(x),p+1)
VC={"bullish":1,"bearish":-1,"neutral":0}
def main2():
    import pickle,os
    from paths import RAW_PKL, SNAP2
    if os.environ.get("CACHE") and os.path.exists(RAW_PKL):
        news,nstatus,raw,srcstat=pickle.load(open(RAW_PKL,"rb")); SY=[]
    else:
        news,nstatus=fetch_news(); raw={}; srcstat=[]; SY=SYMBOLS
    for key,label,src,code,mkt in SY:
        for tf in TFS:
            try:
                df=binance(code,tf) if src=="binance" else yahoo(code,tf)
                if len(df)<150: raise Exception(f"only {len(df)} candles")
                raw[(key,tf)]=df; srcstat.append([key,tf,1,len(df),""])
            except Exception as e: srcstat.append([key,tf,0,0,str(e)[:80]])
    pickle.dump((news,nstatus,raw,srcstat),open(RAW_PKL,"wb"))
    names=None; out=[]; HTF={"1m":"15m","5m":"1h","15m":"4h","1h":"4h","4h":"1d","1d":None}
    for key,label,src,code,mkt in SYMBOLS:
        nf=news_for(key,news)
        sd={"key":key,"label":label,"mkt":mkt,"src":("Binance public API "+code) if src=="binance" else ("Yahoo Finance "+code+(" (COMEX gold futures, proxy for spot XAU/USD)" if code=="GC=F" else "")),
            "news":[[i["src"],i["title"],i["link"],i["t"],datetime.fromtimestamp(i["ts"],TZ).strftime("%d %b %H:%M") if i["ts"] else ""] for i in nf["items"]],"tf":{}}
        for tf in TFS:
            df=raw.get((key,tf))
            if df is None: sd["tf"][tf]={"ok":0,"err":next(s[4] for s in srcstat if s[0]==key and s[1]==tf)}; continue
            htf={}
            for hh in [x for x in ["5m","15m","1h","4h","1d"] if x!=tf]:
                d2=raw.get((key,hh))
                if d2 is not None: htf[hh]=1 if ema(d2.c,20).iloc[-1]>ema(d2.c,50).iloc[-1] else -1
            p=prec(df.c.iloc[-1]); last=df.c.iloc[-1]
            agents,_,_,_,ind=analyze(df,p,htf,nf); a=atr(df).iloc[-1]
            tr,ev,choch,sh,sl=structure(df)
            adv,cat2,lv=advanced(df,p,raw.get((key,"1d")),last,a,tr,ev,choch,sh,sl)
            allag=agents+adv; cats={**CAT,**cat2}
            rg,rk,adxv,er=regime(df)
            num=den=0; rows=[]
            for g in allag:
                w=WT[rk][cats[g["agent"]]]; num+=VC[g["vote"]]*g["conf"]*w; den+=g["conf"]*w
                rows.append([VC[g["vote"]],g["conf"],g["reason"],w])
            score=round(100*num/den) if den else 0
            sig="BUY" if score>=25 else "SELL" if score<=-25 else "WAIT"
            if names is None: names=[[g["agent"],cats[g["agent"]]] for g in allag]
            L,S=strat_signals(df,tf,raw.get((key,HTF[tf])) if HTF[tf] else None,key)
            st=[]
            for k in range(len(STRATS)):
                bt=edge.lez_desk_bt(df,key,tf) if STRATS[k]==edge.LEZ_NAME else backtest(df,L[k],S[k]); lastL=np.where(L[k][-4:])[0]; lastS=np.where(S[k][-4:])[0]
                v=1 if len(lastL) and (not len(lastS) or lastL[-1]>lastS[-1]) else (-1 if len(lastS) else 0)
                ago=None
                if v: ago=3-int((lastL if v>0 else lastS)[-1])
                if k==3 and tf not in ("5m","15m","1h"): bt={"n":0,"na":1}
                if k==5 and HTF[tf] is None: bt={"n":0,"na":1}
                st.append([v,ago,bt])
            pl=plan(df,p,score,a,last,tr,sh,sl,lv)
            N=48; tl=df.tail(N)
            age=int((time.time()*1000-df.t.iloc[-1])/60000)
            sd["tf"][tf]={"ok":1,"p":p,"last":R_(last,p),"chg":round(float((last/df.c.iloc[-2]-1)*100),3),"lt":datetime.fromtimestamp(df.t.iloc[-1]/1000,TZ).strftime("%a %d %b %H:%M"),"age":age,"nb":len(df),
              "score":score,"sig":sig,"reg":[rg,rk,round(float(adxv)),round(float(er),2)],"ag":rows,"st":st,
              "plan":{k:([R_(x,p) for x in v] if isinstance(v,list) and k!="rr" else (R_(v,p) if isinstance(v,(float,np.floating)) else v)) for k,v in pl.items()},
              "atr":R_(a,p),"lv":{"S":R_(ind["S1"],p),"R":R_(ind["R1"],p),"poc":R_(lv["poc"],p),"vah":R_(lv["vah"],p),"val":R_(lv["val"],p),"avwap":R_(lv["avwap"],p),
                    "f50":R_(lv["fib"][0.5],p),"f618":R_(lv["fib"][0.618],p),"e1272":R_(lv["ext"][1.272],p),"e1618":R_(lv["ext"][1.618],p),"P":R_(lv["piv"].get("P"),p)},
              "c":{"t0":datetime.fromtimestamp(tl.t.iloc[0]/1000,TZ).strftime("%d %b %H:%M"),"t1":datetime.fromtimestamp(tl.t.iloc[-1]/1000,TZ).strftime("%d %b %H:%M"),
                   "o":[R_(x,p) for x in tl.o],"h":[R_(x,p) for x in tl.h],"l":[R_(x,p) for x in tl.l],"c":[R_(x,p) for x in tl.c],"e20":[R_(x,p) for x in ind["e20"].tail(N)],"e50":[R_(x,p) for x in ind["e50"].tail(N)]}}
        out.append(sd)
    rates={"USDJPY":raw[("USDJPY","1h")].c.iloc[-1] if ("USDJPY","1h") in raw else None,"USDCHF":raw[("USDCHF","1h")].c.iloc[-1] if ("USDCHF","1h") in raw else None}
    res={"gen":datetime.now(TZ).strftime("%a %d %b %Y, %H:%M"),"names":names,"strats":STRATS,"rules":RULES,"rates":{k:(round(float(v),5) if v else None) for k,v in rates.items()},
         "sym":out,"newsStatus":nstatus,"priceStatus":srcstat,"newsCount":len(news)}
    s=json.dumps(res,separators=(",",":"),default=lambda o: float(o) if isinstance(o,(np.floating,np.integer)) else str(o))
    open(SNAP2,"w").write(s); print("ok",len(s))
if os.environ.get("TRADING_LOAD_ONLY") != "1":
    main2()
