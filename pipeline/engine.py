import json, time, re, math, html, sys, subprocess, urllib.request, urllib.parse
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd
import xml.etree.ElementTree as ET
TZ = ZoneInfo("Europe/Belgrade")
UA = {"User-Agent": "Mozilla/5.0"}
def get(url, t=20):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=t) as r: return r.read()

SYMBOLS = [  # priority order
  ("XAUUSD","Gold (XAU/USD)","yahoo","GC=F","gold"),
  ("BTCUSD","Bitcoin (BTC/USD)","binance","BTCUSDT","crypto"),
  ("EURUSD","EUR/USD","yahoo","EURUSD=X","forex"),
  ("USDJPY","USD/JPY","yahoo","JPY=X","forex"),
  ("USDCHF","USD/CHF","yahoo","CHF=X","forex"),
  ("EURCHF","EUR/CHF","yahoo","EURCHF=X","forex"),
  ("ETHUSD","Ethereum (ETH/USD)","binance","ETHUSDT","crypto"),
  ("SOLUSD","Solana (SOL/USD)","binance","SOLUSDT","crypto"),
  ("XRPUSD","XRP (XRP/USD)","binance","XRPUSDT","crypto"),
  ("GBPUSD","GBP/USD","yahoo","GBPUSD=X","forex"),
]
TFS = ["1m","5m","15m","1h","4h","1d"]
YRANGE = {"1m":("1m","5d"),"5m":("5m","1mo"),"15m":("15m","1mo"),"1h":("60m","3mo"),"4h":("60m","1y"),"1d":("1d","5y")}

def binance(sym, tf):
    d = json.loads(get(f"https://data-api.binance.vision/api/v3/klines?symbol={sym}&interval={tf}&limit=1000"))
    df = pd.DataFrame([[int(r[0]),float(r[1]),float(r[2]),float(r[3]),float(r[4]),float(r[5])] for r in d], columns=["t","o","h","l","c","v"])
    return df.iloc[:-1] if len(df) > 1 else df  # drop forming candle? keep closed only
def yahoo(sym, tf):
    iv, rg = YRANGE[tf]
    d = json.loads(get(f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(sym)}?interval={iv}&range={rg}"))
    r = d["chart"]["result"][0]; q = r["indicators"]["quote"][0]
    df = pd.DataFrame({"t":np.array(r["timestamp"])*1000,"o":q["open"],"h":q["high"],"l":q["low"],"c":q["close"],"v":q["volume"]}).dropna(subset=["o","h","l","c"])
    df["v"] = df["v"].fillna(0)
    step={"1m":60e3,"5m":3e5,"15m":9e5,"60m":3.6e6,"1d":8.64e7}[iv]
    df = df[df.t + step <= time.time()*1000]  # closed bars only
    if tf == "4h":
        idx = pd.to_datetime(df.t, unit="ms", utc=True); df = df.set_index(idx)
        df = df.resample("4h").agg({"t":"first","o":"first","h":"max","l":"min","c":"last","v":"sum"}).dropna()
        df = df.reset_index(drop=True)
    df = df.reset_index(drop=True)
    return df.tail(1000).reset_index(drop=True)

# ---------- indicators ----------
def ema(s,n): return s.ewm(span=n, adjust=False).mean()
def rsi(c,n=14):
    d=c.diff(); up=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean(); dn=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean()
    return (100-100/(1+up/dn.replace(0,np.nan))).where(dn>0,100.0)
def atr(df,n=14):
    pc=df.c.shift(); tr=pd.concat([df.h-df.l,(df.h-pc).abs(),(df.l-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False).mean()
def adx(df,n=14):
    up=df.h.diff(); dn=-df.l.diff()
    pdm=np.where((up>dn)&(up>0),up,0.0); ndm=np.where((dn>up)&(dn>0),dn,0.0)
    a=atr(df,n); pdi=100*pd.Series(pdm).ewm(alpha=1/n,adjust=False).mean()/a; ndi=100*pd.Series(ndm).ewm(alpha=1/n,adjust=False).mean()/a
    dx=100*(pdi-ndi).abs()/(pdi+ndi).replace(0,np.nan)
    return dx.ewm(alpha=1/n,adjust=False).mean(), pdi, ndi
def supertrend(df,n=10,m=3):
    a=atr(df,n); hl=(df.h+df.l)/2; ub=(hl+m*a).values; lb=(hl-m*a).values; c=df.c.values
    fu=ub.copy(); fl=lb.copy(); dirn=np.ones(len(c))
    for i in range(1,len(c)):
        fu[i]=ub[i] if (ub[i]<fu[i-1] or c[i-1]>fu[i-1]) else fu[i-1]
        fl[i]=lb[i] if (lb[i]>fl[i-1] or c[i-1]<fl[i-1]) else fl[i-1]
        if c[i]>fu[i-1]: dirn[i]=1
        elif c[i]<fl[i-1]: dirn[i]=-1
        else: dirn[i]=dirn[i-1]
    line=np.where(dirn>0,fl,fu); return dirn,line

def fmt(x,p):
    if x is None or (isinstance(x,float) and math.isnan(x)): return "n/a"
    return f"{x:,.{p}f}"
def prec(price):
    return 2 if price>=100 else (4 if price>=10 else (5 if price>=0.5 else 6))
def clamp(x,a=0,b=100): return max(a,min(b,x))
def V(name,vote,conf,reason): return {"agent":name,"vote":vote,"conf":int(round(clamp(conf))),"reason":reason}

def find_pivots(h,l,w=5):
    ph=[];pl=[]
    for i in range(w,len(h)-w):
        if h[i]==max(h[i-w:i+w+1]): ph.append((i,h[i]))
        if l[i]==min(l[i-w:i+w+1]): pl.append((i,l[i]))
    return ph,pl

def analyze(df, p, htf_trends, news):
    c=df.c; last=c.iloc[-1]; out=[]
    e20,e50,e200=ema(c,20),ema(c,50),ema(c,200); A=atr(df); a=A.iloc[-1]
    # 1 EMA
    s=(1 if last>e20.iloc[-1] else -1)+(1 if e20.iloc[-1]>e50.iloc[-1] else -1)+(1 if e50.iloc[-1]>e200.iloc[-1] else -1)
    vote="bullish" if s>=2 else "bearish" if s<=-2 else "neutral"
    out.append(V("EMA trend (20/50/200)",vote,40+20*abs(s) if vote!="neutral" else 35,
      f"Price {'above' if last>e20.iloc[-1] else 'below'} EMA20; EMA20 {'>' if e20.iloc[-1]>e50.iloc[-1] else '<'} EMA50 {'>' if e50.iloc[-1]>e200.iloc[-1] else '<'} EMA200"))
    # 2 MACD
    m=ema(c,12)-ema(c,26); sg=ema(m,9); hist=m-sg; h0,h1=hist.iloc[-1],hist.iloc[-2]
    cross = "fresh bullish cross" if h0>0 and h1<=0 else "fresh bearish cross" if h0<0 and h1>=0 else ""
    vote="bullish" if h0>0 else "bearish"
    conf=50+(20 if cross else 0)+(10 if (h0>h1 if h0>0 else h0<h1) else -10)+(10 if (m.iloc[-1]>0)==(h0>0) else 0)
    out.append(V("MACD (12/26/9)",vote,conf,f"Histogram {fmt(h0,p+1)} ({'rising' if h0>h1 else 'falling'}){', '+cross if cross else ''}; MACD {'above' if m.iloc[-1]>0 else 'below'} zero"))
    # 3 RSI
    R=rsi(c); r=R.iloc[-1]
    if r>=70: vote,conf,why="bearish",50+(r-70)*2,"overbought"
    elif r<=30: vote,conf,why="bullish",50+(30-r)*2,"oversold"
    elif r>55: vote,conf,why="bullish",40+(r-55)*1.5,"bullish momentum zone"
    elif r<45: vote,conf,why="bearish",40+(45-r)*1.5,"bearish momentum zone"
    else: vote,conf,why="neutral",30,"mid-range, no edge"
    out.append(V("RSI (14)",vote,conf,f"RSI {r:.1f}: {why}"))
    # 4 Stoch
    ll=df.l.rolling(14).min(); hh=df.h.rolling(14).max(); k=(100*(c-ll)/(hh-ll).replace(0,np.nan)).rolling(3).mean(); d=k.rolling(3).mean()
    k0,d0,k1,d1=k.iloc[-1],d.iloc[-1],k.iloc[-2],d.iloc[-2]
    if k0<20 and k0>d0: vote,conf,why="bullish",70,"oversold and turning up"
    elif k0>80 and k0<d0: vote,conf,why="bearish",70,"overbought and turning down"
    elif k0<20: vote,conf,why="bullish",45,"oversold"
    elif k0>80: vote,conf,why="bearish",45,"overbought"
    elif k0>d0 and k1<=d1: vote,conf,why="bullish",50,"%K crossed above %D"
    elif k0<d0 and k1>=d1: vote,conf,why="bearish",50,"%K crossed below %D"
    else: vote,conf,why=("bullish" if k0>d0 else "bearish"),35,("%K above %D" if k0>d0 else "%K below %D")
    out.append(V("Stochastic (14/3/3)",vote,conf,f"%K {k0:.0f} / %D {d0:.0f}: {why}"))
    # 5 Bollinger
    mb=c.rolling(20).mean(); sd=c.rolling(20).std(); ub,lb=mb+2*sd,mb-2*sd; pb=(last-lb.iloc[-1])/(ub.iloc[-1]-lb.iloc[-1])
    bw=((ub-lb)/mb); squeeze=bw.iloc[-1]<=bw.tail(120).quantile(0.15)
    if pb>1: vote,conf,why="bearish",55,"closed above upper band (stretched)"
    elif pb<0: vote,conf,why="bullish",55,"closed below lower band (stretched)"
    elif pb>0.8: vote,conf,why="bullish",40,"riding upper band"
    elif pb<0.2: vote,conf,why="bearish",40,"riding lower band"
    else: vote,conf,why="neutral",30,"inside bands"
    out.append(V("Bollinger Bands (20/2)",vote,conf,f"%B {pb:.2f}: {why}{'; squeeze — breakout likely' if squeeze else ''}"))
    # 6 ADX
    X,PDI,NDI=adx(df); x=X.iloc[-1]; bull=PDI.iloc[-1]>NDI.iloc[-1]
    if x<20: vote,conf="neutral",35
    else: vote,conf=("bullish" if bull else "bearish"),clamp(30+x*1.2)
    out.append(V("ADX trend strength (14)",vote,conf,f"ADX {x:.0f} ({'weak/ranging' if x<20 else 'trending' if x<40 else 'strong trend'}); +DI {PDI.iloc[-1]:.0f} vs −DI {NDI.iloc[-1]:.0f}"))
    # 7 Ichimoku
    tk=(df.h.rolling(9).max()+df.l.rolling(9).min())/2; kj=(df.h.rolling(26).max()+df.l.rolling(26).min())/2
    sa=((tk+kj)/2).shift(26); sb=((df.h.rolling(52).max()+df.l.rolling(52).min())/2).shift(26)
    top=max(sa.iloc[-1],sb.iloc[-1]); bot=min(sa.iloc[-1],sb.iloc[-1])
    pos="above cloud" if last>top else "below cloud" if last<bot else "inside cloud"
    tkx=tk.iloc[-1]>kj.iloc[-1]
    if pos=="above cloud": vote,conf="bullish",55+(15 if tkx else 0)
    elif pos=="below cloud": vote,conf="bearish",55+(15 if not tkx else 0)
    else: vote,conf="neutral",30
    out.append(V("Ichimoku cloud",vote,conf,f"Price {pos}; Tenkan {'above' if tkx else 'below'} Kijun"))
    # 8 Supertrend
    dirn,stl=supertrend(df); flip=dirn[-1]!=dirn[-2]
    out.append(V("Supertrend (10/3)","bullish" if dirn[-1]>0 else "bearish",55+(20 if flip else 0)+(0),
      f"{'Up' if dirn[-1]>0 else 'Down'}trend, line at {fmt(stl[-1],p)}{' — just flipped' if flip else ''}"))
    # 9 Volume/OBV
    hasvol=df.v.tail(50).sum()>0
    if hasvol:
        obv=(np.sign(c.diff().fillna(0))*df.v).cumsum(); os_=ema(obv,20)
        vr=df.v.iloc[-1]/df.v.tail(20).mean() if df.v.tail(20).mean()>0 else 1
        up=obv.iloc[-1]>os_.iloc[-1]; vote="bullish" if up else "bearish"
        out.append(V("Volume / OBV",vote,40+min(25,abs(vr-1)*20)+(10 if (up==(c.iloc[-1]>c.iloc[-2])) else 0),
          f"OBV {'above' if up else 'below'} its 20-EMA; last bar volume {vr:.1f}× average"))
    else: out.append(V("Volume / OBV","neutral",0,"No volume data from this source (spot FX has no central volume)"))
    # 10 VWAP rolling 50
    if hasvol:
        tp=(df.h+df.l+df.c)/3; vw=((tp*df.v).rolling(50).sum()/df.v.rolling(50).sum()).iloc[-1]
        dist=(last-vw)/a
        vote="bullish" if dist>0.2 else "bearish" if dist<-0.2 else "neutral"
        out.append(V("VWAP (rolling 50 bars)",vote,clamp(35+abs(dist)*12,0,80) if vote!="neutral" else 30,f"Price {abs(dist):.1f} ATR {'above' if dist>0 else 'below'} VWAP {fmt(vw,p)}"))
    else: out.append(V("VWAP (rolling 50 bars)","neutral",0,"Needs volume — not available for this market"))
    # 11 Support/resistance
    ph,pl=find_pivots(df.h.values,df.l.values)
    res=sorted([v for _,v in ph if v>last]); sup=sorted([v for _,v in pl if v<last],reverse=True)
    R1=res[0] if res else df.h.tail(200).max(); S1=sup[0] if sup else df.l.tail(200).min()
    dr=(R1-last)/a; ds=(last-S1)/a
    if ds<0.6 and dr>1.5: vote,conf,why="bullish",55,"sitting on support"
    elif dr<0.6 and ds>1.5: vote,conf,why="bearish",55,"pressing into resistance"
    elif dr<0.6 and last>df.h.iloc[-21:-1].max(): vote,conf,why="bullish",50,"breaking out"
    else: vote,conf,why="neutral",30,"between levels"
    out.append(V("Support / resistance",vote,conf,f"S {fmt(S1,p)} ({ds:.1f} ATR) · R {fmt(R1,p)} ({dr:.1f} ATR): {why}"))
    # 12 Candles
    o,h,l,cl=df.o.values,df.h.values,df.l.values,df.c.values
    body=abs(cl[-1]-o[-1]); rng=h[-1]-l[-1] or 1e-12; uw=h[-1]-max(cl[-1],o[-1]); lw=min(cl[-1],o[-1])-l[-1]
    pat=None
    if cl[-1]>o[-1] and cl[-2]<o[-2] and cl[-1]>=o[-2] and o[-1]<=cl[-2]: pat=("bullish",65,"Bullish engulfing")
    elif cl[-1]<o[-1] and cl[-2]>o[-2] and cl[-1]<=o[-2] and o[-1]>=cl[-2]: pat=("bearish",65,"Bearish engulfing")
    elif lw>2*body and uw<body and body/rng<0.35: pat=("bullish",50,"Hammer / pin bar (long lower wick)")
    elif uw>2*body and lw<body and body/rng<0.35: pat=("bearish",50,"Shooting star (long upper wick)")
    elif body/rng<0.1: pat=("neutral",30,"Doji — indecision")
    elif cl[-1]>o[-1] and cl[-2]>o[-2] and cl[-3]>o[-3]: pat=("bullish",45,"Three white soldiers-style run")
    elif cl[-1]<o[-1] and cl[-2]<o[-2] and cl[-3]<o[-3]: pat=("bearish",45,"Three black crows-style run")
    else: pat=("neutral",20,f"No clear pattern; last candle {'green' if cl[-1]>o[-1] else 'red'}")
    out.append(V("Candlestick patterns",pat[0],pat[1],pat[2]))
    # 13 Divergence over last 30 bars
    win=30; seg=df.tail(win+10).reset_index(drop=True); Rs=R.tail(win+10).reset_index(drop=True)
    sh,sl=find_pivots(seg.h.values,seg.l.values,3); vote,conf,why="neutral",25,"No RSI divergence in last 30 bars"
    if len(sl)>=2:
        (i1,v1),(i2,v2)=sl[-2],sl[-1]
        if v2<v1 and Rs[i2]>Rs[i1]+2: vote,conf,why="bullish",60,"Bullish divergence: lower price low, higher RSI low"
    if vote=="neutral" and len(sh)>=2:
        (i1,v1),(i2,v2)=sh[-2],sh[-1]
        if v2>v1 and Rs[i2]<Rs[i1]-2: vote,conf,why="bearish",60,"Bearish divergence: higher price high, lower RSI high"
    out.append(V("RSI divergence",vote,conf,why))
    # 14 MTF
    if htf_trends:
        sc=sum(htf_trends.values()); n=len(htf_trends)
        vote="bullish" if sc>0 else "bearish" if sc<0 else "neutral"
        out.append(V("Multi-timeframe confirmation",vote,30+50*abs(sc)/n,"Other TFs EMA20 vs EMA50: "+", ".join(f"{k} {'up' if v>0 else 'down'}" for k,v in htf_trends.items())))
    else: out.append(V("Multi-timeframe confirmation","neutral",0,"No higher timeframe above 1d"))
    # 15 ATR
    a_med=A.tail(100).median(); exp=a/a_med; mv=(c.iloc[-1]-c.iloc[-4])/a
    if exp>1.2 and abs(mv)>1: vote,conf="bullish" if mv>0 else "bearish",50
    else: vote,conf="neutral",25
    out.append(V("Volatility (ATR 14)",vote,conf,f"ATR {fmt(a,p)} ({a/last*100:.2f}% of price), {exp:.1f}× its 100-bar median; 3-bar move {mv:+.1f} ATR"))
    # 16 News
    if news["n"]>0:
        sc=news["score"]; vote="bullish" if sc>0.15 else "bearish" if sc<-0.15 else "neutral"
        out.append(V("News sentiment",vote,clamp(25+abs(sc)*60+min(15,news['n']),0,75),f"{news['n']} relevant headlines, net tone {sc:+.2f} ({news['pos']} positive / {news['neg']} negative)"))
    else: out.append(V("News sentiment","neutral",0,"No relevant headlines in fetched feeds"))
    # consensus
    num=sum((1 if v["vote"]=="bullish" else -1 if v["vote"]=="bearish" else 0)*v["conf"] for v in out); den=sum(v["conf"] for v in out) or 1
    score=round(100*num/den)
    sig="BUY" if score>=25 else "SELL" if score<=-25 else "WAIT"
    side=1 if sig!="SELL" else -1
    lv={"entry":last,"stop":last-side*1.5*a,"t1":last+side*2*a,"t2":last+side*3*a,"atr":a,"side":"long" if side>0 else "short"}
    return out,score,sig,lv,{"e20":e20,"e50":e50,"ub":ub,"lb":lb,"rsi":R,"S1":S1,"R1":R1}

# ---------- news ----------
FEEDS=[("CoinDesk","https://www.coindesk.com/arc/outboundfeeds/rss/"),("Cointelegraph","https://cointelegraph.com/rss"),("Decrypt","https://decrypt.co/feed"),
 ("FXStreet","https://www.fxstreet.com/rss/news"),("Investing.com Forex","https://www.investing.com/rss/news_1.rss"),
 ("Google News: gold","https://news.google.com/rss/search?q=gold+price+when:2d&hl=en-US&gl=US&ceid=US:en"),
 ("Google News: forex","https://news.google.com/rss/search?q=(dollar+OR+euro+OR+yen+OR+franc)+forex+when:2d&hl=en-US&gl=US&ceid=US:en"),
 ("Google News: crypto","https://news.google.com/rss/search?q=bitcoin+OR+ethereum+OR+crypto+when:2d&hl=en-US&gl=US&ceid=US:en")]
POS="surge surges soar soars rally rallies gain gains rise rises rising jump jumps climb climbs record high highs bullish boost boosts rebound rebounds recover recovers inflow inflows approve approved approval upgrade strong strength beats optimism adoption breakout buy buying support".split()
NEG="fall falls fell drop drops plunge plunges slump slumps crash crashes decline declines slide slides tumble tumbles sink sinks low lows bearish fear fears hack hacked exploit loss losses selloff sell-off outflow outflows weak weakness ban lawsuit sues risk risks warning concern slip slips cut cuts pressure liquidation liquidations dump".split()
KEYS={"XAUUSD":["gold","xau","bullion","precious metal"],"BTCUSD":["bitcoin","btc"],"ETHUSD":["ethereum","eth ","ether "],"SOLUSD":["solana","sol "],"XRPUSD":["xrp","ripple"],
 "EURUSD":["eur/usd","eurusd","euro","ecb"],"GBPUSD":["gbp/usd","gbpusd","pound","sterling","boe","bank of england"],"USDJPY":["usd/jpy","usdjpy","yen","boj","bank of japan"],
 "USDCHF":["usd/chf","usdchf","franc","snb","swiss"],"EURCHF":["eur/chf","eurchf","franc","snb","swiss","euro"]}
# for pairs where USD is base, positive USD news => bullish; where USD quote => bearish. For simplicity score headlines matching asset keywords.
INVERT={"USDJPY":["yen","boj","bank of japan"],"USDCHF":["franc","snb","swiss"],"EURCHF":["franc","snb","swiss"]}
def fetch_news():
    items=[]; status=[]
    for name,u in FEEDS:
        try:
            root=ET.fromstring(get(u,15)); n=0
            for it in root.iter("item"):
                t=html.unescape((it.findtext("title") or "").strip()); l=(it.findtext("link") or "").strip(); d=it.findtext("pubDate") or ""
                try: ts=pd.to_datetime(d,utc=True).timestamp()
                except Exception: ts=0
                if t: items.append({"src":name,"title":t,"link":l,"ts":ts}); n+=1
                if n>=40: break
            status.append({"source":name,"ok":True,"count":n})
        except Exception as e: status.append({"source":name,"ok":False,"err":str(e)[:80]})
    seen=set(); uniq=[]
    for i in sorted(items,key=lambda x:-x["ts"]):
        k=i["title"].lower()[:70]
        if k in seen: continue
        seen.add(k); uniq.append(i)
    cutoff=time.time()-3*86400
    uniq=[i for i in uniq if i["ts"]==0 or i["ts"]>=cutoff]
    for i in uniq:
        w=re.findall(r"[a-z\-]+",i["title"].lower()); i["tone"]=sum(x in POS for x in w)-sum(x in NEG for x in w)
    return uniq,status
def news_for(sym,items):
    ks=KEYS[sym]; rel=[]
    for i in items:
        tl=" "+i["title"].lower()+" "
        if any(k in tl for k in ks):
            t=i["tone"]
            if sym in INVERT and any(k in tl for k in INVERT[sym]) and not any(k in tl for k in ["usd/","eur/chf","dollar"]): t=-t
            rel.append({**i,"t":t})
    pos=sum(1 for r in rel if r["t"]>0); neg=sum(1 for r in rel if r["t"]<0)
    score=(pos-neg)/max(1,pos+neg) if rel else 0
    return {"n":len(rel),"pos":pos,"neg":neg,"score":round(score,2),"items":rel[:6]}

def r6(x,p): return None if x is None or (isinstance(x,float) and math.isnan(x)) else round(float(x),p+1)

def main():
    news,nstatus=fetch_news()
    data={}; srcstat=[]
    raw={}
    for key,label,src,code,mkt in SYMBOLS:
        for tf in TFS:
            try:
                df=binance(code,tf) if src=="binance" else yahoo(code,tf)
                if len(df)<120: raise Exception(f"only {len(df)} candles")
                raw[(key,tf)]=df; srcstat.append({"sym":key,"tf":tf,"ok":True,"n":len(df)})
            except Exception as e:
                srcstat.append({"sym":key,"tf":tf,"ok":False,"err":str(e)[:80]})
    out=[]
    for key,label,src,code,mkt in SYMBOLS:
        nf=news_for(key,news); sd={"key":key,"label":label,"market":mkt,"source":("Binance public API ("+code+")") if src=="binance" else ("Yahoo Finance chart API ("+code+")"),"news":[{"src":i["src"],"title":i["title"],"link":i["link"],"t":i["t"],"ts":i["ts"]} for i in nf["items"]],"tf":{}}
        for ti,tf in enumerate(TFS):
            df=raw.get((key,tf))
            if df is None: sd["tf"][tf]={"ok":False,"err":next(s.get("err","") for s in srcstat if s["sym"]==key and s["tf"]==tf)}; continue
            htf={}
            for h in [x for x in ["5m","15m","1h","4h","1d"] if x!=tf]:
                d2=raw.get((key,h))
                if d2 is not None: htf[h]=1 if ema(d2.c,20).iloc[-1]>ema(d2.c,50).iloc[-1] else -1
            p=prec(df.c.iloc[-1])
            agents,score,sig,lv,ind=analyze(df,p,htf,nf)
            N=60; tl=df.tail(N)
            age=(time.time()*1000-df.t.iloc[-1])/60000
            sd["tf"][tf]={"ok":True,"p":p,"score":score,"signal":sig,"agents":agents,
              "levels":{k:(r6(v,p) if k!="side" else v) for k,v in lv.items()},
              "S1":r6(ind["S1"],p),"R1":r6(ind["R1"],p),
              "last":r6(df.c.iloc[-1],p),"chg":round(float((df.c.iloc[-1]/df.c.iloc[-2]-1)*100),3),
              "lastTime":datetime.fromtimestamp(df.t.iloc[-1]/1000,TZ).strftime("%a %d %b %H:%M"),"ageMin":int(age),
              "bull":sum(a["vote"]=="bullish" for a in agents),"bear":sum(a["vote"]=="bearish" for a in agents),
              "c":{"t":[datetime.fromtimestamp(x/1000,TZ).strftime("%d %b %H:%M" if tf!="1d" else "%d %b %Y") for x in tl.t],
                   "o":[r6(x,p) for x in tl.o],"h":[r6(x,p) for x in tl.h],"l":[r6(x,p) for x in tl.l],"cl":[r6(x,p) for x in tl.c],
                   "e20":[r6(x,p) for x in ind["e20"].tail(N)],"e50":[r6(x,p) for x in ind["e50"].tail(N)],
                   "ub":[r6(x,p) for x in ind["ub"].tail(N)],"lb":[r6(x,p) for x in ind["lb"].tail(N)],"rsi":[round(float(x),1) for x in ind["rsi"].tail(N)]}}
        out.append(sd)
    res={"generated":datetime.now(TZ).strftime("%a %d %b %Y, %H:%M"),"symbols":out,"newsStatus":nstatus,"priceStatus":srcstat,"newsCount":len(news)}
    from paths import SNAP1
    json.dump(res,open(SNAP1,"w"),separators=(",",":"))
    print("ok",len(json.dumps(res,separators=(",",":"))))

