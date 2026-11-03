"""Strategy rebuild 3 Nov 2026. Pre-registered candidates (rules fixed from the sources before any run; no parameter search).
Harness = round-2 bt2/exp2 (votes /tmp/votes, weights refit per fold on the train half only, VAL = 3 walk-forward folds in train half, TEST = unseen 2nd half),
baseline = 'AFTER: carry + trail 4' (market entry next open, 1.5xATR stop, BE at +1R, 4xATR chandelier, 48 bars, Vantage-like costs)."""
import sys,json
src=open("/workspace/trading/research/exp2.py").read().replace('if __name__=="__main__":','if False:')
src=src.replace('PP="/tmp/prep2.pkl"','FOCUS=[(s,t) for s in SYMS for t in TFL]\nPP="/tmp/prep2_all.pkl"')
exec(src)
GB=("XAUUSD","BTCUSD")
_cache={}
def ind(key):
    if key in _cache: return _cache[key]
    df=raw[key]; c=df.c; h,l=df.h.values,df.l.values; A=atr(df).values
    sh,sl=pivots_confirmed(h,l,3)
    lastH=pd.Series(sh).ffill().values; lastL=pd.Series(sl).ffill().values
    e20,e50,e200=ema(c,20).values,ema(c,50).values,ema(c,200).values
    st,_=supertrend(df,10,3)
    # RSI(14) regular divergence on confirmed 3-bar pivots, active 10 bars after confirmation
    R=rsi(c,14).values; n=len(df); bear=np.zeros(n,bool); bull=np.zeros(n,bool)
    ph=[];pl=[]
    for i in range(n):
        if not np.isnan(sh[i]): ph.append((i-3,sh[i]))
        if not np.isnan(sl[i]): pl.append((i-3,sl[i]))
        if len(ph)>=2:
            (a,pa),(b,pb)=ph[-2],ph[-1]
            if b-a<=30 and pb>pa and R[b]<R[a] and i-(b+3)<=10: bear[i]=True
        if len(pl)>=2:
            (a,pa),(b,pb)=pl[-2],pl[-1]
            if b-a<=30 and pb<pa and R[b]>R[a] and i-(b+3)<=10: bull[i]=True
    hrs=pd.to_datetime(df.t+BARMS[key[1]],unit="ms",utc=True).dt.tz_convert("Europe/London").dt.hour.values
    # session/weekly/monthly anchored VWAP (as in 2 Nov review C2)
    ix=pd.to_datetime(df.t,unit="ms",utc=True).dt.tz_convert("America/New_York"); sess=ix+pd.Timedelta(hours=7); tf=key[1]
    k_=sess.dt.date if tf in("5m","15m","1h") else (sess.dt.to_period("W-SAT").astype(str) if tf=="4h" else ix.dt.to_period("M").astype(str))
    tp=(df.h+df.l+df.c)/3; v=df.v.replace(0,np.nan).fillna(1.0); g=k_.values
    vw=((tp*v).groupby(g).cumsum()/v.groupby(g).cumsum()).values
    # Donchian 20 (prior bars) and TTM squeeze
    dh=df.h.rolling(20).max().shift(1).values; dl=df.l.rolling(20).min().shift(1).values
    sma=c.rolling(20).mean(); sd=c.rolling(20).std(ddof=0); a20=atr(df,20)
    sq=((sma+2*sd)<(sma+1.5*a20))&((sma-2*sd)>(sma-1.5*a20)); sq=sq.values
    fire=np.r_[False,sq[:-1]&~sq[1:]]
    mom=(c-((df.h.rolling(20).max()+df.l.rolling(20).min())/2+sma)/2).rolling(20).apply(lambda y: np.polyval(np.polyfit(np.arange(20),y,1),19),raw=True).values
    X,_,_=adx(df)
    # confirmed pivots known at bar i, last 120 bars (as plan() uses sh[-120:]/sl[-120:])
    phs=[(k,sh[k]) for k in range(n) if not np.isnan(sh[k])]; pls=[(k,sl[k]) for k in range(n) if not np.isnan(sl[k])]
    import bisect
    pk=[k for k,_ in phs]; lk=[k for k,_ in pls]
    ph_list=[phs[bisect.bisect_left(pk,i-119):bisect.bisect_right(pk,i)] for i in range(n)]
    pl_list=[pls[bisect.bisect_left(lk,i-119):bisect.bisect_right(lk,i)] for i in range(n)]
    out=dict(ph_list=ph_list,pl_list=pl_list,A=A,lastH=lastH,lastL=lastL,e20=e20,e50=e50,e200=e200,st=st,bear=bear,bull=bull,hrs=hrs,vw=vw,c=c.values,dh=dh,dl=dl,fire=fire,mom=mom,adx=X.values)
    _cache[key]=out; return out
def single3(df,sym,i,d,p,A):
    o,h,l,c=df.o.values,df.h.values,df.l.values,df.c.values; n=len(df)
    if i+1>=n or not A[i]==A[i]: return None
    e=o[i+1]; j0=i+1
    sm=p.get("stopmode","atr"); I=p.get("_I")
    if sm=="atr": risk=p.get("stopA",1.5)*A[i]
    else:
        piv=I["pl_list"] if d>0 else I["ph_list"]; sw=np.nan
        for (ki,kv) in reversed(piv[i]):
            if (d>0 and kv<e) or (d<0 and kv>e): sw=kv; break
        buf=0.5 if sm=="struct" else 0.15
        raw_=(e-(sw-buf*A[i])) if d>0 else ((sw+buf*A[i])-e)
        if not raw_==raw_: raw_=1.5*A[i]
        if sm=="struct": risk=min(max(raw_,1.0*A[i]),3.0*A[i])
        else:  # replica of live plan(): <0.8 ATR -> 1.2 ATR, >3 ATR -> 2 ATR
            risk=raw_ if 0.8*A[i]<=raw_<=3*A[i] else (1.2*A[i] if raw_<0.8*A[i] else 2*A[i])
    stop=e-d*risk; cst=cost_of(sym,e)/risk
    be=p.get("be",1.0); tr=p.get("trail",4.0); mb=p.get("maxbars",48); stx=p.get("st_exit",False)
    ext=e; armed=False; last=min(n-1,j0+mb-1); j=j0; res=None
    for j in range(j0,last+1):
        if (d>0 and l[j]<=stop) or (d<0 and h[j]>=stop): res=d*(stop-e)/risk; break
        ext=max(ext,h[j]) if d>0 else min(ext,l[j])
        if be is not None and (ext-e)*d>=be*risk: armed=True
        if armed:
            stop=max(stop,e) if d>0 else min(stop,e)
            t_=ext-d*tr*A[i]; stop=max(stop,t_) if d>0 else min(stop,t_)
        if stx and I["st"][j]==-d and j+1<n: res=d*(o[j+1]-e)/risk; j=j+1; break
    if res is None:
        if j>=n-1 and j-j0<mb-1: return None
        res=d*(c[j]-e)/risk
    R=res-cst
    return (R,j,j0,R<0)
def sim3(df,sym,L,S,a,b,p):
    A=atr(df).values; out=[]; i=a
    while i<b and i<len(df)-1:
        d=1 if L[i] else -1 if S[i] else 0
        if d==0: i+=1; continue
        r=single3(df,sym,i,d,p,A)
        if r is None: break
        out.append((i,r[0],r[2],r[1],d)); i=r[1]+1
    return out
# ---- filters: return (allowL, allowS) for gold/BTC; other symbols unchanged ----
def F_ema200(P,I): return I["c"]>I["e200"], I["c"]<I["e200"]
def F_stack(P,I): return (I["e20"]>I["e50"])&(I["e50"]>I["e200"]), (I["e20"]<I["e50"])&(I["e50"]<I["e200"])
def F_st(P,I): return I["st"]>0, I["st"]<0
def F_sess(P,I):
    ok=(I["hrs"]>=7)&(I["hrs"]<20) if P["tf"] in ("5m","15m","1h") else np.ones(P["n"],bool); return ok,ok
def F_div(P,I): return ~I["bear"], ~I["bull"]
def F_vwap1h(P,I):
    if P["tf"]!="1h": o=np.ones(P["n"],bool); return o,o
    return I["c"]>I["vw"], I["c"]<I["vw"]
def F_adx(P,I): o=I["adx"]>=20; return o,o
# ---- standalone new agents (gold/BTC) ----
def A_donch(P,I):
    hb=P["hb"]; hh=(hb>0) if P["hasH"] else True; hl=(hb<0) if P["hasH"] else True
    return (I["c"]>I["dh"])&(I["c"]>I["e200"])&hh, (I["c"]<I["dl"])&(I["c"]<I["e200"])&hl
def A_sqz(P,I):
    return I["fire"]&(I["mom"]>0)&(I["c"]>I["e200"]), I["fire"]&(I["mom"]<0)&(I["c"]<I["e200"])
BASEP={"be":1.0,"trail":4.0,"maxbars":48,"stopmode":"atr","stopA":1.5}
ARMS={
 "BASE":dict(),
 "X1 stop: swing +0.5ATR buffer (1-3 ATR)":dict(p={"stopmode":"struct"}),
 "X2 stop: 2.0xATR":dict(p={"stopA":2.0}),
 "X3 stop: live plan() replica (swing+0.15ATR)":dict(p={"stopmode":"live"}),
 "X4 exit: Supertrend(10,3) flip":dict(p={"st_exit":True}),
 "F1 EMA200 side":dict(f=F_ema200),
 "F2 EMA20/50/200 stack":dict(f=F_stack),
 "F3 Supertrend(10,3) agrees":dict(f=F_st),
 "F4 London+NY session (07-20 London, intraday)":dict(f=F_sess),
 "F5 RSI(14) divergence block":dict(f=F_div),
 "F6 session AVWAP side, 1h only":dict(f=F_vwap1h),
 "F7 ADX>=20":dict(f=F_adx),
 "N1 agent: Donchian20 + EMA200 + HTF":dict(agent=A_donch),
 "N2 agent: TTM squeeze fire + EMA200":dict(agent=A_sqz),
}
def run_arm(args):
    name,key,stage=args; cfg=ARMS[name]; P=PREP[key]; df=raw[key]; out=[]
    gb=key[0] in GB; I=ind(key) if gb else None
    p={**BASEP,**(cfg.get("p",{}) if gb else {}),"_I":I}
    if p["stopmode"]!="atr" or p.get("st_exit"): pass
    for st in (("f0","f1","f2") if stage=="val" else ("test",)):
        a,b=seg(P,st)
        if gb and "agent" in cfg: L,S=cfg["agent"](P,I)
        else:
            L,S=base_sig(P,st); aL,aS=dirmask(P,carryf); L=L&aL; S=S&aS
            if gb and "f" in cfg: fL,fS=cfg["f"](P,I); L=L&fL; S=S&fS
        L=np.asarray(L,bool); S=np.asarray(S,bool)
        out+=[(name,stage,key[0],key[1],*t) for t in sim3(df,key[0],L,S,a,b,p)]
    return out
EDGE=[('XAUUSD','1d'),('XAUUSD','4h'),('XAUUSD','1h'),('BTCUSD','1d'),('BTCUSD','1h'),('USDJPY','1d'),('USDCHF','1d')]
FOC=[(s,t) for s in GB for t in ("1h","4h","1d")]
if __name__=="__main__":
    from multiprocessing import Pool
    keys=list(dict.fromkeys(EDGE+FOC))
    for k in keys:
        if k[0] in GB: ind(k)
    jobs=[(nm,k,s) for nm in ARMS for k in keys for s in ("val","test")]
    with Pool(8) as P_: R=P_.map(run_arm,jobs)
    T=pd.DataFrame([x for r in R for x in r],columns=["v","stage","sym","tf","i","R","entry","exit","d"]); T.to_pickle("cand_trades.pkl")
    def m(x): return metr(list(zip(x.i,x.R)))
    res={}
    for nm in ARMS:
        x=T[T.v==nm]; res[nm]={}
        for st in ("val","test"):
            y=x[x.stage==st]; e=y[[(s,t) in EDGE for s,t in zip(y.sym,y.tf)]]
            res[nm][st]={"edge_book":m(e),**{f"{s} {t}":m(y[(y.sym==s)&(y.tf==t)]) for s,t in FOC}}
    json.dump(res,open("cand_results.json","w"),indent=1)
    for nm in ARMS:
        print("\n##",nm)
        for st in ("val","test"):
            r=res[nm][st]; print(f" {st:4s} BOOK {r['edge_book']}")
            print("      "+" | ".join(f"{k}: n{v['n']} {v.get('exp')}" for k,v in r.items() if k!="edge_book"))
