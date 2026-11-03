# Pattern agents test. Baseline = shipped round-4 (gold 1h Donchian gate, BTC 1h gate+session, display-only; alert book without BTC 1h).
import os,sys
os.chdir("/workspace/strategy-rebuild/research")
exec(open("final.py").read().split("\nif __name__")[0])
os.chdir("/workspace/pattern-agents/research"); sys.path.insert(0,".")
import patterns as pt
PATS=["Harmonic","AB=CD","Wolfe","Wyckoff"]
_pc={}
def pats(key):
    if key in _pc: return _pc[key]
    df=raw[key]; L,S,anc,wt,kind=pt.pattern_signals(df); wl,ws,wa,_=pt.wyckoff_signals(df)
    L["Wyckoff"]=wl; S["Wyckoff"]=ws; anc["Wyckoff"]=wa
    _pc[key]=(L,S,anc,wt); return _pc[key]
def single4(df,sym,i,d,p,A):
    o,h,l,c=df.o.values,df.h.values,df.l.values,df.c.values; n=len(df)
    if i+1>=n or not A[i]==A[i]: return None
    e=o[i+1]; j0=i+1; risk=1.5*A[i]; tp=None
    if p.get("pstop"):
        an=p["anc"][i]
        if an==an: risk=min(max(((e-(an-0.25*A[i])) if d>0 else ((an+0.25*A[i])-e)),0.5*A[i]),3.0*A[i])
        wt=p.get("wt")
        if wt is not None and wt[i] is not None:
            v1,i1,s14=wt[i]; t_=v1+s14*(i+1-i1)
            if (t_-e)*d>=1.0*risk: tp=t_
    stop=e-d*risk; cst=cost_of(sym,e)/risk; ext=e; armed=False; last=min(n-1,j0+47); res=None; j=j0
    for j in range(j0,last+1):
        if (d>0 and l[j]<=stop) or (d<0 and h[j]>=stop): res=d*(stop-e)/risk; break
        if tp is not None and ((d>0 and h[j]>=tp) or (d<0 and l[j]<=tp)): res=d*(tp-e)/risk; break
        ext=max(ext,h[j]) if d>0 else min(ext,l[j])
        if (ext-e)*d>=risk: armed=True
        if armed:
            stop=max(stop,e) if d>0 else min(stop,e); t_=ext-d*4.0*A[i]; stop=max(stop,t_) if d>0 else min(stop,t_)
    if res is None:
        if j>=n-1 and j-j0<47: return None
        res=d*(c[j]-e)/risk
    return (res-cst,j,j0)
def sim4(df,sym,L,S,a,b,p):
    A=atr(df).values; out=[]; i=a
    while i<b and i<len(df)-1:
        d=1 if L[i] else -1 if S[i] else 0
        if d==0: i+=1; continue
        r=single4(df,sym,i,d,p,A)
        if r is None: break
        out.append((i,r[0],r[2],r[1],d)); i=r[1]+1
    return out
def recent(x,k=10): return pd.Series(np.asarray(x,bool)).rolling(k,min_periods=1).max().values.astype(bool)
ARMS=["BASE"]+[f"{pn} {m}" for pn in PATS for m in ("standalone","standalone+pattern stop","confirm","veto")]
def job(args):
    name,key,stage=args; P=PREP[key]; df=raw[key]; gb=key[0] in GB; I=ind(key) if gb else None; out=[]
    for st in (("f0","f1","f2") if stage=="val" else ("test",)):
        a,b=seg(P,st); p={}
        L,S=R2(P,I,st) if gb else base_LS(P,st); L=np.asarray(L,bool); S=np.asarray(S,bool)
        if gb and name!="BASE":
            pn,m=name.split(" ",1); PL,PS,anc,wt=pats(key)
            if m.startswith("standalone"):
                L,S=PL[pn].copy(),PS[pn].copy()
                if "pattern stop" in m: p={"pstop":True,"anc":anc[pn],"wt":wt if pn=="Wolfe" else None}
            elif m=="confirm": L=L&recent(PL[pn]); S=S&recent(PS[pn])
            elif m=="veto": L=L&~recent(PS[pn]); S=S&~recent(PL[pn])
        out+=[(name,stage,key[0],key[1],*t) for t in sim4(df,key[0],L,S,a,b,p)]
    return out
EK=[k for k in EDGE if k!=("BTCUSD","1h")]
if __name__=="__main__":
    from multiprocessing import Pool
    keys=list(dict.fromkeys(EDGE+FOC))
    for k in keys:
        if k[0] in GB: ind(k); pats(k)
    jobs=[(nm,k,s) for nm in ARMS for k in keys for s in ("val","test")]
    with Pool(8) as P_: R=P_.map(job,jobs)
    T=pd.DataFrame([x for r in R for x in r],columns=["v","stage","sym","tf","i","R","entry","exit","d"]); T.to_pickle(f"trades_w{pt.W}.pkl")
    def m(x): return metr(list(zip(x.i,x.R)))
    res={}
    for nm in ARMS:
        r={}
        for st in ("val","test"):
            y=T[(T.v==nm)&(T.stage==st)]
            r[st]={"book":m(y[[(s,t) in EK for s,t in zip(y.sym,y.tf)]]),"gb6":m(y[y.sym.isin(GB)]),**{f"{s} {t}":m(y[(y.sym==s)&(y.tf==t)]) for s,t in FOC}}
        # ship check for filters vs BASE on alert book
        if nm!="BASE" and not nm.split(" ",1)[1].startswith("standalone"):
            ok=True; hurt=[]
            for st in ("val","test"):
                b=res["BASE"][st]; x=r[st]
                ok&=x["book"]["n"]>=0.6*b["book"]["n"] and x["book"]["exp"]>=b["book"]["exp"]-1e-9 and (x["book"].get("pf") or 0)>=b["book"]["pf"]-1e-9
                for s,t in EK:
                    kk=f"{s} {t}"
                    if kk in b and b[kk]["n"]>=8 and (x[kk]["n"]==0 or x[kk]["exp"]<b[kk]["exp"]-1e-9): hurt.append(f"{st}:{kk}")
            r["ship"]=bool(ok and not hurt); r["hurt"]=hurt
        res[nm]=r
    json.dump(res,open(f"results_w{pt.W}.json","w"),indent=1)
    for nm,r in res.items():
        print(f"{nm:38s} VAL {r['val']['gb6' if 'standalone' in nm else 'book']} | TEST {r['test']['gb6' if 'standalone' in nm else 'book']} {'SHIP' if r.get('ship') else ''} {r.get('hurt','')}")
