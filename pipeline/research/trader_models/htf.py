# Higher-TF (1h/4h/1D) versions of the trader models on the SAME unseen half used for the round-2 baseline.
import pickle,sys,numpy as np,pandas as pd
sys.path.insert(0,"/workspace/rebuild/research")
from tm import *
import tm; tm.MAXCOST=1e9
raw=pickle.load(open("/tmp/long.pkl","rb"))
SYMS=["XAUUSD","BTCUSD","EURUSD","USDJPY","USDCHF","EURCHF"]; TFS=["1h","4h","1d"]
HTF={"1h":"4h","4h":"1d","1d":None}; MS={"1h":3.6e6,"4h":1.44e7,"1d":8.64e7}
FOCUS={("XAUUSD","1d"),("BTCUSD","1d"),("EURUSD","1d"),("USDJPY","1d"),("USDCHF","1d"),("EURCHF","1d"),("XAUUSD","4h"),("XAUUSD","1h")}
def split(sym,tf):
    n=len(raw[(sym,tf)]); z=np.load(f"/tmp/votes/{sym}_{tf}.npz"); lo=int(z["idx"][0]); return lo,(lo+n)//2
def exitp(o,h,l,c,A,i,d,e,stop,mode):
    if mode=="CH4": return sim_trade(o,h,l,c,i+1,d,e,stop,None,i+48,be=1.0,trail=4.0,A=A[i])
    if mode=="2R": return sim_trade(o,h,l,c,i+1,d,e,stop,e+d*2*abs(e-stop),i+48)
def sweep_mss_fvg(df,sym,mode="CH4",lb=20,bias=None):
    """ICT/TJR/PB core on one TF: sweep of the prior lb-bar extreme that closes back inside -> displacement close beyond the last swing (MSS) within 10 bars
    -> FVG in that leg -> limit at FVG edge (next 10 bars). Stop beyond swept extreme (0.5..3 ATR). Optional HTF bias array must agree."""
    o,h,l,c=df.o.values,df.h.values,df.l.values,df.c.values; A=atr(df); n=len(df); out=[]
    SH,SL=fractals(h,l,2); lastH=np.full(n,np.nan); lastL=np.full(n,np.nan)
    for conf,piv,p in SH:
        if conf<n: lastH[conf]=p
    for conf,piv,p in SL:
        if conf<n: lastL[conf]=p
    lastH=pd.Series(lastH).ffill().values; lastL=pd.Series(lastL).ffill().values
    hh=pd.Series(h).rolling(lb).max().shift(1).values; ll=pd.Series(l).rolling(lb).min().shift(1).values
    i=lb+5
    while i<n-12:
        d=1 if (l[i]<ll[i] and c[i]>ll[i]) else -1 if (h[i]>hh[i] and c[i]<hh[i]) else 0
        if d==0 or (bias is not None and bias[i]!=d): i+=1; continue
        ext=l[i] if d>0 else h[i]; ref=lastH[i] if d>0 else lastL[i]; got=None
        for k in range(i+1,min(n-2,i+10)):
            ext=min(ext,l[k]) if d>0 else max(ext,h[k])
            if (d>0 and l[k]<ext-1e-12) or (d<0 and h[k]>ext+1e-12): pass
            if ref==ref and ((d>0 and c[k]>ref) or (d<0 and c[k]<ref)):
                for q in range(i+2,k+1):
                    if d>0 and l[q]>h[q-2]: got=(k,l[q])
                    if d<0 and h[q]<l[q-2]: got=(k,h[q])
                break
        if got is None: i+=1; continue
        k,px=got; stop=ext-d*0.1*A[k]; risk=abs(px-stop)
        if risk<0.5*A[k] or risk>3*A[k]: i=k+1; continue
        fl=fill_limit(o,h,l,k+1,k+10,d,px)
        if fl is None: i=k+1; continue
        j0,e=fl
        if (stop-e)*d>=0: i=j0+1; continue
        r=(sim_trade(o,h,l,c,j0,d,e,stop,None,j0+47,be=1.0,trail=4.0,A=A[k],cost=cost_of(sym,e)) if mode=="CH4"
           else sim_trade(o,h,l,c,j0,d,e,stop,e+d*2*abs(e-stop),j0+47,cost=cost_of(sym,e)))
        if r is None: break
        out.append(dict(i=k,d=d,R=r[0],j=r[1])); i=r[1]+1
    return out
def pb_bias(sym,tf):
    """PB step-1 bias: sum of FVG respect/disrespect states on this TF and the higher TFs (1D/4h/1h), |sum|>=2."""
    df=raw[(sym,tf)]; s=fvg_bias_states(df).copy()
    for hh in [x for x in ("1h","4h","1d") if MS[x]>MS[tf]]:
        g=raw[(sym,hh)]; s+=map_htf(df,g,fvg_bias_states(g),MS[hh])
    return np.where(np.abs(s)>=2,np.sign(s),0)
rows=[]
for sym in SYMS:
    for tf in TFS:
        df=raw[(sym,tf)].reset_index(drop=True); lo,mid=split(sym,tf); B=pb_bias(sym,tf)
        for nm,kw in (("ICT/TJR sweep->MSS->FVG, app exit (BE+4ATR trail)",dict(mode="CH4")),("ICT/TJR sweep->MSS->FVG, 2R",dict(mode="2R")),
                      ("PB Blake: same + HTF FVG-respect bias, app exit",dict(mode="CH4",bias=B)),("PB Blake: same + HTF bias, 2R",dict(mode="2R",bias=B))):
            for x in sweep_mss_fvg(df,sym,**kw):
                if x["i"]<lo: continue
                rows.append(dict(sym=sym,tf=tf,model=nm,stage="test" if x["i"]>=mid else "train",R=x["R"],i=x["i"],d=x["d"],focus=(sym,tf) in FOCUS))
        print(sym,tf,flush=True)
T=pd.DataFrame(rows); T.to_pickle("/workspace/rebuild/research/htf_trades.pkl")
# PB bias as a filter on the round-2 baseline trades
E=pd.read_pickle("/tmp/exp7_focus.pkl"); E=E[E.v=="AFTER: carry + trail 4"].copy(); keep=[]
for (sym,tf),g in E.groupby(["sym","tf"]):
    B=pb_bias(sym,tf)
    for idx,r in g.iterrows(): keep.append((idx,B[int(r.i)]==r.d, B[int(r.i)]!=-r.d))
K=pd.DataFrame(keep,columns=["idx","agree","notagainst"]).set_index("idx"); E=E.join(K); E.to_pickle("/workspace/rebuild/research/base_pbfilter.pkl")
for st in ("val","test"):
    g=E[E.stage==st]; print(st,"base",metr(g.R),"| PB agree",metr(g[g.agree].R),"| PB not against",metr(g[g.notagainst].R))
