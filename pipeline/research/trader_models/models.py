# Mechanical readings of 5 traders' published models on 5m data. Each returns a list of trades: dict(i=signal bar, d, R, j=exit).
import numpy as np, pandas as pd
from tm import *
def day_groups(day):
    idx=np.flatnonzero(np.r_[True,day[1:]!=day[:-1]]); return list(zip(idx,np.r_[idx[1:],len(day)]))
class Ctx:
    def __init__(s,df,sym):
        s.df=df.reset_index(drop=True); s.sym=sym; d=s.df
        s.o,s.h,s.l,s.c,s.v=d.o.values,d.h.values,d.l.values,d.c.values,d.v.values; s.n=len(d)
        s.A=atr(d); s.m,s.day,s.wd=et_cols(d,NY); s.mL,s.dayL,_=et_cols(d,LDN)
        s.days=day_groups(s.day)
        s.SH,s.SL=fractals(s.h,s.l,2)
        for nm,lst in (("H",s.SH),("L",s.SL)):
            v=np.full(s.n,np.nan); ix=np.full(s.n,-1)
            for conf,piv,p in lst:
                if conf<s.n: v[conf]=p; ix[conf]=piv
            v=pd.Series(v).ffill().values; ix=pd.Series(np.where(ix>=0,ix,np.nan)).ffill().fillna(-1).astype(int).values
            setattr(s,"last"+nm+"v",v); setattr(s,"last"+nm+"i",ix)
    def cost(s,px): return cost_of(s.sym,px)
    def pipmin(s,pips,px):
        p=PIP[s.sym]; return pips*p if p else px*pips*1e-4   # BTC: treat 1 "pip" as 1bp
# ---------------- Fabio Valentini: IVB (Initial Volume Breakout) ----------------
def fabio_ivb(X,vwap=False,tp_r=1.0,both=True):
    o,h,l,c=X.o,X.h,X.l,X.c; out=[]
    for a,b in X.days:
        if X.wd[a]>=5: continue
        m=X.m[a:b]; orb=np.flatnonzero((m>=570)&(m<600))+a
        if len(orb)<6: continue
        H=h[orb].max(); L=l[orb].min()
        if vwap: tp_=(h[a:b]+l[a:b]+c[a:b])/3; w=np.where(X.v[a:b]>0,X.v[a:b],1.0); s0=orb[0]-a
        flat=a+int(np.flatnonzero(m<900)[-1]) if (m<900).any() else b-1
        for i in range(orb[-1]+1,b-1):
            if not (600<=X.m[i]<=885): continue
            d=1 if c[i]>H else -1 if (both and c[i]<L) else 0
            if d==0: continue
            if vwap:
                vw=np.sum(tp_[s0:i-a+1]*w[s0:i-a+1])/np.sum(w[s0:i-a+1])
                if (d>0 and c[i]<vw) or (d<0 and c[i]>vw): continue
            e=o[i+1]; stop=L if d>0 else H; risk=abs(e-stop)
            if risk<=0.1*X.A[i]: break
            r=sim_trade(o,h,l,c,i+1,d,e,stop,e+d*tp_r*risk,flat,cost=X.cost(e))
            if r: out.append(dict(i=i,d=d,R=r[0],j=r[1]))
            break
    return out
# ---------------- Fabio Valentini: AMT mean reversion (failed break of prior-day value area -> POC) ----------------
def value_area(px,w,bins=40):
    lo,hi=px.min(),px.max()
    if hi<=lo: return None
    hist,edges=np.histogram(px,bins=bins,range=(lo,hi),weights=w); k=int(hist.argmax()); tot=hist.sum(); a=b=k; acc=hist[k]
    while acc<0.7*tot:
        up=hist[b+1] if b+1<bins else -1; dn=hist[a-1] if a-1>=0 else -1
        if up>=dn: b+=1; acc+=hist[b]
        else: a-=1; acc+=hist[a]
    return (edges[a],edges[b+1],(edges[k]+edges[k+1])/2)
def fabio_amt_mr(X,win=(180,720)):
    o,h,l,c=X.o,X.h,X.l,X.c; out=[]; prev=None
    for a,b in X.days:
        if X.wd[a]>=5: prev=None; continue
        if prev is not None:
            VAL,VAH,POC=prev; outside=0; ext=None; done=False
            for i in range(a,b-1):
                if not (win[0]<=X.m[i]<win[1]): continue
                if outside==0:
                    if c[i]>VAH: outside=1; ext=h[i]; k0=i
                    elif c[i]<VAL: outside=-1; ext=l[i]; k0=i
                    continue
                ext=max(ext,h[i]) if outside>0 else min(ext,l[i])
                back=(outside>0 and c[i]<VAH) or (outside<0 and c[i]>VAL)
                if back and i-k0<=12:
                    d=-outside; e=o[i+1]; stop=ext+outside*0.1*X.A[i]; tp=POC
                    if (tp-e)*d<1.0*abs(e-stop) or abs(e-stop)<=0: break
                    flat=min(b-1,i+96)
                    r=sim_trade(o,h,l,c,i+1,d,e,stop,tp,flat,cost=X.cost(e))
                    if r: out.append(dict(i=i,d=d,R=r[0],j=r[1]))
                    break
                if i-k0>12: break
        w=np.where(X.v[a:b]>0,X.v[a:b],1.0); prev=value_area((h[a:b]+l[a:b]+c[a:b])/3,w) if b-a>=100 else None
    return out
# ---------------- Tom Hougaard: School Run (2nd 15m bar of the open) ----------------
def school_run(X,open_="ldn",mode="classic",tp_r=1.0,runner=False):
    o,h,l,c=X.o,X.h,X.l,X.c; out=[]
    mm=X.mL if open_=="ldn" else X.m; dd=X.dayL if open_=="ldn" else X.day
    st=480 if open_=="ldn" else 570          # 08:00 London / 09:30 NY
    for a,b in day_groups(dd):
        if X.wd[a]>=5 and X.sym!="BTCUSD": continue
        sb=np.flatnonzero((mm[a:b]>=st+15)&(mm[a:b]<st+30))+a
        if len(sb)<3: continue
        H=h[sb].max(); L=l[sb].min(); off=0.05*X.A[sb[-1]]; rng=H-L
        if rng<=0: continue
        flat=a+int(np.flatnonzero(mm[a:b]<st+480)[-1])
        for i in range(sb[-1]+1,min(b,flat)):
            up=h[i]>=H+off; dn=l[i]<=L-off
            if not (up or dn): continue
            if up and dn: break  # ambiguous bar -> skip day
            brk=1 if up else -1
            if mode=="classic":
                d=brk; e=max(o[i],H+off) if d>0 else min(o[i],L-off); stop=L-off if d>0 else H+off
            else:  # anti: fade the break
                d=-brk; e=H+off if brk>0 else L-off; e=max(o[i],e) if brk>0 else min(o[i],e); stop=e+brk*rng
            risk=abs(e-stop); tp=None if runner else e+d*tp_r*risk
            r=sim_trade(o,h,l,c,i,d,e,stop,tp,flat,be=1.0 if runner else None,trail=1.0 if runner else None,A=rng,cost=X.cost(e))
            if r: out.append(dict(i=i,d=d,R=r[0],j=r[1]))
            break
    return out
# ---------------- ICT: Silver Bullet ----------------
def silver_bullet(X,window=(600,660),stop_mode="fvg",tp_mode="dol",min_pips=15):
    """Pools: prior-day H/L + today's pre-window H/L. Today a pool (or the 20-bar pre-window extreme) is swept; after the sweep a close beyond the
    latest swing = MSS; first FVG in the MSS direction that forms inside the window -> limit at its 50% (CE), must fill inside the window.
    Stop beyond the FVG (ICT notes) or beyond the sweep; target = nearest opposite pool (>= min framework), else 1R floor. Flat 16:00 NY."""
    o,h,l,c=X.o,X.h,X.l,X.c; out=[]; pdh=pdl=None
    for a,b in X.days:
        if X.wd[a]>=5: pdh=pdl=None; continue
        m=X.m[a:b]; on=np.flatnonzero(m<window[0])+a; wi=np.flatnonzero((m>=window[0])&(m<window[1]))+a
        if pdh is None or len(on)<24 or len(wi)<6:
            if b-a>50: pdh,pdl=h[a:b].max(),l[a:b].min()
            continue
        bsl=[pdh,h[on].max()]; ssl=[pdl,l[on].min()]; k=on[-1]-20; e20h=h[k:on[-1]+1].max(); e20l=l[k:on[-1]+1].min()
        wend=wi[-1]; flat=a+int(np.flatnonzero(m<960)[-1]); taken=False
        for d in (1,-1):
            if taken: break
            sweep=None; mss=False; ext=None
            for i in range(a+3,wend+1):
                if sweep is None:
                    if (d>0 and (l[i]<pdl or (i>on[-1] and l[i]<e20l))) or (d<0 and (h[i]>pdh or (i>on[-1] and h[i]>e20h))): sweep=i; ext=l[i] if d>0 else h[i]; ref=(X.lastHv if d>0 else X.lastLv)[i]
                    continue
                ext=min(ext,l[i]) if d>0 else max(ext,h[i])
                if not mss:
                    if ref==ref and ((d>0 and c[i]>ref) or (d<0 and c[i]<ref)): mss=True
                    else:
                        if (d>0 and l[i]<=ext) or (d<0 and h[i]>=ext): ref=(X.lastHv if d>0 else X.lastLv)[i]
                        continue
                if i<wi[0]: continue
                if (d>0 and l[i]>h[i-2]) or (d<0 and h[i]<l[i-2]):
                    bot,top=(h[i-2],l[i]) if d>0 else (h[i],l[i-2]); ce=(bot+top)/2
                    stop=((bot if d>0 else top) if stop_mode=="fvg" else ext)-d*0.1*X.A[i]
                    pools=[x for x in (bsl if d>0 else ssl) if (x-ce)*d>0]
                    if not pools: break
                    tg=min(pools) if d>0 else max(pools)
                    if abs(tg-ce)<X.pipmin(min_pips,ce): break
                    fl=fill_limit(o,h,l,i+1,wend,d,ce)
                    if fl is None: break
                    j0,e=fl; risk=abs(e-stop)
                    if risk<=0 or (stop-e)*d>=0: break
                    tp=tg if tp_mode=="dol" else e+d*2*risk
                    if (tp-e)*d<risk: tp=e+d*risk
                    r=sim_trade(o,h,l,c,j0,d,e,stop,tp,flat,cost=X.cost(e))
                    if r: out.append(dict(i=i,d=d,R=r[0],j=r[1])); taken=True
                    break
        pdh,pdl=h[a:b].max(),l[a:b].min()
    return out
# ---------------- TJR: sweep -> 5m BOS -> FVG/EQ retrace, 9:30-10:30 ----------------
def tjr(X,entry="fvg",tp_mode="dol"):
    o,h,l,c=X.o,X.h,X.l,X.c; out=[]; pdh=pdl=None
    for a,b in X.days:
        if X.wd[a]>=5: pdh=pdl=None; continue
        m=X.m[a:b]
        lon=np.flatnonzero((m>=120)&(m<300))+a; asia=np.flatnonzero(m<120)+a   # Asia ~ (prev 20:00-) 00:00-02:00 ET proxy, London 02:00-05:00 ET
        if pdh is None or len(lon)<20:
            if b-a>50: pdh,pdl=h[a:b].max(),l[a:b].min()
            continue
        highs=[pdh,h[lon].max()]+([h[asia].max()] if len(asia) else []); lows=[pdl,l[lon].min()]+([l[asia].min()] if len(asia) else [])
        sess=np.flatnonzero((m>=570)&(m<630))+a
        took=False; state=None
        for i in sess:
            if took: break
            # sweep detection
            if state is None:
                for lv in highs:
                    if h[i]>lv and h[a:i].max()<=lv: state=dict(d=-1,lvl=lv,ext=h[i],k=i); break
                if state is None:
                    for lv in lows:
                        if l[i]<lv and l[a:i].min()>=lv: state=dict(d=1,lvl=lv,ext=l[i],k=i); break
                continue
            d=state["d"]; state["ext"]=max(state["ext"],h[i]) if d<0 else min(state["ext"],l[i])
            if "bos" not in state:
                k=state["k"]; li=(X.lastLi if d<0 else X.lastHi)[k]
                if li>=k-30:
                    lv=(X.lastLv if d<0 else X.lastHv)[k]
                    if (d<0 and c[i]<lv) or (d>0 and c[i]>lv): state["bos"]=i
                continue
            # continuation entry
            leg_hi=h[state["k"]:i+1].max(); leg_lo=l[state["k"]:i+1].min()
            fv=None
            for k in range(state["k"]+2,i+1):
                if d>0 and l[k]>h[k-2]: fv=(h[k-2],l[k])
                if d<0 and h[k]<l[k-2]: fv=(h[k],l[k-2])
            px=((fv[1] if d>0 else fv[0]) if (entry=="fvg" and fv) else (leg_hi+leg_lo)/2)
            stop=state["ext"]-d*0.1*X.A[i]
            tg=[x for x in (highs if d>0 else lows) if (x-px)*d>0]
            fl=fill_limit(o,h,l,i+1,a+int(np.flatnonzero(m<630)[-1]),d,px)
            took=True
            if fl is None: break
            j0,e=fl; risk=abs(e-stop)
            if risk<=0 or (stop-e)*d>=0: break
            if tp_mode=="dol":
                tp=(min(tg) if d>0 else max(tg)) if tg else e+d*2*risk
                if (tp-e)*d<risk: tp=e+d*risk
            else: tp=e+d*2*risk
            flat=a+int(np.flatnonzero(m<960)[-1])
            r=sim_trade(o,h,l,c,j0,d,e,stop,tp,flat,cost=X.cost(e))
            if r: out.append(dict(i=i,d=d,R=r[0],j=r[1]))
        pdh,pdl=h[a:b].max(),l[a:b].min()
    return out
# ---------------- PB Blake: HTF FVG bias -> key level (15m/5m FVG) -> 5m IFVG of manipulation leg ----------------
def pb_prepare(X):
    d5=X.df; out={}
    for rule,ms in (("15min",9e5),("1h",3.6e6),("4h",1.44e7),("1D",8.64e7)):
        g=resample(d5,rule); st=fvg_bias_states(g); out[rule]=map_htf(d5,g,st,ms)
    X.bias=np.sign(out["1D"]+out["4h"]+out["1h"]+out["15min"])*((np.abs(out["1D"]+out["4h"]+out["1h"]+out["15min"]))>=2)
    g=resample(d5,"15min"); X.g15=g
    # 15m FVG list with formation close time
    gh,gl=g.h.values,g.l.values; F=[]
    for i in range(2,len(g)):
        if gl[i]>gh[i-2]: F.append((g.t.values[i]+9e5,1,gh[i-2],gl[i]))
        if gh[i]<gl[i-2]: F.append((g.t.values[i]+9e5,-1,gh[i],gl[i-2]))
    X.F15=F; X.Ft=np.array([x[0] for x in F])
    o,h,l,t=X.o,X.h,X.l,d5.t.values; F5=[]
    for i in range(2,len(d5)):
        if l[i]>h[i-2]: F5.append((t[i]+3e5,1,h[i-2],l[i]))
        if h[i]<l[i-2]: F5.append((t[i]+3e5,-1,h[i],l[i-2]))
    X.F5=F5; X.F5t=np.array([x[0] for x in F5])
    gh1=resample(d5,"1h"); SH1,SL1=fractals(gh1.h.values,gh1.l.values,2)
    X.H1=[(gh1.t.values[k]+3.6e6,p) for k,_,p in SH1]; X.L1=[(gh1.t.values[k]+3.6e6,p) for k,_,p in SL1]
def pb_blake(X,window=(570,660),tp_mode="dol",maxn=2,need_bias=True,keys5=False):
    if not hasattr(X,"bias"): pb_prepare(X)
    o,h,l,c,t=X.o,X.h,X.l,X.c,X.df.t.values; out=[]; F=X.F15; fi=0
    for a,b in X.days:
        if X.wd[a]>=5: continue
        m=X.m[a:b]; wi=np.flatnonzero((m>=window[0])&(m<window[1]))+a
        if len(wi)<6: continue
        s0=wi[0]; bias=X.bias[s0-1] if need_bias else 0
        if need_bias and bias==0: continue
        # unmitigated 15m key levels in bias direction formed in last 3 days, known before session start
        t0=t[s0]; keys=[]
        lo_=np.searchsorted(X.Ft,t0-3*8.64e7); hi_=np.searchsorted(X.Ft,t0,"right")
        for (tf_,dd,bot,top) in F[lo_:hi_]:
            if need_bias and dd!=bias: continue
            k=np.searchsorted(t,tf_)
            if k>=s0: continue
            seg=slice(k,s0)
            mitig=(l[seg].min()<=top) if dd>0 else (h[seg].max()>=bot)
            if not mitig: keys.append((dd,bot,top))
        if keys5:
            lo_=np.searchsorted(X.F5t,t0-8.64e7); hi_=np.searchsorted(X.F5t,t0,"right")
            for (tf_,dd,bot,top) in X.F5[lo_:hi_]:
                if need_bias and dd!=bias: continue
                if abs(top-bot)<0.3*X.A[s0-1]: continue
                k=np.searchsorted(t,tf_)
                if k>=s0: continue
                seg=slice(k,s0); mitig=(l[seg].min()<=top) if dd>0 else (h[seg].max()>=bot)
                if not mitig: keys.append((dd,bot,top))
        if not keys: continue
        ntr=0; i=s0; dirs=[bias] if need_bias else [1,-1]
        while i<wi[-1] and ntr<maxn:
            hit=None
            for (dd,bot,top) in keys:
                if dd not in dirs: continue
                if (dd>0 and l[i]<=top and c[i]>=bot) or (dd<0 and h[i]>=bot and c[i]<=top): hit=(dd,bot,top); break
            if hit is None: i+=1; continue
            dd,bot,top=hit
            # manipulation leg: from last swing extreme before touch to running extreme; IFVG = opposite 5m FVG inside the leg closed through
            li=(X.lastHi if dd>0 else X.lastLi)[i]
            k0=li if li>=a-24 else max(a,i-12)
            done=False
            for j in range(i,min(wi[-1],i+12)+1):
                ext=l[k0:j+1].min() if dd>0 else h[k0:j+1].max()
                if (dd>0 and c[j]<bot) or (dd<0 and c[j]>top): break   # key level disrespected
                # opposite-direction FVGs in leg
                inv=None
                for q in range(k0+2,j):
                    if dd>0 and h[q]<l[q-2]: inv=l[q-2]   # top of bearish FVG
                    if dd<0 and l[q]>h[q-2]: inv=h[q-2]   # bottom of bullish FVG
                if inv is None: continue
                if (dd>0 and c[j]>inv) or (dd<0 and c[j]<inv):
                    e=o[j+1]; stop=ext-dd*0.1*X.A[j]; risk=abs(e-stop)
                    if risk<0.2*X.A[j] or (stop-e)*dd>=0: done=True; break
                    tj=t[j]
                    if tp_mode=="dol":
                        cand=[p for (tt,p) in (X.H1 if dd>0 else X.L1) if tt<=tj and (p-e)*dd>0 and tt>tj-10*8.64e7]
                        tg=(min(cand) if dd>0 else max(cand)) if cand else e+dd*2*risk
                        R_=(tg-e)*dd/risk; tp=e+dd*min(max(R_,1.0),3.0)*risk
                    else: tp=e+dd*float(tp_mode)*risk
                    flat=a+int(np.flatnonzero(m<960)[-1])
                    r=sim_trade(o,h,l,c,j+1,dd,e,stop,tp,flat,be=1.0 if (tp-e)*dd>1.01*risk else None,cost=X.cost(e))
                    if r: out.append(dict(i=j,d=dd,R=r[0],j=r[1])); ntr+=1; i=r[1]
                    done=True; break
            keys=[k_ for k_ in keys if k_!=hit]
            i=max(i+1,j+1 if not done else i+1)
    return out
