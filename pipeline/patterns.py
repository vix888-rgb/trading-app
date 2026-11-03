"""Pattern agents (3 Nov 2026), causal. Swings = confirmed 4-bar pivots (a pivot at bar k is usable from bar k+4), alternating zigzag,
each leg >= 1 x ATR(14) at the leg end. A pattern is evaluated only when its last point is confirmed; the signal is at that bar (entry next open).
Rules were fixed before any test run (standard textbook ratios); no tuning after results."""
import numpy as np, pandas as pd
W=4; MINLEG=1.0  # fixed before testing (3 Nov 2026); W=2 was a sensitivity check only
def _atr(df,n=14):
    pc=df.c.shift(); tr=pd.concat([df.h-df.l,(df.h-pc).abs(),(df.l-pc).abs()],axis=1).max(axis=1); return tr.ewm(alpha=1/n,adjust=False).mean().values
def zigzag_events(df,w=W,minleg=MINLEG):
    """Yield (known_at, pts) after each zigzag update; pts = list of (idx, val, +1 high / -1 low), all confirmed by known_at."""
    h,l=df.h.values,df.l.values; n=len(h); A=_atr(df); pts=[]; out=[]
    for k in range(w,n-w):
        isH=h[k]==h[k-w:k+w+1].max(); isL=l[k]==l[k-w:k+w+1].min(); ka=k+w
        for typ,v in ((1,h[k]) if isH else (None,None),(-1,l[k]) if isL else (None,None)):
            if typ is None: continue
            if not pts: pts.append((k,v,typ)); continue
            li,lv,lt=pts[-1]
            if lt==typ:
                if (typ>0 and v>lv) or (typ<0 and v<lv): pts[-1]=(k,v,typ); out.append((ka,list(pts[-7:])))
            elif abs(v-lv)>=minleg*A[k]:
                pts.append((k,v,typ)); out.append((ka,list(pts[-7:])))
    return out,A
def _in(x,a,b): return a<=x<=b
HARM={  # B/XA, C/AB, D: CD/BC, D/XA   (Carney / Pesavento standard ratios, ~±0.03 tolerance on the single-value ratios)
 "Gartley":((0.588,0.648),(0.382,0.886),(1.13,1.618),(0.756,0.816)),
 "Bat":((0.382,0.50),(0.382,0.886),(1.618,2.618),(0.856,0.916)),
 "Butterfly":((0.756,0.816),(0.382,0.886),(1.618,2.24),(1.27,1.618)),
 "Crab":((0.382,0.618),(0.382,0.886),(2.24,3.618),(1.588,1.648)),
}
def harmonic(p5):
    (xi,X,xt),(ai,A_,at),(bi,B,bt),(ci,C,ct),(di,D,dt)=p5
    XA=abs(A_-X); AB=abs(A_-B); BC=abs(C-B); CD=abs(C-D)
    if min(XA,AB,BC,CD)<=0: return None
    d=1 if dt<0 else -1  # pattern ending at a low -> long
    r1,r2,r3,r4=AB/XA,BC/AB,CD/BC,abs(A_-D)/XA
    for nm,(b,c,cd,dx) in HARM.items():
        if _in(r1,*b) and _in(r2,*c) and _in(r3,*cd) and _in(r4,*dx): return nm,d,D
    # Shark (Carney 2011): C extends 1.13-1.618 of XA beyond A? standard: B 0.382-0.618 XA... C = 1.13-1.618 of AB past A, D = 0.886-1.13 of XC
    XC=abs(C-X)
    if _in(r1,0.382,0.618) and _in(BC/AB,1.13,1.618) and XC>0 and _in(abs(C-D)/XC,0.886,1.13): return "Shark",d,D
    return None
def abcd(p4):
    (ai,A_,at),(bi,B,bt),(ci,C,ct),(di,D,dt)=p4
    AB=abs(A_-B); BC=abs(C-B); CD=abs(C-D)
    if min(AB,BC,CD)<=0: return None
    if _in(BC/AB,0.382,0.886) and _in(CD/AB,0.9,1.1) and _in(CD/BC,1.13,2.618):
        tAB=bi-ai; tCD=di-ci
        if tAB>0 and _in(tCD/tAB,0.5,2.0): return "AB=CD",(1 if dt<0 else -1),D
    return None
def wolfe(p5,A5):
    (i1,v1,t1),(i2,v2,t2),(i3,v3,t3),(i4,v4,t4),(i5,v5,t5)=p5
    d=1 if t5<0 else -1
    if d>0:  # bullish: 1,3,5 lows falling, 2,4 highs falling, converging (1-3 steeper), 4 above 1 (overlap), 5 overshoots 1-3 line by <= 1 ATR
        if not (v3<v1 and v5<v3 and v4<v2 and v4>v1): return None
    else:
        if not (v3>v1 and v5>v3 and v4>v2 and v4<v1): return None
    s13=(v3-v1)/(i3-i1); s24=(v4-v2)/(i4-i2)
    if not ((d>0 and s13<s24) or (d<0 and s13>s24)): return None  # converging wedge
    line13=v1+s13*(i5-i1); over=(line13-v5)*d
    if not (0<=over<=1.0*A5): return None
    s14=(v4-v1)/(i4-i1)
    return "Wolfe",d,v5,(v1,i1,s14)  # target line 1-4: v1+s14*(t-i1)
def pattern_signals(df):
    """Per bar: dict of arrays L/S per pattern + stop anchor (pattern extreme) and Wolfe target line."""
    ev,A=zigzag_events(df); n=len(df)
    names=["Harmonic","AB=CD","Wolfe"]; L={k:np.zeros(n,bool) for k in names}; S={k:np.zeros(n,bool) for k in names}
    anchor={k:np.full(n,np.nan) for k in names}; wtgt=[None]*n; kind=[None]*n; seen=set(); pinfo=[]
    for ka,pts in ev:
        if ka>=n: continue
        res=[]
        if len(pts)>=5:
            r=harmonic(pts[-5:]); 
            if r: res.append(("Harmonic",r[1],r[2],r[0]))
            r=wolfe(pts[-5:],A[pts[-1][0]])
            if r: res.append(("Wolfe",r[1],r[2],"Wolfe")); wtgt[ka]=r[3]
        if len(pts)>=4:
            r=abcd(pts[-4:])
            if r: res.append(("AB=CD",r[1],r[2],"AB=CD"))
        for nm,d,anc,lab in res:
            key=(nm,pts[-1][0],d)
            if key in seen: continue  # one signal per pattern end point
            seen.add(key)
            # still valid at the confirmation bar: close has not broken the D/5 extreme
            c=df.c.values[ka]
            if (d>0 and c<=anc) or (d<0 and c>=anc): continue
            (L if d>0 else S)[nm][ka]=True; anchor[nm][ka]=anc
            if nm=="Harmonic": kind[ka]=lab
            npt=5 if nm in ("Harmonic","Wolfe") else 4
            pinfo.append((ka,lab,d,[(int(i_),float(v_)) for i_,v_,_ in pts[-npt:]]))
    pattern_signals.last_info=pinfo
    return L,S,anchor,wtgt,kind
def wyckoff_signals(df,N=40,maxh=8.0,hold=30):
    """Accumulation: trading range = bars [i-N-? ] ... built from confirmed data only.
    Range: highest high / lowest low of the N bars ending 1 bar before the spring bar, height <= maxh x ATR.
    Spring: low < range low (by < 1.5 ATR) and close back >= range low. SOS: within 15 bars a close above the range's upper quartile.
    LPS: first confirmed pivot low (W bars) after the SOS that holds above the range low; entry on its confirmation bar if close > LPS.
    Distribution mirror: UTAD -> SOW (close below lower quartile) -> LPSY pivot high below range high -> short. Signal must come within `hold` bars of the spring."""
    h,l,c=df.h.values,df.l.values,df.c.values; n=len(c); A=_atr(df)
    rh=pd.Series(h).rolling(N).max().shift(1).values; rl=pd.Series(l).rolling(N).min().shift(1).values
    L=np.zeros(n,bool); S=np.zeros(n,bool); anc=np.full(n,np.nan); info=[None]*n
    for d in (1,-1):
        st=None  # (phase, spring_i, lo, hi, spring_extreme, sos_i)
        for i in range(N+1,n):
            if st is None:
                if not (rh[i]==rh[i] and (rh[i]-rl[i])<=maxh*A[i]): continue
                if d>0 and l[i]<rl[i] and rl[i]-l[i]<1.5*A[i] and c[i]>=rl[i]: st=["spring",i,rl[i],rh[i],l[i],None]
                if d<0 and h[i]>rh[i] and h[i]-rh[i]<1.5*A[i] and c[i]<=rh[i]: st=["spring",i,rl[i],rh[i],h[i],None]
                continue
            ph,si,lo,hi,ext,sos=st
            if i-si>hold: st=None; continue
            if (d>0 and l[i]<ext) or (d<0 and h[i]>ext): st=None; continue  # spring low / UTAD high broken -> invalid
            if ph=="spring":
                if i-si<=15 and ((d>0 and c[i]>hi-0.25*(hi-lo)) or (d<0 and c[i]<lo+0.25*(hi-lo))): st[0]="sos"; st[5]=i
                elif i-si>15: st=None
                continue
            k=i-W  # pivot at k confirmed now
            if k<=sos: continue
            if d>0 and l[k]==l[k-W:k+W+1].min() and l[k]>lo and c[i]>l[k]:
                L[i]=True; anc[i]=l[k]; info[i]=("LPS",si,sos,k,lo,hi); st=None
            if d<0 and h[k]==h[k-W:k+W+1].max() and h[k]<hi and c[i]<h[k]:
                S[i]=True; anc[i]=h[k]; info[i]=("LPSY",si,sos,k,lo,hi); st=None
    return L,S,anc,info

_CACHE={}
def all_signals(df):
    """Cached per series state: (L, S, info) with L/S dicts for Harmonic, AB=CD, Wolfe, Wyckoff."""
    key=(len(df),int(df.t.iloc[-1]),float(df.c.iloc[-1]))
    if key in _CACHE: return _CACHE[key]
    L,S,_,_,_=pattern_signals(df); info=list(pattern_signals.last_info)
    wl,ws,_,wi=wyckoff_signals(df); L["Wyckoff"]=wl; S["Wyckoff"]=ws
    for i,x in enumerate(wi):
        if x: info.append((i,x[0],1 if x[0]=="LPS" else -1,[(int(x[1]),float(df.l.values[x[1]] if x[0]=="LPS" else df.h.values[x[1]])),(int(x[3]),float(df.l.values[x[3]] if x[0]=="LPS" else df.h.values[x[3]]))],(float(x[4]),float(x[5]))))
    if len(_CACHE)>64: _CACHE.clear()
    _CACHE[key]=(L,S,sorted(info,key=lambda z:z[0])); return _CACHE[key]
def panel(df,lookback=30):
    """Patterns confirmed in the last `lookback` bars, newest first (display only)."""
    _,_,info=all_signals(df); n=len(df); t=df.t.values; out=[]
    for z in reversed(info):
        ka=z[0]
        if ka<n-lookback: break
        item={"name":z[1],"dir":int(z[2]),"ago":int(n-1-ka),"pts":[[int(t[i]),round(v,8)] for i,v in z[3]]}
        if len(z)>4: item["range"]=[round(z[4][0],8),round(z[4][1],8)]
        out.append(item)
        if len(out)>=4: break
    return out
