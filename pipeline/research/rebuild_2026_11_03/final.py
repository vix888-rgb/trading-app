# Round C: scoped composites + formal ship rule + bootstrap. Ship rule (same as nightly review): pooled exp and PF not lower on VAL and TEST,
# no combo with >=8 base trades loses expectancy, >=60% of base trades kept.
exec(open("cand2.py").read().split("\nif __name__")[0])
def don_conf(P,I,L,S):
    aL,aS=A_donch(P,I); aL=pd.Series(np.asarray(aL,bool)).rolling(10,min_periods=1).max().values.astype(bool); aS=pd.Series(np.asarray(aS,bool)).rolling(10,min_periods=1).max().values.astype(bool)
    return L&aL,S&aS
def R1(P,I,st):
    L,S=base_LS(P,st)
    if P["tf"]=="1h": L,S=don_conf(P,I,L,S)
    return L,S
def R2(P,I,st):
    L,S=R1(P,I,st)
    if P["sym"]=="BTCUSD" and P["tf"]=="1h": ok=(I["hrs"]>=7)&(I["hrs"]<20); L,S=L&ok,S&ok
    return L,S
def R4(P,I,st):  # Donchian confirmation on gold 1h only
    L,S=base_LS(P,st)
    if P["tf"]=="1h" and P["sym"]=="XAUUSD": L,S=don_conf(P,I,L,S)
    return L,S
ARMS2.update({"R1 Donchian-20 breakout confirmation on 1h (gold+BTC)":R1,"R2 = R1 + London/NY session on BTC 1h":R2,"R4 Donchian confirmation gold 1h only":R4,
              "U3 base AND Donchian (all gold/BTC TFs)":U(A_donch,"and"),"S1 session BTC 1h":sessBTC})
for k in ["U1 base OR Donchian20+EMA200+HTF","U2 base OR TTM squeeze+EMA200","U4 base OR Donchian OR squeeze","U3 base AND Donchian breakout within 10 bars","S1 session 07-20 London on BTC 1h only"]: ARMS2.pop(k,None)
if __name__=="__main__":
    from multiprocessing import Pool
    keys=list(dict.fromkeys(EDGE+FOC))
    for k in keys:
        if k[0] in GB: ind(k)
    jobs=[(nm,k,s) for nm in ARMS2 for k in keys for s in ("val","test")]
    with Pool(8) as P_: R=P_.map(run2,jobs)
    T=pd.DataFrame([x for r in R for x in r],columns=["v","stage","sym","tf","i","R","entry","exit","d"]); T.to_pickle("final_trades.pkl")
    def m(x): return metr(list(zip(x.i,x.R)))
    books={"edge7":EDGE,"edge6_noBTC1h":[k for k in EDGE if k!=("BTCUSD","1h")]}
    out={}
    for nm in ARMS2:
        out[nm]={}
        for bk,EK in books.items():
            ok_all=True; o={}
            for st in ("val","test"):
                b=T[(T.v=="BASE")&(T.stage==st)]; x=T[(T.v==nm)&(T.stage==st)]
                bb=b[[(s,t) in EK for s,t in zip(b.sym,b.tf)]]; xx=x[[(s,t) in EK for s,t in zip(x.sym,x.tf)]]
                mb,mx=m(bb),m(xx); hurt=[]
                for (s,t) in EK:
                    cb=m(bb[(bb.sym==s)&(bb.tf==t)]); cx=m(xx[(xx.sym==s)&(xx.tf==t)])
                    if cb["n"]>=8 and (cx["n"]==0 or cx["exp"]<cb["exp"]-1e-9): hurt.append(f"{s} {t}")
                ok=mx["exp"]>=mb["exp"]-1e-9 and (mx.get("pf") or 0)>=(mb.get("pf") or 0)-1e-9 and not hurt and mx["n"]>=0.6*mb["n"]
                ok_all&=ok
                rng=np.random.default_rng(1); d=[rng.choice(xx.R.values,len(xx)).mean()-rng.choice(bb.R.values,len(bb)).mean() for _ in range(3000)]
                o[st]={"base":mb,"new":mx,"hurt":hurt,"ok":ok,"P_no_gain":round(float(np.mean(np.array(d)<=0)),3)}
            o["ship"]=bool(ok_all); out[nm][bk]=o
            if nm!="BASE": print(f"{nm[:50]:50s} {bk:14s} SHIP={o['ship']}  VAL {o['val']['new']} hurt{o['val']['hurt']} | TEST {o['test']['new']} hurt{o['test']['hurt']} P(no gain) {o['test']['P_no_gain']}")
            else: print(f"BASE {bk}: VAL {o['val']['base']} | TEST {o['test']['base']}")
    # per gold/BTC tf table for every arm (TEST and VAL)
    tab={}
    for nm in ARMS2:
        for st in ("val","test"):
            for s,t in FOC:
                x=T[(T.v==nm)&(T.stage==st)&(T.sym==s)&(T.tf==t)]; tab[f"{nm}|{st}|{s} {t}"]=m(x)
    json.dump({"ship":out,"per_tf":tab},open("final_results.json","w"),indent=1)
