# Round B: combinations of round-A survivors (pre-registered after round A, before running): unions with the new agents, BTC-1h session scope, VAL-selected composite.
exec(open("cand.py").read().split("\nif __name__==\"__main__\":\n    from multiprocessing")[0])
def base_LS(P,st):
    L,S=base_sig(P,st); aL,aS=dirmask(P,carryf); return L&aL,S&aS
def U(agent,mode):
    def f(P,I,st):
        L,S=base_LS(P,st); aL,aS=agent(P,I); aL=np.asarray(aL,bool); aS=np.asarray(aS,bool)
        if mode=="or": L2=L|aL; S2=S|aS; c=L2&S2; return L2&~c,S2&~c
        return L&pd.Series(aL).rolling(10,min_periods=1).max().values.astype(bool),S&pd.Series(aS).rolling(10,min_periods=1).max().values.astype(bool)
    return f
def sessBTC(P,I,st):
    L,S=base_LS(P,st)
    if P["sym"]=="BTCUSD" and P["tf"]=="1h": ok=(I["hrs"]>=7)&(I["hrs"]<20); return L&ok,S&ok
    return L,S
ARMS2={"BASE":lambda P,I,st: base_LS(P,st),
 "U1 base OR Donchian20+EMA200+HTF":U(A_donch,"or"),
 "U2 base OR TTM squeeze+EMA200":U(A_sqz,"or"),
 "U3 base AND Donchian breakout within 10 bars":U(A_donch,"and"),
 "U4 base OR Donchian OR squeeze":lambda P,I,st: U(lambda P_,I_:(np.asarray(A_donch(P_,I_)[0],bool)|np.asarray(A_sqz(P_,I_)[0],bool),np.asarray(A_donch(P_,I_)[1],bool)|np.asarray(A_sqz(P_,I_)[1],bool)),"or")(P,I,st),
 "S1 session 07-20 London on BTC 1h only":sessBTC}
def run2(args):
    name,key,stage=args; P=PREP[key]; df=raw[key]; out=[]; gb=key[0] in GB; I=ind(key) if gb else None
    p={**BASEP,"_I":I}
    for st in (("f0","f1","f2") if stage=="val" else ("test",)):
        a,b=seg(P,st)
        L,S=ARMS2[name](P,I,st) if gb else base_LS(P,st)
        out+=[(name,stage,key[0],key[1],*t) for t in sim3(df,key[0],np.asarray(L,bool),np.asarray(S,bool),a,b,p)]
    return out
if __name__=="__main__":
    from multiprocessing import Pool
    keys=list(dict.fromkeys(EDGE+FOC))
    for k in keys:
        if k[0] in GB: ind(k)
    jobs=[(nm,k,s) for nm in ARMS2 for k in keys for s in ("val","test")]
    with Pool(8) as P_: R=P_.map(run2,jobs)
    T=pd.DataFrame([x for r in R for x in r],columns=["v","stage","sym","tf","i","R","entry","exit","d"]); T.to_pickle("cand2_trades.pkl")
    def m(x): return metr(list(zip(x.i,x.R)))
    res={}
    for nm in ARMS2:
        x=T[T.v==nm]; res[nm]={}
        print("\n##",nm)
        for st in ("val","test"):
            y=x[x.stage==st]; e=y[[(s,t) in EDGE for s,t in zip(y.sym,y.tf)]]
            res[nm][st]={"edge_book":m(e),**{f"{s} {t}":m(y[(y.sym==s)&(y.tf==t)]) for s,t in FOC}}
            r=res[nm][st]; print(f" {st:4s} BOOK {r['edge_book']}")
            print("      "+" | ".join(f"{k}: n{v['n']} {v.get('exp')} pf{v.get('pf')}" for k,v in r.items() if k!="edge_book"))
    json.dump(res,open("cand2_results.json","w"),indent=1)
