# Standalone patterns on the FULL history of each series (patterns need no agent votes): TRAIN = 1st half, TEST = 2nd half. Same costs/exit.
import os,sys
exec(open("run.py").read().split("\nEK=")[0])
def fjob(args):
    pn,m,key=args; df=raw[key]; n=len(df); PL,PS,anc,wt=pats(key); out=[]
    p={"pstop":True,"anc":anc[pn],"wt":wt if pn=="Wolfe" else None} if m=="pattern stop" else {}
    for st,(a,b) in (("train",(250,n//2)),("test",(n//2,n))):
        out+=[(pn,m,st,key[0],key[1],*t) for t in sim4(df,key[0],PL[pn],PS[pn],a,b,p)]
    return out
if __name__=="__main__":
    from multiprocessing import Pool
    for k in FOC: pats(k)
    jobs=[(pn,m,k) for pn in PATS for m in ("ATR exit","pattern stop") for k in FOC]
    with Pool(8) as P_: R=P_.map(fjob,jobs)
    T=pd.DataFrame([x for r in R for x in r],columns=["p","m","stage","sym","tf","i","R","entry","exit","d"]); T.to_pickle(f"full_w{pt.W}.pkl")
    def m_(x): return metr(list(zip(x.i,x.R)))
    res={}
    for (pn,mm),g in T.groupby(["p","m"]):
        r={st:{"all":m_(g[g.stage==st]),**{f"{s} {t}":m_(g[(g.stage==st)&(g.sym==s)&(g.tf==t)]) for s,t in FOC}} for st in ("train","test")}
        res[f"{pn} | {mm}"]=r
        print(f"{pn:9s} {mm:13s} TRAIN {r['train']['all']} | TEST {r['test']['all']}")
        print("      TEST "+" | ".join(f"{k}: n{v['n']} {v.get('exp')}" for k,v in r['test'].items() if k!='all'))
    json.dump(res,open(f"full_w{pt.W}.json","w"),indent=1)
