# Round D: agent pruning inside the walk-forward fit. A1 = an agent keeps its weight only if its edge is also positive on the most recent half of
# the fit window (stability check inside train data); otherwise weight 0. Per-agent standalone barrier edge train vs unseen test is reported.
exec(open("final.py").read().split("\nif __name__")[0])
CATS=[CATALL[x] for x in NAMES]
def score_stage(P,up,dn,stage,stab):
    end={"f0":P["folds"][0][0],"f1":P["folds"][1][0],"f2":P["folds"][2][0],"test":P["mid"]}[stage]; lo=P["lo"]; n=P["n"]
    m=np.zeros(n,bool); m[lo:end-13]=True; w=wfit(P["V"],up,dn,m); w[15]=0
    if stab:
        m2=np.zeros(n,bool); m2[(lo+end)//2:end-13]=True; w2=wfit(P["V"],up,dn,m2); w=np.where(w2>0,w,0.0); w[15]=0
    valid=np.zeros(n,bool); valid[lo:]=True
    return np.where(valid,score(P["V"],P["C"],P["RK"],w,CATS),0),w
def runA(args):
    key,stage,stab=args; P=PREP[key]; df=raw[key]; up,dn=barrier(df); I=ind(key) if key[0] in GB else None; out=[]
    for st in (("f0","f1","f2") if stage=="val" else ("test",)):
        sc,w=score_stage(P,up,dn,st,stab); hb=P["hb"]
        L=(sc>=25)&P["volok"]&(P["td"]>0)&((hb>0) if P["hasH"] else True); S=(sc<=-25)&P["volok"]&(P["td"]<0)&((hb<0) if P["hasH"] else True)
        aL,aS=dirmask(P,carryf); L&=aL; S&=aS
        a,b=seg(P,st); out+=[("A1" if stab else "BASEr",stage,key[0],key[1],*t) for t in sim3(df,key[0],L,S,a,b,{**BASEP,"_I":I})]
    ag=[]
    if stage=="test":
        for k in range(28):
            def e(a_,b_):
                sel=np.zeros(P["n"],bool); sel[a_:b_-13]=True; sel&=(P["V"][:,k]!=0)&~np.isnan(up)
                rs=np.where(P["V"][sel,k]>0,up[sel],dn[sel])[::3]; return (len(rs),round(float(rs.mean()),3) if len(rs) else None)
            ag.append((NAMES[k],e(P["lo"],P["mid"]),e(P["mid"],P["n"])))
    return out,(key,ag)
if __name__=="__main__":
    from multiprocessing import Pool
    keys=list(dict.fromkeys(EDGE+FOC))
    for k in keys:
        if k[0] in GB: ind(k)
    jobs=[(k,s,stab) for k in keys for s in ("val","test") for stab in (False,True)]
    with Pool(8) as P_: R=P_.map(runA,jobs)
    T=pd.DataFrame([x for r,_ in R for x in r],columns=["v","stage","sym","tf","i","R","entry","exit","d"]); T.to_pickle("agents_trades.pkl")
    def m(x): return metr(list(zip(x.i,x.R)))
    for v in ("BASEr","A1"):
        for st in ("val","test"):
            x=T[(T.v==v)&(T.stage==st)]; print(v,st,"BOOK",m(x)," | ".join(f"{s} {t}: n{m(x[(x.sym==s)&(x.tf==t)])['n']} {m(x[(x.sym==s)&(x.tf==t)]).get('exp')}" for s,t in FOC))
    AG={f"{k[0]} {k[1]}":a for _,(k,a) in R if a}
    json.dump(AG,open("agent_edges.json","w"),indent=1)
    print("\nAgents with positive train edge but negative unseen edge (barrier R, every 3rd vote):")
    for kk,a in AG.items():
        if kk.split()[0] in GB: print(kk,[ (nm,tr[1],te[1]) for nm,tr,te in a if tr[1] is not None and te[1] is not None and tr[1]>0.05 and te[1]<0])
