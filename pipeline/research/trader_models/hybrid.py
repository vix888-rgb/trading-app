import pickle,sys,numpy as np,pandas as pd
sys.path.insert(0,'.'); from tm import metr
raw=pickle.load(open("/tmp/long.pkl","rb"))
E=pd.read_pickle("/tmp/exp7_focus.pkl"); E=E[E.v=="AFTER: carry + trail 4"].copy()
E["t"]=[int(raw[(s,t)].t.iloc[int(i)]) for s,t,i in zip(E.sym,E.tf,E.i)]; E["src"]="baseline"
H=pd.read_pickle("htf_trades.pkl"); H=H[H.model.str.startswith("ICT/TJR sweep->MSS->FVG, app")].copy()
H["t"]=[int(raw[(s,t)].t.iloc[int(i)]) for s,t,i in zip(H.sym,H.tf,H.i)]; H["src"]="sweep"
tr=H[H.stage=="train"].groupby(["sym","tf"]).R.agg(["mean","count"])
sel=tr[(tr["mean"]>0)&(tr["count"]>=5)].index.tolist(); print("train-selected sweep combos:",sel)
def port(df):
    df=df.sort_values("t"); return metr(df.R.values)
Et=E[E.stage=="test"]; Ht=H[H.stage=="test"]; Hs=Ht[[ (s,t) in sel for s,t in zip(Ht.sym,Ht.tf)]]
Hv=H[H.stage=="train"]; Hvs=Hv[[ (s,t) in sel for s,t in zip(Hv.sym,Hv.tf)]]
print("TEST baseline focus          ",port(Et))
print("TEST sweep, train-selected   ",port(Hs))
print("TEST baseline + sweep(sel)   ",port(pd.concat([Et,Hs])))
print("TEST baseline + sweep(all18) ",port(pd.concat([Et,Ht])))
print("TRAIN sweep sel (in-sample)  ",port(Hvs))
# per-month trade frequency in test
for nm,x in (("baseline",Et),("hybrid",pd.concat([Et,Hs]))):
    span=(x.t.max()-x.t.min())/8.64e7/30.4; print(nm,"trades/month ~",round(len(x)/span,1))
# bootstrap diff
rng=np.random.default_rng(0); a=pd.concat([Et,Hs]).R.values; b=Et.R.values
d=[rng.choice(a,len(a)).sum()-rng.choice(b,len(b)).sum() for _ in range(4000)]; print("total R diff hybrid-baseline",round(a.sum()-b.sum(),1),"P(<=0)",round(float(np.mean(np.array(d)<=0)),3))
pd.concat([Et,Hs]).to_pickle("hybrid_test.pkl")
