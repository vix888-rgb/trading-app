import pickle,sys,time,json,numpy as np,pandas as pd
sys.path.insert(0,"/workspace/rebuild/research")
from models import *
bn=pickle.load(open("/workspace/rebuild/data/bn5m.pkl","rb")); lg=pickle.load(open("/tmp/long.pkl","rb"))
SETS={("BTCUSD","binance BTCUSDT 2023-26"):bn["BTCUSDT"],("XAUUSD","PAXG proxy 2023-26"):bn["PAXGUSDT"],("EURUSD","EURUSDT proxy 2023-26"):bn["EURUSDT"]}
for s in ["XAUUSD","EURUSD","USDJPY","USDCHF","EURCHF","BTCUSD"]: SETS[(s,"yahoo/binance 5m 60d")]=lg[(s,"5m")]
VAR={
 "PB Blake (bias+15m FVG key+5m IFVG, 9:30-11:00, DOL 1-3R)":lambda X:pb_blake(X),
 "PB Blake fixed 2R":lambda X:pb_blake(X,tp_mode=2.0),
 "PB Blake + 5m key levels (premarket FVGs)":lambda X:pb_blake(X,keys5=True),
 "ICT Silver Bullet 10-11 NY (stop beyond FVG, DOL)":lambda X:silver_bullet(X),
 "ICT Silver Bullet 10-11 NY (stop at sweep, DOL)":lambda X:silver_bullet(X,stop_mode="sweep"),
 "ICT Silver Bullet 03-04 NY (London)":lambda X:silver_bullet(X,window=(180,240)),
 "TJR sweep->BOS->FVG 9:30-10:30 (DOL)":lambda X:tjr(X),
 "TJR sweep->BOS->EQ 50% (DOL)":lambda X:tjr(X,entry="eq"),
 "Fabio IVB 30-min ORB 1R (both sides)":lambda X:fabio_ivb(X),
 "Fabio IVB + VWAP bias":lambda X:fabio_ivb(X,vwap=True),
 "Fabio AMT mean-reversion (prior-day VA fail -> POC)":lambda X:fabio_amt_mr(X),
 "Hougaard School Run London classic 1R":lambda X:school_run(X,"ldn"),
 "Hougaard School Run NY classic 1R":lambda X:school_run(X,"ny"),
 "Hougaard School Run London runner (BE+trail)":lambda X:school_run(X,"ldn",runner=True),
 "Hougaard School Run London ANTI 1R":lambda X:school_run(X,"ldn",mode="anti"),
}
rows=[]
for (sym,lab),df in SETS.items():
    df=df[["t","o","h","l","c","v"]].reset_index(drop=True); X=Ctx(df,sym); mid_t=df.t.iloc[len(df)//2]
    for nm,f in VAR.items():
        t0=time.time(); tr=f(X)
        for x in tr: rows.append(dict(set=lab,sym=sym,model=nm,stage="test" if df.t.iloc[x["i"]]>=mid_t else "train",R=x["R"],d=x["d"],t=int(df.t.iloc[x["i"]])))
        print(sym,lab,nm,len(tr),round(time.time()-t0,1),flush=True)
T=pd.DataFrame(rows); T.to_pickle("/workspace/rebuild/research/intraday_trades.pkl")
