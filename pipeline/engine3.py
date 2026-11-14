import os, pickle, json, time, sys
from multiprocessing import Pool
_PIPE = os.path.dirname(os.path.abspath(__file__))
if _PIPE not in sys.path: sys.path.insert(0, _PIPE)
os.environ["TRADING_LOAD_ONLY"] = "1"
exec(compile(open(os.path.join(_PIPE, "engine2.py"), encoding="utf-8").read(), os.path.join(_PIPE, "engine2.py"), "exec"))
os.environ.pop("TRADING_LOAD_ONLY", None)
from paths import STATE, RAW_PKL, SNAP3
import goldfeed
import patterns
os.makedirs(STATE, exist_ok=True)
PARAMS_DEFAULT={"version":1,"horizon_bars":12,"tp_atr":1.5,"sl_atr":1.0,"samples":150,"window":300,"shrink_k":30,"weight_scale":10,"weight_max":2.0,
  "min_samples":10,"low_sample_weight":0.3,"news_weight":0.5,"live_blend":0.5,"live_min":10,"buy_th":25,"sell_th":-25,"regime_mult":WT,"fit_frac":0.6,
  "journal_tfs":["5m","15m","1h","4h","1d"],"journal_expiry_bars":100,
  "filter_mode":True,"exit_mode":"chandelier","trail_atr":4.0,"trail_max_bars":48,"tested_edge_tfs":["1d"],
  "carry_filter":True,"tested_edge":["XAUUSD|1d","XAUUSD|4h","XAUUSD|1h","BTCUSD|1d","BTCUSD|1h","USDJPY|1d","USDCHF|1d","EURUSD|15m"],"params_round":2}  # Nov-2026 walk-forward additions
# Rebuild (Nov 2026, round 3): deterministic scans. News no longer votes (headlines changed the score between scans of the same closed bar),
# limit entries that never fill expire after 12 bars (they blocked new journal entries for up to 100 bars), signals are latched per closed bar.
PARAMS_DEFAULT.update({"news_weight":0.0,"pending_expiry_bars":12,"latch":True,"refit_max_age_days":8,"params_round":3})
# Round 4 (3 Nov 2026 strategy rebuild): BTC 1h leaves the alert list (unseen test -0.05R/trade, PF 0.91, 75 trades; still shown in the app).
# plan_mode "tested": the Entry/SL/TP sent and journaled is the plan the walk-forward measured (market at the signal close, 1.5xATR stop,
# breakeven at +1R, 4xATR chandelier, 48 bars). The old order-block/FVG limit + swing stop ("zone") is shown as an alternative only:
# its stop tested worse on both halves (VAL +0.38R vs +0.53R, TEST +0.27R vs +0.30R) and limit/pullback entries were rejected in round 2.
PARAMS_DEFAULT.update({"tested_edge":["XAUUSD|1d","XAUUSD|4h","XAUUSD|1h","BTCUSD|1d","USDJPY|1d","USDCHF|1d","EURUSD|15m"],"plan_mode":"tested","plan_stop_atr":1.5,"params_round":4})
WPATH=os.path.join(STATE,"weights.json"); JPATH=os.path.join(STATE,"journal.json")
W=json.load(open(WPATH)) if os.path.exists(WPATH) else {}
PRM={**PARAMS_DEFAULT,**W.get("params",{})}
if W.get("params",{}).get("params_round",1)<2:  # round-2 walk-forward update overrides stored values for these keys once
    for _k in ("trail_atr","carry_filter","tested_edge","params_round"): PRM[_k]=PARAMS_DEFAULT[_k]
if W.get("params",{}).get("params_round",1)<3:  # round-3 rebuild overrides once
    for _k in ("news_weight","pending_expiry_bars","latch","refit_max_age_days","params_round"): PRM[_k]=PARAMS_DEFAULT[_k]
if W.get("params",{}).get("params_round",1)<4:  # round-4 rebuild overrides once
    for _k in ("tested_edge","plan_mode","plan_stop_atr","params_round"): PRM[_k]=PARAMS_DEFAULT[_k]
LPATH=os.path.join(STATE,"latch.json")
def refit_due(raw_keys):
    """Agent weights are refit only by the weekly tune-up (TRADING_REFIT=1), when a combo has no stored weights, or when the last refit is too old.
    Between refits every 10-minute scan reuses the stored weights, so the same closed bar always gets the same score."""
    if os.environ.get("TRADING_REFIT")=="1": return True
    if os.environ.get("TRADING_REFIT")=="0": return False
    # Weekly tune-up 14 Nov 2026: a combo missing from by_combo (new combo, or one dropped by an old failed fetch) no longer forces a full
    # refit of every combo; missing_combos() fits only those (see main). Four unscheduled full refits happened 11-13 Nov this way.
    ra=W.get("refit_at")
    if not ra: return True
    try: age=(datetime.now(TZ)-datetime.fromisoformat(ra)).total_seconds()/86400
    except Exception: return True
    return age>PRM["refit_max_age_days"]
def missing_combos(raw_keys):
    """Combos with no stored weights: fitted on their own between weekly refits, so the other combos stay frozen."""
    bc=W.get("by_combo",{})
    return [(k,tf) for k,tf in raw_keys if f"{k}|{tf}" not in bc]
def keep_stored(wout,bc):
    """Carry stored weights for combos not computed this scan (fetch failed), so one failed download never drops a combo from weights.json."""
    out=dict(wout)
    for k,v in (bc or {}).items(): out.setdefault(k,v)
    return out
HSTEP={"1m":6e4,"5m":3e5,"15m":9e5,"1h":3.6e6,"4h":1.44e7,"1d":8.64e7}
NEWS_IDX=15

def votes_at(raw,key,tf,i):
    df=raw[(key,tf)]; w=df.iloc[max(0,i-PRM["window"]+1):i+1].reset_index(drop=True); tcut=w.t.iloc[-1]+HSTEP[tf]
    p=prec(w.c.iloc[-1]); last=w.c.iloc[-1]; a=atr(w).iloc[-1]
    htf={}
    for hh in [x for x in ["5m","15m","1h","4h","1d"] if x!=tf]:
        d2=raw.get((key,hh))
        if d2 is None: continue
        d2=d2[d2.t+HSTEP[hh]<=tcut]
        if len(d2)>60: htf[hh]=1 if ema(d2.c,20).iloc[-1]>ema(d2.c,50).iloc[-1] else -1
    d1=raw.get((key,"1d")); d1=d1[d1.t+HSTEP["1d"]<=tcut+HSTEP["1d"]] if d1 is not None else None  # current day's bar counts as forming → pivots use iloc[-2] = prior day
    ag,_,_,_,_=analyze(w,p,htf,{"n":0}); tr,ev,ch,sh,sl=structure(w); adv,_,_=advanced(w,p,d1,last,a,tr,ev,ch,sh,sl)
    return [VC[g["vote"]] for g in ag+adv],[g["conf"] for g in ag+adv],regime(w)[1]

def outcome(df,i,d):
    A=atr(df.iloc[max(0,i-100):i+1]).iloc[-1]; e=df.c.iloc[i]; H=PRM["horizon_bars"]
    tp=e+d*PRM["tp_atr"]*A; sp=e-d*PRM["sl_atr"]*A; R1=PRM["sl_atr"]*A
    for j in range(i+1,min(len(df),i+1+H)):
        if (d>0 and df.l.iloc[j]<=sp) or (d<0 and df.h.iloc[j]>=sp): return -1.0
        if (d>0 and df.h.iloc[j]>=tp) or (d<0 and df.l.iloc[j]<=tp): return PRM["tp_atr"]/PRM["sl_atr"]
    j=min(len(df)-1,i+H); return float(d*(df.c.iloc[j]-e)/R1)

def stats(rs):
    if not rs: return {"n":0}
    rs=np.array(rs); gw=rs[rs>0].sum(); gl=-rs[rs<0].sum()
    return {"n":int(len(rs)),"hit":round(100*float((rs>0).mean())),"avgR":round(float(rs.mean()),3),"pf":round(float(gw/gl),2) if gl>0 else None}
def weight_from(st):
    if st.get("n",0)<PRM["min_samples"]: return PRM["low_sample_weight"]
    edge=st["avgR"]*st["n"]/(st["n"]+PRM["shrink_k"])
    return round(float(min(PRM["weight_max"],max(0.0,edge*PRM["weight_scale"]))),2)

def adapt_job(args):
    key,tf=args
    raw=pickle.load(open(RAW_PKL,"rb"))[2]
    df=raw[(key,tf)]; n=len(df); H=PRM["horizon_bars"]; S=PRM["samples"]; lo=PRM["window"]; hi=n-H-1
    if hi-lo<30: return (key,tf,None)
    stride=max(1,(hi-lo)//S); idx=list(range(hi,lo,-stride))[:S][::-1]
    rec=[]
    for i in idx:
        v,c,rk=votes_at(raw,key,tf,i)
        rec.append((i,v,c,rk,outcome(df,i,1),outcome(df,i,-1)))
    return (key,tf,rec)

def fit(rec,nag):
    st=[]; 
    for k in range(nag):
        rs=[(r[4] if r[1][k]>0 else r[5]) for r in rec if r[1][k]!=0]
        s=stats(rs); st.append(s)
    return st
def consensus(v,c,rk,w,cats,names):
    num=den=0
    for k in range(len(v)):
        rm=PRM["regime_mult"][rk][cats[k]]; num+=v[k]*c[k]*rm*w[k]; den+=c[k]*rm*max(w[k],0.25)
    return round(100*num/den) if den else 0

def run_journal(raw,curr,names,stratnames):
    J=json.load(open(JPATH)) if os.path.exists(JPATH) else []
    now=time.time()*1000
    # 1) check open outcomes
    for e in J:
        if e.get("status")!="open": continue
        df=raw.get((e["sym"],e["tf"]))
        if df is None: continue
        after=df[df.t>e["bar_t"]]; d=1 if e["side"]=="long" else -1; risk=abs(e["entry"]-e["stop"]); res=None; tp1=False
        if not e.get("filled",True):
            fi=None
            for k_,(_,b) in enumerate(after.iterrows()):
                if (d>0 and b.l<=e["stop"]) or (d<0 and b.h>=e["stop"]): fi=-1; break  # invalidated before fill
                if (d>0 and b.l<=e["entry"]) or (d<0 and b.h>=e["entry"]): fi=k_; break
            if fi is None:
                if len(after)>=PRM.get("pending_expiry_bars",PRM["journal_expiry_bars"]): e.update({"status":"cancelled","exit":"limit not filled"})
                continue
            if fi==-1: e.update({"status":"cancelled","exit":"stop hit before fill"}); continue
            e["filled"]=True; e["fill_bar_t"]=int(after.t.iloc[fi]); after=after.iloc[fi:]
        elif e.get("fill_bar_t"): after=df[df.t>=e["fill_bar_t"]]
        if e.get("exit_mode")=="chandelier":
            # stop -> breakeven once +1R reached, then trail trail_atr x ATR(entry) behind the best price; time exit after trail_max_bars
            st_=e["stop"]; ext=e["entry"]; be=False; k_=0
            for k_,(_,b) in enumerate(after.iterrows()):
                if (d>0 and b.l<=st_) or (d<0 and b.h>=st_):
                    rr=round(float(d*(st_-e["entry"])/risk),2); res=(rr,"stop" if not be else ("breakeven" if abs(rr)<0.01 else "trailing stop")); break
                if (d>0 and b.h>=e["tp"][0]) or (d<0 and b.l<=e["tp"][0]): tp1=True
                ext=max(ext,b.h) if d>0 else min(ext,b.l)
                if (ext-e["entry"])*d>=risk: be=True
                if be:
                    tr_=ext-d*e.get("trail_atr",3.0)*e["atr"]; st_=max(st_,e["entry"],tr_) if d>0 else min(st_,e["entry"],tr_)
            e["trail_stop"]=st_
            if res is None and len(after)>=PRM["trail_max_bars"]:
                res=(round(float(d*(after.c.iloc[PRM["trail_max_bars"]-1]-e["entry"])/risk),2),"time exit")
        else:
          for _,b in after.iterrows():
            if (d>0 and b.l<=e["stop"]) or (d<0 and b.h>=e["stop"]): res=(-1.0,"stop"); break
            if (d>0 and b.h>=e["tp"][0]) or (d<0 and b.l<=e["tp"][0]): tp1=True
            if (d>0 and b.h>=e["tp"][1]) or (d<0 and b.l<=e["tp"][1]): res=(2.0,"TP2"); break
        if res is None and len(after)>=PRM["journal_expiry_bars"]:
            res=(round(float(d*(after.c.iloc[-1]-e["entry"])/risk),2),"expired")
        if res: e.update({"status":"closed","R":res[0],"exit":res[1],"tp1_hit":tp1,"closed_at":datetime.now(TZ).isoformat(timespec="minutes")})
        else: e["tp1_hit"]=tp1; e["bars_open"]=int(len(after))
    # 2) log new signals
    openkeys={(e["sym"],e["tf"]) for e in J if e.get("status")=="open"}
    # NOTE: a market entry is treated as filled at the signal bar's close; a limit entry must be touched first.
    added=0
    for (key,tf),c in curr.items():
        if tf not in PRM["journal_tfs"] or c["sig"]=="WAIT" or (key,tf) in openkeys: continue
        pl=c["plan"]; side="long" if c["sig"]=="BUY" else "short"
        if pl["side"]!=side: continue
        J.append({"id":f"{key}-{tf}-{int(c['bar_t'])}","logged":datetime.now(TZ).isoformat(timespec="minutes"),"bar_t":int(c["bar_t"]),"bar_time":c["lt"],"sym":key,"tf":tf,"side":side,"score":c["score"],
          "entry":pl["entry"],"entry_type":"market" if pl["zoneWhy"].startswith("market") else "limit ("+pl["zoneWhy"]+")","filled":pl["zoneWhy"].startswith("market"),"stop":pl["stop"],"tp":pl["tp"],"inv":pl["inv"],"votes":c["votes"],"weights":c["weights"],"strategies":c["active_strats"],"status":"open",
          "exit_mode":PRM["exit_mode"],"trail_atr":PRM["trail_atr"],"atr":c["atr"],"filters":c.get("flt"),"edge":f"{key}|{tf}" in PRM["tested_edge"]})
        added+=1
    json.dump(J,open(JPATH,"w"),indent=1)
    return J,added

def live_stats(J,names):
    closed=[e for e in J if e.get("status")=="closed"]
    by_agent={}; by_sym={}; by_strat={}
    for e in closed:
        d=1 if e["side"]=="long" else -1
        for k,v in enumerate(e["votes"]):
            if v!=0: by_agent.setdefault(names[k],[]).append(e["R"] if v==d else -e["R"])  # agent's own direction outcome (approx.)
        by_sym.setdefault(e["sym"],[]).append(e["R"])
        for s in e.get("strategies",[]): by_strat.setdefault(s,[]).append(e["R"])
    return {"agent":{k:stats(v) for k,v in by_agent.items()},"sym":{k:stats(v) for k,v in by_sym.items()},"strat":{k:stats(v) for k,v in by_strat.items()},
            "all":stats([e["R"] for e in closed]),"open":sum(1 for e in J if e.get("status")=="open"),"total":len(J)}

def main3():
    global RATES
    try:
        import multiprocessing as mp
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass
    news,nstatus,raw,srcstat=pickle.load(open(RAW_PKL,"rb"))
    RATES=edge.policy_rates(os.path.join(STATE,"rates.json"))
    CAL=edge.load_usd_events(os.path.join(STATE,"calendar.json"))
    jobs=[(k,tf) for (k,tf) in raw.keys()]
    REFIT=refit_due(jobs); t0=time.time()
    MISS=[] if REFIT else missing_combos(jobs)
    if REFIT or MISS:
        with Pool(8) as P: R=P.map(adapt_job,jobs if REFIT else MISS)
    else: R=[]
    print("refit" if REFIT else ("weights frozen; fitted missing "+",".join(f"{k}|{t}" for k,t in MISS) if MISS else "weights frozen (reusing state/weights.json)"),"adapt secs",round(time.time()-t0))
    LATCH=json.load(open(LPATH)) if (PRM.get("latch") and os.path.exists(LPATH)) else {}; LATCH_OUT={}; BC=W.get("by_combo",{})
    J0=json.load(open(JPATH)) if os.path.exists(JPATH) else []
    out=[]; names=None; curr={}; WOUT={}; oos={}
    HTF={"1m":"15m","5m":"1h","15m":"4h","1h":"4h","4h":"1d","1d":None}
    recs={(k,tf):r for k,tf,r in R}
    for key,label,src,code,mkt in SYMBOLS:
        nf=news_for(key,news)
        sd={"key":key,"label":label,"mkt":mkt,"src":goldfeed.feed_label(key,src,code),
            "news":[[i["src"],i["title"],i["link"],i["t"],datetime.fromtimestamp(i["ts"],TZ).strftime("%d %b %H:%M") if i["ts"] else ""] for i in nf["items"]],"tf":{}}
        for tf in TFS:
            df=raw.get((key,tf))
            if df is None: sd["tf"][tf]={"ok":0,"err":next(s[4] for s in srcstat if s[0]==key and s[1]==tf)}; continue
            htf={}
            for hh in [x for x in ["5m","15m","1h","4h","1d"] if x!=tf]:
                d2=raw.get((key,hh))
                if d2 is not None: htf[hh]=1 if ema(d2.c,20).iloc[-1]>ema(d2.c,50).iloc[-1] else -1
            p=prec(df.c.iloc[-1]); last=df.c.iloc[-1]
            agents,_,_,_,ind=analyze(df,p,htf,nf); a=atr(df).iloc[-1]
            tr,ev,choch,sh,sl=structure(df)
            adv,cat2,lv=advanced(df,p,raw.get((key,"1d")),last,a,tr,ev,choch,sh,sl)
            allag=agents+adv; cats={**CAT,**cat2}
            if names is None: names=[g["agent"] for g in allag]; catl=[cats[x] for x in names]
            rg,rk,adxv,er=regime(df)
            rec=recs.get((key,tf)) or []
            # Weekly tune-up (7 Nov 2026): a refit that gets too few samples (e.g. spot gold 1d has ~209 bars < window 300) keeps the stored
            # weights for that combo instead of dropping to the flat low-sample weight.
            frozen=(not REFIT or len(rec)<40) and f"{key}|{tf}" in BC and len(BC[f"{key}|{tf}"].get("weights",{}))==len(names)
            st_all=[BC[f"{key}|{tf}"]["stats"].get(nm_,{"n":0}) for nm_ in names] if frozen else fit(rec,len(names))
            # walk-forward OOS: fit on first part, evaluate consensus on rest
            cut=int(len(rec)*PRM["fit_frac"]); oo=BC[f"{key}|{tf}"].get("oos") if frozen else None
            if len(rec)>=40 and not frozen:
                wfit=[weight_from(s) for s in fit(rec[:cut],len(names))]; wfit[NEWS_IDX]=0
                eq=[1.0]*len(names); eq[NEWS_IDX]=0
                ra=[];re_=[]
                for r in rec[cut:]:
                    for ww,lst in ((wfit,ra),(eq,re_)):
                        s_=consensus(r[1],r[2],r[3],ww,catl,names)
                        if s_>=PRM["buy_th"]: lst.append(r[4])
                        elif s_<=PRM["sell_th"]: lst.append(r[5])
                oo={"adaptive":stats(ra),"equal":stats(re_),"test_samples":len(rec)-cut}
            # live blend
            wts=[]; ls=live_stats(J0,names)["agent"]
            if frozen: wts=[float(BC[f"{key}|{tf}"]["weights"][nm_]) for nm_ in names]; wts[NEWS_IDX]=PRM["news_weight"]
            for k,s in (enumerate(st_all) if not frozen else []):
                w_=weight_from(s)
                if k==NEWS_IDX: w_=PRM["news_weight"]
                l_=ls.get(names[k])
                if l_ and l_["n"]>=PRM["live_min"]: w_=round((1-PRM["live_blend"])*w_+PRM["live_blend"]*weight_from(l_),2)
                wts.append(w_)
            v=[VC[g["vote"]] for g in allag]; cf=[g["conf"] for g in allag]
            score=consensus(v,cf,rk,wts,catl,names)
            sig="BUY" if score>=PRM["buy_th"] else "SELL" if score<=PRM["sell_th"] else "WAIT"
            # quality filters (walk-forward tested): ATR percentile 20-95%, 60-bar trend and higher-TF EMA bias must agree with the signal
            hdf=raw.get((key,HTF[tf])) if HTF[tf] else None; fv,ftd,fhb=edge.filters(df,hdf); apct=edge.atr_pct(df).iloc[-1]
            dsg=1 if sig=="BUY" else -1 if sig=="SELL" else 0
            cdir,cdiff=edge.carry_dir(key,RATES) if PRM["carry_filter"] else (0,None)
            agree=bool(dsg!=0 and fv[-1] and ftd[-1]==dsg and (fhb[-1]==dsg if hdf is not None else True) and (cdir==0 or cdir==dsg))
            flt={"carry":cdiff,"carryDir":cdir,"vol":bool(fv[-1]),"atrPct":None if apct!=apct else round(float(apct)*100),"trend":int(ftd[-1]),"htf":int(fhb[-1]) if hdf is not None else None,"agree":agree,"sigRaw":sig}
            if PRM["filter_mode"] and sig!="WAIT" and not agree: sig="WAIT"
            # Pre-breakout protections. Display fields are always stored. Only filters that held up out of sample change the alert signal.
            ltf_tf=edge.LTF.get(tf); ltf_df=raw.get((key,ltf_tf)) if ltf_tf else None
            gate=edge.protect_last(df,tf,key,ltf_df,ltf_tf,CAL,int(df.t.iloc[-1])+int(HSTEP[tf]))
            sig,why=edge.apply_protect(sig,gate)
            _lez=edge.lez_cached(df,key,tf)
            _dlez=edge.lez_recent_dir(_lez,len(df)-1)
            sig,why2=edge.apply_lez(sig,_dlez,key,tf)
            if why2: why=why2
            # Round 4: Donchian-20 breakout confirmation (gold 1h alerts, BTC 1h display) and London/NY session (BTC 1h).
            _brkL,_brkS,_,_,_=edge.donchian_signals(df,hdf)
            _brk=edge.recent_brk(_brkL,_brkS,len(df)-1); _sess=edge.session_ok(df.t.iloc[-1],HSTEP[tf])
            sig,why3=edge.apply_round4(sig,key,tf,_brk,_sess)
            if why3: why=why3
            # Elder daily tide (10 Nov 2026): gold 1h only. Same closed-bar htf_bias the engine already uses; 0 when 1D is missing.
            _db=edge.htf_bias(df,raw.get((key,"1d")))[-1]
            sig,why4=edge.apply_daily(sig,key,tf,_db)
            if why4: why=why4
            # Weekly tune-up 14 Nov 2026: gold 4h volatility-expansion gate (ATR5 >= ATR20 on the closed bar).
            _vx=edge.volexp_ok(df)
            sig,why5=edge.apply_volexp(sig,key,tf,_vx)
            if why5: why=why5
            flt["range"]=gate["range"]; flt["sweep"]=gate["sweep"]; flt["news"]=gate["news"]; flt["why"]=why
            # latch: a closed bar keeps the signal it got on its first scan (no flip-flop between scans of the same bar)
            lk=f"{key}|{tf}"; bt_=int(df.t.iloc[-1]); prevl=LATCH.get(lk)
            if PRM.get("latch") and prevl and prevl.get("bar_t")==bt_:
                sig,score,flt=prevl["sig"],prevl["score"],prevl.get("flt",flt); flt=dict(flt or {}); flt["latched"]=True
            LATCH_OUT[lk]={"bar_t":bt_,"sig":sig,"score":score,"flt":{k_:v_ for k_,v_ in flt.items() if k_!="latched"}}
            flt=dict(flt or {}); flt["ind"]=edge.indicator_panel(df,hdf,tf,key,HSTEP[tf])  # round 4 indicator panel (display, every scan)
            flt["ind"]["d1"]=int(_db); flt["ind"]["dgate"]=f"{key}|{tf}" in edge.DAILY_GATE; flt["ind"]["vx"]=bool(_vx); flt["ind"]["vxgate"]=f"{key}|{tf}" in edge.VOLEXP_GATE
            flt["pat"]=patterns.panel(df) if key in ("XAUUSD","BTCUSD") and tf in ("1h","4h","1d") else []  # pattern agents, display only
            if oo is not None and not frozen:  # same OOS samples, adaptive consensus but only when the quality filters agree
                rf=[]
                for r in rec[cut:]:
                    s_=consensus(r[1],r[2],r[3],wfit,catl,names); dd_=1 if s_>=PRM["buy_th"] else -1 if s_<=PRM["sell_th"] else 0; i_=r[0]
                    if dd_ and fv[i_] and ftd[i_]==dd_ and (hdf is None or fhb[i_]==dd_): rf.append(r[4] if dd_>0 else r[5])
                oo["filtered"]=stats(rf)
            WOUT[f"{key}|{tf}"]=BC[f"{key}|{tf}"] if frozen else {"weights":dict(zip(names,wts)),"stats":dict(zip(names,st_all)),"oos":oo,"samples":len(rec)}
            L,S=strat_signals(df,tf,raw.get((key,HTF[tf])) if HTF[tf] else None,key)
            st=[]; act=[]
            for k in range(len(STRATS)):
                if STRATS[k]==edge.LEZ_NAME: bt=edge.lez_desk_bt(df,key,tf)
                elif STRATS[k]==edge.TOM_NAME: bt=backtest(df,L[k],S[k],rr=99,maxbars=3)
                else: bt=backtest(df,L[k],S[k])
                lL=np.where(L[k][-4:])[0]; lS=np.where(S[k][-4:])[0]
                vv=1 if len(lL) and (not len(lS) or lL[-1]>lS[-1]) else (-1 if len(lS) else 0); ago=None
                if vv: ago=3-int((lL if vv>0 else lS)[-1]); act.append(STRATS[k])
                if k==3 and tf not in ("5m","15m","1h"): bt={"n":0,"na":1}
                if k==5 and HTF[tf] is None: bt={"n":0,"na":1}
                if STRATS[k]=="Supertrend flip + MTF filter" and HTF[tf] is None: bt={"n":0,"na":1}
                if STRATS[k]==edge.TOM_NAME and tf!="1d": bt={"n":0,"na":1}
                st.append([vv,ago,bt])
            pl=plan(df,p,score,a,last,tr,sh,sl,lv)
            if PRM.get("plan_mode")=="tested":
                _sd=1 if pl["side"]=="long" else -1; _dist=PRM.get("plan_stop_atr",1.5)*a
                pl={**pl,"alt":{"entry":R_(pl["entry"],p),"stop":R_(pl["stop"],p),"why":pl["zoneWhy"],"basis":pl["stopBasis"]},"entry":last,"zone":[last,last],
                    "zoneWhy":"market (tested plan: enter at the close, BE at +1R, 4×ATR trail)","stop":last-_sd*_dist,
                    "stopBasis":f"{PRM.get('plan_stop_atr',1.5)}×ATR (walk-forward tested)","tp":[last+_sd*k_*_dist for k_ in (1,2,3)],"dist":_dist}
            if edge.LEZ_ALERTS and f"{key}|{tf}" in edge.LEZ_EDGE and _dlez!=0 and ((sig=="BUY" and _dlez>0) or (sig=="SELL" and _dlez<0)):
                _lp=edge.lez_trade_plan(_lez,key)
                if _lp is not None: pl=_lp
            N=48; tl=df.tail(N); age=int((time.time()*1000-df.t.iloc[-1])/60000)
            plr={k:([R_(x,p) for x in vv_] if isinstance(vv_,list) and k!="rr" else (R_(vv_,p) if isinstance(vv_,(float,np.floating)) else vv_)) for k,vv_ in pl.items()}
            lt=datetime.fromtimestamp(df.t.iloc[-1]/1000,TZ).strftime("%a %d %b %H:%M")
            curr[(key,tf)]={"sig":sig,"score":score,"plan":plr,"bar_t":df.t.iloc[-1],"lt":lt,"votes":v,"weights":wts,"active_strats":act,"flt":flt,"atr":float(a)}
            sd["tf"][tf]={"ok":1,"p":p,"last":R_(last,p),"chg":round(float((last/df.c.iloc[-2]-1)*100),3),"lt":lt,"age":age,"nb":len(df),
              "score":score,"sig":sig,"reg":[rg,rk,round(float(adxv)),round(float(er),2)],"ag":[[v[k],cf[k],allag[k]["reason"],wts[k],st_all[k]] for k in range(len(v))],"st":st,"oos":oo,
              "plan":plr,"flt":flt,"edge":f"{key}|{tf}" in PRM["tested_edge"],"atr":R_(a,p),"lv":{"S":R_(ind["S1"],p),"R":R_(ind["R1"],p),"poc":R_(lv["poc"],p),"vah":R_(lv["vah"],p),"val":R_(lv["val"],p),"avwap":R_(lv["avwap"],p),
                    "f50":R_(lv["fib"][0.5],p),"f618":R_(lv["fib"][0.618],p),"e1272":R_(lv["ext"][1.272],p),"e1618":R_(lv["ext"][1.618],p),"P":R_(lv["piv"].get("P"),p)},
              "c":{"t0":"","t1":datetime.fromtimestamp(tl.t.iloc[-1]/1000,TZ).strftime("%d %b %H:%M"),
                   "o":[R_(x,p) for x in tl.o],"h":[R_(x,p) for x in tl.h],"l":[R_(x,p) for x in tl.l],"c":[R_(x,p) for x in tl.c],"e20":[R_(x,p) for x in ind["e20"].tail(N)],"e50":[R_(x,p) for x in ind["e50"].tail(N)]}}
        out.append(sd)
    nbase=goldfeed.rebase_open_journal(JPATH)
    if nbase: print("gold journal on spot scale", nbase)
    J,added=run_journal(raw,curr,names,STRATS)
    LS=live_stats(J,names)
    json.dump({"updated":datetime.now(TZ).isoformat(timespec="minutes"),"params":PRM,"agents":names,"categories":dict(zip(names,catl)),
               "note":"Per symbol|timeframe agent weights from walk-forward barrier backtest (TP tp_atr×ATR vs SL sl_atr×ATR within horizon_bars), blended with live journal results once live_min closed signals exist. Weekly review may edit 'params' and 'overrides' (agent name -> fixed weight).",
               "overrides":W.get("overrides",{}),"by_combo":keep_stored(WOUT,BC),"live":LS,"rates":RATES,
               "refit_at":datetime.now(TZ).isoformat(timespec="minutes") if REFIT else W.get("refit_at")},open(WPATH,"w"),indent=1)
    if PRM.get("latch"): json.dump(LATCH_OUT,open(LPATH,"w"),separators=(",",":"))
    uj,uc=raw.get(("USDJPY","1h")),raw.get(("USDCHF","1h"))
    rates={"USDJPY":None if uj is None else round(float(uj.c.iloc[-1]),5),"USDCHF":None if uc is None else round(float(uc.c.iloc[-1]),5)}
    res={"gen":datetime.now(TZ).strftime("%a %d %b %Y, %H:%M"),"names":[[n_,c_] for n_,c_ in zip(names,catl)],"strats":STRATS,"rules":RULES,"rates":{k:(None if v is None else round(float(v),5)) for k,v in rates.items()},
         "sym":out,"newsStatus":nstatus,"priceStatus":srcstat,"newsCount":len(news),"journal":J[-40:],"jadded":added,"live":LS,"gold":goldfeed.public_meta(),
         "prm":{**PRM,"range_alerts":edge.RANGE_ALERTS,"sweep_alerts":edge.SWEEP_ALERTS,"news_alerts":edge.NEWS_ALERTS}}
    s=json.dumps(res,separators=(",",":"),default=lambda o: float(o) if isinstance(o,(np.floating,np.integer)) else str(o))
    open(SNAP3,"w").write(s); print("ok",len(s),"journal added",added,"total",len(J))
if __name__=="__main__": main3()
