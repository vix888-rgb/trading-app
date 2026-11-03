import json,os,sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import RESEARCH, SNAP3, DATA_JSON
def _load(name):
    p=os.path.join(RESEARCH, name)
    return json.load(open(p)) if os.path.exists(p) and os.path.getsize(p) else None
WF=_load("wf_summary.json"); WF2=_load("wf2_summary.json")
d=json.load(open(SNAP3))
PRI=["XAUUSD","BTCUSD","EURUSD","USDJPY","USDCHF","EURCHF","ETHUSD","SOLUSD","XRPUSD","GBPUSD"]; DTF=["1m","5m","15m","1h","4h","1d"]; NB=48
names=[n[0] for n in d["names"]]; det={}; scan=[]; meta={}
agg_ag={n:[] for n in names}; agg_st={s:[] for s in d["strats"]}; agg_sym={}
def comb(lst):
    n=sum(x[0] for x in lst)
    if n==0: return [0,None,None]
    return [n,round(sum(x[0]*x[1] for x in lst)/n),round(sum(x[0]*x[2] for x in lst)/n,3)]
for s in d["sym"]:
    meta[s["key"]]=[s["label"],s["mkt"],s["src"]]
    for tf,t in s["tf"].items():
        if not t["ok"]: scan.append([s["key"],tf,None,t["err"]]); continue
        for k,a in enumerate(t["ag"]):
            if a[4].get("n"): agg_ag[names[k]].append((a[4]["n"],a[4]["hit"],a[4]["avgR"]))
        for k,x in enumerate(t["st"]):
            b=x[2]
            if b.get("n"): agg_st[d["strats"][k]].append((b["n"],b["win"],b["avgR"]))
        o=t.get("oos")
        if o and o["adaptive"].get("n"): agg_sym.setdefault(s["key"],{"a":[],"e":[]})["a"].append((o["adaptive"]["n"],o["adaptive"]["hit"],o["adaptive"]["avgR"]))
        if o and o["equal"].get("n"): agg_sym.setdefault(s["key"],{"a":[],"e":[]})["e"].append((o["equal"]["n"],o["equal"]["hit"],o["equal"]["avgR"]))
        bull=sum(1 for a in t["ag"] if a[0]>0); bear=sum(1 for a in t["ag"] if a[0]<0)
        act=[(i if x[0]>0 else -i-1) for i,x in enumerate(t["st"]) if x[0]!=0]
        scan.append([s["key"],tf,t["score"],t["last"],t["chg"],bull,bear,t["reg"][0],act,t["lt"],t["p"],t["sig"],(t.get("flt") or {}).get("agree"),(t.get("flt") or {}).get("why") or ""])
        if s["key"] in PRI and tf in DTF:
            p=t["p"]; c=t["c"]; allv=[x for k in ("o","h","l","c","e20","e50") for x in c[k][-NB:] if x is not None]; base=min(allv); top=max(allv); sc=999/(top-base)
            enc=lambda arr:[round((x-base)*sc) if x is not None else None for x in arr[-NB:]]
            pl=t["plan"]; o=t.get("oos")
            det[s["key"]+"|"+tf]={"p":p,"last":t["last"],"chg":t["chg"],"lt":t["lt"],"age":t["age"],"nb":t["nb"],"score":t["score"],"reg":t["reg"],
              "ag":[[a[0],a[1],a[2],a[3],a[4].get("n",0),a[4].get("hit"),a[4].get("avgR")] for a in t["ag"]],
              "st":[[x[0],x[1],x[2].get("n",0),x[2].get("win"),x[2].get("avgR"),x[2].get("pf"),x[2].get("na",0),
                     (x[2].get("lez") or {}).get("entry"),(x[2].get("lez") or {}).get("stop"),(x[2].get("lez") or {}).get("tp"),
                     1 if (x[2].get("lez") or {}).get("active") else 0] for x in t["st"]],
              "oos":[o["adaptive"].get("n",0),o["adaptive"].get("hit"),o["adaptive"].get("avgR"),o["equal"].get("n",0),o["equal"].get("hit"),o["equal"].get("avgR"),o["test_samples"]] if o else None,
              "pl":[1 if pl["side"]=="long" else -1,pl["entry"],pl["zone"],pl["zoneWhy"],pl["stop"],pl["stopBasis"],pl["inv"],pl["tp"]],
              "atr":t["atr"],"lv":t["lv"],"flt":t.get("flt"),"plalt":pl.get("alt"),"edge":t.get("edge"),"oosf":(o or {}).get("filtered"),"ch":[base,top,c["t1"],enc(c["o"]),enc(c["h"]),enc(c["l"]),enc(c["c"]),enc(c["e20"]),enc(c["e50"])]}
news={s["key"]:[[n[0],n[1][:110],n[3],n[4]] for n in s["news"][:6]] for s in d["sym"] if s["key"] in PRI}
J=[[e["bar_time"],e["sym"],e["tf"],e["side"],e["entry"],e["stop"],e["tp"][1],e["status"] if e.get("filled",True) or e["status"]!="open" else "pending fill",e.get("R"),e.get("tp1_hit",False),e.get("exit"),e.get("exit_mode","tp2"),e.get("trail_stop")] for e in d["journal"]][-200:]
L=d["live"]
out={"gen":d["gen"],"names":d["names"],"strats":d["strats"],"rules":d["rules"],"rates":d["rates"],"meta":meta,"scan":scan,"det":det,"news":news,
     "ns":[[x["source"],x["ok"],x.get("count",0)] for x in d["newsStatus"]],"nn":d["newsCount"],
     "pa":[[n]+comb(v) for n,v in agg_ag.items()],"ps":[[n]+comb(v) for n,v in agg_st.items()],"py":[[k,comb(v["a"]),comb(v["e"])] for k,v in agg_sym.items()],
     "j":J,"jl":[L["total"],L["open"],L["all"].get("n",0),L["all"].get("hit"),L["all"].get("avgR")],"gold":d.get("gold"),
     "prm":{k:d["prm"][k] for k in ("horizon_bars","tp_atr","sl_atr","samples","shrink_k","weight_scale","weight_max","min_samples","fit_frac","live_min","live_blend")},
     "prm2":{k:d["prm"].get(k) for k in ("filter_mode","exit_mode","trail_atr","trail_max_bars","tested_edge_tfs","carry_filter","tested_edge","range_alerts","sweep_alerts","news_alerts")},"wf":WF,"wf2":WF2}
s=json.dumps(out,separators=(",",":"),ensure_ascii=False)
_parent=os.path.dirname(DATA_JSON)
if _parent: os.makedirs(_parent,exist_ok=True)
open(DATA_JSON,"w").write(s); print(len(s))
print({k:len(json.dumps(v,separators=(",",":"),ensure_ascii=False)) for k,v in out.items()})
