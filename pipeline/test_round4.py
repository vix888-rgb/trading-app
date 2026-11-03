"""Round-4 (3 Nov 2026) gates: Donchian-20 breakout confirmation, session gate, causality of the new series."""
import os, sys, unittest
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import edge
def frame(n=400, seed=3):
    r=np.random.default_rng(seed); c=3900+np.cumsum(r.normal(0,4,n)); o=np.r_[c[0],c[:-1]]
    h=np.maximum(o,c)+r.uniform(0,3,n); l=np.minimum(o,c)-r.uniform(0,3,n); t=1_700_000_000_000+np.arange(n)*3_600_000
    return pd.DataFrame({"t":t,"o":o,"h":h,"l":l,"c":c,"v":np.ones(n)})
class Round4(unittest.TestCase):
    def test_gate(self):
        self.assertEqual(edge.apply_round4("SELL","XAUUSD","1h",(False,False),True),("WAIT","breakout"))
        self.assertEqual(edge.apply_round4("SELL","XAUUSD","1h",(False,True),True),("SELL",""))
        self.assertEqual(edge.apply_round4("BUY","XAUUSD","1h",(False,True),True),("WAIT","breakout"))
        self.assertEqual(edge.apply_round4("SELL","XAUUSD","4h",(False,False),True),("SELL",""))   # 4h/1D not gated
        self.assertEqual(edge.apply_round4("BUY","BTCUSD","1h",(True,False),False),("WAIT","session"))
        self.assertEqual(edge.apply_round4("WAIT","XAUUSD","1h",(False,False),True),("WAIT",""))
    def test_window(self):
        L=np.zeros(30,bool); S=np.zeros(30,bool); S[10]=True
        self.assertEqual(edge.recent_brk(L,S,19),(False,True)); self.assertEqual(edge.recent_brk(L,S,20),(False,False))
    def test_causal(self):
        df=frame(); a=edge.donchian_signals(df.iloc[:300].reset_index(drop=True),None); b=edge.donchian_signals(df,None)
        for k in range(2): self.assertTrue((a[k]==b[k][:300]).all())
        qa=edge.squeeze_signals(df.iloc[:300].reset_index(drop=True)); qb=edge.squeeze_signals(df)
        for k in range(2): self.assertTrue((qa[k]==qb[k][:300]).all())
    def test_session(self):
        t=int(pd.Timestamp("2026-11-03 07:00",tz="Europe/London").value//1_000_000)-3_600_000  # bar 06:00-07:00 London closes 07:00
        self.assertTrue(edge.session_ok(t,3_600_000)); self.assertFalse(edge.session_ok(t-3_600_000,3_600_000))
if __name__=="__main__": unittest.main()
