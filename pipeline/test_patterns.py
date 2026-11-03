"""Pattern agents must not repaint: signals on a prefix of the data equal the signals on the full data for the same bars."""
import os, sys, unittest, pickle
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import patterns
def frame(n=1500, seed=7):
    r=np.random.default_rng(seed); c=3900+np.cumsum(r.normal(0,5,n)); o=np.r_[c[0],c[:-1]]
    h=np.maximum(o,c)+r.uniform(0,4,n); l=np.minimum(o,c)-r.uniform(0,4,n)
    return pd.DataFrame({"t":1_700_000_000_000+np.arange(n)*3_600_000,"o":o,"h":h,"l":l,"c":c,"v":np.ones(n)})
class NoRepaint(unittest.TestCase):
    def test_prefix(self):
        df=frame(); full=patterns.all_signals(df)
        for cut in (600,900,1200):
            pre=patterns.all_signals(df.iloc[:cut].reset_index(drop=True))
            for k in ("Wolfe","Harmonic","AB=CD","Wyckoff"):
                self.assertTrue((pre[0][k]==full[0][k][:cut]).all(),k); self.assertTrue((pre[1][k]==full[1][k][:cut]).all(),k)
    def test_some_patterns(self):
        L,S,_=patterns.all_signals(frame(4000,11)); self.assertGreater(sum(int(L[k].sum()+S[k].sum()) for k in L),0)
if __name__=="__main__": unittest.main()
