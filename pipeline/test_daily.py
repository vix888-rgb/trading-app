"""Daily tide gate (10 Nov 2026): XAUUSD 1h BUY/SELL needs the closed 1D EMA20 vs EMA50 bias."""
import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import edge
class Daily(unittest.TestCase):
    def test_gate(self):
        self.assertEqual(edge.apply_daily("BUY","XAUUSD","1h",1),("BUY",""))
        self.assertEqual(edge.apply_daily("BUY","XAUUSD","1h",-1),("WAIT","daily"))
        self.assertEqual(edge.apply_daily("BUY","XAUUSD","1h",0),("WAIT","daily"))
        self.assertEqual(edge.apply_daily("SELL","XAUUSD","1h",-1),("SELL",""))
        self.assertEqual(edge.apply_daily("SELL","XAUUSD","1h",1),("WAIT","daily"))
        self.assertEqual(edge.apply_daily("SELL","XAUUSD","1h",0),("WAIT","daily"))
        self.assertEqual(edge.apply_daily("BUY","BTCUSD","1h",-1),("BUY",""))       # BTC 1h not gated
        self.assertEqual(edge.apply_daily("SELL","XAUUSD","4h",1),("SELL",""))      # 4h not gated
        self.assertEqual(edge.apply_daily("WAIT","XAUUSD","1h",0),("WAIT",""))
        self.assertEqual(edge.DAILY_GATE,("XAUUSD|1h",))
if __name__=="__main__": unittest.main()
