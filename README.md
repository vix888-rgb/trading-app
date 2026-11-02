# trading-app

## Trading Signal Desk

Installable trading signal dashboard (PWA). It shows consensus signals from analysis agents and strategies for gold, BTC, and FX, plus a scanner, trade plan, and position-size helper.

Signals are not financial advice. They are statistical opinions from historical data, not recommendations or guarantees.

Open the GitHub Pages site, then install it from the browser: Share > Add to Home Screen on iPhone, or the install icon in Chrome or Edge on desktop.

A GitHub Actions scan rebuilds and publishes `data.json` about every 10 minutes. New BUY or SELL signals on the 1h, 4h, and 1D timeframes send a phone alert. The page reloads the snapshot every 5 minutes while it is open.
