#!/bin/bash
# Refresh pipeline: fetch live data -> agents/strategies -> adaptive weights + journal -> compact page data.
# Paths default to this repo (data.json at the root, weights/journal in pipeline/state). Override with TRADING_* env vars.
set -euo pipefail
cd "$(dirname "$0")"
python3 engine2.py
python3 engine3.py
python3 compact3.py
python3 compact_pwa.py
# Display-only trendline break page (gold 15m and 5m). Never fails the scan.
python3 tlbreak.py || echo "tl break skipped (display-only module failed)"
