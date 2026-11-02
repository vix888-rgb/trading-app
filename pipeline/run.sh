#!/bin/bash
# Refresh pipeline: fetch live data -> agents/strategies -> adaptive weights + journal -> compact page data.
# Paths default to this repo (data.json at the root, weights/journal in pipeline/state). Override with TRADING_* env vars.
set -euo pipefail
cd "$(dirname "$0")"
python3 engine2.py
python3 engine3.py
python3 compact3.py
python3 compact_pwa.py
