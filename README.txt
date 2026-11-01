Trading Signal Desk - installable web app (PWA)
Files: index.html, manifest.json, sw.js, icons/, data.json (snapshot built by /workspace/trading/run.sh + compact_pwa.py)
Serve this folder over HTTPS (or http://localhost for testing). Install:
- iPhone (Safari): open the URL > Share > Add to Home Screen.
- PC (Chrome/Edge): open the URL > install icon in the address bar (or menu > Install app).
To refresh data: run /workspace/trading/run.sh then python3 /workspace/trading/compact_pwa.py and re-upload data.json.
