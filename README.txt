Trading Signal Desk - installable web app (PWA)
Files: index.html, manifest.json, sw.js, icons/, data.json (snapshot built by pipeline/run.sh)
Serve this folder over HTTPS (or http://localhost for testing). Install:
- iPhone (Safari): open the URL > Share > Add to Home Screen.
- PC (Chrome/Edge): open the URL > install icon in the address bar (or menu > Install app).
To refresh data: run pipeline/run.sh (writes data.json at the repo root; weights and journal stay in pipeline/state). GitHub Actions scans every 10 minutes.
