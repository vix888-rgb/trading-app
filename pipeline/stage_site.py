#!/usr/bin/env python3
"""Copy the Pages site to _site, leaving the pipeline and git metadata out."""
import os, shutil, sys
from pathlib import Path

SKIP = {"pipeline", ".git", ".github", "_site", "__pycache__"}

def stage(root):
    root = Path(root).resolve()
    site = root / "_site"
    if site.exists():
        shutil.rmtree(site)
    site.mkdir()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP]
        rel = Path(dirpath).resolve().relative_to(root)
        if rel.parts and rel.parts[0] in SKIP:
            continue
        target = site / rel
        target.mkdir(parents=True, exist_ok=True)
        for name in filenames:
            shutil.copy2(Path(dirpath) / name, target / name)
    missing = [name for name in ("index.html", "data.json", "sw.js", ".nojekyll") if not (site / name).exists()]
    if missing or (site / "pipeline").exists():
        raise SystemExit(f"staged site is incomplete: missing={missing} pipeline={(site / 'pipeline').exists()}")
    print(f"staged {site}")

if __name__ == "__main__":
    stage(sys.argv[1] if len(sys.argv) > 1 else ".")
