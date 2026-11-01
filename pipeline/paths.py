"""Repo-relative locations for the signal pipeline.

Defaults assume this file lives in ``<repo>/pipeline``. Override any path with
an environment variable when the checkout or data file is elsewhere.
"""
import os

PIPE = os.environ.get("TRADING_PIPELINE") or os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("TRADING_ROOT") or os.path.abspath(os.path.join(PIPE, ".."))
STATE = os.environ.get("TRADING_STATE") or os.path.join(PIPE, "state")
DATA_JSON = os.environ.get("TRADING_DATA") or os.path.join(ROOT, "data.json")
RAW_PKL = os.environ.get("TRADING_RAW") or os.path.join("/tmp", "raw.pkl")
SNAP1 = os.environ.get("TRADING_SNAP1") or os.path.join(PIPE, "snapshot.json")
SNAP2 = os.environ.get("TRADING_SNAP2") or os.path.join(PIPE, "snap2.json")
SNAP3 = os.environ.get("TRADING_SNAP3") or os.path.join(PIPE, "snap3.json")
RESEARCH = os.environ.get("TRADING_RESEARCH") or os.path.join(PIPE, "research")
