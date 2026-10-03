"""Paths shared by the explainer pipeline. Scripts run from anywhere; the repo root goes on sys.path."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXPLAINER = os.path.dirname(HERE)
REPO = os.path.dirname(EXPLAINER)
DATA = os.path.join(EXPLAINER, "data")          # measurements the page is built from (committed)
WORK = os.path.join(EXPLAINER, "work")          # cached trajectories (not committed)
CLIPS = os.path.join(EXPLAINER, "clips")
FIGS = os.path.join(EXPLAINER, "figs")
ATLAS_JSON = os.path.join(REPO, "search", "atlas.json")

if REPO not in sys.path:
    sys.path.insert(0, REPO)
for d in (DATA, WORK, CLIPS, FIGS):
    os.makedirs(d, exist_ok=True)
