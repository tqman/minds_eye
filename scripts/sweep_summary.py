"""One table for the layer sweep, dose curve and random controls: the Avenue 4
tint metric (mean per-patch |RGB shift| vs the SAME essay's unsteered baseline
tile, floor 71.9 = essay-vs-essay texture) plus signed RGB, CLIP and keyword
hits from each run's results.json. CPU only; reads saved tiles.

usage: sweep_summary.py runs/run2_L6_noeos runs/run2_L18_noeos ...
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "runs" / "run2_L12_noeos"
G, P = 24, 16


def tiles(run, label):
    d = run / "tiles" / f"{label}_argmax"
    ps = sorted(d.glob("*.png"))
    if len(ps) != 8:
        return None
    return np.stack([np.asarray(Image.open(p).convert("RGB")).astype(np.float32) for p in ps])


def patch_rgb(img):
    return img.reshape(G, P, G, P, 3).mean(axis=(1, 3))


def main(runs):
    base_p = np.stack([patch_rgb(b) for b in tiles(BASE, "baseline")])
    print("| run | cond | mean shift | signed RGB (R,G,B) | mean RGB | clip ggb/coffee/circle | kw ggb/coffee/circle | words |")
    print("|---|---|---:|---|---|---|---|---:|")
    for r in runs:
        run = ROOT / r if not Path(r).is_absolute() else Path(r)
        res = json.loads((run / "results.json").read_text()) if (run / "results.json").exists() else {"rows": []}
        rows = {x["label"]: x for x in res["rows"] if x["decode"] == "argmax"}
        labels = sorted({p.name[:-len("_argmax")] for p in (run / "tiles").glob("*_argmax")} - {"baseline", "control_drawstates"})
        for label in labels:
            t = tiles(run, label)
            if t is None:
                continue
            st = np.stack([patch_rgb(x) for x in t])
            diff = st - base_p
            mag = np.linalg.norm(diff, axis=-1).mean()
            signed = diff.mean(axis=(0, 1, 2))
            mrgb = st.mean(axis=(0, 1, 2))
            x = rows.get(label, {})
            clip = "/".join(f"{x.get(f'clip_{c}', float('nan')):.2f}" for c in ("ggb", "coffee", "circle"))
            kw = "/".join(f"{x.get(f'text_kw_{c}', float('nan')):.1f}" for c in ("ggb", "coffee", "circle"))
            print(f"| {run.name} | {label} | {mag:.1f} | {signed[0]:+.0f} {signed[1]:+.0f} {signed[2]:+.0f} | "
                  f"{mrgb[0]:.0f} {mrgb[1]:.0f} {mrgb[2]:.0f} | {clip} | {kw} | {x.get('mean_words', '-')} |")


if __name__ == "__main__":
    main(sys.argv[1:])
