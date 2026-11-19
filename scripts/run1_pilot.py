"""Run 1 pilot: does a 'coffee' language-stream vector reach the image head?

Build the coffee direction, pick one mid layer, draw the neutral free-draw
prompts under three conditions at a few doses:
  - baseline (scale 0)
  - coffee steer
  - random direction at equal norm (control)
Score every image with the CLIP coffee judge, save contact sheets, print a table.

This is the pilot that tells us the sign and rough dose before any full sweep.
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from minds_eye.janus_model import Janus, save_grid  # noqa: E402
from minds_eye.concepts import CONCEPTS, FREE_DRAW_PROMPTS  # noqa: E402
from minds_eye import direction as D  # noqa: E402
from minds_eye.judge import ClipJudge, HoughJudge  # noqa: E402

import os

CONCEPT = os.environ.get("ME_CONCEPT", "ggb")
LAYER = int(os.environ.get("ME_LAYER", "12"))   # mid-depth of 30; dose sized below
# include a low rung: coffee collapsed into grain at 0.6+, so look for the
# regime where the object appears while the scene is still coherent.
RATIOS = [float(x) for x in os.environ.get("ME_RATIOS", "0,0.2,0.3,0.45,0.6").split(",")]  # 0 == baseline
N_PER_PROMPT = 2
SEED = 0
PROMPT_ONLY = os.environ.get("ME_PROMPT_ONLY", "0") == "1"
# Run 2 follow-up (2026-11-16 evening): draw under several random vectors so each
# random seed's DRAW collapse signature can be compared to its own Run 2 WRITE decode.
# random_like() depends only on the seed and the norm, so seed k here is the same
# direction as seed k in run2_decoder.py.
RANDOM_SEEDS = [int(x) for x in os.environ.get("ME_RANDOM_SEEDS", "0").split(",")]
SKIP_CONCEPT = os.environ.get("ME_SKIP_CONCEPT", "0") == "1"
TAG = os.environ.get("ME_TAG", "")
OUT = ROOT / "runs" / f"run1_{CONCEPT}_L{LAYER}{'_po' if PROMPT_ONLY else ''}{TAG}"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    j = Janus()
    pos, ctrl = CONCEPTS[CONCEPT]
    a_pos = j.last_token_acts(pos)
    a_ctrl = j.last_token_acts(ctrl)
    dirs = D.concept_directions(a_pos, a_ctrl)          # (L, d)
    norms = D.layer_norms(np.concatenate([a_pos, a_ctrl]))
    vec = dirs[LAYER]
    vnorm = float(np.linalg.norm(vec))
    resid = float(norms[LAYER])
    conds = [] if SKIP_CONCEPT else [(CONCEPT, vec)]
    for rs in RANDOM_SEEDS:
        conds.append(("random" if rs == 0 else f"random_s{rs}", D.random_like(vec, seed=rs)))
    # scale so that ratio * resid == scale * vnorm  ->  scale = ratio*resid/vnorm
    print(f"layer {LAYER}: dir_norm={vnorm:.1f} resid_norm={resid:.1f} "
          f"base_scale(r=1)={resid / vnorm:.3f}", flush=True)

    judge = ClipJudge()
    hough = HoughJudge() if CONCEPT == "circle" else None
    rows = []
    for ratio in RATIOS:
        scale = ratio * resid / vnorm
        for cond, dvec in conds:
            if ratio == 0.0 and cond != conds[0][0]:
                continue  # baseline is condition-independent
            label = "baseline" if ratio == 0.0 else f"{cond}_r{ratio}"
            imgs = []
            for pi, prompt in enumerate(FREE_DRAW_PROMPTS):
                im = j.draw(prompt, n=N_PER_PROMPT, seed=SEED + pi,
                            layer=LAYER, direction=dvec, scale=scale,
                            prompt_only=PROMPT_ONLY)
                imgs.append(im)
            imgs = np.concatenate(imgs, axis=0)
            probs = judge.concept_prob(imgs, CONCEPT)
            counts = [hough.count(im) for im in imgs] if hough else None
            labels = [f"{p:.2f}" + (f" h{c}" if counts else "")
                      for p, c in zip(probs, counts or [None] * len(probs))]
            save_grid(imgs, OUT / f"{label}.jpg", cols=N_PER_PROMPT * 2, labels=labels,
                      tiles_dir=OUT / "tiles" / label)
            row = {"label": label, "cond": cond, "ratio": ratio,
                   "scale": round(scale, 4), "mean_coffee": round(float(probs.mean()), 3),
                   "frac_over_0.5": round(float((probs > 0.5).mean()), 3)}
            if counts:
                row["hough_mean"] = round(float(np.mean(counts)), 2)
                row["hough_frac_ge1"] = round(float(np.mean([c >= 1 for c in counts])), 3)
            rows.append(row)
            print(rows[-1], flush=True)

    (OUT / "results.json").write_text(json.dumps(
        {"concept": CONCEPT, "layer": LAYER, "prompt_only": PROMPT_ONLY, "dir_norm": vnorm,
         "resid_norm": resid, "rows": rows}, indent=2))
    print("\n== summary ==")
    for r in rows:
        extra = f"  hough={r['hough_mean']:.2f} ge1={r['hough_frac_ge1']:.2f}" if 'hough_mean' in r else ""
        print(f"  {r['label']:>14}  mean={r['mean_coffee']:.3f}  "
              f"frac>0.5={r['frac_over_0.5']:.3f}{extra}")


if __name__ == "__main__":
    t = time.time()
    main()
    print(f"\ntotal {time.time() - t:.0f}s")
