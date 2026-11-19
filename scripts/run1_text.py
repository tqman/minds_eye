"""Run 1 text arm: does the same injection that steers the image head also
steer the text head, at the same layer and the same doses?

Same direction, same layer, same ratio ladder as the image runs. Instead of
drawing, the model writes a short paragraph for each of 8 neutral prompts.
Judge: fraction of outputs containing any concept keyword, plus mean keyword
hits per 100 words. All text is saved so it can be read, because keyword
counts can be gamed just like CLIP (a collapsed "coffee coffee coffee" scores
high and says nothing about coherence).
"""

import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from minds_eye.janus_model import Janus  # noqa: E402
from minds_eye.concepts import CONCEPTS  # noqa: E402
from minds_eye import direction as D  # noqa: E402

CONCEPT = os.environ.get("ME_CONCEPT", "ggb")
LAYER = int(os.environ.get("ME_LAYER", "12"))
RATIOS = [float(x) for x in os.environ.get("ME_RATIOS", "0,0.25,0.4,0.6").split(",")]
PROMPT_ONLY = os.environ.get("ME_PROMPT_ONLY", "0") == "1"
MAX_TOKENS = int(os.environ.get("ME_MAX_TOKENS", "90"))
SEED = 0
OUT = ROOT / "runs" / f"text_{CONCEPT}_L{LAYER}{'_po' if PROMPT_ONLY else ''}"
OUT.mkdir(parents=True, exist_ok=True)

# Neutral writing prompts, parallel to FREE_DRAW_PROMPTS: nothing to do with
# coffee, bridges, or circles.
FREE_WRITE_PROMPTS = [
    "Describe a quiet residential street in two or three sentences.",
    "Describe the objects on a desk in two or three sentences.",
    "Describe a landscape of rolling hills in two or three sentences.",
    "Describe a person reading on a park bench in two or three sentences.",
    "Describe an empty living room in two or three sentences.",
    "Describe a bookshelf in two or three sentences.",
    "Describe a bicycle on a sidewalk in two or three sentences.",
    "Describe a harbor with boats in two or three sentences.",
]

KEYWORDS = {
    "coffee": r"\b(coffee|espresso|latte|cappuccino|mug|caf[eé]|brew|barista|roast)\w*",
    "ggb": r"\b(golden gate|bridge|suspension|san francisco|cable|tower|bay|fog)\w*",
    "circle": r"\b(circle|circular|round|ring|disc|disk|sphere|orb|wheel|loop)\w*",
}


def score(text: str, concept: str):
    hits = re.findall(KEYWORDS[concept], text, flags=re.I)
    words = max(1, len(text.split()))
    return len(hits), 100.0 * len(hits) / words, [h if isinstance(h, str) else h[0] for h in hits]


def main():
    j = Janus()
    pos, ctrl = CONCEPTS[CONCEPT]
    a_pos = j.last_token_acts(pos)
    a_ctrl = j.last_token_acts(ctrl)
    dirs = D.concept_directions(a_pos, a_ctrl)
    norms = D.layer_norms(np.concatenate([a_pos, a_ctrl]))
    vec = dirs[LAYER]
    vnorm = float(np.linalg.norm(vec))
    resid = float(norms[LAYER])
    rand = D.random_like(vec, seed=SEED)
    print(f"layer {LAYER}: dir_norm={vnorm:.1f} resid_norm={resid:.1f}", flush=True)

    rows, samples = [], {}
    for ratio in RATIOS:
        scale = ratio * resid / vnorm
        for cond, dvec in ((CONCEPT, vec), ("random", rand)):
            if ratio == 0.0 and cond == "random":
                continue
            label = "baseline" if ratio == 0.0 else f"{cond}_r{ratio}"
            outs = []
            for pi, prompt in enumerate(FREE_WRITE_PROMPTS):
                txt = j.write(prompt, max_new_tokens=MAX_TOKENS, seed=SEED + pi,
                              layer=LAYER, direction=dvec, scale=scale,
                              prompt_only=PROMPT_ONLY)
                n, per100, found = score(txt, CONCEPT)
                outs.append({"prompt": prompt, "text": txt, "hits": n,
                             "per100": round(per100, 2), "found": found})
            samples[label] = outs
            row = {"label": label, "cond": cond, "ratio": ratio, "scale": round(scale, 4),
                   "frac_any": round(float(np.mean([o["hits"] > 0 for o in outs])), 3),
                   "hits_per100": round(float(np.mean([o["per100"] for o in outs])), 2),
                   "mean_words": round(float(np.mean([len(o["text"].split()) for o in outs])), 1)}
            rows.append(row)
            print(row, flush=True)

    (OUT / "results.json").write_text(json.dumps(
        {"concept": CONCEPT, "layer": LAYER, "prompt_only": PROMPT_ONLY,
         "dir_norm": vnorm, "resid_norm": resid, "rows": rows}, indent=2))
    with open(OUT / "samples.md", "w") as f:
        f.write(f"# text steer: {CONCEPT} L{LAYER}{' prompt-only' if PROMPT_ONLY else ''}\n\n")
        for label, outs in samples.items():
            f.write(f"## {label}\n\n")
            for o in outs:
                f.write(f"**{o['prompt']}** (hits {o['hits']}: {', '.join(o['found'])})\n\n")
                f.write(o["text"].strip() + "\n\n")
    print("\n== summary ==")
    for r in rows:
        print(f"  {r['label']:>14}  frac_any={r['frac_any']:.3f}  "
              f"hits/100w={r['hits_per100']:.2f}  words={r['mean_words']}")


if __name__ == "__main__":
    t = time.time()
    main()
    print(f"\ntotal {time.time() - t:.0f}s")
