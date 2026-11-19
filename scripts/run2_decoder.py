"""Run 2: is there a picture in the hidden states when nobody asked for one?

Run 1 showed a concept vector added to the shared residual stream reaches the
image head when the model is DRAWING. Run 2 asks the mind's-eye question
proper: steer the same vector, make the model WRITE about something with no
picture in it (tax law), and decode the hidden states it produces while
writing through the model's own image path anyway.

Decoder: none trained. Janus's ``gen_head`` IS the model's native map from a
final-layer hidden state to a VQ image token (it is what ``draw`` calls at
every step), and the VQ decoder turns 576 tokens into a 384x384 image. So we
generate exactly 576 text tokens, treat their 576 hidden states as the 24x24
raster grid, and run gen_head + VQ decoder on them unchanged. Argmax and one
sampled decode per state grid.

Conditions, all at one layer:
  baseline (no steer); concept steer at two doses; random vector at equal norm.
  Two concepts (coffee, ggb). Prompts: 8 non-visual essay topics.
Positive control: hidden states captured while the model actually draws
(unsteered) go through the same decode; if that is not a picture, the decode
path is broken and nothing else means anything.

Judge: CLIP concept prob (both concepts scored on every image, so the cross
condition is visible), plus the text keyword score so we know whether the
TEXT carried the concept at that dose. And look at the sheets.

Pre-registered prediction (2026-11-05): brown smears on the text side,
i.e. no picture when nobody asked for one. Second prediction (2026-11-16, before
the first text condition finished): "I have no idea. I'm curious to see what we
get."

Avenue 3 (2026-11-16 20:05, ``ME_PROMPT_ONLY=1``): the vector is added only on
the prefill (prompt positions); every generated token is unsteered, so the
concept can only reach the writing states through attention to the prompt. In
Run 1 this regime is the one that COMPOSED real bridges while drawing. Doses
r1.0/r2.0 as in the Run 1 prompt-only draws. Prediction on record: the
text will drift toward the concept more than full-position did at equal ratio
(the text head reads the prompt), and the image-decode tint will be WEAKER or
absent, because the tint in Run 2 came from the per-position bias on every
generated state; any tint that does appear should look like Follow-up A's topic
palette (the essay is now about the concept), not Run 1's collapse signature.
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

from minds_eye.janus_model import Janus, save_grid, N_IMG_TOKENS  # noqa: E402
from minds_eye.concepts import CONCEPTS, FREE_DRAW_PROMPTS  # noqa: E402
from minds_eye import direction as D  # noqa: E402
from minds_eye.judge import ClipJudge  # noqa: E402

CONCEPTS_TO_RUN = os.environ.get("ME_CONCEPTS", "coffee,ggb").split(",")
LAYER = int(os.environ.get("ME_LAYER", "12"))
RATIOS = [float(x) for x in os.environ.get("ME_RATIOS", "0.6,1.0").split(",") if x]
RANDOM_RATIO = float(os.environ.get("ME_RANDOM_RATIO", "1.0"))
RANDOM_SEEDS = [int(x) for x in os.environ.get("ME_RANDOM_SEEDS", "0").split(",") if x]
SKIP_CONTROL = os.environ.get("ME_SKIP_CONTROL", "0") == "1"
SKIP_BASELINE = os.environ.get("ME_SKIP_BASELINE", "0") == "1"
TEMP = float(os.environ.get("ME_TEMP", "0.7"))
SUPPRESS_EOS = os.environ.get("ME_SUPPRESS_EOS", "0") == "1"
PROMPT_ONLY = os.environ.get("ME_PROMPT_ONLY", "0") == "1"
TAG = os.environ.get("ME_TAG", "")
SEED = 0
OUT = ROOT / "runs" / f"run2_L{LAYER}{TAG}"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "states").mkdir(exist_ok=True)

# Non-visual essay prompts. Long enough that 576 tokens is a natural length.
NONVISUAL_PROMPTS = [
    "Write a detailed essay explaining how income tax brackets work.",
    "Write a detailed essay on the principles of contract law.",
    "Write a detailed essay on how central banks set interest rates.",
    "Write a detailed essay explaining double-entry bookkeeping.",
    "Write a detailed essay on the rules of parliamentary procedure.",
    "Write a detailed essay on how insurance underwriting decides premiums.",
    "Write a detailed essay on the history of the metric system.",
    "Write a detailed essay on the ethics of informed consent in medicine.",
]

# Run 2 follow-up (2026-11-16: "if the ocean essay's states are blue with nothing
# injected, that's the model's mind's eye on its own"). Visual topics, no steering.
# Each has a strong, well-known palette so a decoded tint can be read against it by eye.
# Golden Gate and coffee are included so the unsteered WRITE decode can be compared to
# the steered Run 1/Run 2 signatures for the same concepts.
VISUAL_PROMPTS = [
    "Write a detailed essay describing the open ocean on a clear day.",
    "Write a detailed essay describing a forest fire at night.",
    "Write a detailed essay describing a wheat field in late summer.",
    "Write a detailed essay describing the Golden Gate Bridge in the fog.",
    "Write a detailed essay describing a cup of black coffee on a wooden table.",
    "Write a detailed essay describing a snow-covered mountain range.",
    "Write a detailed essay describing a lavender field in Provence.",
    "Write a detailed essay describing a sunset over the desert.",
]
if os.environ.get("ME_PROMPTS", "nonvisual") == "visual":
    NONVISUAL_PROMPTS = VISUAL_PROMPTS

KEYWORDS = {
    "coffee": r"\b(coffee|espresso|latte|cappuccino|mug|caf[eé]|brew|barista|roast)\w*",
    "ggb": r"\b(golden gate|bridge|suspension|san francisco|cable|tower|bay|fog)\w*",
    "circle": r"\b(circle|circular|round|ring|disc|disk|sphere|orb|wheel|loop)\w*",
}


def kw_hits(text, concept):
    return len(re.findall(KEYWORDS[concept], text, flags=re.I))


def main():
    j = Janus()
    judge = ClipJudge()
    score_concepts = ["coffee", "ggb", "circle"]

    # Directions for every concept we steer with.
    dirs, resid = {}, None
    for c in CONCEPTS_TO_RUN:
        pos, ctrl = CONCEPTS[c]
        a_pos, a_ctrl = j.last_token_acts(pos), j.last_token_acts(ctrl)
        dirs[c] = D.concept_directions(a_pos, a_ctrl)[LAYER]
        resid = float(D.layer_norms(np.concatenate([a_pos, a_ctrl]))[LAYER])
        print(f"{c}: dir_norm={np.linalg.norm(dirs[c]):.1f} resid_norm={resid:.1f}", flush=True)

    rows, samples = [], {}

    def judge_grid(label, imgs_argmax, imgs_sampled, texts_meta):
        for kind, imgs in (("argmax", imgs_argmax), ("sampled", imgs_sampled)):
            imgs = np.stack(imgs)
            scores = {c: judge.concept_prob(imgs, c) for c in score_concepts}
            labels = [" ".join(f"{scores[c][i]:.2f}{c[0]}" for c in score_concepts) for i in range(len(imgs))]
            save_grid(imgs, OUT / f"{label}_{kind}.jpg", cols=4, labels=labels,
                      tiles_dir=OUT / "tiles" / f"{label}_{kind}")
            row = {"label": label, "decode": kind,
                   **{f"clip_{c}": round(float(scores[c].mean()), 3) for c in score_concepts},
                   **{f"frac_{c}>0.5": round(float((scores[c] > 0.5).mean()), 3) for c in score_concepts},
                   "per_image": {c: [round(float(x), 3) for x in scores[c]] for c in score_concepts}}
            row.update(texts_meta)
            rows.append(row)
            print({k: v for k, v in row.items() if k != "per_image"}, flush=True)

    # ---- positive control: in-regime (draw) states through the same decode
    t0 = time.time()
    drawn, argm, samp = [], [], []
    for pi, prompt in enumerate(FREE_DRAW_PROMPTS if not SKIP_CONTROL else []):
        img, hid = j.draw(prompt, n=1, seed=SEED + pi, return_hidden=True)
        drawn.append(img[0])
        a, _ = j.states_to_image(hid)
        s, _ = j.states_to_image(hid, sample=True, seed=SEED + pi)
        argm.append(a)
        samp.append(s)
    if not SKIP_CONTROL:
        save_grid(np.stack(drawn), OUT / "control_drawn.jpg", cols=4,
                  tiles_dir=OUT / "tiles" / "control_drawn")
        judge_grid("control_drawstates", argm, samp, {"cond": "draw_states", "ratio": 0.0})
        print(f"positive control done {time.time() - t0:.0f}s", flush=True)

    # ---- text conditions
    conds = [] if SKIP_BASELINE else [("baseline", None, 0.0)]
    for c in CONCEPTS_TO_RUN:
        for r in RATIOS:
            conds.append((f"{c}_r{r}", dirs[c], r))
    for rs in RANDOM_SEEDS:
        rand = D.random_like(dirs[CONCEPTS_TO_RUN[0]], seed=rs)
        tag = "" if rs == 0 else f"_s{rs}"
        conds.append((f"random{tag}_r{RANDOM_RATIO}", rand, RANDOM_RATIO))

    for label, vec, ratio in conds:
        t0 = time.time()
        scale = 0.0 if vec is None else ratio * resid / float(np.linalg.norm(vec))
        argm, samp, texts = [], [], []
        for pi, prompt in enumerate(NONVISUAL_PROMPTS):
            txt, hid = j.write(prompt, max_new_tokens=N_IMG_TOKENS, seed=SEED + pi,
                               temperature=TEMP, layer=LAYER, direction=vec, scale=scale,
                               return_hidden=True, stop_on_eos=False,
                               suppress_eos=SUPPRESS_EOS, prompt_only=PROMPT_ONLY)
            assert hid.shape[0] == N_IMG_TOKENS
            np.savez_compressed(OUT / "states" / f"{label}_{pi}.npz", h=hid.astype(np.float16))
            a, _ = j.states_to_image(hid)
            s, _ = j.states_to_image(hid, sample=True, seed=SEED + pi)
            argm.append(a)
            samp.append(s)
            texts.append({"prompt": prompt, "text": txt, "eos_at": j.last_eos,
                          "words": len(txt.split()),
                          **{f"kw_{c}": kw_hits(txt, c) for c in score_concepts}})
        samples[label] = texts
        meta = {"cond": label.split("_r")[0], "ratio": ratio, "scale": round(scale, 4),
                "mean_words": round(float(np.mean([t["words"] for t in texts])), 1),
                "eos_mean": round(float(np.mean([t["eos_at"] for t in texts])), 1),
                **{f"text_kw_{c}": round(float(np.mean([t[f"kw_{c}"] for t in texts])), 2)
                   for c in score_concepts}}
        judge_grid(label, argm, samp, meta)
        print(f"{label} done {time.time() - t0:.0f}s", flush=True)

        (OUT / "results.json").write_text(json.dumps(
            {"layer": LAYER, "concepts": CONCEPTS_TO_RUN, "ratios": RATIOS, "temp": TEMP,
             "suppress_eos": SUPPRESS_EOS, "prompt_only": PROMPT_ONLY,
             "resid_norm": resid, "rows": rows}, indent=2))
        with open(OUT / "samples.md", "w") as f:
            f.write(f"# run2 text samples, L{LAYER}\n\n")
            for lab, outs in samples.items():
                f.write(f"## {lab}\n\n")
                for o in outs:
                    kw = " ".join(f"{c} {o[f'kw_{c}']}" for c in score_concepts)
                    f.write(f"**{o['prompt']}** (words {o['words']}, eos_at {o['eos_at']}, kw {kw})\n\n")
                    f.write(o["text"].strip() + "\n\n")

    print("\n== summary (argmax decode) ==")
    for r in rows:
        if r["decode"] == "argmax":
            clip = "  ".join(f"clip_{c}={r[f'clip_{c}']:.3f}" for c in score_concepts)
            kw = "  ".join(f"kw_{c}={r.get(f'text_kw_{c}', '-')}" for c in score_concepts)
            print(f"  {r['label']:>22}  {clip}  {kw}")


if __name__ == "__main__":
    t = time.time()
    main()
    print(f"\ntotal {time.time() - t:.0f}s")
