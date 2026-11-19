"""Run 3: the mirror of Run 2. Push states captured while DRAWING through the
LANGUAGE head and read the words.

Run 2 took states produced while the model was writing (tax law) and decoded
them through the image head: the concept's image-space features were there
even when the text had no concept word. Run 3 asks the opposite question: when
the model is drawing (a bridge, under the GGB vector, or a street with nothing
injected), what does the language head say the same states "mean"? If the
drawing states read out as bridge words, the dissociation runs both ways and the
two heads share the concept at the state level; if they read out as nothing,
the image head is seeing something the text head cannot name.

Readout per condition (8 free-draw prompts x 1 image, 576 states each):
  * top-1 token at each position through ``lm_head`` (argmax), decoded;
  * probability mass on each concept's keyword tokens, averaged over positions
    (more sensitive than argmax);
  * top-30 tokens by frequency, and the tokens most over-represented relative
    to the unsteered baseline;
  * the drawn images themselves (contact sheet + CLIP) so we know the steer
    landed in the picture.

Conditions at one layer (L12): baseline; ggb full-position r0.3/r0.45 and
prompt-only r1.0 (the dose that composed real bridges in Run 1); coffee
r0.3/r0.45; circle r0.45; random seeds 0 and 2 at r0.45.

Prediction on record before running (2026-11-16 19:20): the language
head reads image-regime states as junk (image-token-ish or boilerplate), and
the concept words do not rise above baseline except possibly under prompt-only
GGB, where the prompt itself was steered. Second prediction: not asked yet; fill in.
"""

import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from minds_eye.janus_model import Janus, save_grid  # noqa: E402
from minds_eye.concepts import CONCEPTS, FREE_DRAW_PROMPTS  # noqa: E402
from minds_eye import direction as D  # noqa: E402
from minds_eye.judge import ClipJudge  # noqa: E402

LAYER = int(os.environ.get("ME_LAYER", "12"))
TAG = os.environ.get("ME_TAG", "")
SEED = 0
OUT = ROOT / "runs" / f"run3_mirror_L{LAYER}{TAG}"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "states").mkdir(exist_ok=True)

KEYWORDS = {
    "coffee": ["coffee", "espresso", "latte", "mug", "cafe", "brew", "roast", "cup", "beans"],
    "ggb": ["bridge", "Golden", "Gate", "suspension", "cable", "cables", "tower", "towers",
            "San", "Francisco", "bay", "fog"],
    "circle": ["circle", "circles", "circular", "round", "ring", "disc", "sphere", "orb", "wheel", "loop"],
}
KW_RE = {
    "coffee": r"\b(coffee|espresso|latte|cappuccino|mug|caf[eé]|brew|barista|roast)\w*",
    "ggb": r"\b(golden gate|bridge|suspension|san francisco|cable|tower|bay|fog)\w*",
    "circle": r"\b(circle|circular|round|ring|disc|disk|sphere|orb|wheel|loop)\w*",
}


def keyword_token_ids(tok, words):
    """First sub-token of each keyword, with and without a leading space, both cases."""
    ids = set()
    for w in words:
        for form in (w, " " + w, w.lower(), " " + w.lower(), w.capitalize(), " " + w.capitalize()):
            enc = tok.encode(form, add_special_tokens=False)
            if enc:
                ids.add(enc[0])
    return sorted(ids)


def main():
    j = Janus()
    judge = ClipJudge()
    tok = j.tok
    kw_ids = {c: torch.tensor(keyword_token_ids(tok, ws), device=j.device) for c, ws in KEYWORDS.items()}
    print({c: len(v) for c, v in kw_ids.items()}, flush=True)

    dirs, resid = {}, None
    for c in ("coffee", "ggb", "circle"):
        pos, ctrl = CONCEPTS[c]
        a_pos, a_ctrl = j.last_token_acts(pos), j.last_token_acts(ctrl)
        dirs[c] = D.concept_directions(a_pos, a_ctrl)[LAYER]
        resid = float(D.layer_norms(np.concatenate([a_pos, a_ctrl]))[LAYER])
        print(f"{c}: dir_norm={np.linalg.norm(dirs[c]):.1f} resid_norm={resid:.1f}", flush=True)

    conds = [
        ("baseline", None, 0.0, False),
        ("ggb_r0.3", dirs["ggb"], 0.3, False),
        ("ggb_r0.45", dirs["ggb"], 0.45, False),
        ("ggb_po_r1.0", dirs["ggb"], 1.0, True),
        ("ggb_po_r2.0", dirs["ggb"], 2.0, True),
        ("coffee_r0.3", dirs["coffee"], 0.3, False),
        ("coffee_r0.45", dirs["coffee"], 0.45, False),
        ("circle_r0.45", dirs["circle"], 0.45, False),
        ("random_r0.45", D.random_like(dirs["ggb"], seed=0), 0.45, False),
        ("random_s2_r0.45", D.random_like(dirs["ggb"], seed=2), 0.45, False),
    ]
    extra = os.environ.get("ME_ONLY")
    if extra:
        keep = set(extra.split(","))
        conds = [c for c in conds if c[0] in keep]

    rows, token_tables = [], {}
    base_counts = None

    for label, vec, ratio, po in conds:
        t0 = time.time()
        scale = 0.0 if vec is None else ratio * resid / float(np.linalg.norm(vec))
        imgs, counts, mass, argmax_seqs = [], Counter(), {c: [] for c in KEYWORDS}, []
        for pi, prompt in enumerate(FREE_DRAW_PROMPTS):
            img, hid = j.draw(prompt, n=1, seed=SEED + pi, layer=LAYER, direction=vec,
                              scale=scale, prompt_only=po, return_hidden=True)
            imgs.append(img[0])
            np.savez_compressed(OUT / "states" / f"{label}_{pi}.npz", h=hid.astype(np.float16))
            with torch.inference_mode():
                h = torch.as_tensor(hid, device=j.device, dtype=torch.bfloat16)
                logits = j.lm.lm_head(h).float()  # (576, vocab)
                probs = torch.softmax(logits, -1)
                top1 = logits.argmax(-1).tolist()
                for c, ids in kw_ids.items():
                    mass[c].append(float(probs[:, ids].sum(-1).mean()))
            toks = [tok.decode([t]) for t in top1]
            counts.update(toks)
            argmax_seqs.append(toks)
        imgs = np.stack(imgs)
        clip = {c: float(judge.concept_prob(imgs, c).mean()) for c in KEYWORDS}
        save_grid(imgs, OUT / f"{label}_drawn.jpg", cols=4,
                  labels=[f"{clip['ggb']:.2f}g" for _ in imgs], tiles_dir=OUT / "tiles" / label)
        # argmax-token keyword hits over all 8x576 positions
        joined = " ".join(" ".join(s) for s in argmax_seqs)
        kw_argmax = {c: len(re.findall(KW_RE[c], joined, flags=re.I)) for c in KEYWORDS}
        if label == "baseline":
            base_counts = counts
        row = {"label": label, "ratio": ratio, "prompt_only": po, "scale": round(scale, 4),
               **{f"clip_{c}": round(v, 3) for c, v in clip.items()},
               **{f"mass_{c}": round(float(np.mean(mass[c])) * 1e3, 3) for c in KEYWORDS},  # x1e3
               **{f"argmax_kw_{c}": kw_argmax[c] for c in KEYWORDS},
               "distinct_top1": len(counts)}
        rows.append(row)
        token_tables[label] = {"top30": counts.most_common(30), "seq0": argmax_seqs[0], "counts": counts}
        print(row, flush=True)
        print(f"  top-15 tokens: {counts.most_common(15)}", flush=True)
        print(f"{label} done {time.time() - t0:.0f}s", flush=True)

        (OUT / "results.json").write_text(json.dumps({"layer": LAYER, "resid_norm": resid, "rows": rows}, indent=2))
        with open(OUT / "tokens.md", "w") as f:
            f.write(f"# run3 mirror: draw states through lm_head, L{LAYER}\n\n")
            f.write("mass_* = mean softmax probability on the concept's keyword tokens, x1e3.\n")
            f.write("argmax_kw_* = keyword regex hits over the 8x576 argmax tokens.\n\n")
            for lab, tt in token_tables.items():
                f.write(f"## {lab}\n\n")
                f.write("top-30 argmax tokens: " + ", ".join(f"{repr(t)}x{n}" for t, n in tt["top30"]) + "\n\n")
                if base_counts is not None and lab != "baseline":
                    enriched = sorted(((t, n, base_counts.get(t, 0)) for t, n in tt["counts"].items()),
                                      key=lambda x: -(x[1] - x[2]))[:20]
                    f.write("most over-represented vs baseline (token, count, baseline count): "
                            + ", ".join(f"{repr(t)} {n} vs {b}" for t, n, b in enriched) + "\n\n")
                f.write("argmax sequence, prompt 0, first 96 positions:\n\n`"
                        + " ".join(repr(t) for t in tt["seq0"][:96]) + "`\n\n")

    print("\n== summary ==")
    for r in rows:
        print(f"  {r['label']:>16}  clip g/c/o={r['clip_ggb']:.2f}/{r['clip_coffee']:.2f}/{r['clip_circle']:.2f}"
              f"  mass g/c/o={r['mass_ggb']:.3f}/{r['mass_coffee']:.3f}/{r['mass_circle']:.3f}"
              f"  argmax_kw g/c/o={r['argmax_kw_ggb']}/{r['argmax_kw_coffee']}/{r['argmax_kw_circle']}")


if __name__ == "__main__":
    t = time.time()
    main()
    print(f"\ntotal {time.time() - t:.0f}s")
