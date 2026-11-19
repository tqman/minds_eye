"""Run 2 follow-up C: is the unsteered visual-essay tint carried by the colour
WORDS or by the TOPIC?

Follow-up A found that unsteered essays on visual topics decode (through
gen_head) to the topic's palette 4.5/8 (ocean -> aqua, wheat -> gold, snow ->
white). The obvious confound: the essays are full of colour words, and the
state at a colour token may simply be "the state about to say blue". This
script regenerates the same eight essays with the same seeds (the baseline was
tile-for-tile identical across Run 2 passes, so the seeds hold), this time
keeping the token ids, and decodes three grids per essay:

  full      : all 576 states, as Follow-up A (should reproduce A's tiles);
  mask_color: states at every colour-word token (and the position after it,
              where the colour word is the input) replaced by the essay's mean
              state;
  mask_rand : the same NUMBER of positions, chosen at random away from colour
              tokens, replaced the same way (control for how much masking per
              se changes the tile).

Readout: the tiles, plus the mean RGB of each decoded image and its distance
from the non-visual baseline's mean colour. If the tint survives mask_color
about as well as mask_rand, it is the topic; if mask_color kills it and
mask_rand does not, it is the colour tokens.

Prediction on record (2026-11-16 19:25): the tint mostly survives; the
colour words are ~5% of positions and the Run 2 steered decodes carried tint
with zero concept words, so the palette is in the topic state, not the token.
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

LAYER = 12  # unused (no steer) but kept in the path for symmetry with run2
OUT = ROOT / "runs" / "run2_L12_visual_colormask"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 0
TEMP = 0.7

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
SHORT = ["ocean", "fire", "wheat", "ggb_fog", "coffee", "snow", "lavender", "desert"]

COLOR_RE = re.compile(
    r"^\s*(blue|azure|turquoise|aqua|aquamarine|cyan|teal|navy|cobalt|sapphire|"
    r"green|emerald|olive|lime|jade|"
    r"red|crimson|scarlet|ruby|maroon|burgundy|"
    r"orange|amber|gold|golden|yellow|saffron|ochre|"
    r"brown|tan|beige|caramel|chocolate|mahogany|copper|bronze|rust|"
    r"black|white|grey|gray|silver|charcoal|ivory|cream|"
    r"purple|violet|lavender|lilac|mauve|indigo|pink|magenta|rose|"
    r"dark|darkness|light|bright|pale|glow|glowing|fiery|golden|snowy|ashen|inky)\w*$",
    re.I,
)


def color_positions(tok, ids):
    """Positions whose token (decoded alone) is a colour word, plus the next position."""
    pos = set()
    for i, t in enumerate(ids):
        s = tok.decode([t])
        if COLOR_RE.match(s):
            pos.add(i)
            if i + 1 < len(ids):
                pos.add(i + 1)
    return sorted(pos)


def mean_rgb(img):
    return [round(float(x), 1) for x in img.reshape(-1, 3).mean(0)]


def main():
    j = Janus()
    tok = j.tok
    rng = np.random.default_rng(0)
    rows = []
    sheets = {"full": [], "mask_color": [], "mask_rand": []}
    base_rgb = None
    base_path = ROOT / "runs" / "run2_L12_noeos" / "tiles" / "baseline_argmax"
    if base_path.exists():
        from PIL import Image
        arr = np.stack([np.asarray(Image.open(p).convert("RGB")) for p in sorted(base_path.glob("*.png"))])
        base_rgb = np.array(mean_rgb(arr))
        print("non-visual baseline mean rgb", base_rgb, flush=True)

    for pi, prompt in enumerate(VISUAL_PROMPTS):
        t0 = time.time()
        txt, hid = j.write(prompt, max_new_tokens=N_IMG_TOKENS, seed=SEED + pi, temperature=TEMP,
                           return_hidden=True, stop_on_eos=False, suppress_eos=True)
        ids = j.last_ids
        assert hid.shape[0] == N_IMG_TOKENS and len(ids) == N_IMG_TOKENS
        cpos = color_positions(tok, ids)
        others = [i for i in range(N_IMG_TOKENS) if i not in set(cpos)]
        rpos = sorted(rng.choice(others, size=len(cpos), replace=False).tolist()) if cpos else []
        mean_state = hid.mean(0, keepdims=True)

        grids = {"full": hid.copy()}
        hm = hid.copy(); hm[cpos] = mean_state; grids["mask_color"] = hm
        hr = hid.copy(); hr[rpos] = mean_state; grids["mask_rand"] = hr
        row = {"essay": SHORT[pi], "n_color_pos": len(cpos),
               "color_tokens": [tok.decode([ids[i]]).strip() for i in cpos if COLOR_RE.match(tok.decode([ids[i]]))],
               "words": len(txt.split())}
        for k, g in grids.items():
            img, _ = j.states_to_image(g)
            sheets[k].append(img)
            row[f"rgb_{k}"] = mean_rgb(img)
            if base_rgb is not None:
                row[f"dist_from_nonvisual_{k}"] = round(float(np.linalg.norm(np.array(row[f"rgb_{k}"]) - base_rgb)), 1)
        row["rgb_shift_color_vs_full"] = round(float(np.linalg.norm(np.array(row["rgb_mask_color"]) - np.array(row["rgb_full"]))), 1)
        row["rgb_shift_rand_vs_full"] = round(float(np.linalg.norm(np.array(row["rgb_mask_rand"]) - np.array(row["rgb_full"]))), 1)
        rows.append(row)
        print({k: v for k, v in row.items() if k != "color_tokens"}, flush=True)
        print(f"  colour tokens: {row['color_tokens']}", flush=True)
        print(f"{SHORT[pi]} done {time.time() - t0:.0f}s", flush=True)
        (OUT / "results.json").write_text(json.dumps(rows, indent=2))

    for k, imgs in sheets.items():
        save_grid(np.stack(imgs), OUT / f"{k}.jpg", cols=4, labels=SHORT, tiles_dir=OUT / "tiles" / k)
    # one combined sheet: rows = full / mask_color / mask_rand
    combined = np.concatenate([np.stack(sheets["full"]), np.stack(sheets["mask_color"]), np.stack(sheets["mask_rand"])])
    save_grid(combined, OUT / "combined.jpg", cols=8, labels=SHORT + [f"{s} -col" for s in SHORT] + [f"{s} -rnd" for s in SHORT])

    print("\n== summary ==")
    print(f"{'essay':>9} {'ncol':>4} {'shift_color':>11} {'shift_rand':>10}  rgb_full / rgb_mask_color / rgb_mask_rand")
    for r in rows:
        print(f"{r['essay']:>9} {r['n_color_pos']:>4} {r['rgb_shift_color_vs_full']:>11} {r['rgb_shift_rand_vs_full']:>10}"
              f"  {r['rgb_full']} / {r['rgb_mask_color']} / {r['rgb_mask_rand']}")


if __name__ == "__main__":
    t = time.time()
    main()
    print(f"\ntotal {time.time() - t:.0f}s")
