"""Run 5: is there a palette while the model READS about the ocean?

Follow-up A: unsteered essays the model WRITES about visual topics decode
(through gen_head, nothing trained) to the topic's palette. Run 5 asks whether
the same thing is in the states when the model is only reading. Three grids per
essay, all from a single prefill forward, no generation, states at the 576
essay positions decoded with ``Janus.states_to_image``:

  assist  : the essay in the ASSISTANT span after its own prompt, i.e. the
            teacher-forced replay of Follow-up A. Causal attention over the same
            tokens is the same computation as generating them, so this is a
            CONTROL for the read path, not a result: it must reproduce A's tiles
            (cosine to the generation states ~1).
  user    : the essay as USER content ("<|User|>: <essay>") with no visual
            prompt anywhere. Someone else's text about the ocean; the model is
            reading, not producing. THE question.
  shuffle : the same 576 tokens in the user span in a fixed random order. Same
            bag of words, no coherent reading. Does the palette need the essay
            or just the vocabulary?

Alignment: write() returns the k-th state at the position that predicted essay
token k (the first one sits on the role header's ':'), so every grid here takes
states from (first essay token index - 1) onward. Essay token ids are recovered
by regenerating the eight Follow-up A essays with their seeds (the baseline was
tile-for-tile identical across Run 2 passes); the in-process generation states
double as the cosine reference and are also checked against the saved
runs/run2_L12_visual/states/baseline_{i}.npz.

Readout: contact sheets + tiles, per-tile mean RGB, mean-colour distance of each
grid from the assist grid and from the non-visual baseline's mean colour (the
run2_L12_noeos baseline tiles), and per-pixel mean abs diff vs assist.

PREDICTIONS on record (2026-11-16 20:55, before the run):
  assist : identical to Follow-up A (cosine > 0.99, tiles equal within noise).
  user   : the palette SURVIVES. The final-layer state at a user-span position
           does the same next-token job over the same text; only the role header
           differs. 6/8 tiles within ~20 RGB units of assist in mean colour. If
           it vanishes instead, the palette is tied to the production role, which
           would be the bigger result.
  shuffle: weakened, not gone. Topic survives as bag-of-words at reduced
           strength: distance from the non-visual grey roughly halves and the
           texture goes incoherent.
Second prediction: to be asked for before the sheet is shown.
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from minds_eye.janus_model import Janus, save_grid, N_IMG_TOKENS  # noqa: E402

OUT = ROOT / "runs" / "run5_reading"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "states").mkdir(exist_ok=True)
SEED, TEMP = 0, 0.7
REF = ROOT / "runs" / "run2_L12_visual"
BASE_NONVIS = ROOT / "runs" / "run2_L12_noeos" / "tiles" / "baseline_argmax"

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


def mean_rgb(img):
    return np.asarray(img, dtype=np.float64).reshape(-1, 3).mean(0)


def main():
    j = Janus()
    tok = j.tok
    enc = lambda s, sp=False: tok.encode(s, add_special_tokens=sp)  # noqa: E731
    user_prefix = enc("<|User|>:", True)          # BOS <|User|> :
    user_suffix = enc("\n\n<|Assistant|>:")       # closes the user turn
    nonvis = np.stack([mean_rgb(Image.open(BASE_NONVIS / f"{i:02d}.png")) for i in range(8)]).mean(0)

    grids = {"gen": [], "assist": [], "user": [], "shuffle": []}
    rows = []
    rng = np.random.default_rng(SEED)
    for pi, prompt in enumerate(VISUAL_PROMPTS):
        t0 = time.time()
        txt, hid_gen = j.write(prompt, max_new_tokens=N_IMG_TOKENS, seed=SEED + pi,
                               temperature=TEMP, return_hidden=True, stop_on_eos=False,
                               suppress_eos=True)
        ids = list(j.last_ids)
        assert hid_gen.shape[0] == N_IMG_TOKENS and len(ids) == N_IMG_TOKENS
        saved = np.load(REF / "states" / f"baseline_{pi}.npz")["h"].astype(np.float32)
        cos_saved = float(np.mean(np.sum(hid_gen * saved, 1) /
                                  (np.linalg.norm(hid_gen, axis=1) * np.linalg.norm(saved, axis=1) + 1e-6)))

        # assist: teacher-forced replay in the assistant span
        pre = enc(j.chat_prompt(prompt), True)
        h_assist = j.read_states(pre + ids, start=len(pre) - 1)
        # user: the essay as user content, no visual prompt anywhere
        h_user = j.read_states(user_prefix + ids + user_suffix, start=len(user_prefix) - 1)
        # shuffle: same tokens, fixed random order, user span
        perm = rng.permutation(N_IMG_TOKENS)
        sh_ids = [ids[k] for k in perm]
        h_shuf = j.read_states(user_prefix + sh_ids + user_suffix, start=len(user_prefix) - 1)

        def cos(a, b):
            return float(np.mean(np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-6)))

        imgs = {}
        for name, h in (("gen", hid_gen), ("assist", h_assist), ("user", h_user), ("shuffle", h_shuf)):
            img, _ = j.states_to_image(h)
            imgs[name] = img
            grids[name].append(img)
            np.savez_compressed(OUT / "states" / f"{name}_{pi}.npz", h=h.astype(np.float16))
        ref_tile = np.asarray(Image.open(REF / "tiles" / "baseline_argmax" / f"{pi:02d}.png"))
        m = {k: mean_rgb(v) for k, v in imgs.items()}
        row = {"i": pi, "topic": SHORT[pi], "words": len(txt.split()),
               "cos_gen_vs_saved": round(cos_saved, 4),
               "cos_assist_vs_gen": round(cos(h_assist, hid_gen), 4),
               "cos_user_vs_gen": round(cos(h_user, hid_gen), 4),
               "cos_shuffle_vs_gen": round(cos(h_shuf, hid_gen), 4),
               "pix_gen_vs_A": round(float(np.abs(imgs["gen"].astype(int) - ref_tile.astype(int)).mean()), 2),
               "pix_assist_vs_gen": round(float(np.abs(imgs["assist"].astype(int) - imgs["gen"].astype(int)).mean()), 2),
               "pix_user_vs_assist": round(float(np.abs(imgs["user"].astype(int) - imgs["assist"].astype(int)).mean()), 2),
               "pix_shuffle_vs_assist": round(float(np.abs(imgs["shuffle"].astype(int) - imgs["assist"].astype(int)).mean()), 2),
               **{f"rgb_{k}": [round(float(x)) for x in v] for k, v in m.items()},
               **{f"d_{k}_vs_assist": round(float(np.linalg.norm(m[k] - m["assist"])), 1) for k in ("user", "shuffle")},
               **{f"d_{k}_vs_nonvis": round(float(np.linalg.norm(m[k] - nonvis)), 1) for k in m}}
        rows.append(row)
        print(row, flush=True)
        print(f"{SHORT[pi]} done {time.time() - t0:.0f}s", flush=True)
        (OUT / "results.json").write_text(json.dumps({"nonvis_mean_rgb": [round(float(x)) for x in nonvis],
                                                      "rows": rows}, indent=2))

    for name, imgs in grids.items():
        save_grid(np.stack(imgs), OUT / f"{name}.jpg", cols=4, labels=SHORT,
                  tiles_dir=OUT / "tiles" / name)
    # combined: one row per condition, columns = topics
    allimgs = np.stack([im for name in ("assist", "user", "shuffle") for im in grids[name]])
    save_grid(allimgs, OUT / "combined.jpg", cols=8,
              labels=[f"{n}:{s}" for n in ("assist", "user", "shuffle") for s in SHORT])

    print("\n== summary ==")
    print(f"{'topic':>9} {'cosA':>6} {'cosU':>6} {'cosS':>6} {'pixU':>6} {'pixS':>6} {'dU':>6} {'dS':>6} {'nvA':>6} {'nvU':>6} {'nvS':>6}")
    for r in rows:
        print(f"{r['topic']:>9} {r['cos_assist_vs_gen']:6.3f} {r['cos_user_vs_gen']:6.3f} {r['cos_shuffle_vs_gen']:6.3f} "
              f"{r['pix_user_vs_assist']:6.1f} {r['pix_shuffle_vs_assist']:6.1f} {r['d_user_vs_assist']:6.1f} "
              f"{r['d_shuffle_vs_assist']:6.1f} {r['d_assist_vs_nonvis']:6.1f} {r['d_user_vs_nonvis']:6.1f} {r['d_shuffle_vs_nonvis']:6.1f}")


if __name__ == "__main__":
    t = time.time()
    main()
    print(f"\ntotal {time.time() - t:.0f}s")
