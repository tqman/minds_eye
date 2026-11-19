"""Avenue 4: WHICH of the 576 writing positions carry the tint?

Run 2 decodes 576 writing states as a 24x24 VQ grid. Position p lands at
patch (p // 24, p % 24) of the 384x384 image, 16x16 pixels each. So a
per-position map of the tint is a per-patch map of the decoded tile: for each
steered condition, take the argmax tile for essay i, take the baseline tile for
the same essay i (same prompt, same seed, so the unsteered text and its states
are the natural reference), compute the per-patch mean RGB in both, and record
the shift. Average the 8 essays -> one 24x24 heatmap per condition.

Reads only saved tiles (runs/run2_L12_noeos/tiles/*_argmax/NN.png and the rand
set), so it needs no GPU and can run while GPU1 is busy.

Readouts:
  * heatmap of |RGB shift| per position, per condition (raster order = token
    order, so the top rows are the first ~24 tokens of the essay, the bottom
    rows the last);
  * row profile (early vs late in the essay) and column profile (position
    within a run of 24 tokens; there is no reason for any column structure, so
    a flat column profile is the sanity check);
  * signed per-channel shift, so "bluer" / "warmer" is readable;
  * token-class split: re-tokenize each essay's text from samples.md; where the
    re-tokenization gives exactly 576 ids (it should: the text was decoded from
    those ids), classify each position (newline / punctuation / word-start /
    word-continuation) and report the mean shift per class.

Prediction on record (2026-11-16 20:10): the tint is spatially UNIFORM
(flat row and column profiles; the heatmap is noise around one level), because
the steer is a constant per-position bias and the VQ decoder smears locally.
If anything, the first row (the essay's opening, where the state is closest to
the prompt) is a little weaker. Token class will not matter: punctuation and
word positions shift alike (same reason the colour-word mask did nothing).
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "runs" / "run2_L12_noeos"
RAND = ROOT / "runs" / "run2_L12_noeos_rand"
OUT = ROOT / "runs" / "run4_posmap_L12"
OUT.mkdir(parents=True, exist_ok=True)
G, P = 24, 16

CONDS = {
    "coffee_r0.6": RUN, "coffee_r1.0": RUN, "ggb_r0.6": RUN, "ggb_r1.0": RUN,
    "circle_r0.6": RUN, "circle_r1.0": RUN, "random_r1.0": RUN,
    "random_s1_r1.0": RAND, "random_s2_r1.0": RAND, "random_s3_r1.0": RAND,
}


def tiles(run, label):
    d = run / "tiles" / f"{label}_argmax"
    ps = sorted(d.glob("*.png"))
    assert len(ps) == 8, (d, len(ps))
    return np.stack([np.asarray(Image.open(p).convert("RGB")).astype(np.float32) for p in ps])


def patch_rgb(img):
    """(384,384,3) -> (24,24,3) mean RGB per 16x16 patch."""
    return img.reshape(G, P, G, P, 3).mean(axis=(1, 3))


def samples_texts(run, label):
    """Essay texts for a condition from samples.md, in prompt order."""
    md = (run / "samples.md").read_text()
    sec = re.split(r"^## ", md, flags=re.M)
    for s in sec:
        if s.startswith(label + "\n"):
            parts = re.split(r"^\*\*.*?\*\* \(words \d+, eos_at -?\d+, kw .*?\)\n\n", s, flags=re.M)
            return [p.strip() for p in parts[1:]]
    return None


def token_classes(tok, text):
    ids = tok.encode(text, add_special_tokens=False) if hasattr(tok, "encode") else None
    if ids is None or len(ids) != G * G:
        return None, len(ids) if ids is not None else -1
    cls = []
    for t in ids:
        s = tok.decode([t])
        if "\n" in s:
            cls.append("newline")
        elif re.fullmatch(r"\s*[^\w\s]+\s*", s):
            cls.append("punct")
        elif s.startswith(" ") or s.startswith("▁"):
            cls.append("word_start")
        else:
            cls.append("word_cont")
    return cls, len(ids)


def main():
    base = tiles(RUN, "baseline")
    base_p = np.stack([patch_rgb(b) for b in base])  # (8,24,24,3)

    tok = None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("deepseek-ai/Janus-Pro-7B")
    except Exception as e:  # noqa: BLE001
        print("tokenizer unavailable, skipping token-class split:", e)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    for ax, (label, run) in zip(axes.ravel(), CONDS.items()):
        st = np.stack([patch_rgb(t) for t in tiles(run, label)])  # (8,24,24,3)
        diff = st - base_p                                        # signed, per essay
        mag = np.linalg.norm(diff, axis=-1)                       # (8,24,24)
        heat = mag.mean(0)                                        # (24,24)
        signed = diff.mean(axis=(0, 1, 2))                        # (3,) mean per channel
        row_prof = heat.mean(1)                                   # (24,) by grid row = token//24
        col_prof = heat.mean(0)                                   # (24,) by token%24
        # how uniform? spread of the heatmap vs its level, and early/late ratio
        rec = {
            "cond": label,
            "mean_shift": round(float(heat.mean()), 1),
            "heat_std_over_mean": round(float(heat.std() / heat.mean()), 3),
            "row_first4": round(float(row_prof[:4].mean()), 1),
            "row_mid": round(float(row_prof[8:16].mean()), 1),
            "row_last4": round(float(row_prof[-4:].mean()), 1),
            "col_range": round(float(col_prof.max() - col_prof.min()), 1),
            "row_range": round(float(row_prof.max() - row_prof.min()), 1),
            "signed_rgb": [round(float(x), 1) for x in signed],
            "row_profile": [round(float(x), 1) for x in row_prof],
            "col_profile": [round(float(x), 1) for x in col_prof],
        }
        # essay-to-essay agreement: is the SAME position hot across essays?
        # correlation of each essay's map with the mean of the others
        cors = []
        for i in range(8):
            others = np.delete(mag, i, 0).mean(0)
            cors.append(np.corrcoef(mag[i].ravel(), others.ravel())[0, 1])
        rec["pos_consistency_r"] = round(float(np.mean(cors)), 3)

        if tok is not None:
            texts = samples_texts(run, label)
            btexts = samples_texts(RUN, "baseline")
            if texts and btexts:
                agg = {}
                nlen = []
                for i, t in enumerate(texts):
                    cls, n = token_classes(tok, t)
                    nlen.append(n)
                    if cls is None:
                        continue
                    for p, c in enumerate(cls):
                        agg.setdefault(c, []).append(float(mag[i, p // G, p % G]))
                rec["retok_lengths"] = nlen
                rec["by_token_class"] = {c: [round(float(np.mean(v)), 1), len(v)] for c, v in agg.items()}
        rows.append(rec)
        print({k: v for k, v in rec.items() if k not in ("row_profile", "col_profile")}, flush=True)

        im = ax.imshow(heat, cmap="magma", vmin=0)
        ax.set_title(f"{label}\nmean {rec['mean_shift']} std/mean {rec['heat_std_over_mean']}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Per-position |RGB shift| vs baseline tile, mean of 8 essays (row = token//24, top = essay start)")
    fig.tight_layout()
    fig.savefig(OUT / "heatmaps.png", dpi=110)

    # profiles figure
    fig2, (a1, a2) = plt.subplots(1, 2, figsize=(16, 5))
    for r in rows:
        a1.plot(r["row_profile"], label=r["cond"])
        a2.plot(r["col_profile"], label=r["cond"])
    a1.set_title("row profile: shift by grid row (token index // 24; left = essay start)")
    a2.set_title("column profile: shift by token index % 24 (should be flat)")
    a1.legend(fontsize=7); a2.legend(fontsize=7)
    fig2.tight_layout()
    fig2.savefig(OUT / "profiles.png", dpi=110)

    (OUT / "results.json").write_text(json.dumps(rows, indent=2))
    with open(OUT / "results.md", "w") as f:
        f.write("| cond | mean shift | std/mean | rows 0-3 | rows 8-15 | rows 20-23 | row range | col range | pos consistency r | signed RGB | by token class (mean, n) |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|\n")
        for r in rows:
            f.write(f"| {r['cond']} | {r['mean_shift']} | {r['heat_std_over_mean']} | {r['row_first4']} | {r['row_mid']} | "
                    f"{r['row_last4']} | {r['row_range']} | {r['col_range']} | {r['pos_consistency_r']} | {r['signed_rgb']} | "
                    f"{r.get('by_token_class', '-')} |\n")
    print(open(OUT / "results.md").read())


if __name__ == "__main__":
    main()
