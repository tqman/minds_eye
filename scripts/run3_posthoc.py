"""Run 3 post-hoc: whole-vocabulary differential readout of the saved draw states.

For each condition in runs/run3_mirror_L12/states, push all 8x576 states through
lm_head, average the logits over positions, and subtract the baseline average.
The tokens with the largest positive shift are what the language head would
say MORE under that steer; no keyword list, no argmax. Also lists the most
common second-choice token (first choice is always <end_of_image>).
"""

import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from minds_eye.janus_model import Janus  # noqa: E402

LAYER = int(os.environ.get("ME_LAYER", "12"))
RUN = ROOT / "runs" / f"run3_mirror_L{LAYER}"
STATES = RUN / "states"
TOPK = 25


def main():
    j = Janus()
    tok = j.tok
    eoi = tok.convert_tokens_to_ids("<end_of_image>")
    labels = sorted({p.name.rsplit("_", 1)[0] for p in STATES.glob("*.npz")})
    labels = ["baseline"] + [l for l in labels if l != "baseline"]
    mean_logits, second = {}, {}
    with torch.inference_mode():
        for lab in labels:
            acc, n, sec = None, 0, Counter()
            for p in sorted(STATES.glob(f"{lab}_*.npz")):
                h = torch.as_tensor(np.load(p)["h"], device=j.device, dtype=torch.bfloat16)
                lg = j.lm.lm_head(h).float()
                acc = lg.sum(0) if acc is None else acc + lg.sum(0)
                n += lg.shape[0]
                lg2 = lg.clone()
                lg2[:, eoi] = -1e9
                sec.update(lg2.argmax(-1).tolist())
            mean_logits[lab] = (acc / n).cpu()
            second[lab] = sec
    base = mean_logits["baseline"]
    out = {}
    lines = [f"# run3 post-hoc: mean lm_head logit shift vs baseline, L{LAYER}\n"]
    lines.append("## baseline second-choice tokens (after <end_of_image>)\n")
    lines.append(", ".join(f"{tok.decode([t])!r}x{c}" for t, c in second["baseline"].most_common(20)) + "\n")
    for lab in labels[1:]:
        d = mean_logits[lab] - base
        up = torch.topk(d, TOPK).indices.tolist()
        dn = torch.topk(-d, TOPK).indices.tolist()
        out[lab] = {"up": [(tok.decode([t]), round(float(d[t]), 3)) for t in up],
                    "down": [(tok.decode([t]), round(float(d[t]), 3)) for t in dn],
                    "second_choice": [(tok.decode([t]), c) for t, c in second[lab].most_common(15)]}
        lines.append(f"\n## {lab}\n")
        lines.append("UP:   " + ", ".join(f"{t!r} {v:+.2f}" for t, v in out[lab]["up"]) + "\n")
        lines.append("DOWN: " + ", ".join(f"{t!r} {v:+.2f}" for t, v in out[lab]["down"]) + "\n")
        lines.append("2nd:  " + ", ".join(f"{t!r}x{c}" for t, c in out[lab]["second_choice"]) + "\n")
        print(f"{lab:>16} UP: " + ", ".join(f"{t!r}" for t, _ in out[lab]["up"][:12]), flush=True)
    (RUN / "posthoc.md").write_text("".join(lines))
    (RUN / "posthoc.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
