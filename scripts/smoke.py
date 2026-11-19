"""Smoke test: load, write a sentence, draw four images, report timings."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from minds_eye.janus_model import Janus, save_grid  # noqa: E402

t0 = time.time()
j = Janus()
print(f"loaded in {time.time() - t0:.0f}s; layers={j.n_layers} d={j.d_model}; "
      f"vram={torch.cuda.memory_allocated() / 2**30:.1f}GB", flush=True)
print("sft prompt repr:", repr(j.chat_prompt("hello")), flush=True)

t0 = time.time()
print("text:", repr(j.write("What is the capital of France? One sentence.", max_new_tokens=30)))
print(f"write {time.time() - t0:.1f}s", flush=True)

t0 = time.time()
imgs = j.draw("A red bicycle leaning against a brick wall.", n=4, seed=1)
print(f"draw 4 in {time.time() - t0:.1f}s; peak vram={torch.cuda.max_memory_allocated() / 2**30:.1f}GB")
out = Path(__file__).resolve().parent.parent / "runs" / "smoke" / "bicycle.jpg"
save_grid(imgs, out, cols=4)
print("saved", out)

acts = j.last_token_acts(["I love a cup of coffee in the morning."])
print("acts", acts.shape, "per-layer norms", [round(float(x), 1) for x in (acts[0] ** 2).sum(-1) ** 0.5])
