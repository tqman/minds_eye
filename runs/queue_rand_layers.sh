#!/bin/bash
# 2026-11-16 20:45. Random-vector control for the layer sweep: L6 GGB r1.0 came back
# with per-essay palette drift (purple, rust, ice blue) and no cable structure, and
# without a random at L6 I can't call that drift concept-specific. s0 at r1.0 and
# r0.6 at each swept layer, tagged separately so results.json isn't clobbered.
# PREDICTION (written before the run): random at L6 drifts the palettes about as much as GGB did
# (so L6 GGB is "any perturbation", not the concept); at L24 random gives a flat
# single-colour wash while GGB gives orange/red specifically.
cd ~/src/minds_eye
while [ ! -f runs/run5.done ]; do sleep 20; done
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1
for L in 6 18 24; do
  for R in 0.6 1.0; do
    ME_LAYER=$L ME_CONCEPTS=ggb ME_RATIOS= ME_RANDOM_SEEDS=0 ME_RANDOM_RATIO=$R ME_SUPPRESS_EOS=1 \
      ME_SKIP_CONTROL=1 ME_SKIP_BASELINE=1 ME_TAG=_noeos_rand${R} \
      .venv/bin/python scripts/run2_decoder.py > runs/run2_L${L}_noeos_rand${R}.log 2>&1
  done
done
touch runs/rand_layers.done
