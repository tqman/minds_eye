#!/bin/bash
# Run 2 follow-ups (2026-11-16 evening), GPU1.
# 1. Unsteered essays on VISUAL topics, decoded through gen_head: the model's mind's eye
#    with nothing injected.                               -> runs/run2_L12_visual/
# 2. DRAW under random seeds 1-3 at the Run 1 GGB collapse doses (r0.3, r0.45), L12,
#    full-position, so each random's draw signature can be compared to its own Run 2
#    write decode (seed 2 wrote blue streaks).           -> runs/run1_ggb_L12_rand123/
cd ~/src/minds_eye
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1
ME_LAYER=12 ME_PROMPTS=visual ME_CONCEPTS=coffee ME_RATIOS= ME_RANDOM_SEEDS= \
  ME_SUPPRESS_EOS=1 ME_SKIP_CONTROL=1 ME_TAG=_visual \
  .venv/bin/python scripts/run2_decoder.py > runs/run2_L12_visual.log 2>&1
touch runs/fu_visual.done
ME_CONCEPT=ggb ME_LAYER=12 ME_RATIOS=0.3,0.45 ME_RANDOM_SEEDS=1,2,3 ME_SKIP_CONCEPT=1 ME_TAG=_rand123 \
  .venv/bin/python scripts/run1_pilot.py > runs/run1_ggb_L12_rand123.log 2>&1
touch runs/fu.done
