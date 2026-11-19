#!/bin/bash
# 2026-11-16 evening, GPU1. Avenue 1 then avenue 2 from the handoff.
# 1. Run 3 mirror: draw states through lm_head              -> runs/run3_mirror_L12/
# 2. Colour-word mask on the unsteered visual essays        -> runs/run2_L12_visual_colormask/
cd ~/src/minds_eye
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1
ME_LAYER=12 .venv/bin/python scripts/run3_mirror.py > runs/run3_mirror_L12.log 2>&1
touch runs/run3.done
.venv/bin/python scripts/run2_colormask.py > runs/run2_L12_visual_colormask.log 2>&1
touch runs/colormask.done
