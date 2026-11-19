#!/bin/bash
# 2026-11-16 20:05, GPU1. Avenue 3: prompt-only steer in the WRITE regime, decoded through gen_head.
cd ~/src/minds_eye
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1
ME_LAYER=12 ME_CONCEPTS=ggb,coffee,circle ME_RATIOS=1.0,2.0 ME_RANDOM_RATIO=2.0 ME_RANDOM_SEEDS=0,2 \
  ME_SUPPRESS_EOS=1 ME_PROMPT_ONLY=1 ME_SKIP_CONTROL=1 ME_SKIP_BASELINE=1 ME_TAG=_noeos_po \
  .venv/bin/python scripts/run2_decoder.py > runs/run2_L12_noeos_po.log 2>&1
touch runs/po_write.done
