#!/bin/bash
cd ~/src/minds_eye
for C in circle coffee; do
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ME_CONCEPT=$C ME_LAYER=12 ME_PROMPT_ONLY=1 ME_RATIOS=0,1,2 \
    .venv/bin/python scripts/run1_pilot.py > runs/run1_${C}_L12_po.log 2>&1
done
touch runs/po2.done
