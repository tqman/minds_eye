#!/bin/bash
cd ~/src/minds_eye
until [ -f runs/circle.done ]; do sleep 15; done
for L in 12 18; do
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ME_CONCEPT=ggb ME_LAYER=$L ME_PROMPT_ONLY=1 ME_RATIOS=0,0.5,1,2 \
    .venv/bin/python scripts/run1_pilot.py > runs/run1_ggb_L${L}_po.log 2>&1
done
touch runs/po.done
