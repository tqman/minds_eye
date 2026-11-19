#!/bin/bash
cd ~/src/minds_eye
until command grep -q '^total' runs/run1_ggb_L24.log 2>/dev/null; do sleep 15; done
for L in 12 18; do
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ME_CONCEPT=circle ME_LAYER=$L ME_RATIOS=0,0.25,0.4,0.6 \
    .venv/bin/python scripts/run1_pilot.py > runs/run1_circle_L$L.log 2>&1
done
touch runs/circle.done
