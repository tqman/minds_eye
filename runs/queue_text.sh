#!/bin/bash
cd ~/src/minds_eye
for C in ggb coffee circle; do
  for L in 12 18; do
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ME_CONCEPT=$C ME_LAYER=$L ME_RATIOS=0,0.25,0.4,0.6 \
      .venv/bin/python scripts/run1_text.py > runs/text_${C}_L${L}.log 2>&1
  done
done
touch runs/text.done
