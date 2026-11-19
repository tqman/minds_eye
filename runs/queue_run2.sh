#!/bin/bash
cd ~/src/minds_eye
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ME_LAYER=12 ME_CONCEPTS=coffee,ggb ME_RATIOS=0.6,1.0 \
  .venv/bin/python scripts/run2_decoder.py > runs/run2_L12.log 2>&1
touch runs/run2.done
