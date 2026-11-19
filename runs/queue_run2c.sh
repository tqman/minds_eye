#!/bin/bash
cd ~/src/minds_eye
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ME_LAYER=12 ME_CONCEPTS=coffee ME_RATIOS= \
  ME_RANDOM_SEEDS=1,2,3 ME_SUPPRESS_EOS=1 ME_TAG=_noeos_rand ME_SKIP_CONTROL=1 ME_SKIP_BASELINE=1 \
  .venv/bin/python scripts/run2_decoder.py > runs/run2_L12_noeos_rand.log 2>&1
touch runs/run2c.done
