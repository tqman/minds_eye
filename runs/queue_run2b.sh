#!/bin/bash
cd ~/src/minds_eye
# wait for the first run2 pass to release the GPU
while [ ! -f runs/run2.done ]; do sleep 10; done
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ME_LAYER=12 ME_CONCEPTS=coffee,ggb,circle ME_RATIOS=0.6,1.0 \
  ME_SUPPRESS_EOS=1 ME_TAG=_noeos \
  .venv/bin/python scripts/run2_decoder.py > runs/run2_L12_noeos.log 2>&1
touch runs/run2b.done
