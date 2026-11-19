#!/bin/bash
# Run 5 (reading regime), chained behind the layer sweep + dose curve (GPU1 holds one 7B).
cd ~/src/minds_eye
while [ ! -f runs/dose.done ]; do sleep 20; done
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/run5_reading.py > runs/run5_reading.log 2>&1
touch runs/run5.done
