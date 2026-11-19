#!/bin/bash
# 2026-11-16 20:40. Experiments 1 and 2 from the 20:22 handoff, chained on GPU1.
#
# 1. LAYER SWEEP for Run 2 (kills the "one layer" caveat). ggb and coffee at r0.6 and
#    r1.0, EOS masked, L6/L18/L24. Baseline and positive control skipped: both are
#    unsteered and layer-independent, already in runs/run2_L12_noeos/.
#    PREDICTION (written before the run): L6 = no tint (Run 1 L6 drew untouched
#    scenes; the bias is a small share of the residual and gets washed by 24 blocks).
#    L18 = weaker and cleaner tint than L12 (Run 1 L18 was coherent feature drift, no
#    object). L24 = the strongest and crudest tint of all, one block before gen_head:
#    GGB a red/orange wash (Run 1 L24 "everything turns red"), coffee brown, and the
#    essays likely degrade at r1.0 since the direction is a large share of the residual
#    that deep. Text keyword hits stay ~0 for ggb/coffee at every layer.
# 2. DOSE CURVE at L12: ggb, coffee, circle at r0.2/0.4/0.8 (0.6 and 1.0 exist in
#    runs/run2_L12_noeos/). PREDICTION (written before the run): image-side signal (tile RGB shift from
#    the baseline tiles above the texture floor of 72; CLIP ggb rising) is detectable
#    by r0.4 for ggb and by r0.4-0.6 for coffee, while text keyword hits for both stay
#    at zero through r0.8. Circle is the mirror: text hits rise from r0.4, image flat at
#    every dose. So the image head's threshold is ~0.4 and the text head never crosses
#    below r1.0 for ggb/coffee in the writing regime.
cd ~/src/minds_eye
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1
for L in 6 18 24; do
  ME_LAYER=$L ME_CONCEPTS=ggb,coffee ME_RATIOS=0.6,1.0 ME_RANDOM_SEEDS= ME_SUPPRESS_EOS=1 \
    ME_SKIP_CONTROL=1 ME_SKIP_BASELINE=1 ME_TAG=_noeos \
    .venv/bin/python scripts/run2_decoder.py > runs/run2_L${L}_noeos.log 2>&1
done
touch runs/sweep.done
ME_LAYER=12 ME_CONCEPTS=ggb,coffee,circle ME_RATIOS=0.2,0.4,0.8 ME_RANDOM_SEEDS= ME_SUPPRESS_EOS=1 \
  ME_SKIP_CONTROL=1 ME_SKIP_BASELINE=1 ME_TAG=_noeos_dose \
  .venv/bin/python scripts/run2_decoder.py > runs/run2_L12_noeos_dose.log 2>&1
touch runs/dose.done
