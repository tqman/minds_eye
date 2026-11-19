# Figure captions

- `fig1a_baseline_street.png` (cropped from runs/run1_ggb_L12/baseline.jpg, top-left cell; this run saved no per-tile PNGs): the unsteered residential-street draw, used as the every-position baseline. The small yellow number in the corner is the CLIP bridge score printed on the contact sheet.
- `fig1a_ggb_L12_r0.3_street.png` (cropped from runs/run1_ggb_L12/ggb_r0.3.jpg, top-left cell): the same street under the bridge direction added at every position (L12, r=0.3), utility poles, overhead wires and red brick with the scene intact and no tower, the "features, not objects" claim.
- `fig1b_baseline_harbor.png` (runs/run1_ggb_L12_po/tiles/baseline/14.png): the unsteered harbor draw, used as the prompt-only baseline.
- `fig1b_ggb_L12_po_r1.0_harbor.png` (runs/run1_ggb_L12_po/tiles/ggb_r1.0/14.png): the same harbor with the bridge direction added to the prompt alone (L12, r=1.0), composing full Golden Gate towers with deck and cables among the masts while the scene stays coherent.
- `fig1c_baseline_bookkeeping.png` (runs/run2_L12_noeos/tiles/baseline_argmax/03.png): the image-head decode of an unsteered bookkeeping essay's writing states, a blue-gray smear with no object.
- `fig1c_ggb_L12_noeos_r1.0_bookkeeping.png` (runs/run2_L12_noeos/tiles/ggb_r1.0_argmax/03.png): the same essay written under the bridge direction (L12, r=1.0), whose states decode to blue with red and orange cable streaks though the text holds no bridge word.
- `fig1d_ocean_visual.png` (runs/run2_L12_visual/tiles/baseline_argmax/00.png): the image-head decode of an unsteered ocean essay, turquoise and aqua with white glints, the topic's palette with no steering.
- `fig1d_taxlaw_baseline.png` (runs/run2_L12_noeos/tiles/baseline_argmax/00.png): the image-head decode of an unsteered tax-law essay, a featureless smear, the non-visual baseline the ocean tile is read against.
