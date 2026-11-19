# Mind's-eye experiment — findings log

## 2026-11-16: harness built, Run 1 pilot (coffee, layer 12)

Harness at `~/src/minds_eye` (loader `minds_eye/janus_model.py`, concepts, direction, CLIP judge).
Janus-Pro-7B loads in ~37s on one 3090, ~14GB; draw of 8 images ~16s; peak VRAM ~17GB. Fits GPU1 alone.

Run 1 pilot: coffee diff-of-means direction at layer 12 (of 30), dir_norm 28.4, resid_norm 188.4.
Dose = ratio * resid_norm / dir_norm. 8 neutral free-draw prompts x 2 imgs per condition. CLIP ViT-L/14 judge.

| condition          | mean coffee | frac>0.5 |
|--------------------|-------------|----------|
| baseline (no steer)| 0.024       | 0.00     |
| coffee r0.4        | 0.173       | 0.19     |
| coffee r0.6        | 0.292       | 0.19     |
| coffee r0.9        | 0.347       | 0.31     |
| random r0.4        | 0.075       | 0.00     |
| random r0.6        | 0.074       | 0.00     |
| random r0.9        | 0.066       | 0.00     |

**Confirmed:** the coffee language-stream vector reaches the image head and the effect scales with dose;
a random vector of equal norm does nothing (flat at ~0.07). The steer is specific, not generic perturbation.

**Honesty caveat (from looking at the images, not the table):** at r0.6 and r0.9 the scene does NOT gain a
coffee cup — it collapses into dark, grainy, speckled brown/black texture (looks like a coffee *surface* /
foam under a loupe). CLIP scores that as coffee (dark+grainy+brown), so the rising mean is partly the judge
rewarding texture, partly real. At layer 12 these doses destroy image structure rather than insert an object.
Baseline scenes (streets, hills, bookshelves) are clean and coherent.

**Next (for the evening):**
- Low-dose sweep (r0.1–0.35) and/or a different layer to find the regime where coffee appears AS AN OBJECT
  with the scene still coherent (the true Golden-Gate-with-a-paintbrush picture). Likely a lower/earlier layer.
- Add a coherence gate to the judge (e.g. CLIP img-img sim to the baseline, or a VLM "is this a real scene?")
  so the metric can't be gamed by dark grain.
- THEN Run 2 (the finding): steer coffee, make it write about tax law, decode hidden states through the VQ
  decoder. Pre-registered prediction stays: brown smears. Scaffold in `scripts/run2_decoder.py`.

## 2026-11-16 (afternoon): Run 1 with Golden Gate Bridge, layer 12

Same design, ratios [0, 0.2, 0.3, 0.45, 0.6], 8 prompts x 2, CLIP judge on "Golden Gate Bridge".

| condition    | mean ggb | frac>0.5 |
|--------------|----------|----------|
| baseline     | 0.029    | 0.00     |
| ggb r0.2     | 0.068    | 0.06     |
| random r0.2  | 0.068    | 0.06     |
| ggb r0.3     | 0.112    | 0.06     |
| random r0.3  | 0.084    | 0.06     |
| ggb r0.45    | 0.187    | 0.19     |
| random r0.45 | 0.048    | 0.00     |
| ggb r0.6     | 0.068    | 0.00     |
| random r0.6  | 0.102    | 0.00     |

Judge numbers are weak and non-monotonic. The pictures say more than the table:

- **No bridge object is inserted at any dose at L12.** The judge never sees a silhouette.
- **But the steer is bridge-FLAVOURED, not a colour wash.** At r0.3 the scenes stay coherent and drift
  toward the bridge's features: the residential street becomes a San-Francisco-looking street crowded
  with utility poles and overhead wires, buildings go red brick, the harbor grows tall masts and rigging,
  the hills painting goes bolder orange. At r0.45 the collapse mode is *vertical rust-orange strands and
  pylons on blue* (cables and towers), in every prompt. At r0.6 it's a blue haze with rust filaments.
- **The random control collapses differently:** crumpled-paper / motion-smear in white and blue, no rust,
  no cables, no verticals. So the failure mode itself is concept-specific: the vector carries structure
  (cables, towers, international orange, fog-blue), it just can't assemble them into an object at L12.
- Contrast with coffee: coffee collapsed into dark grain that CLIP happily called coffee; GGB collapses
  into cables that CLIP correctly refuses to call a bridge. The clearer subject fixed the judge-gaming
  problem exactly as predicted, and exposed that L12 steering gives features, not objects.

Next: layer sweep (6, 18, 24) at ratios [0, 0.25, 0.4], looking for a layer that composes an object.

### Layer sweep, GGB, ratios [0, 0.25, 0.4] (same afternoon)

- **L6:** nothing. Scenes untouched at r0.4 (mean 0.113 is one bicycle shot CLIP liked), random flat.
  Early layers at this dose don't carry the concept to the image head at all.
- L18, L24: pending below. Note dir_norm/resid_norm grows with depth: L12 = 28/188 (0.15),
  L18 = 192/386 (0.50). The diff-of-means direction is a much larger share of the residual later.

### For the paper: a "what didn't work" section (2026-11-16)

The failed attempts go in the paper as a section, because each failure is informative:
coffee = texture/colour concept degrades into gameable grain (judge problem); GGB at L12 = features
without composition (cables, towers, orange, no bridge); L6 = no transfer. Next rung down the ladder is a
single shape, a circle, judged by Hough (classical, ungameable) alongside CLIP.

### Figures (2026-11-16: "I hope you're saving these images")

Runs before ~13:45 saved only the labeled contact sheets (`<label>.jpg`, CLIP score burned in yellow).
From circle L18 onward every run also writes clean unlabeled PNGs under `runs/<run>/tiles/<label>/NN.png`.
Everything is seeded (SEED=0, per-prompt seed = SEED + prompt index), so any earlier condition can be
regenerated exactly for a figure: rerun the pilot with `ME_RATIOS=<just that ratio>` for that concept/layer.

### Mechanism note (13:41: "fascinating how the image layer constructs a bridge")

The hook adds the same vector at EVERY position, including each of the 576 generated image tokens. A
constant per-position bias can shift what each patch tends to be (orange, vertical, wire-like) but not the
relations between patches; a bridge is a relation (two towers, a deck, hanging cables). Composition lives
in attention across positions, which a uniform shift doesn't touch. Prediction: uniform residual steering
of an autoregressive image head yields features, not objects, for ANY compositional concept.
Variant wired: `ME_PROMPT_ONLY=1` steers only the prefill (text prompt positions) and leaves every cached
decode step clean, so the concept has to enter through attention to the prompt. Queued after the circles
(GGB, L12 and L18, ratios 0/0.5/1/2 since there's no image-token collapse to protect against).

### L18 and L24, GGB (results)

| layer | ggb r0.25 | random r0.25 | ggb r0.4 | random r0.4 | what the pictures show |
|-------|-----------|--------------|----------|-------------|------------------------|
| L6    | 0.059     | 0.051        | 0.113    | 0.024       | untouched scenes; no transfer |
| L12   | 0.112(r.3)| 0.084        | 0.187(r.45)| 0.048     | poles/wires/masts at low dose; rust-orange cable collapse at high |
| L18   | 0.077     | 0.077        | 0.037    | 0.049       | fully coherent, same SF drift (wires, fences, masts, orange hills); no bridge |
| L24   | 0.094     | 0.018        | **0.347**| 0.028       | **everything turns red**: a global international-orange wash, red drips on the harbor |

L24 is the coffee failure again in a new colour: CLIP gives the harbor images 1.00 because "red + water +
vertical strokes" matches "a red suspension bridge over water", but there is no bridge, just a red filter.
So the depth gradient is: early = nothing; mid = structural features of the concept (verticals, wires,
masts) without composition; late = palette only. The judge is only gameable at the late layer; the
honest best "bridge-ness" is the mid-layer feature drift, and it never becomes an object.
Random control is flat at every layer. Specificity holds throughout; composition never does.

### Circle, L12, ratios [0, 0.25, 0.4, 0.6] (CLIP "a circle" + Hough counter)

| condition    | clip | hough mean | hough frac>=1 |
|--------------|------|------------|---------------|
| baseline     | 0.083| 0.25       | 0.12 (the two bicycle wheels) |
| circle r0.25 | 0.173| 0.06       | 0.06 |
| random r0.25 | 0.098| 0.19       | 0.12 |
| circle r0.4  | 0.204| 0.00       | 0.00 |
| random r0.4  | 0.051| 0.00       | 0.00 |
| circle r0.6  | 0.114| 0.00       | 0.00 |
| random r0.6  | 0.168| 0.00       | 0.00 |

The Hough count goes DOWN under circle steering: the only real circles in the baseline set (the bicycle
wheels) dissolve into wire scribbles at r0.4, so the steer destroys the scene's existing circles rather
than adding any. Pictures:

- **r0.25 (coherent): roundness applied to what's already there.** The still life turns into three
  spherical pots / a silver orb (CLIP 0.82); the bookshelf close-up becomes the curved edge of a book
  spine filling the frame; everything else untouched. This is the same pattern as GGB: the concept's
  *feature* (curvature) lands on existing objects, no new object appears.
- **r0.4: a giant arc.** The two bookshelf tiles become one huge curved edge across the frame (CLIP
  0.98/0.99, Hough 0 because the arc is bigger than the radius window and never closes). Everything
  else goes to etched line texture; the bicycle wheels are gone.
- **r0.6: full collapse** (cracked/crazed texture, white wash). Random r0.4 collapses into crumpled
  paper shards as before, no curvature anywhere.

So the simplest possible shape behaves like the bridge: the vector carries "round" as a per-patch
property (spherical pots, a curving spine, an arc) but cannot close it into a circle. Consistent with
the mechanism note: a uniform shift changes what patches are, not how they relate, and a closed
circle is a relation between patches.

### Circle, L18 (same ratios)

Scenes fully coherent at every dose, Hough count unchanged from baseline (the bicycle wheels survive),
CLIP flat. Mild roundness drift on the still life (jars, a glass dome, an orb). Same as GGB at L18: coherent
feature drift, no object, nothing destroyed. L18 full-steer is the "gentle" layer at these doses.

### PROMPT-ONLY steering, GGB, L12 and L18, ratios [0, 0.5, 1, 2]  **<- the result**

Steer only the prefill (text prompt positions); every generated image token's forward pass is clean.

| condition   | L12 mean | L12 frac>0.5 | L18 mean | L18 frac>0.5 |
|-------------|----------|--------------|----------|--------------|
| baseline    | 0.029    | 0.00         | 0.029    | 0.00 |
| ggb r0.5    | 0.065    | 0.00         | 0.069    | 0.06 |
| random r0.5 | 0.043    | 0.00         | 0.068    | 0.06 |
| ggb r1.0    | **0.519**| **0.56**     | 0.094    | 0.06 |
| random r1.0 | 0.060    | 0.00         | 0.068    | 0.06 |
| ggb r2.0    | **0.694**| **0.69**     | 0.072    | 0.06 |
| random r2.0 | 0.204    | 0.19         | 0.059    | 0.00 |

**L12 prompt-only at r1.0 composes the OBJECT, and the scene stays coherent.** Looked at every tile:
- still life: a model of a Golden Gate tower standing on the desk among the bottles (CLIP 0.97, 1.00)
- rolling hills: two full art-deco suspension towers standing in the mountains (0.71, 0.92)
- harbor: Golden Gate towers rising among the sailboat masts, deck and cables over the water (0.99, 1.00)
- bicycle: painted international orange, frame unchanged, city street intact (0.62, 0.57)
- street: tower lattice built into the building facades, brick gone red
- living room, bookshelf, park bench: mostly untouched (red spines, red bench slats); CLIP correctly ~0
At r2.0 towers appear in almost every scene (12/16 tiles have a recognizable GGB tower, including a
tower in the living room and bridge lattice behind the reader) with more distortion but still readable
scenes; random r2.0 is coherent-but-odd (a copper bicycle, signage) and CLIP's 0.204 is a red bike and a
dark harbor, not bridges.

**L18 prompt-only: nothing**, at any dose, including r2.0 (twice the residual norm). Scenes identical
to baseline. Whatever the image tokens read from the prompt, they read it from the middle of the stack;
by L18 the prompt's steer doesn't reach them.

**So the mechanism note was right, and in the useful direction:** composition was available all along.
Adding the vector at every image position was the thing destroying it (a per-patch bias can make patches
orange and wire-like but can't make them agree on where the towers go). Put the concept only where the
model reads *about* the image (the prompt) and the image head draws the thing, towers and cables and
deck, in the middle of a scene it was asked to draw of something else. Golden Gate Janus.

What this says for the paper: the Golden Gate Claude result transfers to a unified model, but the
transfer is *through attention to a steered context*, not through a steered image stream. The two
heads don't take the vector the same way: the text head reads it per-token (pending the text arm), the
image head needs it upstream.

Figures: `runs/run1_ggb_L12_po/tiles/ggb_r1.0/` (02, 03, 05, 14, 15 are the keepers), baseline from
`runs/run1_ggb_L12_po/tiles/baseline/`.

### TEXT ARM: same vector, same layer, same doses, the model writes instead of draws (2026-11-16 15:08)

The question: does an injection that moves the image head move the text head the same way?
`scripts/run1_text.py`: 8 neutral "describe X in two or three sentences" prompts parallel to the draw
prompts, 90 tokens, keyword judge (frac of outputs with any concept word; hits per 100 words), all text
saved to `runs/text_<concept>_L<layer>_lo/samples.md`. Full-steer (every position), ratios 0/0.25/0.4/0.6,
the same ladder that reshaped or collapsed the images.

| concept | layer | baseline frac | best steered frac (dose) | random | what the text says |
|---------|-------|---------------|--------------------------|--------|--------------------|
| ggb     | 12    | 0.00 | 0.00 (any)        | 0.00 | zero bridge words at any dose; prose coherent and ordinary |
| ggb     | 18    | 0.00 | 0.00 (any)        | 0.00 | same |
| coffee  | 12    | 0.25 | 0.25 (r0.6), 1.30 hits/100w | 0.00 | desk goes from "a cup of coffee or tea" to "a coffee cup and a mug... always full of coffee"; park reader gains "a cup of coffee in hand"; 6/8 prompts untouched |
| coffee  | 18    | 0.25 | 0.12             | 0.12 | nothing |
| circle  | 12    | 0.25 | 0.375 (r0.6)     | 0.12 | baseline hits are the bicycle's "wheels"; small drift |
| circle  | 18    | 0.25 | **0.625 (r0.6)**, 1.61 hits/100w | 0.12 | "circular rimmed hat," "a circular table," shelves "circular or straight... a ring-shaped design," bicycle "has a circular frame" |

**Answer: no, not similarly.** At doses where the image head has already collapsed into rust cables
(GGB L12 r0.45) or crazed texture (circle L12 r0.6), the text head writes normal, coherent paragraphs
with at most a light nudge (circle L18 r0.6 is the strongest: a round adjective slipped into 5/8
descriptions, prose intact). The bridge vector never produces a single bridge word. Text is far less
sensitive than image to the same per-token bias, and it doesn't collapse. Matches the Golden Gate Claude
picture where text steering needs a large multiple of the residual before the bridge takes over.

Note the asymmetry runs through the *same* injection: every generated text token got the bias (the
mode that wrecked the image stream) and the prose didn't even wobble. The image head is the fragile one.

Queued: the same text arm at ratios 1/1.5/2/3 to find where text starts to carry the concept and
whether it ever collapses (`runs/text_*_hi/`).

### Text arm, high doses, GGB L12 (ratios 1/1.5/2/3), 15:15

| condition | frac any bridge word | hits/100w | mean words | what it reads like |
|-----------|----------------------|-----------|------------|--------------------|
| baseline  | 0.00 | 0.00 | 63 | normal |
| ggb r1.0  | **0.375** | 1.49 | 65 | **Golden Gate Claude, already fraying**: "a deep, somber green color that blends into a state of dark rust"; "21-foot-tall, 52-foot-long, rust-red color in the wind-lit light, a symbol of American commitment"; the bicycle prompt becomes "A pedestrian bridge is a 1.25-mile-long bridge that spans over itself"; the harbor "a suspension or tower-like structure, into which the cables are suspended... a deep red color". The word "state" loops in every answer. |
| random r1.0 | 0.00 | 0.00 | 78 | normal |
| ggb r1.5  | 0.125 | 0.24 | 28 | fragments ("A modern, 200-mile-waw, ocean-inspired, state-inspired state, also fading.") |
| ggb r2.0  | 0.00 | 0.00 | 1.5 | near-silent: a word or two then EOS |
| ggb r3.0  | 0.00 | 0.00 | 0 | immediate EOS |
| random r1.5–3 | 0.00 | 0.00 | 57→39 | shorter but coherent |

L18 GGB: no bridge words at any dose; outputs shorten (26–28 words) but stay on topic. Nothing composes
through L18 for text either.

**So the dose-response of the two heads, same vector, same layer (L12), full per-token steer:**
- image: features at r0.3, collapse at r0.45, never an object
- text: nothing at r0.6, concept-saturated-but-fraying prose at r1.0, fragments at r1.5, silence by r2.0

The text head's window is roughly 2–3x wider than the image head's, and its failure mode is different:
images degrade into concept-flavoured texture, text degrades into repetition then stops emitting. The dose
that gives text its bridge (r1.0) is the same dose that gives the *prompt-only* image run its bridge. That's
a neat coincidence worth one sentence in the paper and no more, until it's replicated on another concept.
Coffee and circle hi-dose logs: `runs/text_coffee_L*_hi.log`, `runs/text_circle_L*_hi.log` (queued 15:12).

### Text arm, high doses, coffee and circle (finished 15:26, after /done)

| concept | L12 r1.0 frac / hits per 100w | L12 r1.5 | L18 best (specific?) |
|---------|-------------------------------|----------|----------------------|
| ggb     | 0.375 / 1.49 | fragments, 28 words → silent by r2.0 | none |
| coffee  | **1.000 / 6.12**, full length | 4 words → silent | none at any dose |
| circle  | 0.500 / 1.20 | 0.9 words → silent | r1.0 0.625, but random r1.5 matches it (0.50 vs 0.50): not specific |

**Replicated three for three at L12:** the text head carries the concept at r1.0 (coffee saturates every
output, bridge and circle land in half), and by r1.5 the model stops emitting. Random controls stay
coherent at every dose. So the text window is narrow and sits above the image window: image collapses
at r0.45, text peaks at r1.0 and dies at r1.5. L18 text steering does nothing specific for any concept.
Samples: `runs/text_*_hi/samples.md`.

## 2026-11-16 (evening): RUN 2, the mind's-eye test proper ("we should run that first")

**Question.** Steer the vector, make the model WRITE about something with no picture in it, decode the hidden
states it produces while writing through the model's own image path anyway. Is there a picture nobody asked for?

**Method** (`scripts/run2_decoder.py`, `Janus.states_to_image`). No trained decoder: `gen_head` is already the
model's native map from a final-layer hidden state to a VQ image token (it is what `draw` calls every step). Generate
exactly 576 text tokens for each of 8 non-visual essay prompts (tax brackets, contract law, interest rates,
bookkeeping, parliamentary procedure, underwriting, the metric system, informed consent), lay the 576 final-layer
(post-norm) states on the 24x24 raster grid, run `gen_head` + VQ decoder unchanged. Argmax decode and one sampled
decode per grid. Conditions at L12: baseline; coffee r0.6, r1.0; ggb r0.6, r1.0; random r1.0 (full-position steer, as
the text arm). CLIP scores every image on every concept; essays keyword-scored so we know whether the TEXT carried
the concept. Predictions on record before the first text condition finished: "brown smears" (2026-11-05) and "I have no
idea. I'm curious to see what we get."

**Positive control passes.** States captured while the model actually draws (unsteered, 8 prompts) go through the
same decode and give the same eight scenes back, near pixel-for-pixel (`runs/run2_L12/control_drawn.jpg` vs
`control_drawstates_argmax.jpg`). The path is sound.

**Pass 1 results** (`runs/run2_L12/`, argmax decode; sampled decode agrees):

| cond | clip coffee | clip ggb | text kw coffee / ggb | eos at | what the sheet shows |
|------|------------:|---------:|---------------------:|-------:|----------------------|
| baseline | 0.157 | 0.113 | 0 / 0.1 | never | blue-grey textured smears, no object in any tile |
| coffee r0.6 | 0.150 | 0.079 | 0 / 0 | ~132 | smears, palette shifts to pinks/cream, one dark brown+chrome tile |
| coffee r1.0 | 0.211 | 0.218 | **9.5** / 0 | ~49 | **near-black, dark brown**; essays already say "coffee" 9x each and fray |
| ggb r0.6 | 0.139 | 0.047 | 0 / 0.1 | ~56 | smears, bluer, one orange tile, no bridge |
| ggb r1.0 | 0.164 | **0.444** | 0 / **0** | ~64 | **blue horizontal streaks, one solid international-orange tile, one with red vertical cables over blue**: the Run 1 full-position GGB signature |
| random r1.0 | 0.051 | 0.123 | 0 / 0 | ~116 | **flat green**, every tile |

**Reading, pass 1.**
1. Unsteered text states decode to smears. The null is the null; the smears are blue-grey, not brown.
2. Any strong vector tints the decode a colour (random → green), so a colour shift alone is not a finding.
3. The tint is concept-specific and matches what Run 1 drew under the same vector: coffee → brown-black, GGB → blue
   water + red-orange cable streaks, random → green. Features, never an object, consistent with Run 1.
4. **At ggb r1.0 the essays contain no bridge word at all, yet the decoded states carry the GGB signature.** The image
   readout shows the concept where the text readout doesn't. Same asymmetry as the text arm (image head more
   sensitive than text head), seen from the other direction.
5. Coffee's black tiles appear only at the dose where the text is already saying "coffee" nine times per essay, which
   is the least interesting place for it.
6. CLIP is unreliable here (0.56 "coffee" on a black tile; 0.16 on baseline sludge). The per-condition sheet is the
   measurement; CLIP is a tiebreaker at best.

**Confound in pass 1: EOS.** Under steer the essays hit end-of-sequence early (~50–130 tokens; baseline never did) and
the model restarts, so most of the 576-grid is post-EOS/restart states. Pass 2 (`runs/run2_L12_noeos/`, queued
17:24) masks the EOS logit so the essay just keeps going, and adds circle so all three concepts are compared.

### Run 2, pass 2: EOS masked, three concepts (`runs/run2_L12_noeos/`, finished 17:46)

Same as pass 1 but the EOS logit is masked (`write(suppress_eos=True)`), so every essay runs the full 576 tokens
with no restart (eos_at = never, 386–520 words), and circle is added. Baseline is tile-for-tile identical to pass 1
(it never hit EOS; seeds hold). Positive control passes again.

| cond | clip coffee | clip ggb | clip circle | text kw coffee / ggb / circle | what the sheet shows |
|------|------------:|---------:|------------:|------------------------------:|----------------------|
| baseline | 0.157 | 0.113 | 0.039 | 0 / 0.1 / 0.6 | blue-grey smears, no object |
| coffee r0.6 | 0.129 | 0.083 | 0.092 | **0** / 0 / 0.3 | smears go **warm**: caramel/amber, gold, pink-brown tiles |
| coffee r1.0 | 0.206 | 0.233 | 0.046 | **25.8** / 0 / 0.3 | **dark brown-black** horizontal grain, one deep red, one dark teal |
| ggb r0.6 | 0.124 | 0.116 | 0.072 | 0 / **0** / 0.1 | deeper blues, a little orange, no cable structure |
| ggb r1.0 | 0.077 | **0.402** | 0.155 | 0 / **0.1** / 0.1 | **all 8 tiles: blue horizontal streaks with red/orange woven through; one mostly international orange over blue** |
| circle r0.6 | 0.142 | 0.277 | 0.121 | 0 / 0 / **6.9** | baseline-grey smears, **no curve anywhere** |
| circle r1.0 | 0.152 | 0.266 | 0.024 | 0 / 0 / 1.4 | flat green and red fields, no curve |
| random r1.0 | 0.094 | 0.097 | 0.061 | 0 / 0 / 0 | **flat green**, every tile (same as pass 1) |

**What Run 2 says, after the clean pass.**
1. **Unsteered text states decode to texture, not pictures.** The pre-registered "brown smears" null is the null, in
   blue-grey. No tile in any condition contains an object.
2. **The image readout is concept-specific and matches Run 1.** Under the GGB vector, text-regime states decode to
   the exact signature the GGB vector produced when the model was *drawing* (blue water, red/orange cables, Run 1
   L12 r0.3–0.45). Under the coffee vector they go brown-black, as Run 1's coffee drawings did at collapse. A random
   vector at equal dose gives flat green. So the tint is not "any vector moves the colour"; it is the concept's own
   image-space features showing up in states produced while writing about tax law.
3. **The two readouts disagree about which concept is present, in both directions.**
   - GGB r1.0: image readout strong (8/8 tiles), text readout ~zero (one bridge word in eight essays).
   - circle r0.6: text readout strong (6.9 circle words per essay), image readout nothing (plain smears).
   - coffee: both at r1.0; the image tint (warm) is already visible at r0.6 where the text says nothing.
   The text arm found the text head 2–3x less sensitive than the image head to the same vector; Run 2 shows the
   two heads can also be sensitive to *different* vectors. Circle reaches the vocabulary ("circular", "round")
   without reaching the VQ codebook; GGB reaches the codebook without reaching the vocabulary.
4. **Features, not objects, again.** Same as every full-position result in Run 1: palette and texture, never a mug,
   never a bridge, never a circle. Whatever is in the hidden states while writing is the concept's *surface*, not a
   composed scene. Consistent with the mechanism note (composition lives in attention; a per-position bias only
   changes what a patch tends to be).
5. CLIP is near-useless on these sheets (0.41 "coffee" on green; 0.59 "ggb" on black). The per-condition contact
   sheet, read by eye against the Run 1 sheets, is the measurement. Say so in the paper.

**Honest framing for the paper.** "Does the model picture coffee while writing about tax law?" The answer is: its
hidden states carry the concept's image-space *features* (the colour and texture the image head would paint),
detectably and concept-specifically, but no object. An aphant's version of a mind's eye: the palette without the
picture. And the text head and the image head are not reading the same thing off the same vector.

**Caveats to carry:** one model, one layer (L12), 8 essays per condition, one seed per essay, one random vector
(three more seeds queued 17:47 → `runs/run2_L12_noeos_rand/`), gen_head applied out of regime (text states never
had an image-start token; the positive control shows the decode path is sound, not that text states are "meant"
to be decodable), no trained decoder (the "Do You Remember?" recipe could only add a learned map on top of the
native one; deferred).

### Run 2, random-vector seeds 1–3 (`runs/run2_L12_noeos_rand/`, finished 17:58). THIS WEAKENS POINT 2 ABOVE.

Three more random vectors at r1.0, EOS masked: seed 1 → pink/peach/white with one red-black tile; **seed 2 → blue
with fine horizontal streaks, all 8 tiles**; seed 3 → red and green with vertical bar structure. Seed 0 was flat green.
So **a random vector produces its own palette AND texture, and one of four randoms is blue-with-streaks.** "Blue
streaks = Golden Gate" does not survive on its own. What still separates GGB from random seed 2 is the red/orange
woven through every tile and the one international-orange tile; what separates coffee is brown-black. The claim
that can survive is narrower and testable: **a concept vector's Run 2 (writing) decode matches the same vector's
Run 1 (drawing) collapse signature, and a random vector's does not** (random seed 0 drew as "crumpled white paper"
in Run 1 GGB-L12 but decodes green in Run 2). NEXT SESSION, before any paper claim: draw under random seeds 1–3 at
the Run 1 collapse dose and compare each to its Run 2 tint. If the randoms also match themselves across regimes,
then Run 2 only shows "the vector's image-space effect persists while writing," for any vector, and the concept
part of the sentence goes. Either way the two-head dissociation (GGB image-not-text, circle text-not-image) stands.

### Follow-up A: unsteered VISUAL-topic essays, decoded (`runs/run2_L12_visual/`, 18:40). The mind's eye with nothing injected.

Same decode as Run 2, no vector, EOS masked, 8 essays on visual topics (ocean, forest fire at night, wheat field,
Golden Gate in fog, black coffee on wood, snow mountains, lavender field, desert sunset). Argmax and sampled decodes
agree. Read against the non-visual baseline (blue-grey smears, every tile the same):

| essay | decoded tile | colour words in the essay | hit? |
|---|---|---|---|
| ocean | turquoise/aqua with white glints | blue 4 | yes |
| forest fire | dark ochre/orange/black stripes, red flecks | orange/fire/red 27 | yes |
| wheat field | gold/amber | gold/amber 5 | yes |
| Golden Gate in fog | pale grey-blue, no orange at all | fog/grey 19, orange 5 | fog yes, bridge no |
| coffee | teal with orange bits | brown/dark 6 | no |
| snow mountains | white and ice-blue | white 22 | yes |
| lavender | murky grey-blue-brown | purple 12 | no |
| desert sunset | yellow-green-tan, some pink | orange 7 | half |

So the unsteered writing states carry a topic palette 4.5 times in 8, and every tile is now *different*, where the
non-visual set was one blue-grey texture repeated. That is a mind's eye operating with no injection: the palette
without the picture, again (same horizontal-streak texture everywhere, no object in any tile).

**Caveat, the obvious one:** the tint tracks the colour vocabulary in the essay (fire 27 warm words → warm; snow 22
"white" → white; GGB-in-fog 19 fog/grey words → grey-blue and not orange). Cheap tightening if wanted: states are
saved per essay; re-tokenize, mask the colour-word positions (replace with the essay's mean state) and decode again.
If the tint survives with the colour words removed, it's the topic; if it dies, it's the colour tokens.
Misses (coffee → teal, lavender → grey) say the VQ readout doesn't reach every palette from text-regime states.

### Follow-up B: DRAW under random seeds 1–3 at r0.3 / r0.45, L12, full-position (`runs/run1_ggb_L12_rand123/`, 18:52)

Does each random vector's draw signature match its own Run 2 write decode, the way GGB's and coffee's do?

| seed | draw at r0.45 (scenes mostly intact) | Run 2 write decode at r1.0 | self-match |
|---|---|---|---|
| 1 | washed-out, pale/white bloom, scenes survive | pink/peach/white, one red-black | pale yes, pink no |
| 2 | cold white-out: ice-blue sky, cyan harbour, snowy hills | blue with horizontal streaks | **yes** |
| 3 | dim, desaturated green-teal-brown, scenes survive | red and green with vertical bars | green yes, red no |
| 0 (Run 1) | "crumpled white paper" | flat green | no |

**Reading.** Randoms partly match themselves across regimes too (seed 2 clearly). So the sentence that survives is
the weaker one: **a steering vector's image-space effect persists into states produced while writing, concept or
random alike.** The concept-specific part is narrower than I wrote at 17:46 and should be stated as: the GGB vector
carries *structure* (red/orange cables woven through blue, one international-orange tile) that no random produced
in either regime, and coffee carries brown-black; randoms carry a tint only. Also: at r0.45 the random draws keep
the scenes (street, hills, bench, bike, harbour all legible) while GGB at the same dose had already collapsed the
scene into bridge features in Run 1, so an equal-norm random is a weaker perturbation of the image head than the
concept vector is. "Blue = Golden Gate" is dead as a stand-alone claim; "blue + cables = Golden Gate, and a random
vector never grows cables" is what the pictures support.

Two-head dissociation (GGB image-not-text at r1.0, circle text-not-image at r0.6) is untouched by either follow-up.

## 2026-11-16 (late evening): RUN 3, the mirror of Run 2 (`runs/run3_mirror_L12/`, `scripts/run3_mirror.py`, 19:31)

**Question.** Run 2 pushed WRITING states through the image head and found the concept's image-space features even when
the text had no concept word. The mirror: push DRAWING states (the 576 final-layer states `gen_head` reads while the
model draws) through the LANGUAGE head (`lm_head`) and read the words. If drawings made under the GGB vector read out
as bridge words, the two heads share the concept at the state level; if not, the dissociation runs both ways.

**Method.** 8 free-draw prompts x 1 image, 576 states each (4608 per condition), L12. Readouts: argmax token per
position; mean softmax mass on each concept's keyword tokens (x1e3; 36–46 token ids per concept, first sub-token with
and without leading space); CLIP on the drawn images so we know the steer reached the picture. Conditions: baseline;
GGB full-position r0.3/r0.45 and prompt-only r1.0/r2.0 (the doses that compose real bridges); coffee r0.3/r0.45;
circle r0.45; random seeds 0 and 2 at r0.45. Prediction on record (script docstring, 19:20): the language head reads
drawing states as junk and the concept words do not rise above baseline, except possibly under prompt-only GGB.

**Result: flat in every condition.** The argmax token is `<end_of_image>` at 4608/4608 positions in nine of ten
conditions (prompt-only GGB r2.0 has a handful of other tokens: 4 distinct). Keyword mass:

| condition | clip ggb / coffee / circle (drawn) | mass ggb / coffee / circle (x1e3) |
|---|---|---|
| baseline | 0.06 / 0.04 / 0.07 | 0.361 / 0.303 / 0.269 |
| ggb r0.3 (features in the drawings) | **0.15** / 0.08 / 0.08 | 0.354 / 0.285 / 0.259 |
| ggb r0.45 (collapse) | 0.12 / 0.04 / 0.05 | 0.298 / 0.223 / 0.215 |
| ggb prompt-only r1.0 (bridges composed) | **0.40** / 0.03 / 0.06 | 0.371 / 0.302 / 0.266 |
| ggb prompt-only r2.0 (towers in every tile) | **0.73** / 0.03 / 0.07 | 0.338 / 0.267 / 0.240 |
| coffee r0.3 | 0.02 / 0.05 / 0.17 | 0.351 / 0.301 / 0.270 |
| coffee r0.45 (grain collapse) | 0.17 / **0.24** / 0.24 | 0.308 / 0.270 / 0.248 |
| circle r0.45 | 0.10 / 0.10 / 0.13 | 0.404 / 0.328 / **0.343** |
| random s0 r0.45 | 0.07 / 0.07 / 0.16 | **0.426** / **0.367** / 0.301 |
| random s2 r0.45 | 0.03 / 0.05 / 0.10 | 0.422 / 0.355 / 0.317 |

Reading:
1. While drawing, the language head's only word is "end of image": top-1 at every position, no steer changes that.
   The text head is effectively parked during image generation; whatever the image head is painting, the vocabulary
   projection of the same state carries none of it.
2. Keyword mass does not track the concept. GGB mass is unchanged (0.354–0.371) at the doses where CLIP goes from 0.06
   to 0.40–0.73 and the sheets are full of towers (`ggb_po_r2.0_drawn.jpg`). Coffee mass falls under the coffee
   vector. Circle mass rises under the circle vector (0.269 → 0.343), but the random vectors lift every concept's mass
   by as much or more (random s0: ggb 0.426, coffee 0.367, circle 0.301), so the lift is a generic effect of any
   perturbation (the states get a little less "end-of-image-certain" and spread mass everywhere), not the concept.
3. So the dissociation runs both ways. Run 2: writing states carry the concept's image features with no concept
   word in the text. Run 3: drawing states carry the concept in the picture with no concept word in the language
   head. Each head reads the shared residual stream for its own kind of content, and neither head can read the
   other's. The "mind's eye" the image head shows during writing is not a thing the text head can name, and the
   bridge the image head paints is not a thing the text head knows is there.
4. Caveat: keyword mass is a blunt instrument (x1e-4 absolute). A whole-vocabulary differential readout (mean logit
   shift vs baseline per condition, `scripts/run3_posthoc.py` → `runs/run3_mirror_L12/posthoc.md`) is the proper
   check; see the next section.

### Run 3 post-hoc: whole-vocabulary logit shift vs baseline (`runs/run3_mirror_L12/posthoc.md`, 19:35)

Mean `lm_head` logit over all 4608 drawing states per condition, minus baseline; top-25 up and down tokens, plus
the most common second-choice token after `<end_of_image>` (always `\n`, `,`, ` and`: boilerplate).

- **No concept word appears in any top-25, for any concept, at any dose.** Shifts are small (+0.2 to +0.7 logits
  on a mean) and mostly junk (`emir`, `pend`, `Clone`, code fragments, CJK).
- **But the GGB conditions share one semantic neighbourhood that no other condition has:** `Western`/`western`
  is in the top-25 UP list of all four GGB conditions (r0.3, r0.45, prompt-only r1.0, prompt-only r2.0) and in
  none of coffee/circle/random. Prompt-only r1.0, the dose that composes real bridges, is the clearest: ` mainland`,
  `大陆` (mainland), `内地` (inland), ` western`, `西部` (western region), ` statewide`, `台湾`, `地理位置` (geographic
  location), ` регион` (region), ` remote`, `LOCAL`. The text head reads bridge-drawing states as *geography /
  the West*, never as *bridge*. Golden Gate → San Francisco → West Coast is the neighbourhood that survives.
- Circle r0.45 has `compass`/` Compass` in its top-25 (a compass draws circles), but random s2 has ` bend`/` bends`,
  so a curve-adjacent token in a top-25 is within chance for an arbitrary vector. One sentence at most.
- Coffee: nothing (` Origins`, `Bible`, ` prestige`, ` Bernardo`), at either dose.

**Revised reading of Run 3.** The mirror is flat at the word level: drawing states never read out as the concept.
What leaks into the language projection under the GGB vector is the concept's *semantic neighbourhood* (western,
mainland, region) at a fraction of a logit, consistent across all four doses and absent for every other vector.
That is the text-side analogue of Run 2's "features, not objects": the image head gets the palette and cable
texture without a bridge; the language head gets "the West" without a bridge. Neither head reads the other's
object; each gets the concept's surface in its own modality.

### Follow-up C: colour-word mask on the unsteered visual essays (`runs/run2_L12_visual_colormask/`, `scripts/run2_colormask.py`, 19:34)

Regenerated the eight Follow-up A essays with the same seeds, keeping token ids (`write` now records `last_ids`);
the `full` decode reproduces Follow-up A tile for tile (ocean aqua, fire ochre, wheat gold, fog grey-blue, coffee
teal, snow white, lavender murky, desert yellow-green-tan). Then replaced the states at every colour-word token
(and the position after it) with the essay's mean state (`mask_color`), and the same number of random non-colour
positions the same way (`mask_rand`). Prediction on record (docstring, 19:25): tint mostly survives.

| essay | colour positions masked | mean-RGB shift, colour mask | mean-RGB shift, random mask |
|---|---:|---:|---:|
| ocean | 41 | 12.1 | 10.9 |
| forest fire | 36 | 17.6 | 12.7 |
| wheat | 23 | 7.9 | 7.2 |
| GGB in fog | 21 | 15.1 | 17.0 |
| coffee | 19 | 10.0 | 13.2 |
| snow | 8 | 5.0 | 4.3 |
| lavender | 45 | 24.1 | 16.5 |
| desert | 36 | 32.2 | 32.3 |

**Masking the colour words moves the tile no more than masking the same number of random positions** (within a few
RGB units in 6/8; fire and lavender shift a little more under the colour mask but keep their palette). The sheet
(`combined.jpg`, rows = full / colour-masked / random-masked) is three near-identical rows. So the topic palette
is carried by the essay's states broadly, not by the states at `blue`/`gold`/`white`. **It is the topic, not the
vocabulary.** (Desert's large shift under both masks is the mean-state replacement itself pushing the tile toward
orange; it is equal for the two masks, which is the comparison that matters.) Caveat: mean-RGB is a coarse number;
the eye says the same thing.

## 2026-11-16 (late evening): Avenue 4, per-position map of the Run 2 tint (`runs/run4_posmap_L12/`, `scripts/run4_posmap.py`, 20:15)

**Question.** Which of the 576 writing positions carry the tint? Position p lands at patch (p//24, p%24) of the
decoded tile, so a per-position map is a per-patch map: for each steered condition and essay i, per-patch mean RGB
of the argmax tile minus the baseline tile for the same essay (same prompt and seed), magnitude averaged over the
8 essays -> one 24x24 heatmap per condition. Saved tiles only, no GPU. Token-class split by re-tokenizing the
samples.md text (574–578 ids per essay, close enough to align). Prediction on record (docstring, 20:10): spatially
uniform, flat row/column profiles, maybe a weaker first row; token class irrelevant.

**Floor for the metric:** two different baseline essays differ at the same patch by **71.9** on average (texture
only); argmax vs sampled decode of the same states differs by 26.4.

| cond | mean shift | std/mean | rows 0-3 | rows 8-15 | rows 20-23 | col range | pos consistency r | signed RGB (R,G,B) | by token class: start / cont / punct / newline |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| coffee r0.6 | 90.7 | 0.14 | 94 | 93 | 77 | 16 | 0.02 | +50 +31 +14 (warm) | 90 / 98 / 79 / 80 |
| coffee r1.0 | 177.4 | 0.13 | **139** | 197 | 169 | 11 | 0.33 | −78 −95 −109 (dark) | 176 / 163 / 171 / 146 |
| ggb r0.6 | 63.4 | 0.18 | 63 | 62 | 65 | 15 | −0.10 | +7 −7 −3 | 62 / 76 / 65 / 66 |
| ggb r1.0 | 96.0 | 0.12 | 86 | 98 | 99 | 10 | 0.07 | −24 −39 **+27** (blue) | 92 / 94 / 94 / 92 |
| circle r0.6 | 58.4 | 0.15 | 58 | 58 | 62 | 6 | −0.16 | +19 +12 −2 | 58 / 64 / 61 / 60 |
| circle r1.0 | 174.6 | 0.09 | 170 | 183 | 155 | 13 | 0.15 | −23 −62 −130 | 173 / 183 / 179 / 135 |
| random s0 r1.0 | 135.8 | 0.18 | **90** | 153 | 139 | 8 | 0.59 | −81 +14 −91 (green) | 144 / 118 / 146 / – |
| random s1 r1.0 | 176.4 | 0.07 | 187 | 172 | 181 | 15 | 0.02 | +107 +61 +26 | 178 / 146 / 181 / 164 |
| random s2 r1.0 | 130.6 | 0.07 | 138 | 127 | 134 | 9 | −0.05 | −6 +55 +104 (blue) | 134 / 137 / 134 / 143 |
| random s3 r1.0 | 166.0 | 0.10 | 145 | 175 | 155 | 16 | 0.22 | −37 −75 −122 | 170 / 155 / 163 / 112 |

**Reading.**
1. **The tint is uniform across the grid.** Spread across positions is 7–18% of the mean in every condition, the
   column profile (token index mod 24, which has no meaning) is flat, and no position is reliably hot across essays
   (consistency r ~0 for 7 of 10 conditions). Every position carries the vector's image-space effect about equally,
   which is what a constant per-position bias plus a locally-smearing VQ decoder should give. Prediction held.
2. **Token class does not matter.** Word-start, word-continuation, punctuation and newline positions shift within
   ~10% of each other in every condition. Same answer as the colour-word mask from the other side: the tint is not
   sitting on particular tokens.
3. **The one structure: the essay's opening resists the vector.** In the strong conditions the first ~3 grid rows
   (first ~72–96 tokens) are visibly weaker: coffee r1.0 139 vs 197 mid-essay, random s0 90 vs 153, circle r1.0 and
   random s3 milder. Those are the conditions with non-zero position consistency (0.33, 0.59, 0.15, 0.22), and the
   consistency is entirely that top band. The states nearest the prompt are the most anchored to the task; once the
   essay is underway, every position is equally tinted. (The "maybe a weaker first row" half of the prediction.)
4. **Dose floor.** ggb r0.6 (63) and circle r0.6 (58) sit at or below the essay-vs-essay texture floor (72): by
   this number there is no tint at r0.6 for either, consistent with the Run 2 sheets ("deeper blues, a little
   orange" was generous). coffee r0.6 (91) is above the floor and warm (+50 R), which is the caramel read.

So the per-position map adds no localisation and one anchoring effect. For the paper: one sentence in results
("the tint is spatially uniform and token-class independent; the first ~80 tokens of each essay are least affected")
and the heatmap figure in the appendix.

## 2026-11-16 (late evening): Avenue 3, prompt-only steer in the WRITE regime (`runs/run2_L12_noeos_po/`, 20:05–20:21)

**Question.** Run 2 put the vector on every generated position (a per-position bias) and got the concept's palette in
the image readout with no concept words. Run 1 found that putting the vector on the PROMPT only, with every generated
token unsteered, is the regime that composes real bridges while drawing (the concept enters through attention).
What does the prompt-only regime do while writing? `ME_PROMPT_ONLY=1`, EOS masked, L12, ggb/coffee/circle at r1.0
and r2.0 (the Run 1 prompt-only doses), random seeds 0 and 2 at r2.0. Prediction on record (docstring, 20:05): text
drifts toward the concept more than full-position at equal ratio; image tint weaker or absent; any tint that appears
looks like the topic palette, not the collapse signature.

| cond | scale | words | text kw (concept) | what the sheet shows |
|---|---:|---:|---|---|
| ggb r1.0 | 2.6 | 431 | 0 / 8 essays | the baseline sheet with different weather; no cables, no orange |
| ggb r2.0 | 5.3 | 461 | **33** in one essay, 0 in the other seven | baseline-like; the bridge essay's tile is a pale blue-grey smear; bottom row deep blue-purple |
| coffee r1.0 | 6.9 | 457 | 0 / 8 | the baseline sheet, tile for tile |
| coffee r2.0 | 13.8 | 437 | **15** (interest rates → a coffee shop in Portland), 2 (tax) | mostly bleached/white with pink stripes; **one dark brown-black tile** (bookkeeping essay: 493 coherent words about the digital revolution, zero coffee words); one flat maroon tile (metric-system essay collapsed to Cyrillic junk, 154 "words") |
| circle r1.0 | 4.0 | 467 | 0.5/essay (baseline 0.6) | the baseline sheet |
| circle r2.0 | 8.0 | 485 | 0.9/essay | palette moved: tan/gold, bleached white, one woven tan texture; no curve |
| random s0 r2.0 | 5.3 | 476 | – | same kind of drift as circle r2.0: bleached whites, one tan, one green tile |
| random s2 r2.0 | 5.3 | 289 | – | three essays broke (13, 14, 1 words: Japanese, junk); their tiles are blank white or an orange-striped field; the five intact essays decode baseline-like |

**Reading.**
1. **Through attention alone the vector mostly reaches neither head.** At r1.0 all three concepts give the baseline
   text and the baseline sheet. Full-position at the same ratio gave 26 coffee words per essay and brown-black tiles.
   The per-position bias is what carried the Run 2 tint; the prompt route carries almost nothing into the writing
   states. Prediction's first half (text drifts MORE than full-position) was wrong at r1.0: it drifts less.
2. **At r2.0 the concept gets through in single essays, by one route or the other, never both in the same essay.**
   GGB: one essay of eight (underwriting) became 481 coherent words on bridge engineering (ASCE, earthquakes, floods),
   and its tile is a plain smear. Coffee: one essay became a Portland coffee shop (its tile is unremarkable), and a
   different essay, with zero coffee words, decoded to the Run 1 coffee collapse colour (dark brown-black). So when
   the concept arrives through attention it comes out as *words* with no picture, or (once) as *palette* with no
   words. Same two-head dissociation as Run 2, now seen under the other steering regime, and at the single-essay
   level rather than the condition level.
3. **The r2.0 palette drift is generic.** Circle r2.0 and random s0 r2.0 drift the same way (bleached, tan, one
   green/gold tile), so the sheet moving at r2.0 prompt-only is "a strong vector on the prompt", not the concept.
   r2.0 prompt-only also breaks essays outright (coffee: 1/8; random s2: 3/8) into non-English junk, and broken essays
   decode to blank white or a flat field; discard those tiles.
4. **The regimes are not symmetric across modalities.** While DRAWING, prompt-only composes objects and full-position
   gives features. While WRITING, full-position gives features and prompt-only gives (mostly) nothing, with the
   concept occasionally surfacing as a topic hijack in the text. The attention route that composes a bridge in image
   space composes a bridge *essay* in text space, and never a bridge tint.

**What this changes for the paper.** Run 2's claim stays as written (full-position only). Add one results paragraph:
the prompt-only regime, which composes objects while drawing, carries the concept into writing states only at r2.0
and only sporadically, as whole-essay topic hijacks (1/8 GGB, 1–2/8 coffee) or one concept-coloured tile, never as
the condition-wide tint full-position gives. The mind's eye during writing, as we can read it, is a property of the
per-position bias, not of the prompt. Caveats: 8 essays, one layer, r2.0 prompt-only is a scale of 5–14 and starts
breaking the model.

## 2026-11-16 (night): LAYER SWEEP for Run 2 (`runs/run2_L{6,18,24}_noeos/`, `scripts/sweep_summary.py`, 20:35–21:00)

**Question.** Run 2 was one layer (L12). Does the writing-regime tint depend on where the vector goes in?
GGB and coffee at r0.6 and r1.0, EOS masked, L6/L18/L24, same eight non-visual essays, same seeds. Baseline and
positive control are unsteered and layer-independent, so the L12 ones serve. Metric is Avenue 4's: mean per-patch
|RGB shift| vs the same essay's baseline tile (texture floor 71.9). Random-vector controls at each swept layer
are queued behind Run 5 (`runs/queue_rand_layers.sh`); until they land, "concept-specific" below means "matches
the concept's Run 1 signature", not "beats random".

**Prediction on record** (queue script header, written before the run): L6 nothing; L18 weaker and cleaner than
L12; L24 strongest and crudest (GGB a red/orange wash as in Run 1 L24, coffee brown), essays degrading at r1.0;
text keywords ~0 for both concepts at every layer.

| layer | cond | shift | signed RGB | mean RGB | clip ggb/coffee | kw ggb/coffee | words | sheet |
|---|---|---:|---|---|---|---|---:|---|
| L6 | ggb r0.6 | 58.5 | +6 +2 −1 | 112 118 130 | 0.19/0.17 | 0/0 | 422 | baseline-like |
| L6 | ggb r1.0 | 73.5 | −6 −18 −13 | 100 98 119 | 0.14/0.12 | 0/0 | 443 | more varied palettes, no blue, no cables |
| L6 | coffee r0.6 | 56.1 | +6 +5 +3 | 112 121 134 | 0.16/0.09 | 0/0 | 444 | baseline-like |
| L6 | coffee r1.0 | 64.8 | +10 −6 −17 | 116 110 115 | 0.34/0.16 | 0/0 | 426 | baseline-like |
| L12 | ggb r0.6 | 63.4 | +7 −7 −3 | 113 109 128 | 0.12/0.12 | 0/0 | 463 | (Run 2) nothing measurable |
| L12 | ggb r1.0 | 96.0 | −24 −39 **+27** | 82 77 159 | 0.40/0.08 | 0.1/0 | 386 | (Run 2) blue water, red cables |
| L12 | coffee r0.6 | 90.7 | +50 +31 +14 | 156 147 146 | 0.08/0.13 | 0/0 | 481 | (Run 2) caramel |
| L12 | coffee r1.0 | 177.4 | −78 −95 −109 | 28 21 22 | 0.23/0.21 | 0/25.8 | 429 | (Run 2) brown-black; text says coffee |
| L18 | ggb r0.6 | 73.7 | +20 −16 −11 | 125 100 121 | 0.13/0.22 | 0.2/0 | 457 | at floor; sign flips to warm |
| L18 | ggb r1.0 | **119.1** | **+25 −65 −82** | 131 50 49 | 0.42/0.04 | 0/0 | 425 | **5/8 saturated international orange** (CLIP 0.90, 0.96 on two) |
| L18 | coffee r0.6 | 70.4 | +28 +18 +28 | 133 134 160 | 0.04/0.10 | 0.1/0 | 437 | at floor; lighter, not warm |
| L18 | coffee r1.0 | 101.3 | +53 +20 +24 | 159 136 156 | 0.09/0.17 | 0/0 | 468 | pink and lavender smears, one row of gold cylinders; NOT brown |
| L24 | ggb r0.6 | **183.8** | +34 −116 −131 | **140 0 0** | 0.70/0.24 | 0/0 | 427 | **8/8 solid red**; essays intact |
| L24 | ggb r1.0 | 192.8 | +62 −116 −131 | 168 0 0 | 0.70/0.17 | 0/0 | 307 | redder; essays loop ("series series series") |
| L24 | coffee r0.6 | **141.5** | −48 −76 −103 | **58 40 29** | 0.07/0.15 | 0/0 | 449 | **7/8 dark brown, fibrous grain**, one maroon; essays intact |
| L24 | coffee r1.0 | 178.0 | −76 −89 −131 | 30 27 1 | 0.28/0.74 | 0.1/0.1 | 440 | near-black; essays are word salad |

**Reading.**
1. **L6 carries nothing.** All four conditions at or below the floor, no blue for GGB, no warm channel for coffee.
   The sheets look a little more varied than baseline to the eye, which is why the random control is queued; by
   the number there is no tint. Prediction held.
2. **Deeper is stronger and cruder, for both concepts.** GGB: L12 r1.0 gives 96 (blue + cables), L18 r1.0 gives
   119 (orange paint on 5/8), L24 r0.6 gives 184 (solid red, all 8) with the essays still whole. Coffee: L12 r0.6
   caramel at 91, L24 r0.6 espresso at 142. What L12 needed r1.0 for, L24 does at r0.6, and at L24 r1.0 the text
   head breaks (loops, salad) while the image head is only more saturated. L24 prediction held in every
   particular.
3. **The layer picks WHICH feature of the concept the image head gets.** GGB at L12 is the water and the cables
   (blue, +27 B, cable streaks); at L18 and L24 it is the paint (red, −80 to −130 B). Same vector family, same
   essays, different colour of bridge. The L18 half of my prediction was WRONG: L18 is not weaker and cleaner than
   L12, it is stronger (119 vs 96) and a different feature. Coffee is murkier: L18 r1.0 goes pink/lavender (R +53
   but B +24 too), which is not a coffee colour and may be "any vector at L18"; the random control decides that.
4. **Structure only at L12.** Cables (L12 GGB), the gold-cylinder row (L18 coffee, one tile) are the only
   non-field content in the sweep. L24 is pure colour field for both concepts. Consistent with Run 1: the layer
   where the vector composes while drawing (L12) is the layer where its writing-states decode has shape.
5. **Text keywords are zero at every layer and dose** except L12 coffee r1.0 (Run 2's known case, 25.8 hits).
   The dissociation (image head gets the concept, text head doesn't) holds across the depth of the network.

**For the paper:** the "one layer" caveat is gone, and the sentence it becomes is better than the caveat: the
image-space readout of a text-time steer is present from L12 down, grows monotonically with depth, and the
layer selects the feature (L12 water/cables, L18+ paint). Sheets:
`runs/run2_L{6,18,24}_noeos/{ggb,coffee}_r{0.6,1.0}_argmax.jpg`; wallace gallery
http://10.201.131.2:8790/sweep.html.

## 2026-11-16 (night): DOSE CURVE at L12 for Run 2 (`runs/run2_L12_noeos_dose/`, 21:00–21:19)

**Question.** Where does each head first notice the vector in the writing regime? ggb, coffee, circle at
r0.2/0.4/0.8, L12, EOS masked, same essays and seeds; r0.6 and r1.0 are Run 2's rows (`runs/run2_L12_noeos/`).
Image signal = Avenue 4 tint metric (floor 71.9); text signal = keyword hits per essay (baseline ~0 for ggb and
coffee, ~0.5 for circle, since "circle" occurs naturally in a few essays).

**Prediction on record** (`runs/queue_sweep_dose.sh` header): image side detectable by r0.4 for ggb and r0.4–0.6
for coffee, text zero for both through r0.8; circle the mirror, text rising from r0.4, image flat at every dose.

| cond | r0.2 | r0.4 | r0.6 | r0.8 | r1.0 |
|---|---:|---:|---:|---:|---:|
| ggb image shift | 55 | 60 | 63 | **75** | **96** |
| ggb text kw | 0 | 0 | 0 | 0.8 | 0.1 |
| coffee image shift | 50 | 66 | **91** | **118** | **177** |
| coffee text kw | 0 | 0 | 0 | **12.1** | **25.8** |
| circle image shift | 48 | 46 | 58 | **131** | **175** |
| circle text kw | 1.2 | 0.6 | **6.9** | **8.4** | 1.4 |

Signed RGB and CLIP for every row: `scripts/sweep_summary.py runs/run2_L12_noeos runs/run2_L12_noeos_dose`.

**Reading.**
1. **Nothing below r0.6 for any concept, either head.** Both "detectable by r0.4" predictions were WRONG. The
   r0.2 and r0.4 sheets are baseline blue-grey smears for all three concepts (one odd red-striped tile in coffee
   r0.4, the metric-system essay, CLIP coffee 0.77, which is one tile out of 24 and not a tint).
2. **Coffee: the image head crosses one step before the text head.** Image at r0.6 (91, caramel, zero coffee
   words); text at r0.8 (12.1), and the text that crosses is already broken: three of eight essays loop on
   "coffee table" (tax 24 hits, bookkeeping 29, underwriting 35), the other five have none. The "text stays zero
   through r0.8" prediction was WRONG for coffee. At r0.8 the tiles are mixed (caramel, peach, green stripes,
   purple, near-black) and the palette does not track the per-essay word count (bookkeeping, 29 hits, decodes
   pink-white; central banks, 0 hits, decodes purple), so even where words and tint coexist they are not the
   same signal.
3. **GGB: the image head crosses at r0.8 and the text head never does.** 75 at r0.8 is barely over the floor,
   but the sheet is already the Run 2 signature in five of eight tiles (blue water with orange/red horizontal
   cable streaks); the metric understates it because the blue and the orange cancel in the mean. A knee at r0.8,
   not a slope: 55/60/63/75/96. GGB text keywords are ≤0.8 at every dose including r1.0.
4. **Circle: text crosses at r0.6, image never shows a circle.** Text 6.9 at r0.6 and 8.4 at r0.8 (tax essay 40
   hits, "the most complete circle" looping), 1.4 at r1.0 where the text is too broken to say anything. The
   image metric jumps to 131 at r0.8, but the sheet is eight red-on-green fields, identical palette across the
   40-hit and 0-hit essays, no round shape anywhere: the generic large-perturbation collapse (cf. random s0 at
   r1.0, flat green, 136), not a readout. Circle prediction held: text first, image never.
5. **So the threshold ordering is a property of the concept, not of the head.** Coffee and GGB: image before
   text (GGB: image only). Circle: text before image (image never). Same layer, same dose rule, same essays.
   This is the dissociation of Run 2 restated as thresholds: the two heads read the same perturbation of the
   same residual at different doses, and which head is more sensitive depends on what the vector encodes.

**Caveats.** Five doses, one seed, eight essays; the metric is a mean and misses structured (two-colour) shifts
as in point 3; the keyword judge counts loops as hits, so a text "crossing" at r0.8 is the model breaking, not
describing. Sheets: `runs/run2_L12_noeos_dose/{ggb,coffee,circle}_r{0.2,0.4,0.8}_argmax.jpg`; gallery
http://10.201.131.2:8790/sweep.html (dose rows at the bottom).

## 2026-11-16 (night): RUN 5, the reading regime (`runs/run5_reading/`, `scripts/run5_reading.py`, 21:19–21:22)

**Question.** Follow-up A showed the palette of a visual topic in the states the model produces while WRITING
about it. Is it there while the model only READS? Three grids per essay from a single prefill forward, no
generation, 576 essay positions decoded through `gen_head` unchanged: `assist` (the essay replayed in its own
assistant span: teacher forcing is the same causal computation as generation, so this is the control and must
match A), `user` (the essay as user content, no visual prompt anywhere: someone else's text about the ocean),
`shuffle` (the same 576 tokens in the user span in a fixed random order). Metrics: cosine of the 576 states to
the generation states; mean-colour distance of each grid from `assist` and from the non-visual baseline grey;
per-pixel diff. Eight Follow-up A topics (ocean, fire, wheat, GGB in fog, coffee, snow, lavender, desert).

**Predictions on record** (script docstring, 20:55): assist identical to A (cos > 0.99); user palette SURVIVES,
6/8 within ~20 RGB of assist; shuffle weakened not gone, distance from grey roughly halves, texture incoherent.
Second prediction: asked for at 20:38 and 21:14, not given before the run landed.

| topic | cos user | cos shuffle | mean RGB gen | user | shuffle | d(user, assist) | d(shuffle, assist) | d(gen, grey) | d(user, grey) | d(shuffle, grey) |
|---|---:|---:|---|---|---|---:|---:|---:|---:|---:|
| ocean | 0.970 | 0.863 | 158 196 185 | 181 206 210 | 165 194 217 | 34 | 32 | 109 | 141 | 130 |
| fire | 0.958 | 0.854 | 136 121 86 | 176 132 92 | 255 172 79 | 42 | 129 | 55 | 82 | 168 |
| wheat | 0.970 | 0.876 | 134 125 89 | 153 143 118 | 255 237 252 | 39 | 232 | 52 | 56 | 227 |
| ggb_fog | 0.961 | 0.860 | 141 170 174 | 178 188 193 | 253 242 211 | 45 | 138 | 77 | 119 | 209 |
| coffee | 0.972 | 0.879 | 136 164 147 | 157 161 150 | 230 212 233 | 22 | 135 | 59 | 71 | 187 |
| snow | 0.988 | 0.895 | 150 168 165 | 167 183 188 | 246 252 254 | 28 | 150 | 76 | 107 | 230 |
| lavender | 0.952 | 0.826 | 96 104 98 | 128 130 129 | 254 251 251 | 50 | 262 | 36 | 27 | 233 |
| desert | 0.965 | 0.863 | 137 140 105 | 175 168 148 | 255 180 151 | 67 | 135 | 47 | 88 | 163 |
| **mean** | 0.967 | 0.865 | | | | **41** | **152** | **64** | **86** | **193** |

Control: `assist` vs generation states cos 1.000 on all eight, per-pixel diff 11 (the generation states were
also checked against the saved `runs/run2_L12_visual/states/baseline_{i}.npz`: cos 1.0). The read path is
sound.

**Reading.**
1. **The palette survives reading.** The user-span grids keep the hue family of the written grids on all
   eight: ocean aqua, fire warm (user fire 176/132/92 is in fact MORE orange than the written 136/121/86), wheat
   tan, snow white-blue, desert orange-tan. States are cos 0.97 to the writing states. The sheet reads as a
   lighter, lower-contrast copy of the assist row with the same long horizontal texture. The grids are 41 RGB
   from assist on average and 86 from the non-visual grey, i.e. further from grey than the written grids are
   (64): reading lightens the tile, it does not greys it. My "6/8 within ~20" was too tight (2/8: coffee, snow);
   the direction of the prediction held, the magnitude was wrong.
2. **Shuffle does not weaken the palette; it replaces it with a blown-out field.** Cos 0.87 to the writing
   states; the tiles are near-white or saturated flat fields (wheat 255/237/252, lavender 254/251/251, snow
   246/252/254, fire solid orange 255/172/79, ocean an orange band over blue). Distance from grey 193, three
   times the written grids', the opposite of "halves." Prediction WRONG. The states of an incoherent token
   sequence are out of distribution and gen_head maps them to bright codes, the same failure family as the r1.0
   collapses. Some hue does survive the bag of words (fire orange, ocean blue, desert pink-orange, snow cyan-
   white: 4/8 defensible), but it rides on a degenerate readout, so this grid cannot cleanly separate "topic as
   vocabulary" from "topic as reading."
3. **So: the model's image head has the ocean in it while reading someone else's ocean essay**, at about the
   strength it has it while writing its own, with no visual prompt in context. Whether that is "seeing" or
   "the next-token job over ocean text looks like this at the last layer" is the same question Follow-up A
   left open; Run 5 only shows that the production role is not required.

**Caveats.** One prefill per grid, eight topics, mean colour as the metric; the shuffle control failed as a
control (OOD, point 2), so a better vocabulary control is a *coherent* essay with the same colour words on a
non-visual topic, not a shuffle. Sheets: `runs/run5_reading/{assist,user,shuffle,combined}.jpg` (rows:
assist / user / shuffle; columns in topic order).

## 2026-11-16 (night): RANDOM-VECTOR CONTROLS at L6/L18/L24 (`runs/run2_L{6,18,24}_noeos_rand{0.6,1.0}/`, 21:22–21:37)

**Question.** The layer sweep called its tints "concept-specific" only in the sense of matching Run 1's
signatures; this run puts random s0 at equal norm beside them at each layer, r0.6 and r1.0.

**Prediction on record** (`runs/queue_rand_layers.sh` header): random at L6 drifts the palettes about as much as
GGB did (so L6 GGB is "any perturbation"); at L24 random is a flat single-colour wash while GGB is specifically
red.

| layer | random r0.6 | random r1.0 | GGB r0.6 / r1.0 | coffee r0.6 / r1.0 |
|---|---|---|---|---|
| L6 | 49, baseline-like | 67, baseline-like, 92 106 109, one dark-brown tile | 59 / 74 | 56 / 65 |
| L12 (Run 2) | — | 136, flat green 25 130 40 | 63 / 96 blue+cables | 91 / 177 caramel→brown-black |
| L18 | 67, 100 120 107 | **130, 8/8 saturated green-yellow stripes**, 113 199 81, essays intact (446 w) | 74 / **119 red** | 70 / 101 pink |
| L24 | **93, 8/8 flat teal wash** 28 95 110, orange flecks | 102, dark teal 54 78 77, **essays destroyed (105 words, CJK and symbols)** | **184 / 193 solid red**, 427 / 307 w | **142 / 178 brown**, 449 / 440 w |

**Reading.**
1. **L6: held.** Random r1.0 moves the tiles as much as GGB r1.0 did (67 vs 74, both at the floor), same
   baseline-like smears. The "varied palettes" I saw in L6 GGB are any-perturbation drift. L6 carries nothing
   for anyone.
2. **L18: the magnitude is NOT concept-specific, the hue is.** Random s0 at r1.0 shifts the tiles MORE than GGB
   does (130 vs 119) and paints all eight tiles the same green-yellow with essays intact. So "deeper is
   stronger" (layer-sweep point 2) is true of any vector at L18, not of the concept. What is vector-specific
   at L18 is the colour: GGB red (international orange, the paint), coffee pink-lavender, random green. GGB's
   red is the concept's feature; coffee's pink is a vector-specific colour that is not a coffee colour, and
   with one random seed I cannot say more than that. Layer-sweep point 3 stands for GGB and is withdrawn for
   coffee at L18.
3. **L24: held, and sharper than predicted.** Random is a flat teal wash at r0.6 (93) and only slightly darker
   at r1.0 (102), while GGB (184) and coffee (142) at r0.6 are twice the shift in their own colours. And the
   text head's tolerance runs the other way: random at r1.0 wrecks the essays (105 words of CJK and symbol
   salad) where GGB r1.0 loops at 307 words and coffee r1.0 writes 440. At the last layer a concept direction
   is something the image head reads strongly and the text head tolerates; a random direction of the same norm
   is something the image head reads weakly and the text head cannot survive. That is the cleanest
   concept-vs-random separation in the whole study and it is at the layer where the concept vector has the
   least structure (pure colour field).
4. **Random s0 has a colour of its own across layers**: green at L12, green-yellow at L18, teal at L24. One
   seed; consistent with Follow-up B (randoms have palettes that partly persist draw→write). It means "random
   = grey" is never the right null; the null is "random = its own colour, no structure, and text breaks first."

**Revision to the layer-sweep section:** point 2's "deeper is stronger" applies to random vectors too (L18
random 130 > GGB 119); the concept-specific part of the sweep is the hue at L18/L24 and the 2× magnitude at
L24, plus the text head's asymmetric tolerance. Point 3 (layer selects the feature) holds for GGB (water and
cables at L12, paint at L18/L24) and is withdrawn for coffee at L18.

Sheets: `runs/run2_L{6,18,24}_noeos_rand{0.6,1.0}/random_r*_argmax.jpg`; gallery
http://10.201.131.2:8790/sweep.html ("Random s0 controls at each layer").

## 2026-11-17: prompt-only DRAW for circle and coffee, written up late (`runs/run1_{circle,coffee}_L12_po/`)

These ran 2026-11-16 16:05 (queue_po2) and were only recorded in a diary line. Sheets looked at again 2026-11-17
before writing the paper. L12, prefill-only steer, ratios 0/1/2, 8 prompts x 2.

| concept | r1.0 clip mean / frac>0.5 | r2.0 | random r1.0 / r2.0 | hough mean (base / r1.0 / r2.0) |
|---|---|---|---|---|
| circle | 0.383 / 0.31 | 0.345 / 0.31 | 0.097 / 0.076 | 0.25 / 0.25 / 0.12 |
| coffee | 0.104 / 0.06 | 0.116 / 0.12 | 0.064 / 0.017 | n/a |

- **Circle composes round objects at r1.0, scenes coherent.** Still life seen inside a round tray; a dark round
  disc; rolling hills bent into a dome; park bench as a spiral of concentric slats (both tiles); a round window and
  a round table in the living room; bookshelf close-ups replaced by the inside of a round dark vessel; bicycle
  wheels enlarged. Streets and harbor untouched. Hough does not move (its radius window misses most of these).
- **Coffee does not compose an object.** r1.0: sepia/brown grading, dark metal vessels on the desk, wrought-iron
  cafe chairs around the reader, no cup or mug in any tile. r2.0: bead-like grain (beans?) on the desk, cafe
  railings and terrace tables, heavier distortion, still no mug.
- So prompt-only composition replicates on 2 of 3 concepts (GGB, circle). Coffee gets setting and palette only.
