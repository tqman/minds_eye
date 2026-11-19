# Features, not objects: an aphantic mind's eye in a unified multimodal model

Brett A. Thomas

## Abstract

Unified multimodal models generate text and images from one residual stream through two output heads. We ask whether a concept placed in that stream reaches both heads alike, and whether the image head registers anything while the model writes. In Janus-Pro-7B we steer with difference-of-means directions for three concepts (the Golden Gate Bridge, coffee, a circle). Added at every position, a direction gives the image head the concept's features and never the object: cable-like rust streaks on blue, brown-black grain, curvature on things already in the scene. Added to the prompt alone, so that it reaches the image tokens only through attention, the same direction composes recognizable bridge towers inside otherwise coherent scenes; this holds for two of the three concepts. We then decode hidden states produced during text generation through the model's own image head, with no trained probe. Essays on tax law written under the bridge direction contain no bridge words, yet their states decode to the bridge's palette and cable texture; the circle direction does the reverse, filling the text with "circular" while the image readout shows nothing. Read the other way, drawing states never yield the concept's name through the language head. With no steering at all, essays on visual topics decode to the topic's palette, during writing and during reading alike, and the effect survives masking the color words. Random directions carry palettes of their own, so we limit concept-specific claims to structure and hue. The model's mind's eye, as far as its own image head can read it, is aphantic: palette and texture without a picture.

## 1. Introduction

A unified multimodal model such as Janus-Pro (Chen et al., 2025) writes and draws from one residual stream. Text tokens and image tokens pass through the same transformer blocks, and two output heads read the final state: a language head over words and an image head over the codebook of a VQ image tokenizer. Golden Gate Claude (Templeton et al., 2024) showed that a direction in a language model's stream, pushed hard enough, takes over the one head that model has: every answer becomes about the bridge. A model with two heads lets us ask what happens if we put a concept into the stream and ask each head what it got.

We ask three questions in Janus-Pro-7B, steering with difference-of-means directions for the Golden Gate Bridge, coffee and a circle, and reading the results off contact sheets by eye.

First, what does the image head draw when a direction is added? Added at every position, a direction gives the image head the concept's features and never the object: rust-orange cables on blue, brown-black grain, curvature on objects already in the scene, at every layer and dose, with an equal-norm random direction collapsing the scenes into a different material. Added to the prompt alone, so that it reaches the image tokens only through attention, the same direction at the same layer composes bridge towers and round objects inside otherwise coherent scenes.

Second, do the two heads read a direction the same way? They do not. The language head's dose window is two to three times wider than the image head's and fails into silence rather than texture. Which head registers a direction first depends on the direction. And when we decode the states the model produces while writing about tax law through its own image head, with no trained probe, the bridge direction paints the bridge's palette and cables into states whose text contains no bridge word, while the circle direction fills the text with "circular" and paints nothing. The mirror holds too: drawing states full of towers give the language head "the West" and never a bridge.

Third, with nothing injected, does the image head see anything while the model writes? Essays on visual topics decode to the topic's palette (turquoise for the ocean, gold for wheat, ochre and black for a fire at night), the effect survives masking the color words, and it is present while the model only reads the essay. The decode never contains an object.

Random directions carry palettes of their own, so we limit every concept-specific claim to structure and hue measured against a random direction of the same norm (Section 7), and we give the failed experiments their own section (Section 8), since several of them set the floors and controls the results depend on. The model's mind's eye, as far as its own image head can read it, is aphantic: palette and texture without a picture.

## 2. Related work

**Unified models.** Janus (Wu et al., 2024) and Janus-Pro (Chen et al., 2025) decouple visual encoding for understanding and generation while keeping one autoregressive transformer for both text and image tokens; the image tokenizer and the autoregressive image head follow LlamaGen (Sun et al., 2024). We use Janus-Pro-7B unchanged and treat its two heads as two readers of one stream.

**Activation steering.** Adding a fixed direction to the residual stream to change behavior goes back at least to activation addition (Turner et al., 2023) and contrastive activation addition (Panickssery et al., 2023), which builds the direction as a difference of means over contrasting prompts, the recipe we use. Golden Gate Claude (Templeton et al., 2024) is the result ours is named after: a sparse-autoencoder feature for the bridge in Claude 3 Sonnet, clamped to ten times its maximum activation, made the model's every answer about the bridge. That model had one output head; the steering method also differs from ours (an SAE feature clamped, not a difference-of-means direction added), so the comparison in Section 5 is to the shape of the result, not the method.

**Reading intermediate states with the model's own heads.** The logit lens (nostalgebraist, 2020) applies the final unembedding to intermediate states and reads the resulting distributions. Our native decode is the same move at the final layer with the other head: writing states through the image head, drawing states through the language head, no probe trained.

**Vision-language models without an image.** Mirage Probes (Ben-Levi et al., 2026) finds that vision-language models often answer image questions correctly with no image supplied, and that this can be linearly decoded from their activations; the authors distinguish language priors from "spurious images," false visual content constructed internally, and detect it with probes rather than decoding it. Our Section 6 is the nearest thing to decoding such content, in a model that happens to own a decoder for it. Line of Sight (Rajaram et al., 2025) finds linearly decodable, causal ImageNet-class features in a VLLM's residual stream, with text and image representations largely disjoint early and converging deeper; What's in the Image? (Kaduri et al., 2024) finds that query-text tokens accumulate global image information and that cross-modal flow is concentrated in the middle quarter of the layers. Both work with a real image on the input side; our depth results (L12 composes and carries structure, L6 carries nothing, L24 carries only color) are consistent with a middle band where the two modalities meet.

**Visual states without pixels, and the limits of decoding them.** Machine Mental Imagery (Yang et al., 2025) trains latent visual tokens into a model's reasoning chain so that it can carry visual steps without emitting pixels; the imagery is engineered in, where we ask what is already there. Do You Remember? (Yu et al., 2026) finds that a multimodal LLM's hidden states retain almost no recoverable visual information about an image it was shown, even with a trained diffusion decoder, and builds an external memory to fix it. That is a contrast to our result rather than a method we could borrow: we recover palette and texture, not a picture, from states produced with no image in context at all, through a decoder the model already has.

**Aphantasia.** Zeman, Dewar and Della Sala (2015) named aphantasia for a lifelong reduction or absence of voluntary visual imagery, in people who otherwise know what things look like. We use the word for the result in Section 6 and 9 as an analogy only.

## 3. Setup

**Model.** Janus-Pro-7B (Chen et al., 2025) is a decoder-only transformer of 30 blocks with a 4096-dimensional residual stream, read by two output heads: a language head over the text vocabulary, and an image head (`gen_head`) over the 16,384-entry codebook of a VQ image tokenizer in the LlamaGen family (Sun et al., 2024). An image is 576 codebook tokens on a 24×24 grid, decoded to 384×384 pixels. We use the released weights unchanged, in bfloat16, on one 24 GB GPU.

**Directions.** For each concept we wrote 16 sentences that mention it and 16 matched control sentences of the same register and length that do not ("She cradled a steaming mug of black coffee in both hands" / "She cradled a worn paperback in both hands"). Each sentence is wrapped in the model's chat template as a user turn, and we record the residual-stream state at the final token of the empty assistant header, at the output of block L. The direction is the mean over concept sentences minus the mean over controls. The three concepts are the Golden Gate Bridge, coffee, and a circle, the last built from sentences about round everyday objects rather than geometry.

**Steering and dose.** A direction is added as a fixed vector to the output of block L (blocks are zero-indexed; we report L6, L12, L18 and L24). Because a direction's raw norm is small against the stream (at L12 the coffee direction has norm 28 and the mean residual norm is 188), we state dose as a ratio r: the added vector is rescaled so its norm is r times the mean residual norm at that layer. Under this rule r is comparable across concepts and layers.

**Two regimes.** In the *every-position* regime the vector is added at every position, prompt and generated tokens alike, on every forward pass, so each generated state carries the same bias. In the *prompt-only* regime the vector is added only at the prompt positions during prefill; generated tokens are unsteered, and the concept can reach them only through attention to the prompt.

**Drawing.** Eight free-draw prompts name scenes unrelated to any concept (a quiet residential street, a still life on a desk, rolling hills, a person reading on a park bench, an empty living room, a bookshelf, a bicycle on a sidewalk, a harbor with boats). The prompt is wrapped in the chat template and followed by the image-start tag; 576 image tokens are sampled at temperature 1.0 with classifier-free guidance of 5.0. The steer is applied to both the conditional and unconditional guidance rows, so a shift common to both passes through once rather than being amplified by the guidance. Two images per prompt per condition.

**Writing.** Eight essay prompts name topics with no picture in them (tax brackets, contract law, interest rates, bookkeeping, parliamentary procedure, underwriting, the metric system, informed consent). The model writes exactly 576 tokens at temperature 0.7 with the end-of-sequence logit masked, so every grid position holds a state produced mid-essay. An early pass without the mask ended steered essays after 50 to 130 tokens and restarted them, filling most of the grid with post-restart states (Section 8).

**Native decode.** The final-layer, post-norm hidden state at each of the 576 generated text positions is passed through `gen_head` exactly as it is during drawing. We take the argmax codebook entry at each position (and, separately, one sampled entry), lay the 576 entries on the 24×24 raster, and run the VQ decoder. Nothing is trained and no probe is fit; the readout is the model's own. As a positive control, states captured while the model draws the eight free-draw prompts unsteered, sent through the same path, return the eight drawn scenes near pixel-for-pixel.

**Random directions.** Every concept condition is paired with a Gaussian vector rescaled to the concept direction's norm and applied at the same layer and ratio. Where we report more than one random direction, they differ by seed.

**Measurement.** The unit of measurement is a contact sheet of every image in a condition, read by eye; the sheets are the data and all of them appear in the appendix. We also report a CLIP ViT-L/14 (Radford et al., 2021) zero-shot score for each image against the concept name, and for the circle a Hough circle count, as tiebreakers only; Section 8 shows where CLIP misleads. Essays are scored by counting concept words. Seeds are fixed (a base seed plus the prompt index): two images per prompt per drawing condition, one essay per prompt per writing condition, so each cell of a sheet is a single sample and no condition is averaged over seeds.

## 4. Drawing: features, not objects

We begin with the regime closest to Golden Gate Claude: steer the stream and ask the image head what it draws. All three concepts were added in the every-position regime at L12 while the model drew the eight free-draw prompts, the Golden Gate Bridge direction was also swept across L6, L18 and L24, and the circle direction across L18.

**Every position: the concept arrives as features.** At no layer and no dose does a steered scene gain the object. What it gains is the object's parts, applied to whatever is already there. Under the bridge direction at L12 (r = 0.3) the residential street fills with utility poles and overhead wires and its buildings go red brick, the harbor grows tall masts and rigging, and the hills painting turns a bolder orange, all with the scene intact. Raise the dose (r = 0.45) and every prompt collapses into the same material: vertical rust-orange strands and pylons on blue, cables and towers with nothing to hang them from. At r = 0.6 it is blue haze with rust filaments. The coffee direction does the same with its own material: at r = 0.6 and above the scene dissolves into dark, grainy brown-black speckle, a coffee surface with no cup in it. The circle direction, chosen as the simplest concept we could think of, behaves the same way. At r = 0.25 roundness lands on existing objects (the still life becomes three spherical pots and a silver orb; the bookshelf close-up becomes the curved edge of one spine filling the frame); at r = 0.4 the two bookshelf tiles become a single arc larger than the frame, and the only true circles in the baseline set, the bicycle's wheels, dissolve into wire scribble. The Hough circle count falls under circle steering, from 0.25 per image at baseline to zero at r = 0.4: the direction destroys the scene's circles rather than adding one.

**The failure is concept-specific.** An equal-norm random direction at the same layer and dose collapses the scenes too, but into a different material: crumpled-paper and motion-smear textures in white and blue, with no rust, no cables, no verticals and no curvature. At r = 0.45 the random draws also keep their scenes legible (street, hills, bench, bicycle, harbor all readable) where the bridge direction had already dissolved them, so an equal-norm random vector is a weaker perturbation of the image head than the concept direction is. The direction carries structure, cables and towers and international orange or curvature, and cannot assemble the structure into a thing.

**Depth selects the feature.** At L6 nothing transfers: scenes are untouched at r = 0.4 and the random control is flat. At L18 the scenes stay fully coherent at every dose we tried and drift gently toward the concept (the same San Francisco look of wires, fences and orange hills under the bridge direction; jars, a glass dome and an orb under the circle), with no object and nothing destroyed; the baseline bicycle wheels survive. At L24 the bridge direction becomes a palette: at r = 0.4 everything turns international orange, with red drips over the harbor. So the gradient with depth runs from nothing, through structural features without composition, to color alone. (The CLIP score of 0.35 at L24 against 0.19 at L12 is the judge rewarding red water and vertical strokes, and is the first of the cases in Section 8 where the number and the picture disagree.)

**Prompt only: the object appears.** The every-position regime adds the same vector to each of the 576 image positions. A constant per-position bias can shift what each patch tends to be (orange, vertical, wire-like) but says nothing about the relations between patches, and a bridge is a relation: two towers, a deck, cables between them. That reasoning predicted that steering only the prompt, so the concept reaches the image tokens through attention, would compose where the uniform bias could not. It did. With the bridge direction added at L12 to the prompt positions alone (r = 1.0), the generated scenes stay coherent and contain the bridge: a model of a Golden Gate tower standing among the bottles of the still life, two full art-deco suspension towers in the rolling hills, towers rising among the sailboat masts in the harbor with deck and cables over the water, a bicycle painted international orange on an unchanged street, tower lattice built into the facades of the residential street. Three prompts (living room, bookshelf, park bench) are mostly untouched apart from red spines and red bench slats. At r = 2.0, twelve of sixteen tiles hold a recognizable tower, including one in the living room and lattice behind the reader on the bench, with more distortion but readable scenes. The random control at r = 2.0 is coherent but odd (a copper bicycle, stray signage) and composes nothing. CLIP agrees for once: mean bridge score 0.52 at r = 1.0 and 0.69 at r = 2.0, against 0.03 at baseline and 0.06 for the random direction.

The same regime at L18 does nothing at any dose up to r = 2.0, twice the residual norm; the scenes are identical to baseline. Whatever the image tokens read from the prompt, they read it from the middle of the stack.

**Two of three concepts compose.** The circle direction, prompt-only at L12 (r = 1.0), produces round objects in coherent scenes: the still life seen inside a round tray, the hills bent into a dome, the park bench as a spiral of concentric slats, a round window and a round table in the living room, the bookshelf close-up replaced by the inside of a round dark vessel, the bicycle's wheels enlarged. Coffee does not. At r = 1.0 it gives a sepia grading, dark metal vessels on the desk and wrought-iron cafe chairs around the reader; at r = 2.0 bead-like grain on the desk and cafe railings and terrace tables; in no tile is there a cup or a mug. Coffee's direction, built from sentences about cafes and mugs and roasting, seems to carry the setting and the palette rather than a single object to place.

**What we claim.** The same direction at the same layer gives the image head features when it is added at every position and the object when it is added to the prompt alone. We read this as the uniform bias overwriting, at each patch, the very relations that attention to the prompt would otherwise build, but we have not tested that mechanism directly, and Section 9 states the claim with its hedges. What the pictures show without inference is that composition was available all along, and that adding the vector where the image is made is what destroyed it.

## 5. Writing: the two heads read different things

Section 4 steered the stream while the image head was active. Here the language head is active instead: the same directions, at the same layer (L12) and in the every-position regime, while the model writes. We ask two things. First, does a dose that moves the image head move the language head the same way? Second, while the model writes under a direction, what does its own image head make of the states it is producing?

**The language head's dose window sits above the image head's.** The first text experiment used eight short description prompts parallel to the free-draw prompts ("describe a quiet residential street in two or three sentences"), 90 tokens each, scored by the fraction of outputs containing any concept word. At the doses that had already reshaped or collapsed the images (r = 0.25 to 0.6) the prose is normal, coherent and almost untouched. The bridge direction produces no bridge word at any of these doses at L12 or L18. Coffee at L12 nudges two of eight descriptions (a desk "always full of coffee," a park reader with "a cup of coffee in hand"). Circle at L18, r = 0.6, is the strongest: a round adjective slips into five of eight descriptions ("a circular rimmed hat," "a circular table," a bicycle that "has a circular frame") with the prose intact. The injection is the same one that wrecked the image stream, applied to every generated token, and the text does not wobble.

Raising the dose finds the language head's window. At L12, r = 1.0, all three concepts land: coffee saturates every output (6.1 concept words per 100), the bridge reaches three outputs of eight and the circle four. The bridge outputs at this dose read like Golden Gate Claude already fraying: "a deep, somber green color that blends into a state of dark rust"; the bicycle prompt becomes "a pedestrian bridge is a 1.25-mile-long bridge that spans over itself"; the harbor gains "a suspension or tower-like structure, into which the cables are suspended... a deep red color," and the word "state" loops through every answer. By r = 1.5 the outputs are fragments, and by r = 2.0 the model emits a word or two and stops. Random directions at the same doses leave the prose coherent, only somewhat shorter. At L18 no concept word rises above the random control at any dose. So for the same direction at the same layer, the image head gives features at r = 0.3 and collapses at r = 0.45; the language head gives nothing at r = 0.6, concept-saturated prose at r = 1.0, fragments at r = 1.5 and silence at r = 2.0. The text window is two to three times wider than the image window and fails differently: images degrade into concept-flavored texture, text degrades into repetition and then stops.

**Reading writing states through the image head.** The main experiment of this section decodes the states produced during writing through `gen_head` and the VQ decoder, as described in Section 3. The model writes 576 tokens on each of the eight non-visual topics; the 576 final-layer states are laid on the image grid and decoded. The positive control (drawing states through the same path return the drawing) passes.

Unsteered, the essays decode to smears. Every tile is a blue-gray textured field; no tile in any condition of this experiment contains an object. That is the null, and it is the null we recorded before running: no picture when nobody asked for one.

Under the bridge direction at r = 1.0, all eight tiles change to the same thing: blue horizontal streaks with red and orange woven through, one tile mostly international orange over blue. This is the signature the same direction painted in Section 4 when the model was drawing (blue water, rust-orange cables, L12 at r = 0.3 to 0.45). The eight essays that produced these states contain, between them, one bridge word. They are about tax brackets and underwriting and the metric system, and their hidden states carry the bridge's palette and cable texture. Under the coffee direction at r = 0.6 the smears go warm (caramel, amber, gold, pink-brown) while the essays contain no coffee word; at r = 1.0 they go dark brown-black with horizontal grain, as the coffee drawings did at collapse, and by then the essays say "coffee" about 26 times each. Under the circle direction the pattern reverses. At r = 0.6 the essays carry 6.9 circle words each (the tax essay loops on "the most complete circle") while the decoded states are baseline-gray smears with no curve anywhere; at r = 1.0 they are flat green and red fields, still with no curve. A random direction at r = 1.0 gives flat green in every tile, so a color shift alone proves nothing (Section 7 takes this up); what distinguishes the bridge condition is structure, red and orange cables woven through blue, which no random direction produced in either regime.

**Thresholds by concept.** A dose curve at r = 0.2, 0.4, 0.6, 0.8 and 1.0 locates where each head first registers each direction. The image signal is a mean per-patch color shift of the decoded grid from baseline, whose floor is set by the shift between two different unsteered essays (71.9); the text signal is concept words per essay. Below r = 0.6 neither head registers any concept; the r = 0.2 and 0.4 sheets are baseline smears for all three. Coffee crosses on the image side first (r = 0.6, caramel, zero coffee words) and on the text side at r = 0.8, where the text that crosses is already broken (three of eight essays loop on "coffee table" while the other five have none, and the per-essay tint does not track the per-essay word count). The bridge crosses on the image side at r = 0.8, a knee rather than a slope (the metric reads 55, 60, 63, 75, 96, and understates a two-color signature whose blue and orange cancel in a mean; the sheet at r = 0.8 already shows the cable pattern in five of eight tiles), and on the text side never exceeds 0.8 bridge words per essay at any dose, against 12 to 26 coffee words where coffee crosses. The circle crosses on the text side at r = 0.6 and never produces a round shape on the image side; the metric's jump at r = 0.8 is eight red-on-green fields, identical across the 40-hit and 0-hit essays, the same large-perturbation collapse a random direction gives. Which head notices a direction first is a property of the direction, not of the head.

**The mirror.** If writing states carry image features the text cannot name, do drawing states carry words the image does not show? We took the 576 final-layer states the image head reads while drawing each free-draw prompt (4,608 states per condition) and passed them through the language head (the logit lens applied at the final layer; nostalgebraist, 2020). Conditions: baseline; the bridge direction every-position at r = 0.3 and 0.45 and prompt-only at r = 1.0 and 2.0, the doses that compose real bridges in Section 4; coffee at r = 0.3 and 0.45; circle at r = 0.45; two random directions at r = 0.45. The language head's top token is `<end_of_image>` at 4,608 of 4,608 positions in nine of ten conditions, and softmax mass on each concept's keyword tokens does not track the concept: bridge mass is unchanged across conditions where CLIP rises from 0.06 to 0.73 and the sheets fill with towers, and the small lift in circle mass under the circle direction (0.27 to 0.34) is the size of the lift the random directions give every concept's mass (bridge 0.36 to 0.43, coffee 0.30 to 0.37), the generic effect of any perturbation making the states a little less certain of "end of image." During image generation the language head is parked, and no steer unparks it.

A whole-vocabulary readout (mean logit shift from baseline over all 4,608 states, top 25 tokens up) finds no concept word in any condition. It does find one neighborhood. "Western" is in the top 25 of all four bridge conditions and of no other condition; at prompt-only r = 1.0, the dose that composes bridges, the list is "mainland," 大陆 (mainland), 内地 (inland), "western," 西部 (the western region), "statewide," 台湾, 地理位置 (geographic location), регион, "remote." The language head reads bridge-drawing states as geography, the West Coast, and never as a bridge. The shifts are a fraction of a logit on a mean. We report them as the text-side counterpart of features without objects, and no more: a curve-adjacent token ("compass") appears in the circle condition's top 25, but one random direction has "bend" and "bends" in its own, so a single suggestive token is within chance.

**What this section shows.** The two heads read the shared residual stream for different things, and neither reads the other's. Writing states under the bridge direction decode to the bridge's palette and cables while the text has no bridge word; writing states under the circle direction fill the text with "circular" while the image readout shows nothing; drawing states full of towers give the language head nothing but "the West." Each head gets the concept's surface in its own modality.

## 6. No steering at all

Everything so far put a direction into the stream. The question the readout was built for is what the image head sees when nothing is injected and the model simply writes about something visible.

**Visual topics decode to their palettes.** We wrote eight essays, unsteered, on visual topics (the ocean, a forest fire at night, a wheat field, the Golden Gate Bridge in fog, black coffee on wood, snow mountains, a lavender field, a desert sunset) and decoded their states as in Section 5. Against the non-visual baseline, where every tile is the same blue-gray texture, every tile is now different. The ocean essay decodes to turquoise and aqua with white glints; the forest fire to dark ochre, orange and black stripes with red flecks; the wheat field to gold and amber; the snow mountains to white and ice blue; the bridge in fog to a pale gray-blue with no orange in it, which is the fog; the desert sunset to yellow-green-tan with some pink, which we count as half a hit. Two miss: the coffee essay decodes to teal with orange bits, and the lavender field to a murky gray-blue-brown. Argmax and sampled decodes agree. The texture is the same long horizontal streaking in every tile, and no tile contains an object. With nothing injected, the model's writing states carry the topic's palette in four and a half essays of eight, and the picture in none.

**The palette is the topic, not the color words.** The obvious objection is that the essays are full of color vocabulary (27 warm words in the fire essay, 22 occurrences of "white" in the snow essay, 19 fog and gray words in the bridge essay) and the tint simply rides on those tokens. We regenerated the eight essays with the same seeds, confirmed the decode reproduces tile for tile, then replaced the state at every color-word position (and the position after it) with the essay's mean state and decoded again; as a control we replaced the same number of randomly chosen non-color positions the same way. Masking the color words moves the tile no more than masking the same number of random positions: the mean RGB shift is within a few units of the random mask in six essays of eight, and the fire and lavender tiles shift a little more under the color mask but keep their palette. The contact sheet is three near-identical rows. The topic palette is carried broadly by the essay's states, not by the states at "blue" and "gold" and "white."

**It is there while the model reads.** Writing is a production task; the states above were produced while the model chose each next token. Is the palette present when the model only reads? From a single prefill forward pass with no generation, we decoded the 576 essay positions in three placements: the essay replayed in its own assistant span (teacher forcing is the same causal computation as generation, so this is the control and must reproduce the written grid), the essay placed as user content with no visual prompt anywhere in context, and the same 576 tokens in the user span in a fixed random order. The control reproduces the written grid (cosine 1.000 between state sets on all eight essays; per-pixel difference 11). In the user span, the states are cosine 0.97 to the writing states and the grids keep the hue family on all eight topics: ocean aqua, fire warm (the read fire tile is in fact more orange than the written one), wheat tan, snow white-blue, desert orange-tan. The sheet reads as a lighter, lower-contrast copy of the control row with the same horizontal texture; the read grids are 41 RGB from the control on average and 86 from the non-visual gray, further from gray than the written grids are (64). Reading lightens the tile; it does not gray it. The image head has the ocean in it while the model reads someone else's ocean essay, at about the strength it has it while writing its own.

The shuffled placement was meant to separate topic from vocabulary and did not work as a control. Its states are cosine 0.87 to the writing states and decode to blown-out near-white or saturated flat fields, three times as far from gray as the written grids, the same out-of-distribution failure the image head shows under large perturbations in Section 7. Some hue survives the bag of words (fire orange, ocean blue), but it rides on a degenerate readout, so this grid says nothing clean about vocabulary. A proper vocabulary control is a coherent essay with the same color words on a non-visual topic; we have not run it.

**What this section shows.** Without steering, the states a unified model produces while writing or reading about a visible thing already decode, through its own image head, to that thing's palette. The decode never yields the thing. The color-word mask says the palette belongs to the topic rather than to the color tokens, and the reading placement says production is not required. Whether this is anything like seeing, or is only what the next-token computation over ocean text looks like at the last layer, the readout cannot say, and Section 9 says so.

## 7. What a random direction does

Every result above was paired with a Gaussian direction of the same norm at the same layer and dose. This section collects what those controls did, because they set the null for every claim in the paper, and the null is not "nothing."

**A random direction has a palette of its own.** In the writing readout at L12 (Section 5), random seed 0 at r = 1.0 decodes to flat green in every tile. Three more seeds at the same dose decode to pink, peach and white with one red-black tile (seed 1); blue with fine horizontal streaks in all eight tiles (seed 2); red and green with vertical bars (seed 3). One of four random directions is blue with streaks. "Blue streaks mean the bridge" does not survive this control on its own; what separates the bridge condition from random seed 2 is the red and orange woven through every tile and the one international-orange tile, and what separates coffee is brown-black.

**The palette partly persists across regimes, for randoms too.** Section 5 noted that a concept direction's writing-time decode matches the same direction's drawing-time collapse (bridge: blue and cables in both; coffee: brown-black in both). We drew under random seeds 1 to 3 at the drawing doses (r = 0.3 and 0.45) and compared each to its own writing-time decode. Seed 2 matches itself clearly: cold ice-blue skies, a cyan harbor and snowy hills when drawing, blue with horizontal streaks when writing. Seed 1 is pale and washed out in both, though pink only when writing; seed 3 is dim green-teal-brown when drawing and red and green when writing; seed 0 drew as crumpled white paper and decodes green. So the sentence that survives is the weaker one: a steering vector's image-space effect persists into states produced while writing, concept or random alike. The concept-specific part is narrower, and we state it as structure and hue: the bridge direction carries cables (red and orange streaks woven through blue) that no random direction produced in either regime, and coffee carries brown-black; random directions carry a tint. At r = 0.45 the random draws also keep their scenes legible where the bridge direction had dissolved them, so an equal-norm random is the weaker perturbation of the image head.

**Depth: magnitude is not concept-specific, hue is.** We repeated the writing readout for the bridge and coffee directions at L6, L18 and L24 (r = 0.6 and 1.0), with random seed 0 at each layer. The image signal is the mean per-patch color shift from the same essay's baseline tile (floor 71.9, Section 5).

At L6 nothing carries for anyone. The bridge, coffee and random directions all sit at or below the floor (bridge r = 1.0: 74; random: 67), with baseline-like smears and no blue channel for the bridge or warm channel for coffee. The slightly more varied palettes we saw in the L6 bridge sheet before the random control landed are any-perturbation drift.

At L18 the random direction at r = 1.0 shifts the tiles more than the bridge direction does (130 against 119) and paints all eight the same saturated green-yellow stripes, with the essays intact. "Deeper is stronger" is therefore true of any vector at L18, not of the concept. What is vector-specific at L18 is the hue: the bridge direction paints five of eight tiles saturated international orange, coffee gives pink and lavender smears with one row of gold cylinders, random gives green. The bridge's red is a feature of the concept, the paint rather than the water; coffee's pink is a vector-specific color that is not a coffee color, and with one random seed we cannot say more than that.

At L24 the separation is the cleanest in the study, at the layer where the concept direction has the least structure. The random direction is a flat teal wash at r = 0.6 (93) and only slightly darker at r = 1.0 (102). The bridge direction at r = 0.6 turns all eight tiles solid red (184) and coffee turns seven of eight dark brown with fibrous grain (142), each about twice the random shift, in its own color, with the essays whole. The text head's tolerance runs the other way: random at r = 1.0 wrecks the essays (105 words of CJK and symbol salad) where the bridge direction at the same dose loops at 307 words and coffee writes 440. At the last layer a concept direction is something the image head reads strongly and the language head tolerates; a random direction of the same norm is something the image head reads weakly and the language head cannot survive.

**The layer selects the feature.** For the bridge direction the readout at L12 is the water and the cables (blue, with red and orange streaks: the only writing-time decode in the sweep with structure in it); at L18 and L24 it is the paint (red). Same direction family, same essays, a different color of bridge. This held for the bridge and not for coffee, whose L18 decode is pink; we make the claim for the bridge only. Across the sweep, L12 is the only layer where the writing-time decode has shape, and it is also the layer where the prompt-only steer composes while drawing (Section 4).

**The null we use.** Random seed 0 is green at L12, green-yellow at L18 and teal at L24; the three other seeds we ran at L12 each have a color. "Random equals gray" is never the right null. The null is: a random direction has its own color, no structure, and breaks the language head before it moves the image head much. Everything we call concept-specific in this paper is measured against that: cables and brown-black grain at L12, hue and a twofold magnitude at L24, and the language head surviving a concept direction that a random direction of the same norm destroys.

## 8. What didn't work

Each entry is one result and the one thing it taught us.

**Coffee as the primary concept.** Coffee was chosen first, as the Golden Gate Claude case with a paintbrush. Under every-position steering at L12 it never produced a cup; at r = 0.6 and above the scene dissolved into dark, grainy brown-black speckle, and CLIP scored that speckle as coffee (mean 0.35 at r = 0.9, against 0.02 at baseline) because dark, grainy and brown is what "coffee" looks like to a contrastive judge. The rising curve was partly the judge rewarding texture. Lesson: a concept whose features are a texture and a color cannot be judged by a model that scores texture and color; the bridge, whose collapse (cables on blue) CLIP correctly refuses to call a bridge, fixed the judge problem by choice of subject.

**L6.** Steering at the output of block 6 does nothing to either head at any dose we tried, and the random control does the same nothing. The slightly varied palettes we noticed in the L6 bridge sheet before the random control ran were any-perturbation drift. Lesson: look at the control before describing the sheet.

**CLIP on smears.** The writing-time decodes are textured fields, and CLIP's scores on them are not usable: 0.56 "coffee" on a near-black tile, 0.41 "coffee" on flat green, 0.59 "bridge" on black, 1.00 "bridge" on the L24 drawings that are a red filter over a harbor. Lesson: the contact sheet read against the other sheets is the measurement; a number that cannot tell red water from a bridge is a tiebreaker at best, and we report it as one.

**The EOS confound.** In the first writing-readout pass, steered essays hit end-of-sequence after 50 to 130 tokens (the baseline never did) and the model restarted, so most of each 576-position grid held post-restart states. The signatures reported in Section 5 were visible in that pass, but they could not be separated from "what restart states look like." Masking the EOS logit so every essay runs the full 576 tokens removed the confound and left the result; the baseline is tile-for-tile identical across the two passes because it never hit EOS. Lesson: check where the tokens came from before reading the grid.

**The shuffle control.** To separate topic from vocabulary in the reading regime (Section 6), we placed the same 576 tokens in random order and decoded. The states of an incoherent token sequence are out of distribution and the image head maps them to bright, blown-out codes, three times as far from gray as the written grids. Some hue survives the bag of words, but on a degenerate readout, so the control cannot say what it was built to say. Lesson: a control has to stay in regime; the right vocabulary control is a coherent essay with the same color words on a non-visual topic.

**Prompt-only while writing.** The prompt-only regime that composes bridges while drawing (Section 4) carries almost nothing into writing states. At r = 1.0, all three concepts give the baseline text and the baseline sheet, where every-position at the same dose gave 26 coffee words per essay and brown-black tiles. At r = 2.0 the concept gets through in single essays, by one route or the other and never both in the same essay: one essay of eight under the bridge direction became 481 coherent words on bridge engineering (seismic codes, flood loads) while its tile stayed a plain smear; under coffee, one essay became a Portland coffee shop with an unremarkable tile, and a different essay with zero coffee words decoded to the coffee collapse color. The rest of the r = 2.0 palette drift is generic (circle and random drift the same way, bleached and tan), and r = 2.0 on the prompt breaks essays outright into non-English junk whose tiles are blank. Lesson: the mind's eye we can read during writing is a property of the per-position bias, not of the prompt; the attention route that composes a bridge in image space composes a bridge essay in text space, and never a bridge tint. The regimes are not symmetric across modalities.

**The per-position map.** We asked which of the 576 writing positions carry the tint, by differencing each steered tile from its baseline tile patch by patch and averaging over essays. The map is spatially uniform and independent of token class (word starts, continuations, punctuation, newlines all the same), with one effect: the first few rows, the states nearest the prompt, are least tinted, and positional consistency across essays lives entirely in that top band. Lesson: once the essay is underway every position is equally tinted; the map adds no localization and we did not pursue it further. (The map's floor, the per-patch difference between two unsteered essays, 71.9, became the floor for the dose and layer metrics in Sections 5 and 7.)

## 9. Discussion

**An aphantic mind's eye.** Zeman, Dewar and Della Sala (2015) named aphantasia for people with a lifelong absence of voluntary visual imagery who nonetheless know what things look like: they can tell you the color of a lemon without seeing one. The analogy is to the result, not to any person. When this model writes about tax law under the bridge direction, or writes or reads about the ocean with nothing injected, its own image head finds in the writing states the hue and texture of the thing, and never the thing. Palette and grain are present; a composed picture is not. We do not know whether the states "contain" more than the image head can read; the readout is one linear map and a VQ decoder, applied where they were not trained to be applied. What we can say is that, as far as the model's own image head can report, what accompanies its writing is features without a picture.

**Features, not objects.** The same phrase describes the every-position drawing results. A direction added at every position gives cables, international orange, brown grain or curvature on existing objects, at every layer and dose; it never gives a bridge, a mug or a circle. Added to the prompt alone at L12, the same direction composes bridge towers and round objects in coherent scenes. We offer one mechanism claim, hedged. A constant bias added to every image position can change what each patch tends to be but carries no information about relations between patches, and an object is a relation (two towers, a deck, cables between them; a closed curve). Composition in an autoregressive image head has to come through attention across positions, and a steer that enters only through the prompt is read by exactly that route. On this account, the every-position bias is not failing to compose; it is overwriting, patch by patch, what attention would have composed. We have not tested this directly (for instance by steering a subset of positions, or by ablating attention to the prompt), and the coffee direction, which composes nothing in either regime, shows that the account needs the direction to carry an object in the first place. We state it as a reading of the pictures, consistent across three concepts and four layers, and not as a demonstrated mechanism.

**Two heads, one stream, different readers.** The dissociations in Section 5 run in both directions and across regimes. The image head is two to three times more sensitive than the language head to the same per-position bias and fails differently (texture rather than silence). Which head registers a direction first depends on the direction: the bridge reaches the codebook and never the vocabulary, the circle reaches the vocabulary and never the codebook, coffee reaches both at different doses. Drawing states full of towers give the language head "the West" and never "bridge." Each head reads the shared residual stream for its own kind of content, and neither can read the other's object. This bears on how steering results transfer from language-only models: Golden Gate Claude (Templeton et al., 2024) steered one head by clamping a sparse-autoencoder feature; in a unified model, the direction's effect on the image head is not a translation of its effect on the language head but a different thing, with its own dose window, its own failure mode and its own route in.

**What this does not show.** It does not show that the model imagines, pictures, or sees anything while it writes; it shows that a linear readout the model already owns, applied to its writing states, returns the palette of the topic and nothing else. It does not show that the states carry no picture; a trained decoder might recover more, and we did not train one. It does not show that the palette is an epiphenomenon of color vocabulary, since the color-word mask did not move it, but a coherent-essay vocabulary control remains to be run. It does not separate "topic palette" from "whatever the final-layer computation over text about the ocean looks like when forced through a codebook," which may be the same statement. And random directions carry palettes of their own, so every concept-specific claim in this paper is about structure (cables, grain) and hue measured against a random direction of the same norm, never about a tint alone.

## 10. Limitations

One model, Janus-Pro-7B, with one image tokenizer; we do not know whether a unified model with a different image path (for instance one with a diffusion head) would read its own writing states the same way. Three concepts, chosen by hand; coffee behaves differently from the other two in ways we can describe but not explain. Eight drawing prompts and eight essay prompts, one seed per prompt, so each contact sheet cell is a single sample and no claim rests on a within-condition average. The primary measurement is a human reading a contact sheet against other contact sheets; CLIP and a Hough transform are tiebreakers that we have shown to fail in specific ways, and we have no automated metric for "cables" or "a composed tower." The image head is applied out of regime to writing states, which never followed an image-start token; the positive control shows the decode path is sound, not that writing states are meant to be decodable. Most random-direction controls use one seed (four at L12 in the writing readout; one at the other layers), so "no random direction produced cables" is a claim about a handful of directions. Dose is stated as a ratio to the mean residual norm at a layer, a rule we chose for comparability and have not validated against alternatives. All steering is difference-of-means over sixteen sentence pairs; a sparse-autoencoder feature, as in Golden Gate Claude, might carry the object more cleanly or differently. Nothing here was replicated by a second person.

## 11. Reproduction

Code, directions, prompt sets, every contact sheet and every essay are in the repository accompanying this paper. The model is `deepseek-ai/Janus-Pro-7B` from the Hugging Face hub, loaded in bfloat16; all runs fit on one 24 GB GPU (about 17 GB peak), and a drawing condition of sixteen images takes about half a minute.

`minds_eye/` holds the harness: `janus_model.py` (loading, the steering hook, `draw`, `write`, `read_states`, `states_to_image`), `concepts.py` (the sixteen concept and sixteen control sentences per concept and the eight free-draw prompts), `direction.py` (difference-of-means directions and equal-norm random directions), `judge.py` (CLIP and Hough). `scripts/` holds one script per experiment: `run1_pilot.py` (drawing under steer; Section 4), `run1_text.py` (short descriptions under steer; Section 5), `run2_decoder.py` (writing states through the image head, with the EOS mask, the visual-topic essays and the prompt-only variant; Sections 5, 6, 8), `run2_colormask.py` (the color-word mask; Section 6), `run3_mirror.py` and `run3_posthoc.py` (drawing states through the language head; Section 5), `run4_posmap.py` (the per-position map; Section 8), `run5_reading.py` (the reading regime; Section 6), `sweep_summary.py` (the shift metric over layers and doses; Sections 5 and 7). Conditions are selected by environment variables (`ME_CONCEPT`/`ME_CONCEPTS`, `ME_LAYER`, `ME_RATIOS`, `ME_PROMPT_ONLY`, `ME_SUPPRESS_EOS`, `ME_RANDOM_SEEDS`), and the exact invocation for every run reported here is a shell script under `runs/queue_*.sh`, with the run's output under `runs/<run>/` beside it. Seeds are fixed (base seed 0 plus prompt index), and we have confirmed that rerunning a condition reproduces its sheet tile for tile. Each script's docstring records the question it was written to answer.

## Appendix

This index lists every contact sheet, every essay and description file, and every invocation under `runs/`, grouped by the paper section that reports it. Links are relative so they resolve on GitHub. The smoke test is omitted.

### A. Contact sheets

#### Section 4. Drawing: features, not objects

Every-position and prompt-only drawing under each concept direction, swept across layers.

| Condition | Run | Sheet |
| --- | --- | --- |
| baseline | run1_pilot | [baseline](runs/run1_pilot/baseline.jpg) |
| coffee r=0.4 | run1_pilot | [coffee r=0.4](runs/run1_pilot/coffee_r0.4.jpg) |
| coffee r=0.6 | run1_pilot | [coffee r=0.6](runs/run1_pilot/coffee_r0.6.jpg) |
| coffee r=0.9 | run1_pilot | [coffee r=0.9](runs/run1_pilot/coffee_r0.9.jpg) |
| random r=0.4 | run1_pilot | [random r=0.4](runs/run1_pilot/random_r0.4.jpg) |
| random r=0.6 | run1_pilot | [random r=0.6](runs/run1_pilot/random_r0.6.jpg) |
| random r=0.9 | run1_pilot | [random r=0.9](runs/run1_pilot/random_r0.9.jpg) |
| baseline | run1_ggb_L12 | [baseline](runs/run1_ggb_L12/baseline.jpg) |
| ggb r=0.2 | run1_ggb_L12 | [ggb r=0.2](runs/run1_ggb_L12/ggb_r0.2.jpg) |
| ggb r=0.3 | run1_ggb_L12 | [ggb r=0.3](runs/run1_ggb_L12/ggb_r0.3.jpg) |
| ggb r=0.45 | run1_ggb_L12 | [ggb r=0.45](runs/run1_ggb_L12/ggb_r0.45.jpg) |
| ggb r=0.6 | run1_ggb_L12 | [ggb r=0.6](runs/run1_ggb_L12/ggb_r0.6.jpg) |
| random r=0.2 | run1_ggb_L12 | [random r=0.2](runs/run1_ggb_L12/random_r0.2.jpg) |
| random r=0.3 | run1_ggb_L12 | [random r=0.3](runs/run1_ggb_L12/random_r0.3.jpg) |
| random r=0.45 | run1_ggb_L12 | [random r=0.45](runs/run1_ggb_L12/random_r0.45.jpg) |
| random r=0.6 | run1_ggb_L12 | [random r=0.6](runs/run1_ggb_L12/random_r0.6.jpg) |
| baseline | run1_ggb_L6 | [baseline](runs/run1_ggb_L6/baseline.jpg) |
| ggb r=0.25 | run1_ggb_L6 | [ggb r=0.25](runs/run1_ggb_L6/ggb_r0.25.jpg) |
| ggb r=0.4 | run1_ggb_L6 | [ggb r=0.4](runs/run1_ggb_L6/ggb_r0.4.jpg) |
| random r=0.25 | run1_ggb_L6 | [random r=0.25](runs/run1_ggb_L6/random_r0.25.jpg) |
| random r=0.4 | run1_ggb_L6 | [random r=0.4](runs/run1_ggb_L6/random_r0.4.jpg) |
| baseline | run1_ggb_L18 | [baseline](runs/run1_ggb_L18/baseline.jpg) |
| ggb r=0.25 | run1_ggb_L18 | [ggb r=0.25](runs/run1_ggb_L18/ggb_r0.25.jpg) |
| ggb r=0.4 | run1_ggb_L18 | [ggb r=0.4](runs/run1_ggb_L18/ggb_r0.4.jpg) |
| random r=0.25 | run1_ggb_L18 | [random r=0.25](runs/run1_ggb_L18/random_r0.25.jpg) |
| random r=0.4 | run1_ggb_L18 | [random r=0.4](runs/run1_ggb_L18/random_r0.4.jpg) |
| baseline | run1_ggb_L24 | [baseline](runs/run1_ggb_L24/baseline.jpg) |
| ggb r=0.25 | run1_ggb_L24 | [ggb r=0.25](runs/run1_ggb_L24/ggb_r0.25.jpg) |
| ggb r=0.4 | run1_ggb_L24 | [ggb r=0.4](runs/run1_ggb_L24/ggb_r0.4.jpg) |
| random r=0.25 | run1_ggb_L24 | [random r=0.25](runs/run1_ggb_L24/random_r0.25.jpg) |
| random r=0.4 | run1_ggb_L24 | [random r=0.4](runs/run1_ggb_L24/random_r0.4.jpg) |
| baseline | run1_circle_L12 | [baseline](runs/run1_circle_L12/baseline.jpg) |
| circle r=0.25 | run1_circle_L12 | [circle r=0.25](runs/run1_circle_L12/circle_r0.25.jpg) |
| circle r=0.4 | run1_circle_L12 | [circle r=0.4](runs/run1_circle_L12/circle_r0.4.jpg) |
| circle r=0.6 | run1_circle_L12 | [circle r=0.6](runs/run1_circle_L12/circle_r0.6.jpg) |
| random r=0.25 | run1_circle_L12 | [random r=0.25](runs/run1_circle_L12/random_r0.25.jpg) |
| random r=0.4 | run1_circle_L12 | [random r=0.4](runs/run1_circle_L12/random_r0.4.jpg) |
| random r=0.6 | run1_circle_L12 | [random r=0.6](runs/run1_circle_L12/random_r0.6.jpg) |
| baseline | run1_circle_L18 | [baseline](runs/run1_circle_L18/baseline.jpg) |
| circle r=0.25 | run1_circle_L18 | [circle r=0.25](runs/run1_circle_L18/circle_r0.25.jpg) |
| circle r=0.4 | run1_circle_L18 | [circle r=0.4](runs/run1_circle_L18/circle_r0.4.jpg) |
| circle r=0.6 | run1_circle_L18 | [circle r=0.6](runs/run1_circle_L18/circle_r0.6.jpg) |
| random r=0.25 | run1_circle_L18 | [random r=0.25](runs/run1_circle_L18/random_r0.25.jpg) |
| random r=0.4 | run1_circle_L18 | [random r=0.4](runs/run1_circle_L18/random_r0.4.jpg) |
| random r=0.6 | run1_circle_L18 | [random r=0.6](runs/run1_circle_L18/random_r0.6.jpg) |
| baseline | run1_ggb_L12_po | [baseline](runs/run1_ggb_L12_po/baseline.jpg) |
| ggb r=0.5 | run1_ggb_L12_po | [ggb r=0.5](runs/run1_ggb_L12_po/ggb_r0.5.jpg) |
| ggb r=1.0 | run1_ggb_L12_po | [ggb r=1.0](runs/run1_ggb_L12_po/ggb_r1.0.jpg) |
| ggb r=2.0 | run1_ggb_L12_po | [ggb r=2.0](runs/run1_ggb_L12_po/ggb_r2.0.jpg) |
| random r=0.5 | run1_ggb_L12_po | [random r=0.5](runs/run1_ggb_L12_po/random_r0.5.jpg) |
| random r=1.0 | run1_ggb_L12_po | [random r=1.0](runs/run1_ggb_L12_po/random_r1.0.jpg) |
| random r=2.0 | run1_ggb_L12_po | [random r=2.0](runs/run1_ggb_L12_po/random_r2.0.jpg) |
| baseline | run1_ggb_L18_po | [baseline](runs/run1_ggb_L18_po/baseline.jpg) |
| ggb r=0.5 | run1_ggb_L18_po | [ggb r=0.5](runs/run1_ggb_L18_po/ggb_r0.5.jpg) |
| ggb r=1.0 | run1_ggb_L18_po | [ggb r=1.0](runs/run1_ggb_L18_po/ggb_r1.0.jpg) |
| ggb r=2.0 | run1_ggb_L18_po | [ggb r=2.0](runs/run1_ggb_L18_po/ggb_r2.0.jpg) |
| random r=0.5 | run1_ggb_L18_po | [random r=0.5](runs/run1_ggb_L18_po/random_r0.5.jpg) |
| random r=1.0 | run1_ggb_L18_po | [random r=1.0](runs/run1_ggb_L18_po/random_r1.0.jpg) |
| random r=2.0 | run1_ggb_L18_po | [random r=2.0](runs/run1_ggb_L18_po/random_r2.0.jpg) |
| baseline | run1_circle_L12_po | [baseline](runs/run1_circle_L12_po/baseline.jpg) |
| circle r=1.0 | run1_circle_L12_po | [circle r=1.0](runs/run1_circle_L12_po/circle_r1.0.jpg) |
| circle r=2.0 | run1_circle_L12_po | [circle r=2.0](runs/run1_circle_L12_po/circle_r2.0.jpg) |
| random r=1.0 | run1_circle_L12_po | [random r=1.0](runs/run1_circle_L12_po/random_r1.0.jpg) |
| random r=2.0 | run1_circle_L12_po | [random r=2.0](runs/run1_circle_L12_po/random_r2.0.jpg) |
| baseline | run1_coffee_L12_po | [baseline](runs/run1_coffee_L12_po/baseline.jpg) |
| coffee r=1.0 | run1_coffee_L12_po | [coffee r=1.0](runs/run1_coffee_L12_po/coffee_r1.0.jpg) |
| coffee r=2.0 | run1_coffee_L12_po | [coffee r=2.0](runs/run1_coffee_L12_po/coffee_r2.0.jpg) |
| random r=1.0 | run1_coffee_L12_po | [random r=1.0](runs/run1_coffee_L12_po/random_r1.0.jpg) |
| random r=2.0 | run1_coffee_L12_po | [random r=2.0](runs/run1_coffee_L12_po/random_r2.0.jpg) |

#### Section 5. Writing: the two heads read different things

Writing states decoded through the image head at L12, the dose curve, and the mirror readout of drawing states through the language head.

| Condition | Run | Sheet |
| --- | --- | --- |
| baseline argmax | run2_L12_noeos | [baseline argmax](runs/run2_L12_noeos/baseline_argmax.jpg) |
| baseline sampled | run2_L12_noeos | [baseline sampled](runs/run2_L12_noeos/baseline_sampled.jpg) |
| circle r=0.6 argmax | run2_L12_noeos | [circle r=0.6 argmax](runs/run2_L12_noeos/circle_r0.6_argmax.jpg) |
| circle r=0.6 sampled | run2_L12_noeos | [circle r=0.6 sampled](runs/run2_L12_noeos/circle_r0.6_sampled.jpg) |
| circle r=1.0 argmax | run2_L12_noeos | [circle r=1.0 argmax](runs/run2_L12_noeos/circle_r1.0_argmax.jpg) |
| circle r=1.0 sampled | run2_L12_noeos | [circle r=1.0 sampled](runs/run2_L12_noeos/circle_r1.0_sampled.jpg) |
| coffee r=0.6 argmax | run2_L12_noeos | [coffee r=0.6 argmax](runs/run2_L12_noeos/coffee_r0.6_argmax.jpg) |
| coffee r=0.6 sampled | run2_L12_noeos | [coffee r=0.6 sampled](runs/run2_L12_noeos/coffee_r0.6_sampled.jpg) |
| coffee r=1.0 argmax | run2_L12_noeos | [coffee r=1.0 argmax](runs/run2_L12_noeos/coffee_r1.0_argmax.jpg) |
| coffee r=1.0 sampled | run2_L12_noeos | [coffee r=1.0 sampled](runs/run2_L12_noeos/coffee_r1.0_sampled.jpg) |
| control drawn | run2_L12_noeos | [control drawn](runs/run2_L12_noeos/control_drawn.jpg) |
| control drawstates argmax | run2_L12_noeos | [control drawstates argmax](runs/run2_L12_noeos/control_drawstates_argmax.jpg) |
| control drawstates sampled | run2_L12_noeos | [control drawstates sampled](runs/run2_L12_noeos/control_drawstates_sampled.jpg) |
| ggb r=0.6 argmax | run2_L12_noeos | [ggb r=0.6 argmax](runs/run2_L12_noeos/ggb_r0.6_argmax.jpg) |
| ggb r=0.6 sampled | run2_L12_noeos | [ggb r=0.6 sampled](runs/run2_L12_noeos/ggb_r0.6_sampled.jpg) |
| ggb r=1.0 argmax | run2_L12_noeos | [ggb r=1.0 argmax](runs/run2_L12_noeos/ggb_r1.0_argmax.jpg) |
| ggb r=1.0 sampled | run2_L12_noeos | [ggb r=1.0 sampled](runs/run2_L12_noeos/ggb_r1.0_sampled.jpg) |
| random r=1.0 argmax | run2_L12_noeos | [random r=1.0 argmax](runs/run2_L12_noeos/random_r1.0_argmax.jpg) |
| random r=1.0 sampled | run2_L12_noeos | [random r=1.0 sampled](runs/run2_L12_noeos/random_r1.0_sampled.jpg) |
| circle r=0.2 argmax | run2_L12_noeos_dose | [circle r=0.2 argmax](runs/run2_L12_noeos_dose/circle_r0.2_argmax.jpg) |
| circle r=0.2 sampled | run2_L12_noeos_dose | [circle r=0.2 sampled](runs/run2_L12_noeos_dose/circle_r0.2_sampled.jpg) |
| circle r=0.4 argmax | run2_L12_noeos_dose | [circle r=0.4 argmax](runs/run2_L12_noeos_dose/circle_r0.4_argmax.jpg) |
| circle r=0.4 sampled | run2_L12_noeos_dose | [circle r=0.4 sampled](runs/run2_L12_noeos_dose/circle_r0.4_sampled.jpg) |
| circle r=0.8 argmax | run2_L12_noeos_dose | [circle r=0.8 argmax](runs/run2_L12_noeos_dose/circle_r0.8_argmax.jpg) |
| circle r=0.8 sampled | run2_L12_noeos_dose | [circle r=0.8 sampled](runs/run2_L12_noeos_dose/circle_r0.8_sampled.jpg) |
| coffee r=0.2 argmax | run2_L12_noeos_dose | [coffee r=0.2 argmax](runs/run2_L12_noeos_dose/coffee_r0.2_argmax.jpg) |
| coffee r=0.2 sampled | run2_L12_noeos_dose | [coffee r=0.2 sampled](runs/run2_L12_noeos_dose/coffee_r0.2_sampled.jpg) |
| coffee r=0.4 argmax | run2_L12_noeos_dose | [coffee r=0.4 argmax](runs/run2_L12_noeos_dose/coffee_r0.4_argmax.jpg) |
| coffee r=0.4 sampled | run2_L12_noeos_dose | [coffee r=0.4 sampled](runs/run2_L12_noeos_dose/coffee_r0.4_sampled.jpg) |
| coffee r=0.8 argmax | run2_L12_noeos_dose | [coffee r=0.8 argmax](runs/run2_L12_noeos_dose/coffee_r0.8_argmax.jpg) |
| coffee r=0.8 sampled | run2_L12_noeos_dose | [coffee r=0.8 sampled](runs/run2_L12_noeos_dose/coffee_r0.8_sampled.jpg) |
| ggb r=0.2 argmax | run2_L12_noeos_dose | [ggb r=0.2 argmax](runs/run2_L12_noeos_dose/ggb_r0.2_argmax.jpg) |
| ggb r=0.2 sampled | run2_L12_noeos_dose | [ggb r=0.2 sampled](runs/run2_L12_noeos_dose/ggb_r0.2_sampled.jpg) |
| ggb r=0.4 argmax | run2_L12_noeos_dose | [ggb r=0.4 argmax](runs/run2_L12_noeos_dose/ggb_r0.4_argmax.jpg) |
| ggb r=0.4 sampled | run2_L12_noeos_dose | [ggb r=0.4 sampled](runs/run2_L12_noeos_dose/ggb_r0.4_sampled.jpg) |
| ggb r=0.8 argmax | run2_L12_noeos_dose | [ggb r=0.8 argmax](runs/run2_L12_noeos_dose/ggb_r0.8_argmax.jpg) |
| ggb r=0.8 sampled | run2_L12_noeos_dose | [ggb r=0.8 sampled](runs/run2_L12_noeos_dose/ggb_r0.8_sampled.jpg) |
| baseline drawn | run3_mirror_L12 | [baseline drawn](runs/run3_mirror_L12/baseline_drawn.jpg) |
| circle r=0.45 drawn | run3_mirror_L12 | [circle r=0.45 drawn](runs/run3_mirror_L12/circle_r0.45_drawn.jpg) |
| coffee r=0.3 drawn | run3_mirror_L12 | [coffee r=0.3 drawn](runs/run3_mirror_L12/coffee_r0.3_drawn.jpg) |
| coffee r=0.45 drawn | run3_mirror_L12 | [coffee r=0.45 drawn](runs/run3_mirror_L12/coffee_r0.45_drawn.jpg) |
| ggb po r=1.0 drawn | run3_mirror_L12 | [ggb po r=1.0 drawn](runs/run3_mirror_L12/ggb_po_r1.0_drawn.jpg) |
| ggb po r=2.0 drawn | run3_mirror_L12 | [ggb po r=2.0 drawn](runs/run3_mirror_L12/ggb_po_r2.0_drawn.jpg) |
| ggb r=0.3 drawn | run3_mirror_L12 | [ggb r=0.3 drawn](runs/run3_mirror_L12/ggb_r0.3_drawn.jpg) |
| ggb r=0.45 drawn | run3_mirror_L12 | [ggb r=0.45 drawn](runs/run3_mirror_L12/ggb_r0.45_drawn.jpg) |
| random r=0.45 drawn | run3_mirror_L12 | [random r=0.45 drawn](runs/run3_mirror_L12/random_r0.45_drawn.jpg) |
| random s2 r=0.45 drawn | run3_mirror_L12 | [random s2 r=0.45 drawn](runs/run3_mirror_L12/random_s2_r0.45_drawn.jpg) |

#### Section 6. No steering at all

Unsteered visual-topic essays, the color-word mask, and the reading regime.

| Condition | Run | Sheet |
| --- | --- | --- |
| baseline argmax | run2_L12_visual | [baseline argmax](runs/run2_L12_visual/baseline_argmax.jpg) |
| baseline sampled | run2_L12_visual | [baseline sampled](runs/run2_L12_visual/baseline_sampled.jpg) |
| combined | run2_L12_visual_colormask | [combined](runs/run2_L12_visual_colormask/combined.jpg) |
| full | run2_L12_visual_colormask | [full](runs/run2_L12_visual_colormask/full.jpg) |
| mask color | run2_L12_visual_colormask | [mask color](runs/run2_L12_visual_colormask/mask_color.jpg) |
| mask rand | run2_L12_visual_colormask | [mask rand](runs/run2_L12_visual_colormask/mask_rand.jpg) |
| assist | run5_reading | [assist](runs/run5_reading/assist.jpg) |
| combined | run5_reading | [combined](runs/run5_reading/combined.jpg) |
| gen | run5_reading | [gen](runs/run5_reading/gen.jpg) |
| shuffle | run5_reading | [shuffle](runs/run5_reading/shuffle.jpg) |
| user | run5_reading | [user](runs/run5_reading/user.jpg) |

#### Section 7. What a random direction does

Random-direction controls and the bridge and coffee writing readout repeated at L6, L18 and L24.

| Condition | Run | Sheet |
| --- | --- | --- |
| random s1 r=0.3 | run1_ggb_L12_rand123 | [random s1 r=0.3](runs/run1_ggb_L12_rand123/random_s1_r0.3.jpg) |
| random s1 r=0.45 | run1_ggb_L12_rand123 | [random s1 r=0.45](runs/run1_ggb_L12_rand123/random_s1_r0.45.jpg) |
| random s2 r=0.3 | run1_ggb_L12_rand123 | [random s2 r=0.3](runs/run1_ggb_L12_rand123/random_s2_r0.3.jpg) |
| random s2 r=0.45 | run1_ggb_L12_rand123 | [random s2 r=0.45](runs/run1_ggb_L12_rand123/random_s2_r0.45.jpg) |
| random s3 r=0.3 | run1_ggb_L12_rand123 | [random s3 r=0.3](runs/run1_ggb_L12_rand123/random_s3_r0.3.jpg) |
| random s3 r=0.45 | run1_ggb_L12_rand123 | [random s3 r=0.45](runs/run1_ggb_L12_rand123/random_s3_r0.45.jpg) |
| random s1 r=1.0 argmax | run2_L12_noeos_rand | [random s1 r=1.0 argmax](runs/run2_L12_noeos_rand/random_s1_r1.0_argmax.jpg) |
| random s1 r=1.0 sampled | run2_L12_noeos_rand | [random s1 r=1.0 sampled](runs/run2_L12_noeos_rand/random_s1_r1.0_sampled.jpg) |
| random s2 r=1.0 argmax | run2_L12_noeos_rand | [random s2 r=1.0 argmax](runs/run2_L12_noeos_rand/random_s2_r1.0_argmax.jpg) |
| random s2 r=1.0 sampled | run2_L12_noeos_rand | [random s2 r=1.0 sampled](runs/run2_L12_noeos_rand/random_s2_r1.0_sampled.jpg) |
| random s3 r=1.0 argmax | run2_L12_noeos_rand | [random s3 r=1.0 argmax](runs/run2_L12_noeos_rand/random_s3_r1.0_argmax.jpg) |
| random s3 r=1.0 sampled | run2_L12_noeos_rand | [random s3 r=1.0 sampled](runs/run2_L12_noeos_rand/random_s3_r1.0_sampled.jpg) |
| coffee r=0.6 argmax | run2_L6_noeos | [coffee r=0.6 argmax](runs/run2_L6_noeos/coffee_r0.6_argmax.jpg) |
| coffee r=0.6 sampled | run2_L6_noeos | [coffee r=0.6 sampled](runs/run2_L6_noeos/coffee_r0.6_sampled.jpg) |
| coffee r=1.0 argmax | run2_L6_noeos | [coffee r=1.0 argmax](runs/run2_L6_noeos/coffee_r1.0_argmax.jpg) |
| coffee r=1.0 sampled | run2_L6_noeos | [coffee r=1.0 sampled](runs/run2_L6_noeos/coffee_r1.0_sampled.jpg) |
| ggb r=0.6 argmax | run2_L6_noeos | [ggb r=0.6 argmax](runs/run2_L6_noeos/ggb_r0.6_argmax.jpg) |
| ggb r=0.6 sampled | run2_L6_noeos | [ggb r=0.6 sampled](runs/run2_L6_noeos/ggb_r0.6_sampled.jpg) |
| ggb r=1.0 argmax | run2_L6_noeos | [ggb r=1.0 argmax](runs/run2_L6_noeos/ggb_r1.0_argmax.jpg) |
| ggb r=1.0 sampled | run2_L6_noeos | [ggb r=1.0 sampled](runs/run2_L6_noeos/ggb_r1.0_sampled.jpg) |
| random r=0.6 argmax | run2_L6_noeos_rand0.6 | [random r=0.6 argmax](runs/run2_L6_noeos_rand0.6/random_r0.6_argmax.jpg) |
| random r=0.6 sampled | run2_L6_noeos_rand0.6 | [random r=0.6 sampled](runs/run2_L6_noeos_rand0.6/random_r0.6_sampled.jpg) |
| random r=1.0 argmax | run2_L6_noeos_rand1.0 | [random r=1.0 argmax](runs/run2_L6_noeos_rand1.0/random_r1.0_argmax.jpg) |
| random r=1.0 sampled | run2_L6_noeos_rand1.0 | [random r=1.0 sampled](runs/run2_L6_noeos_rand1.0/random_r1.0_sampled.jpg) |
| coffee r=0.6 argmax | run2_L18_noeos | [coffee r=0.6 argmax](runs/run2_L18_noeos/coffee_r0.6_argmax.jpg) |
| coffee r=0.6 sampled | run2_L18_noeos | [coffee r=0.6 sampled](runs/run2_L18_noeos/coffee_r0.6_sampled.jpg) |
| coffee r=1.0 argmax | run2_L18_noeos | [coffee r=1.0 argmax](runs/run2_L18_noeos/coffee_r1.0_argmax.jpg) |
| coffee r=1.0 sampled | run2_L18_noeos | [coffee r=1.0 sampled](runs/run2_L18_noeos/coffee_r1.0_sampled.jpg) |
| ggb r=0.6 argmax | run2_L18_noeos | [ggb r=0.6 argmax](runs/run2_L18_noeos/ggb_r0.6_argmax.jpg) |
| ggb r=0.6 sampled | run2_L18_noeos | [ggb r=0.6 sampled](runs/run2_L18_noeos/ggb_r0.6_sampled.jpg) |
| ggb r=1.0 argmax | run2_L18_noeos | [ggb r=1.0 argmax](runs/run2_L18_noeos/ggb_r1.0_argmax.jpg) |
| ggb r=1.0 sampled | run2_L18_noeos | [ggb r=1.0 sampled](runs/run2_L18_noeos/ggb_r1.0_sampled.jpg) |
| random r=0.6 argmax | run2_L18_noeos_rand0.6 | [random r=0.6 argmax](runs/run2_L18_noeos_rand0.6/random_r0.6_argmax.jpg) |
| random r=0.6 sampled | run2_L18_noeos_rand0.6 | [random r=0.6 sampled](runs/run2_L18_noeos_rand0.6/random_r0.6_sampled.jpg) |
| random r=1.0 argmax | run2_L18_noeos_rand1.0 | [random r=1.0 argmax](runs/run2_L18_noeos_rand1.0/random_r1.0_argmax.jpg) |
| random r=1.0 sampled | run2_L18_noeos_rand1.0 | [random r=1.0 sampled](runs/run2_L18_noeos_rand1.0/random_r1.0_sampled.jpg) |
| coffee r=0.6 argmax | run2_L24_noeos | [coffee r=0.6 argmax](runs/run2_L24_noeos/coffee_r0.6_argmax.jpg) |
| coffee r=0.6 sampled | run2_L24_noeos | [coffee r=0.6 sampled](runs/run2_L24_noeos/coffee_r0.6_sampled.jpg) |
| coffee r=1.0 argmax | run2_L24_noeos | [coffee r=1.0 argmax](runs/run2_L24_noeos/coffee_r1.0_argmax.jpg) |
| coffee r=1.0 sampled | run2_L24_noeos | [coffee r=1.0 sampled](runs/run2_L24_noeos/coffee_r1.0_sampled.jpg) |
| ggb r=0.6 argmax | run2_L24_noeos | [ggb r=0.6 argmax](runs/run2_L24_noeos/ggb_r0.6_argmax.jpg) |
| ggb r=0.6 sampled | run2_L24_noeos | [ggb r=0.6 sampled](runs/run2_L24_noeos/ggb_r0.6_sampled.jpg) |
| ggb r=1.0 argmax | run2_L24_noeos | [ggb r=1.0 argmax](runs/run2_L24_noeos/ggb_r1.0_argmax.jpg) |
| ggb r=1.0 sampled | run2_L24_noeos | [ggb r=1.0 sampled](runs/run2_L24_noeos/ggb_r1.0_sampled.jpg) |
| random r=0.6 argmax | run2_L24_noeos_rand0.6 | [random r=0.6 argmax](runs/run2_L24_noeos_rand0.6/random_r0.6_argmax.jpg) |
| random r=0.6 sampled | run2_L24_noeos_rand0.6 | [random r=0.6 sampled](runs/run2_L24_noeos_rand0.6/random_r0.6_sampled.jpg) |
| random r=1.0 argmax | run2_L24_noeos_rand1.0 | [random r=1.0 argmax](runs/run2_L24_noeos_rand1.0/random_r1.0_argmax.jpg) |
| random r=1.0 sampled | run2_L24_noeos_rand1.0 | [random r=1.0 sampled](runs/run2_L24_noeos_rand1.0/random_r1.0_sampled.jpg) |

#### Section 8. What didn't work

The EOS-confound first writing pass and the prompt-only-while-writing run; the per-position map is text-only below.

| Condition | Run | Sheet |
| --- | --- | --- |
| baseline argmax | run2_L12 | [baseline argmax](runs/run2_L12/baseline_argmax.jpg) |
| baseline sampled | run2_L12 | [baseline sampled](runs/run2_L12/baseline_sampled.jpg) |
| coffee r=0.6 argmax | run2_L12 | [coffee r=0.6 argmax](runs/run2_L12/coffee_r0.6_argmax.jpg) |
| coffee r=0.6 sampled | run2_L12 | [coffee r=0.6 sampled](runs/run2_L12/coffee_r0.6_sampled.jpg) |
| coffee r=1.0 argmax | run2_L12 | [coffee r=1.0 argmax](runs/run2_L12/coffee_r1.0_argmax.jpg) |
| coffee r=1.0 sampled | run2_L12 | [coffee r=1.0 sampled](runs/run2_L12/coffee_r1.0_sampled.jpg) |
| control drawn | run2_L12 | [control drawn](runs/run2_L12/control_drawn.jpg) |
| control drawstates argmax | run2_L12 | [control drawstates argmax](runs/run2_L12/control_drawstates_argmax.jpg) |
| control drawstates sampled | run2_L12 | [control drawstates sampled](runs/run2_L12/control_drawstates_sampled.jpg) |
| ggb r=0.6 argmax | run2_L12 | [ggb r=0.6 argmax](runs/run2_L12/ggb_r0.6_argmax.jpg) |
| ggb r=0.6 sampled | run2_L12 | [ggb r=0.6 sampled](runs/run2_L12/ggb_r0.6_sampled.jpg) |
| ggb r=1.0 argmax | run2_L12 | [ggb r=1.0 argmax](runs/run2_L12/ggb_r1.0_argmax.jpg) |
| ggb r=1.0 sampled | run2_L12 | [ggb r=1.0 sampled](runs/run2_L12/ggb_r1.0_sampled.jpg) |
| random r=1.0 argmax | run2_L12 | [random r=1.0 argmax](runs/run2_L12/random_r1.0_argmax.jpg) |
| random r=1.0 sampled | run2_L12 | [random r=1.0 sampled](runs/run2_L12/random_r1.0_sampled.jpg) |
| circle r=1.0 argmax | run2_L12_noeos_po | [circle r=1.0 argmax](runs/run2_L12_noeos_po/circle_r1.0_argmax.jpg) |
| circle r=1.0 sampled | run2_L12_noeos_po | [circle r=1.0 sampled](runs/run2_L12_noeos_po/circle_r1.0_sampled.jpg) |
| circle r=2.0 argmax | run2_L12_noeos_po | [circle r=2.0 argmax](runs/run2_L12_noeos_po/circle_r2.0_argmax.jpg) |
| circle r=2.0 sampled | run2_L12_noeos_po | [circle r=2.0 sampled](runs/run2_L12_noeos_po/circle_r2.0_sampled.jpg) |
| coffee r=1.0 argmax | run2_L12_noeos_po | [coffee r=1.0 argmax](runs/run2_L12_noeos_po/coffee_r1.0_argmax.jpg) |
| coffee r=1.0 sampled | run2_L12_noeos_po | [coffee r=1.0 sampled](runs/run2_L12_noeos_po/coffee_r1.0_sampled.jpg) |
| coffee r=2.0 argmax | run2_L12_noeos_po | [coffee r=2.0 argmax](runs/run2_L12_noeos_po/coffee_r2.0_argmax.jpg) |
| coffee r=2.0 sampled | run2_L12_noeos_po | [coffee r=2.0 sampled](runs/run2_L12_noeos_po/coffee_r2.0_sampled.jpg) |
| ggb r=1.0 argmax | run2_L12_noeos_po | [ggb r=1.0 argmax](runs/run2_L12_noeos_po/ggb_r1.0_argmax.jpg) |
| ggb r=1.0 sampled | run2_L12_noeos_po | [ggb r=1.0 sampled](runs/run2_L12_noeos_po/ggb_r1.0_sampled.jpg) |
| ggb r=2.0 argmax | run2_L12_noeos_po | [ggb r=2.0 argmax](runs/run2_L12_noeos_po/ggb_r2.0_argmax.jpg) |
| ggb r=2.0 sampled | run2_L12_noeos_po | [ggb r=2.0 sampled](runs/run2_L12_noeos_po/ggb_r2.0_sampled.jpg) |
| random r=2.0 argmax | run2_L12_noeos_po | [random r=2.0 argmax](runs/run2_L12_noeos_po/random_r2.0_argmax.jpg) |
| random r=2.0 sampled | run2_L12_noeos_po | [random r=2.0 sampled](runs/run2_L12_noeos_po/random_r2.0_sampled.jpg) |
| random s2 r=2.0 argmax | run2_L12_noeos_po | [random s2 r=2.0 argmax](runs/run2_L12_noeos_po/random_s2_r2.0_argmax.jpg) |
| random s2 r=2.0 sampled | run2_L12_noeos_po | [random s2 r=2.0 sampled](runs/run2_L12_noeos_po/random_s2_r2.0_sampled.jpg) |

#### Runs not discussed in the text

None; every run with a contact sheet is referenced in one of the sections above.

### B. Texts

Full essays and descriptions (`samples.md`), the language-head readouts of drawing states (`posthoc.md`, `tokens.md`), and the per-position map writeup (`results.md`).

#### Section 4. Drawing: features, not objects

No text files; this section's runs are contact sheets only.

#### Section 5. Writing: the two heads read different things

- [run2_L12_noeos/samples.md](runs/run2_L12_noeos/samples.md)
- [run2_L12_noeos_dose/samples.md](runs/run2_L12_noeos_dose/samples.md)
- [run3_mirror_L12/posthoc.md](runs/run3_mirror_L12/posthoc.md)
- [run3_mirror_L12/tokens.md](runs/run3_mirror_L12/tokens.md)
- [text_circle_L12_hi/samples.md](runs/text_circle_L12_hi/samples.md)
- [text_circle_L12_lo/samples.md](runs/text_circle_L12_lo/samples.md)
- [text_circle_L18_hi/samples.md](runs/text_circle_L18_hi/samples.md)
- [text_circle_L18_lo/samples.md](runs/text_circle_L18_lo/samples.md)
- [text_coffee_L12_hi/samples.md](runs/text_coffee_L12_hi/samples.md)
- [text_coffee_L12_lo/samples.md](runs/text_coffee_L12_lo/samples.md)
- [text_coffee_L18_hi/samples.md](runs/text_coffee_L18_hi/samples.md)
- [text_coffee_L18_lo/samples.md](runs/text_coffee_L18_lo/samples.md)
- [text_ggb_L12_hi/samples.md](runs/text_ggb_L12_hi/samples.md)
- [text_ggb_L12_lo/samples.md](runs/text_ggb_L12_lo/samples.md)
- [text_ggb_L18_hi/samples.md](runs/text_ggb_L18_hi/samples.md)
- [text_ggb_L18_lo/samples.md](runs/text_ggb_L18_lo/samples.md)

#### Section 6. No steering at all

- [run2_L12_visual/samples.md](runs/run2_L12_visual/samples.md)

#### Section 7. What a random direction does

- [run2_L12_noeos_rand/samples.md](runs/run2_L12_noeos_rand/samples.md)
- [run2_L6_noeos/samples.md](runs/run2_L6_noeos/samples.md)
- [run2_L6_noeos_rand0.6/samples.md](runs/run2_L6_noeos_rand0.6/samples.md)
- [run2_L6_noeos_rand1.0/samples.md](runs/run2_L6_noeos_rand1.0/samples.md)
- [run2_L18_noeos/samples.md](runs/run2_L18_noeos/samples.md)
- [run2_L18_noeos_rand0.6/samples.md](runs/run2_L18_noeos_rand0.6/samples.md)
- [run2_L18_noeos_rand1.0/samples.md](runs/run2_L18_noeos_rand1.0/samples.md)
- [run2_L24_noeos/samples.md](runs/run2_L24_noeos/samples.md)
- [run2_L24_noeos_rand0.6/samples.md](runs/run2_L24_noeos_rand0.6/samples.md)
- [run2_L24_noeos_rand1.0/samples.md](runs/run2_L24_noeos_rand1.0/samples.md)

#### Section 8. What didn't work

- [run2_L12/samples.md](runs/run2_L12/samples.md)
- [run2_L12_noeos_po/samples.md](runs/run2_L12_noeos_po/samples.md)
- [run4_posmap_L12/results.md](runs/run4_posmap_L12/results.md)
- [run4_posmap_L12/heatmaps.png](runs/run4_posmap_L12/heatmaps.png) and [profiles.png](runs/run4_posmap_L12/profiles.png), the per-position map figures

### C. Invocations

The exact shell command for every run reported here.

- [queue_circle.sh](runs/queue_circle.sh)
- [queue_fu.sh](runs/queue_fu.sh)
- [queue_po.sh](runs/queue_po.sh)
- [queue_po2.sh](runs/queue_po2.sh)
- [queue_po_write.sh](runs/queue_po_write.sh)
- [queue_rand_layers.sh](runs/queue_rand_layers.sh)
- [queue_run2.sh](runs/queue_run2.sh)
- [queue_run2b.sh](runs/queue_run2b.sh)
- [queue_run2c.sh](runs/queue_run2c.sh)
- [queue_run3.sh](runs/queue_run3.sh)
- [queue_run5.sh](runs/queue_run5.sh)
- [queue_sweep_dose.sh](runs/queue_sweep_dose.sh)
- [queue_text.sh](runs/queue_text.sh)
- [queue_text_hi.sh](runs/queue_text_hi.sh)

