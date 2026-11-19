"""Concept prompt sets for building steering directions by difference-of-means.

For a concept C we need sentences that mention C and matched control sentences
that are the same kind of everyday statement without C. The direction is
mean(act[C]) - mean(act[control]); a random direction at equal norm is the
control condition in every run.

``coffee`` is the pre-registered primary concept (the Golden-Gate-with-a-
paintbrush case). The others let us check the effect is not coffee-specific.
"""

COFFEE_POS = [
    "I poured myself a hot cup of coffee this morning.",
    "The cafe's espresso machine hissed as it pulled a shot.",
    "She cradled a steaming mug of black coffee in both hands.",
    "The smell of fresh-ground coffee beans filled the kitchen.",
    "He ordered a latte and sat by the window.",
    "A barista frothed the milk for a cappuccino.",
    "The diner refilled my coffee without being asked.",
    "Nothing beats a strong cup of coffee before work.",
    "The French press sat next to two ceramic mugs.",
    "Steam rose off the dark roast in the paper cup.",
    "We met for coffee at the corner espresso bar.",
    "The office coffee pot had been brewing since dawn.",
    "He added cream and sugar to his morning coffee.",
    "The roastery smelled of beans and caramel.",
    "A cappuccino with cocoa dusted on the foam.",
    "Her thermos kept the coffee hot until noon.",
]

# Matched everyday statements, no coffee, same register and length band.
COFFEE_CTRL = [
    "I laced up my running shoes this morning.",
    "The printer whirred as it spat out the page.",
    "She cradled a worn paperback in both hands.",
    "The smell of fresh-cut grass filled the yard.",
    "He ordered a sandwich and sat by the window.",
    "A clerk stacked the apples in a neat pyramid.",
    "The waiter refilled my water without being asked.",
    "Nothing beats a brisk walk before work.",
    "The toolbox sat next to two folding chairs.",
    "Steam rose off the hot soup in the bowl.",
    "We met for lunch at the corner deli.",
    "The office radio had been playing since dawn.",
    "He added salt and pepper to his morning eggs.",
    "The bakery smelled of bread and cinnamon.",
    "A milkshake with sprinkles on the whipped cream.",
    "Her bottle kept the water cold until noon.",
]

FOREST_POS = [
    "Sunlight filtered through the tall pine trees.",
    "We hiked deep into the old-growth forest.",
    "Moss covered the roots of the ancient oaks.",
    "A deer stepped quietly through the woods.",
    "The forest floor was thick with fallen leaves.",
    "Birdsong echoed between the towering trunks.",
    "Ferns unfurled in the shade of the canopy.",
    "The trail wound between mossy evergreens.",
]
FOREST_CTRL = [
    "Sunlight glinted off the parked cars.",
    "We walked deep into the crowded market.",
    "Dust covered the shelves of the old shop.",
    "A cat slipped quietly through the alley.",
    "The warehouse floor was slick with oil.",
    "Chatter echoed between the office cubicles.",
    "Posters peeled in the heat of the subway.",
    "The road wound between concrete buildings.",
]

GGB_POS = [
    "The Golden Gate Bridge stood red against the fog.",
    "We drove across the Golden Gate Bridge at sunset.",
    "The Golden Gate Bridge towers rose above the bay.",
    "Tourists photographed the Golden Gate Bridge from the overlook.",
    "The Golden Gate Bridge's cables stretched toward the headlands.",
    "Fog rolled under the Golden Gate Bridge in the morning.",
    "A ship passed beneath the Golden Gate Bridge.",
    "The Golden Gate Bridge glowed orange in the evening light.",
    "Cyclists crossed the Golden Gate Bridge on the walkway.",
    "The Golden Gate Bridge spanned the strait in international orange.",
    "From Marin the Golden Gate Bridge framed the city skyline.",
    "The Golden Gate Bridge's deck disappeared into low cloud.",
    "Sailboats drifted past the Golden Gate Bridge piers.",
    "The red suspension towers of the Golden Gate Bridge caught the sun.",
    "The Golden Gate Bridge arched over the cold green water.",
    "Cars streamed across the Golden Gate Bridge at dusk.",
]

# Matched scenes: same register, outdoor/landmark feel, no bridge.
GGB_CTRL = [
    "The old lighthouse stood white against the fog.",
    "We drove along the coast highway at sunset.",
    "The cathedral towers rose above the square.",
    "Tourists photographed the waterfall from the overlook.",
    "The power lines stretched toward the hills.",
    "Fog rolled over the meadow in the morning.",
    "A ship passed along the open harbor.",
    "The sandstone cliffs glowed orange in the evening light.",
    "Cyclists crossed the park on the walkway.",
    "The water tower stood painted in faded blue.",
    "From the hill the valley framed the distant town.",
    "The mountain ridge disappeared into low cloud.",
    "Sailboats drifted past the stone jetty.",
    "The red barn roof caught the afternoon sun.",
    "The footpath arched over the cold green river.",
    "Cars streamed along the freeway at dusk.",
]

CONCEPTS = {
    "coffee": (COFFEE_POS, COFFEE_CTRL),
    "forest": (FOREST_POS, FOREST_CTRL),
    "ggb": (GGB_POS, GGB_CTRL),
}

# Neutral free-draw prompts for Run 1: the model is told to draw something that
# has nothing to do with the concept, so any concept content is the steer.
FREE_DRAW_PROMPTS = [
    "A photograph of a quiet residential street.",
    "A still life of objects on a desk.",
    "A landscape painting of rolling hills.",
    "A photo of a person reading on a park bench.",
    "An interior of an empty living room.",
    "A close-up of a bookshelf.",
    "A photo of a bicycle on a sidewalk.",
    "A painting of a harbor with boats.",
]

# Super-simple shape concept (2026-11-16): round things in everyday scenes,
# not geometry homework, so the vector is "round" rather than "math class".
# Controls are the same scenes with a non-round object in the same slot.
CIRCLE_POS = [
    "A big round clock hung on the wall above the counter.",
    "The full moon was a perfect circle over the rooftops.",
    "A round red ball sat in the middle of the lawn.",
    "She drew a circle on the whiteboard with a marker.",
    "The pizza arrived, a perfect round disc of cheese.",
    "A circular mirror hung in the hallway.",
    "The wheel was a wide round hoop with thick spokes.",
    "A round porthole looked out onto the sea.",
    "The coin was a small bright circle on the table.",
    "Ripples spread in circles across the pond.",
    "A round dinner plate sat empty on the tablecloth.",
    "The sun was a flat orange circle in the haze.",
    "A circular rug covered the center of the floor.",
    "The target was a set of concentric circles on the board.",
    "A round lollipop stuck out of the child's mouth.",
    "The manhole cover was a heavy iron circle in the road.",
]
CIRCLE_CTRL = [
    "A long wooden shelf hung on the wall above the counter.",
    "The crescent streetlight was a thin line over the rooftops.",
    "A square cardboard box sat in the middle of the lawn.",
    "She wrote a word on the whiteboard with a marker.",
    "The sandwich arrived, a thick wedge of bread and cheese.",
    "A rectangular painting hung in the hallway.",
    "The ladder was a tall frame with wooden rungs.",
    "A square window looked out onto the sea.",
    "The key was a small bright sliver on the table.",
    "Waves broke in lines along the shore.",
    "A folded napkin sat empty on the tablecloth.",
    "The sun was a thin orange band in the haze.",
    "A rectangular rug covered the center of the floor.",
    "The target was a tall post at the end of the board.",
    "A thin pretzel stick stuck out of the child's mouth.",
    "The drain grate was a heavy iron rectangle in the road.",
]

CONCEPTS["circle"] = (CIRCLE_POS, CIRCLE_CTRL)
