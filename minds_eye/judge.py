"""CLIP zero-shot judge: does an image depict a concept?

Scores each image against a pair of prompts ("a photo of coffee ..." vs "a photo
with no coffee ...") and returns the softmax probability of the concept prompt.
A per-concept score in [0,1]; aggregate is the mean over a batch. OpenAI
CLIP ViT-L/14 is already in the HF cache.
"""

from __future__ import annotations

import numpy as np
import torch

CLIP_ID = "openai/clip-vit-large-patch14"

CONCEPT_PROMPTS = {
    "coffee": (
        ["a photo of a cup of coffee", "a photo of coffee", "a mug of coffee",
         "an espresso", "a latte"],
        ["a photo of a street", "a photo of a landscape", "a photo of a room",
         "a photo of a bicycle", "a photo of a book"],
    ),
    "forest": (
        ["a photo of a forest", "trees in a forest", "woodland"],
        ["a photo of a city", "a photo of a room", "a photo of a street"],
    ),
    "ggb": (
        ["a photo of the Golden Gate Bridge", "the Golden Gate Bridge",
         "a red suspension bridge", "a large bridge over water"],
        ["a photo of a landscape with no bridge", "a photo of a street",
         "a photo of a room", "a photo of a coastline", "a photo of hills"],
    ),
}


class ClipJudge:
    def __init__(self, device: str = "cuda:0"):
        from transformers import CLIPModel, CLIPProcessor

        self.device = device
        self.model = CLIPModel.from_pretrained(CLIP_ID).to(device).eval()
        self.proc = CLIPProcessor.from_pretrained(CLIP_ID)

    @torch.inference_mode()
    def concept_prob(self, imgs: np.ndarray, concept: str) -> np.ndarray:
        """(n,H,W,3) uint8 -> (n,) prob the image shows ``concept``.

        Prob = softmax over [best concept prompt sim, best anti prompt sim] after
        averaging image-text cosine within each group, temperature from CLIP's
        logit scale. Returns per-image probabilities.
        """
        from PIL import Image

        pos, neg = CONCEPT_PROMPTS[concept]
        texts = pos + neg
        pil = [Image.fromarray(im) for im in imgs]
        inp = self.proc(text=texts, images=pil, return_tensors="pt", padding=True)
        inp = {k: v.to(self.device) for k, v in inp.items()}
        out = self.model(**inp)
        logits = out.logits_per_image  # (n, len(texts)), already * logit_scale
        npos = len(pos)
        pos_max = logits[:, :npos].max(dim=1).values
        neg_max = logits[:, npos:].max(dim=1).values
        stacked = torch.stack([pos_max, neg_max], dim=1)
        return torch.softmax(stacked, dim=1)[:, 0].float().cpu().numpy()


CONCEPT_PROMPTS["circle"] = (
    ["a photo of a circle", "a round object", "a circular shape", "a ball",
     "a round disc"],
    ["a photo of a street", "a photo of a landscape", "a photo of a room",
     "a photo of a bookshelf", "a photo with square shapes"],
)


class HoughJudge:
    """Classical circle detector: no learned judge to game.

    Tuned 2026-11-16 against synthetic shapes and the GGB baseline tiles:
    dp=1.5, Gaussian blur 9, param1=200, param2=90 gives filled circle=1,
    ring=1, square=0, white noise=0, and on the 16 baseline tiles 0 everywhere
    except the two bicycle shots (2 each: the wheels). Score per image =
    min(1, n_circles / 3); the absolute count is arbitrary, the
    baseline-vs-steer-vs-random delta on fixed settings is the measurement.
    """

    def __init__(self, min_r_frac: float = 0.08, max_r_frac: float = 0.45,
                 blur: int = 9, param1: int = 200, param2: int = 90,
                 dp: float = 1.5):
        self.min_r_frac = min_r_frac
        self.max_r_frac = max_r_frac
        self.blur = blur
        self.param1 = param1
        self.param2 = param2
        self.dp = dp

    def count(self, img: np.ndarray) -> int:
        import cv2

        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        g = cv2.GaussianBlur(g, (self.blur, self.blur), 0)
        h = min(g.shape[:2])
        c = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=self.dp, minDist=h // 6,
                             param1=self.param1, param2=self.param2,
                             minRadius=int(h * self.min_r_frac),
                             maxRadius=int(h * self.max_r_frac))
        return 0 if c is None else int(c.shape[1])

    def concept_prob(self, imgs: np.ndarray, concept: str = "circle") -> np.ndarray:
        return np.array([min(1.0, self.count(im) / 3.0) for im in imgs],
                        dtype=np.float32)
