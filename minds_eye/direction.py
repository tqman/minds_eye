"""Build a concept direction from last-token activations, per layer.

Difference-of-means at each layer, plus a seeded random direction scaled to the
same L2 norm as the concept direction at that layer (the control condition).
Kept deliberately simple: the point of Run 1 is whether a known concept vector
reaches the image head at all, not a denoising contest. The tasp denoise/paired
machinery is available if Run 1 needs tightening.
"""

from __future__ import annotations

import numpy as np


def concept_directions(acts_pos: np.ndarray, acts_ctrl: np.ndarray) -> np.ndarray:
    """(N+,L,d) and (N-,L,d) -> (L,d) difference of means per layer."""
    return acts_pos.mean(axis=0) - acts_ctrl.mean(axis=0)


def random_like(direction_layer: np.ndarray, seed: int) -> np.ndarray:
    """A random vector with the same norm as ``direction_layer`` (1-D, d)."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(direction_layer.shape)
    return v / np.linalg.norm(v) * np.linalg.norm(direction_layer)


def layer_norms(acts: np.ndarray) -> np.ndarray:
    """Mean final-token residual norm per layer, (L,). Used to size the dose
    as a fraction of the typical residual norm, like the tasp ladder."""
    return np.linalg.norm(acts, axis=2).mean(axis=0)
