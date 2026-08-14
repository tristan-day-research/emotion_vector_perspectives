"""Deterministic seeding helpers.

We deliberately keep two notions of randomness separate:

* **Structural randomness** (topic splits, story subsampling) uses explicit
  ``numpy.random.Generator`` instances derived from a config seed, so the same
  config always yields the same splits regardless of import order or how many
  other random calls happened first.
* **Global RNG state** (torch/python/numpy) is seeded once per process for
  reproducibility of anything incidental.
"""

from __future__ import annotations

import hashlib
import os
import random

import numpy as np


def set_global_seeds(seed: int, deterministic_torch: bool = True) -> None:
    """Seed python/numpy/torch global RNGs.

    ``deterministic_torch`` also disables cuDNN autotuning and requests
    deterministic kernels where available. Forward-only inference is already
    close to deterministic, but batch composition and reduction order can still
    perturb low-order bits; see README "Reproducibility caveats".
    """
    random.seed(seed)
    np.random.seed(seed % (2**32))
    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    try:
        import torch
    except ImportError:  # dry-run / no-torch environments
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def stable_hash(*parts: object) -> int:
    """A process-independent 64-bit hash of the given parts.

    ``hash()`` in Python is salted per process; this is not. Used to derive
    sub-seeds from names (e.g. per-emotion subsampling) so that adding or
    removing an emotion from the config does not perturb the others.
    """
    joined = "\x1f".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.blake2b(joined, digest_size=8).digest()
    return int.from_bytes(digest, "big")


def rng_for(seed: int, *parts: object) -> np.random.Generator:
    """A ``Generator`` deterministically derived from ``seed`` and ``parts``."""
    return np.random.default_rng(np.random.SeedSequence([seed, stable_hash(*parts)]))
