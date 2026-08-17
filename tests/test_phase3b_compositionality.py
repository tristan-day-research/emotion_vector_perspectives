"""Phase 3b compositionality math on a known synthetic factorial geometry."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emotion_pca_jlens import phase3b_compositionality as c


def main() -> int:
    rng = np.random.default_rng(4)
    d = 24
    mu = rng.normal(size=d)
    gv = rng.normal(size=d); gv /= np.linalg.norm(gv)
    ga = rng.normal(size=d); ga -= np.dot(ga, gv) * gv; ga /= np.linalg.norm(ga)
    gi = 0.08 * rng.normal(size=d); gi /= max(np.linalg.norm(gi), 1e-12) / 0.08
    labels = np.asarray([(v, a) for v, a in c.QUADRANTS for _ in range(4)])
    v, a = labels[:, 0], labels[:, 1]
    matrix = np.asarray([
        mu + vv * gv + aa * ga + vv * aa * gi + 0.03 * rng.normal(size=d)
        for vv, aa in labels
    ])

    geom = c.geometry_metrics(matrix, v, a)
    heldout, summary = c.cv_summary(matrix, v, a)
    components = c.factorial_components(matrix, v, a)
    checks = {
        "arousal contrast is reused": geom["arousal_contrast_cosine"] > 0.95,
        "valence contrast is reused": geom["valence_contrast_cosine"] > 0.95,
        "interaction is small": geom["interaction_to_main_ratio"] < 0.15,
        "held-out additive reconstruction works": summary["additive_mean_cosine"] > 0.9,
        "one row per held-out emotion and model": len(heldout) == 32,
        "supervised components recover their generators": (
            c.cosine(components["valence"], gv) > 0.98
            and c.cosine(components["arousal"], ga) > 0.98
        ),
    }
    failures = [name for name, passed in checks.items() if not passed]
    for name, passed in checks.items():
        print(("  ok   " if passed else "  FAIL ") + name)
    if failures:
        print("failures:", failures)
        return 1
    print("all Phase 3b tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
