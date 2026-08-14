"""Canonical project paths.

Everything large (datasets, HF caches, activations) lives under ``data/`` or
``outputs/``, both of which are git-ignored.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ANALYSIS_DIR = PROJECT_ROOT / "analysis"
CORE_DIR = PROJECT_ROOT / "core"
DATA_DIR = PROJECT_ROOT / "data"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
EXTRACT_DIR = PROJECT_ROOT / "extract_emotion_vectors"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

#: Canonical list of the 171 emotion words used by Anthropic (Sofroniew et al., 2026),
#: transcribed from the paper's appendix. Used only to validate config/dataset overlap.
EMOTIONS_171_FILE = DATA_DIR / "emotions_171.txt"


def hf_cache_dir() -> Path:
    """Where Hugging Face artefacts are cached.

    Respects ``HF_HOME`` if set (recommended on RunPod: point it at a volume),
    otherwise falls back to ``data/hf_cache`` inside the project.
    """
    env = os.environ.get("HF_HOME")
    if env:
        return Path(env)
    return DATA_DIR / "hf_cache"


def run_dir(run_name: str) -> Path:
    """Output directory for a named run."""
    return OUTPUTS_DIR / run_name


def load_emotions_171() -> list[str]:
    """The canonical 171 emotion words, in the paper's alphabetical order."""
    text = EMOTIONS_171_FILE.read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
