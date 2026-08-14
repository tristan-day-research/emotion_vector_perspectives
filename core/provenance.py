"""Reproducibility metadata: environment capture and human-readable run records.

Every run writes two files into its output directory:

* ``run_config.txt``   — everything needed to re-run, in plain text
* ``run_manifest.json`` — the same content, machine-readable
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PACKAGES_OF_INTEREST = (
    "torch",
    "transformers",
    "datasets",
    "safetensors",
    "accelerate",
    "numpy",
    "scipy",
    "pandas",
    "pyarrow",
    "scikit-learn",
    "huggingface_hub",
    "bitsandbytes",
    "boto3",
    "matplotlib",
)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_stamp() -> str:
    """Compact timestamp suitable for directory names."""
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def package_versions() -> dict[str, str]:
    import importlib.metadata as md

    out = {}
    for name in PACKAGES_OF_INTEREST:
        try:
            out[name] = md.version(name)
        except Exception:
            out[name] = "not installed"
    return out


def git_info() -> dict[str, str]:
    """Commit, branch and dirty state of this repository."""

    def run(*args: str) -> str:
        try:
            return subprocess.run(
                args,
                cwd=Path(__file__).resolve().parent.parent,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            ).stdout.strip()
        except Exception:
            return ""

    status = run("git", "status", "--porcelain")
    return {
        "commit": run("git", "rev-parse", "HEAD") or "unknown",
        "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD") or "unknown",
        "dirty": str(bool(status)),
        "n_dirty_files": str(len(status.splitlines())) if status else "0",
    }


def hardware_info() -> dict[str, object]:
    info: dict[str, object] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "hostname": platform.node(),
    }
    try:
        import torch

        info["torch_cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            info["gpus"] = [
                {
                    "name": torch.cuda.get_device_name(i),
                    "total_memory_gib": round(
                        torch.cuda.get_device_properties(i).total_memory / 1024**3, 2
                    ),
                    "capability": ".".join(map(str, torch.cuda.get_device_capability(i))),
                }
                for i in range(torch.cuda.device_count())
            ]
    except ImportError:
        info["torch_cuda_available"] = "torch not installed"
    return info


def environment_snapshot() -> dict[str, object]:
    """Everything about the environment we care to record."""
    return {
        "timestamp_utc": utc_timestamp(),
        "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
        "cwd": str(Path.cwd()),
        "git": git_info(),
        "hardware": hardware_info(),
        "packages": package_versions(),
        "env_vars": {
            k: os.environ.get(k)
            for k in ("HF_HOME", "HF_HUB_OFFLINE", "CUDA_VISIBLE_DEVICES", "R2_BUCKET")
            if os.environ.get(k)
        },
    }


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def _render(value: object, indent: int = 0) -> list[str]:
    pad = "  " * indent
    lines: list[str] = []
    if isinstance(value, dict):
        for key in value:
            item = value[key]
            if isinstance(item, (dict, list, tuple)) and item:
                lines.append(f"{pad}{key}:")
                lines += _render(item, indent + 1)
            else:
                lines.append(f"{pad}{key}: {_scalar(item)}")
    elif isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, (dict, list, tuple)):
                lines += _render(item, indent)
                lines.append("")
            else:
                lines.append(f"{pad}- {_scalar(item)}")
    else:
        lines.append(f"{pad}{_scalar(value)}")
    return lines


def _scalar(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def write_run_record(
    out_dir: str | Path,
    title: str,
    sections: dict[str, object],
    txt_name: str = "run_config.txt",
    json_name: str = "run_manifest.json",
) -> tuple[Path, Path]:
    """Write the human-readable ``.txt`` and machine-readable ``.json`` records.

    ``sections`` maps section headings to arbitrary nested dict/list/scalar data.
    An ``environment`` section is added automatically.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {**sections, "environment": environment_snapshot()}

    lines = [
        "=" * 78,
        title,
        "=" * 78,
        f"written: {payload['environment']['timestamp_utc']}",
        "",
        "This file records every parameter and piece of metadata needed to reproduce",
        "the run. Section order is stable; values are literal.",
        "",
    ]
    for heading, content in payload.items():
        lines.append("-" * 78)
        lines.append(heading.upper())
        lines.append("-" * 78)
        lines += _render(content)
        lines.append("")

    txt_path = out_dir / txt_name
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    json_path = out_dir / json_name
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=False, default=str), encoding="utf-8")
    return txt_path, json_path
