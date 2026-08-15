"""On-disk store for pooled residual-stream activations.

Layout
------
::

    <run>/activations/
        manifest.json                     compatibility fingerprint + schema
        shard000/
            chunk_00000.safetensors       one tensor per layer: (n_rows, hidden)
            chunk_00000.index.parquet     one row per example, aligned by row_in_chunk
            chunk_00001.safetensors
            ...
            skipped.jsonl                 examples excluded, with reasons
        shard001/
            ...

Design notes
------------
* **One tensor per layer per chunk** (not one ``(n, n_layers, hidden)`` tensor).
  Direction fitting and evaluation both iterate layer-by-layer over the whole
  dataset, so per-layer tensors let us stream one layer at a time instead of
  paging in every layer to use one.
* **The index parquet is the commit record.** Tensors are written first, index
  second, both via atomic rename. A chunk missing its index is an incomplete
  write and gets discarded on resume, so a killed pod never corrupts a run.
* **Resume is by example id**, not by counter, so it survives changes in shard
  assignment or ordering.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

MANIFEST_NAME = "manifest.json"
INDEX_COLUMNS = [
    "example_id",
    "source",
    "emotion",
    "topic",
    "topic_id",
    "story_idx",
    "split",
    "content_sha1",
    "n_tokens",
    "n_pooled_tokens",
    "row_in_chunk",
    "chunk",
    "shard",
]


def layer_key(layer: int) -> str:
    return f"layer_{layer:03d}"


def estimate_storage_bytes(n_examples: int, n_layers: int, hidden_size: int, nbytes: int) -> int:
    """Raw activation bytes (index/manifest overhead is negligible by comparison)."""
    return int(n_examples) * int(n_layers) * int(hidden_size) * int(nbytes)


def human_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:.2f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.2f} TiB"  # pragma: no cover


# --------------------------------------------------------------------------- #
# Manifest / compatibility
# --------------------------------------------------------------------------- #

class IncompatibleRunError(RuntimeError):
    """Raised when an existing run's settings differ from the current ones."""


class MissingChunkError(FileNotFoundError):
    """An activation chunk is indexed but absent locally (it lives in R2 only)."""


def _diff_fingerprints(existing: dict, current: dict) -> list[str]:
    diffs = []
    for key in sorted(set(existing) | set(current)):
        old, new = existing.get(key, "<absent>"), current.get(key, "<absent>")
        if old != new:
            diffs.append(f"  {key}: existing={old!r} current={new!r}")
    return diffs


def read_manifest(activations_dir: Path) -> dict | None:
    path = Path(activations_dir) / MANIFEST_NAME
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(activations_dir: Path, manifest: dict) -> None:
    path = Path(activations_dir) / MANIFEST_NAME
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)


def init_or_check_manifest(
    activations_dir: Path,
    fingerprint: dict,
    extra: dict | None = None,
    allow_overwrite: bool = False,
) -> dict:
    """Create the manifest, or verify an existing one matches ``fingerprint``.

    Raises :class:`IncompatibleRunError` rather than silently mixing activations
    produced under different settings.
    """
    activations_dir = Path(activations_dir)
    activations_dir.mkdir(parents=True, exist_ok=True)
    existing = read_manifest(activations_dir)

    if existing is None:
        manifest = {"fingerprint": fingerprint, **(extra or {})}
        write_manifest(activations_dir, manifest)
        return manifest

    diffs = _diff_fingerprints(existing.get("fingerprint", {}), fingerprint)
    if diffs:
        if not allow_overwrite:
            raise IncompatibleRunError(
                "Existing activations in\n"
                f"  {activations_dir}\n"
                "were produced with different settings:\n"
                + "\n".join(diffs)
                + "\n\nUse a new run_name, or pass --overwrite to delete and re-extract."
            )
        shutil.rmtree(activations_dir)
        activations_dir.mkdir(parents=True, exist_ok=True)
        manifest = {"fingerprint": fingerprint, **(extra or {})}
        write_manifest(activations_dir, manifest)
        return manifest

    # Compatible: refresh non-fingerprint metadata (timestamps, package versions).
    merged = {**existing, **(extra or {}), "fingerprint": fingerprint}
    write_manifest(activations_dir, merged)
    return merged


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #

@dataclass
class ActivationWriter:
    """Buffers pooled activations and flushes them as chunks.

    Args:
        activations_dir: the run's ``activations/`` directory.
        shard_index: which shard this process owns.
        layers: hidden-state indices being saved (must match every ``add`` call).
        dtype: storage dtype name, e.g. ``"bfloat16"``.
        chunk_size: examples per chunk file.
        on_chunk_written: callback receiving the paths of each completed chunk
            (used to mirror to R2).
    """

    activations_dir: Path
    shard_index: int
    layers: Sequence[int]
    dtype: str = "bfloat16"
    chunk_size: int = 512
    on_chunk_written: Callable[[list[Path]], None] | None = None

    _records: list[dict] = field(default_factory=list, init=False)
    _buffers: dict[int, list] = field(default_factory=dict, init=False)
    _next_chunk: int = field(default=0, init=False)
    _n_written: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.activations_dir = Path(self.activations_dir)
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        self._discard_incomplete_chunks()
        self._next_chunk = len(self.existing_chunks())
        self._buffers = {int(l): [] for l in self.layers}

    # -- paths ------------------------------------------------------------- #

    @property
    def shard_dir(self) -> Path:
        return self.activations_dir / f"shard{self.shard_index:03d}"

    @property
    def skipped_path(self) -> Path:
        return self.shard_dir / "skipped.jsonl"

    def existing_chunks(self) -> list[Path]:
        return sorted(self.shard_dir.glob("chunk_*.safetensors"))

    def _discard_incomplete_chunks(self) -> None:
        """Remove tensor files whose index never landed, plus stale temp files."""
        for tmp in self.shard_dir.glob("*.tmp"):
            tmp.unlink()
        for tensors in self.shard_dir.glob("chunk_*.safetensors"):
            if not tensors.with_suffix(".index.parquet").exists():
                tensors.unlink()

    # -- writing ----------------------------------------------------------- #

    def add(self, records: Sequence[dict], pooled: dict[int, "object"]) -> None:
        """Buffer a batch. ``pooled[layer]`` rows must align with ``records``."""
        if set(int(k) for k in pooled) != set(self._buffers):
            raise ValueError(
                f"pooled layers {sorted(int(k) for k in pooled)} != writer layers "
                f"{sorted(self._buffers)}"
            )
        for layer, tensor in pooled.items():
            if tensor.shape[0] != len(records):
                raise ValueError(
                    f"layer {layer}: {tensor.shape[0]} rows for {len(records)} records"
                )
            self._buffers[int(layer)].append(tensor.detach().to("cpu"))
        self._records.extend(records)

        while len(self._records) >= self.chunk_size:
            self._flush_chunk(self.chunk_size)

    def flush(self) -> None:
        """Write out whatever is buffered (called at end of shard)."""
        while self._records:
            self._flush_chunk(min(self.chunk_size, len(self._records)))

    def _flush_chunk(self, n: int) -> None:
        import torch
        from safetensors.torch import save_file

        chunk_id = self._next_chunk
        stem = f"chunk_{chunk_id:05d}"
        tensors_path = self.shard_dir / f"{stem}.safetensors"
        index_path = self.shard_dir / f"{stem}.index.parquet"

        tensors = {}
        leftover: dict[int, list] = {}
        for layer, parts in self._buffers.items():
            if not parts:
                raise RuntimeError(
                    f"layer {layer} has no buffered activations but {len(self._records)} "
                    "records are pending; writer state is inconsistent"
                )
            full = torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]
            tensors[layer_key(layer)] = full[:n].contiguous()
            rest = full[n:]
            leftover[layer] = [rest] if rest.shape[0] else []

        records = self._records[:n]
        index = pd.DataFrame(records)
        index["row_in_chunk"] = np.arange(n, dtype=np.int64)
        index["chunk"] = stem
        index["shard"] = self.shard_index
        for col in INDEX_COLUMNS:
            if col not in index.columns:
                index[col] = pd.NA
        index = index[INDEX_COLUMNS]

        # Tensors first, index second: the index is what marks a chunk complete.
        tmp_tensors = tensors_path.with_suffix(".safetensors.tmp")
        save_file(
            tensors,
            str(tmp_tensors),
            metadata={
                "shard": str(self.shard_index),
                "chunk": stem,
                "dtype": self.dtype,
                "layers": json.dumps(sorted(int(l) for l in self.layers)),
                "n_rows": str(n),
            },
        )
        os.replace(tmp_tensors, tensors_path)

        tmp_index = index_path.with_suffix(".parquet.tmp")
        index.to_parquet(tmp_index, index=False)
        os.replace(tmp_index, index_path)

        self._buffers = leftover
        self._records = self._records[n:]
        self._next_chunk += 1
        self._n_written += n
        if self.on_chunk_written:
            self.on_chunk_written([tensors_path, index_path])

    def record_skipped(self, entries: Iterable[dict]) -> None:
        """Append skipped-example records (example_id, reason, counts)."""
        entries = list(entries)
        if not entries:
            return
        with self.skipped_path.open("a", encoding="utf-8") as fh:
            for entry in entries:
                fh.write(json.dumps(entry, default=str) + "\n")

    def close(self) -> None:
        self.flush()

    def __enter__(self) -> "ActivationWriter":
        return self

    def __exit__(self, *exc) -> None:
        # Always flush buffered rows, even on error: partial results are useful
        # and resume-by-example-id makes them safe to keep.
        self.close()


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #

class ActivationStore:
    """Read-side view over a run's activations, across all shards."""

    def __init__(self, activations_dir: str | Path):
        self.activations_dir = Path(activations_dir)
        manifest = read_manifest(self.activations_dir)
        if manifest is None:
            raise FileNotFoundError(
                f"no {MANIFEST_NAME} in {self.activations_dir}; run extraction first"
            )
        self.manifest = manifest
        fp = manifest.get("fingerprint", {})
        self.layers: list[int] = [int(l) for l in fp.get("layers", [])]
        self.hidden_size: int = int(fp.get("hidden_size", 0))
        self.dtype: str = fp.get("activation_dtype", "bfloat16")
        self.index = self._load_index()

    def _load_index(self) -> pd.DataFrame:
        parts = []
        for index_path in sorted(self.activations_dir.glob("shard*/chunk_*.index.parquet")):
            df = pd.read_parquet(index_path)
            df["chunk_path"] = str(index_path.with_suffix("").with_suffix(".safetensors"))
            parts.append(df)
        if not parts:
            raise FileNotFoundError(f"no completed chunks under {self.activations_dir}")
        index = pd.concat(parts, ignore_index=True)
        dupes = int(index["example_id"].duplicated().sum())
        if dupes:
            raise RuntimeError(
                f"{dupes} duplicate example_ids across chunks in {self.activations_dir}; "
                "shards likely overlapped. Re-check --shard-index/--num-shards."
            )
        return index.sort_values(["source", "emotion", "topic_id", "story_idx"],
                                na_position="first").reset_index(drop=True)

    # -- queries ----------------------------------------------------------- #

    def subset(
        self,
        source: str | None = None,
        splits: Sequence[str] | None = None,
        emotions: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Rows of the index matching the given filters (order preserved)."""
        df = self.index
        if source is not None:
            df = df[df["source"] == source]
        if splits is not None:
            df = df[df["split"].isin(list(splits))]
        if emotions is not None:
            df = df[df["emotion"].isin(list(emotions))]
        return df.reset_index(drop=True)

    def completed_example_ids(self) -> set[str]:
        return set(self.index["example_id"].tolist())

    def load_layer(self, layer: int, rows: pd.DataFrame | None = None) -> np.ndarray:
        """Load one layer's activations as ``(n, hidden)`` float32.

        ``rows`` is a slice of :attr:`index` (from :meth:`subset`); the returned
        array is in that slice's row order. Only the requested layer is read from
        each chunk.
        """
        import torch
        from safetensors import safe_open

        if layer not in self.layers:
            raise KeyError(f"layer {layer} not in stored layers {self.layers}")
        rows = self.index if rows is None else rows
        if len(rows) == 0:
            return np.zeros((0, self.hidden_size), dtype=np.float32)

        out = np.empty((len(rows), self.hidden_size), dtype=np.float32)
        key = layer_key(layer)
        work = rows.assign(_pos=np.arange(len(rows)))

        for chunk_path, group in work.groupby("chunk_path", sort=True):
            if not Path(chunk_path).exists():
                # Normal state after a run with delete_local_after_sync=True: the
                # index parquet is local but the tensors live only in R2. Say how to
                # fix it rather than leaving a bare FileNotFoundError.
                raise MissingChunkError(
                    f"activation chunk is not on this machine:\n  {chunk_path}\n\n"
                    "The index is present, so this run was extracted with "
                    "delete_local_after_sync=True and the tensors are in R2 only.\n"
                    "Download them first:\n\n"
                    f"  python run.py r2 pull {self.activations_dir} "
                    f"--prefix <r2_root>/<run_name>\n\n"
                    "(the run's manifest.json records the run name; see the README "
                    'section "Sharing activations" for credentials)'
                )
            with safe_open(str(chunk_path), framework="pt") as fh:
                if key not in fh.keys():
                    raise KeyError(f"{key} missing from {chunk_path}")
                tensor = fh.get_tensor(key)
            # .copy() so torch gets a writable array (it warns otherwise).
            local = group["row_in_chunk"].to_numpy().copy()
            out[group["_pos"].to_numpy()] = tensor[local].to(torch.float32).numpy()
        return out

    def load_layers(self, layers: Sequence[int], rows: pd.DataFrame | None = None) -> dict[int, np.ndarray]:
        return {int(l): self.load_layer(int(l), rows) for l in layers}

    def skipped(self) -> pd.DataFrame:
        """All skipped-example records across shards."""
        records = []
        for path in sorted(self.activations_dir.glob("shard*/skipped.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    records.append(json.loads(line))
        return pd.DataFrame(records)

    def total_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.activations_dir.rglob("*") if p.is_file())


def completed_example_ids(activations_dir: str | Path) -> set[str]:
    """Example ids already written, without constructing a full store.

    Used by extraction to resume; tolerates a directory with no chunks yet.
    """
    ids: set[str] = set()
    for index_path in sorted(Path(activations_dir).glob("shard*/chunk_*.index.parquet")):
        ids.update(pd.read_parquet(index_path, columns=["example_id"])["example_id"].tolist())
    return ids
