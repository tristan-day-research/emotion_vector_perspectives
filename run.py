#!/usr/bin/env python
"""Single entrypoint for the shared ``scripts/`` RunPod workflow.

The workflow's ``RUN_ENTRYPOINT`` is one fixed string; everything you type after
the command name is ``printf %q``-escaped and appended to it. So this file takes a
**stage name** as its first argument and forwards the rest verbatim to that
stage's ``main()``:

    python run.py extract_activations --set batch_size=8
    python run.py compute_directions
    python run.py evaluate_directions --set eval_bootstrap_n=1000

which via ``RUN_ENTRYPOINT="python run.py"`` becomes:

    run-emotionvectors extract_activations --set batch_size=8

This is a router, not a second config system: the ``--set field=value``
convention of :class:`VectorExtractionConfig` is untouched, and every stage keeps
working when invoked directly as ``python -m extract_emotion_vectors.<stage>``.

The ``all`` stage runs extract -> directions -> evaluate in sequence, which is
what you usually want for one pod session, since the pod is ephemeral and each
stage's arguments are mostly shared.
"""

from __future__ import annotations

import sys

STAGES = {
    "extract_activations": "extract_emotion_vectors.extract_activations",
    "compute_directions": "extract_emotion_vectors.compute_directions",
    "evaluate_directions": "extract_emotion_vectors.evaluate_directions",
    "r2": "core.r2",
}

#: Stage sequence for ``run.py all``.
PIPELINE = ("extract_activations", "compute_directions", "evaluate_directions")

ALIASES = {
    "extract": "extract_activations",
    "activations": "extract_activations",
    "directions": "compute_directions",
    "compute": "compute_directions",
    "evaluate": "evaluate_directions",
    "eval": "evaluate_directions",
    "pipeline": "all",
}

USAGE = f"""usage: python run.py <stage> [args...]

stages:
  extract_activations   pooled residual-stream activations      (needs the GPU)
  compute_directions    mean-difference directions + neutral-PC removal
  evaluate_directions   held-out metrics, stability, plots
  all                   the three above, in order
  r2                    Cloudflare R2 mirroring CLI (push/pull/ls/check)

aliases: {', '.join(f'{k}->{v}' for k, v in ALIASES.items())}

Arguments after the stage name are passed through unchanged, so the usual
--set field=value overrides, --dry-run, --limit, --num-shards etc. all work:

  python run.py extract_activations --dry-run
  python run.py extract_activations --limit 256
  python run.py all --set emotions=joyful,sad,angry --set stories_per_emotion=300
  python run.py r2 push outputs/<run>/activations --prefix runs/<run>/activations

With `all`, arguments valid only for some stages are dropped for the others;
--dry-run and --limit are understood by all three.
"""

# Flags that every stage in PIPELINE accepts, so `all` can forward them safely.
SHARED_FLAGS = ("--set", "--config-json", "--dry-run", "--limit")


def _resolve(stage: str) -> str:
    stage = ALIASES.get(stage, stage)
    if stage != "all" and stage not in STAGES:
        raise SystemExit(
            f"unknown stage {stage!r}\n\n{USAGE}"
        )
    return stage


def _filter_for_all(args: list[str]) -> list[str]:
    """Keep only arguments that all three pipeline stages understand.

    ``all`` is a convenience, so a stage-specific flag like ``--num-shards``
    should not make the later stages fail. We report what we drop rather than
    dropping it silently.
    """
    kept: list[str] = []
    dropped: list[str] = []
    i = 0
    while i < len(args):
        arg = args[i]
        name = arg.split("=", 1)[0]
        if name in SHARED_FLAGS:
            kept.append(arg)
            # Consume a following value for flags given as `--set k=v` style pairs.
            if "=" not in arg and name in ("--set", "--config-json", "--limit"):
                if i + 1 < len(args):
                    kept.append(args[i + 1])
                    i += 1
        else:
            dropped.append(arg)
        i += 1
    if dropped:
        print(f"note: dropping stage-specific args for 'all': {' '.join(dropped)}",
              file=sys.stderr)
    return kept


def _run(stage: str, args: list[str]) -> int:
    import importlib

    module = importlib.import_module(STAGES[stage])
    main = getattr(module, "main", None) or getattr(module, "_main")
    # core.r2's CLI reads sys.argv directly rather than taking argv.
    if stage == "r2":
        sys.argv = [f"run.py {stage}", *args]
        main()
        return 0
    return int(main(args) or 0)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(USAGE)
        return 0 if argv else 2

    # Before any stage runs: r2.env carries HF_TOKEN as well as the R2 credentials,
    # so loading it here (not just inside core.r2) means a gated checkpoint
    # downloads without a separate `export` step.
    from core.env_file import load_env_file

    load_env_file()

    stage = _resolve(argv[0])
    args = argv[1:]

    if stage != "all":
        return _run(stage, args)

    shared = _filter_for_all(args)
    for step in PIPELINE:
        print(f"\n{'=' * 78}\nrun.py: stage {step}\n{'=' * 78}", flush=True)
        code = _run(step, list(shared))
        if code != 0:
            print(f"\nrun.py: stage {step} exited {code}; stopping.", file=sys.stderr)
            return code
        if step == "extract_activations" and "--dry-run" not in shared:
            _ensure_activations_local(shared)
    return 0


def _ensure_activations_local(shared_args: list[str]) -> None:
    """Re-download activation chunks if extraction deleted them after uploading.

    With ``delete_local_after_sync=True`` (the default) the tensors live only in R2
    once extraction finishes, but stages 2 and 3 read them from disk. Pulling here
    keeps ``run.py all`` a single command. No-op when the chunks are already local.
    """
    from extract_emotion_vectors.extract_activations import build_parser, load_config

    # Reuse the stage parser so --set overrides resolve exactly as the stages saw them.
    try:
        args = build_parser().parse_args(shared_args)
        config = load_config(args)
    except SystemExit:
        return

    indexed = sorted(config.activations_dir.glob("shard*/chunk_*.index.parquet"))
    missing = [
        p for p in (i.with_suffix("").with_suffix(".safetensors") for i in indexed)
        if not p.exists()
    ]
    if not missing:
        return

    print(
        f"\nrun.py: {len(missing)} activation chunk(s) are in R2 only "
        "(delete_local_after_sync=True).\n"
        "        Pulling them back so direction fitting can read them...",
        flush=True,
    )
    try:
        from core.r2 import R2Client

        stats = R2Client.from_env().sync_down(
            config.activations_dir, config.resolved_r2_prefix(), verbose=False
        )
        print(f"        downloaded {stats['downloaded']}, already present {stats['skipped']}, "
              f"{stats['bytes'] / 1024**3:.2f} GiB", flush=True)
    except Exception as exc:
        print(f"        WARNING could not pull from R2: {exc}", file=sys.stderr)
        print("        The next stage will fail with instructions; pull manually and re-run "
              "just that stage.", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
