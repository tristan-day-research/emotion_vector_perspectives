"""Extend the published Jacobian lens with additional fitting prompts.

Why this exists
---------------
The published ``qwen3-32b`` lens is an **interrupted fit**. Phase 0's divisor
cross-check surfaced it and the fit's own convergence CSV confirms it:

* the checkpoint carries ``n_done = 80``, not the ``prompts_fitted: 615``
  recorded in ``config.yaml`` (that metadata is simply wrong -- the CSV has 81
  rows, i.e. 80 prompts);
* ``fitting.fit`` stops at ``mean_rel_change < 0.002``; this fit stopped at
  **0.026**, thirteen times its own threshold;
* for comparison, the ``gemma-2-27b`` lens in the same repo ran to 780 prompts
  and did converge (0.0018).

The lens still works -- Phase 0 GATE B reproduces the model's own next token at
block 62, and GATE A resolves ``tennis`` at rank 1 and ``chess`` at rank 0. But
the *emotion* association items read out weakly (``anger`` at rank 4230), and
with an under-converged lens there is no way to tell lens noise from genuine
non-lexicalisation. That ambiguity is a direct confound for Phase 4: a weak PC
readout would be uninterpretable. Adding prompts removes the confound.

Method: merge, not resume
-------------------------
``fit(resume=True)`` skips prompts by *index* into the list you hand it, so
resuming correctly requires reproducing Neuronpedia's exact prompt sequence.
Their ``fit_lens.py`` is not public and used flags (``--max_chars 2000``) whose
filtering we cannot reproduce exactly, so index 80 in our list is probably not
index 80 in theirs -- we would silently re-use prompts already averaged in.

Instead this fits a *fresh* lens on a slice of WikiText taken well downstream of
anything the original touched, then combines with
``JacobianLens.merge``, which is the library's own documented path for
"fit on disjoint slices and combine" and does the ``n_prompts``-weighted mean
correctly. Disjointness is structural: both loaders read the same split in
stream order, the original stopped after 80 prompts near the beginning, and
``--skip`` (default 1000) starts far past that.

Cost -- read this before starting
---------------------------------
Backward work per prompt is ``4 * P * d_model * seq_len`` ~= 86 PFLOP for a 32B
model, and is **independent of ``dim_batch``**; ``dim_batch`` only buys
utilisation. The published fit used ``dim_batch=128`` on a B200 (179 GB) and got
51 s/prompt. On an 80 GiB H100, 66 GiB of weights leaves ~14 GiB, and the
retained backward graph costs roughly 1.5 GiB *per batch element*, so
``dim_batch`` is capped near 8 -- 640 backward passes per prompt, at poor
utilisation. Budget **5-10 min/prompt**, i.e. 10-20 h for 120 prompts.

This is an overnight job. Do not block a one-day sprint on it; run the pipeline
on the 80-prompt lens, and use this to produce a confirmatory re-run.

Usage::

    python run.py refit_lens --dry-run              # plan + cost, no model
    python run.py refit_lens --n-prompts 120        # the real thing
    python run.py refit_lens --n-prompts 120 --dim-batch 24 --set device_map=auto

The last form shards the model across every visible GPU, which frees enough
memory for a much larger ``dim_batch`` and is substantially faster -- but it
occupies both cards, so nothing else can run meanwhile.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from core import env_file, jlens_lens, model_utils, paths, provenance
from core.seeds import set_global_seeds
from emotion_pca_jlens.pca_jlens_config import load_config

RULE = "=" * 78

#: Matches the published fit's ``--max_chars 2000`` so the added prompts come
#: from the same distribution as the 80 already averaged in.
MAX_CHARS = 2000
MIN_CHARS = 600
FIT_MAX_SEQ_LEN = 128


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extend the published J-lens with more fitting prompts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n-prompts", type=int, default=120,
                   help="additional prompts to fit on")
    p.add_argument("--skip", type=int, default=1000,
                   help="WikiText records to skip before collecting, so the new "
                        "prompts are disjoint from the published fit's first 80")
    p.add_argument("--dim-batch", type=int, default=8,
                   help="output dims per backward pass; raise only if VRAM allows "
                        "(~1.5 GiB of retained graph per unit on this model)")
    p.add_argument("--checkpoint-every", type=int, default=20,
                   help="prompts between checkpoint writes. The default in jlens is 1, "
                        "which would write ~6.6 GiB every prompt")
    p.add_argument("--dry-run", action="store_true",
                   help="resolve the lens, show the plan and cost estimate, load no model")
    p.add_argument("--config-json", type=Path, default=None)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return p


def wikitext_slice(n_prompts: int, skip: int) -> list[str]:
    """``n_prompts`` WikiText records, skipping the first ``skip`` that qualify.

    Mirrors ``jlens.examples.load_wikitext_prompts`` (same dataset, split and
    ``min_chars``) but offsets into the stream, which is what makes the result
    disjoint from the prompts already in the published checkpoint.
    """
    from datasets import load_dataset

    stream = load_dataset(
        "Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True
    )
    prompts: list[str] = []
    qualified = 0
    for record in stream:
        text = record["text"]
        if len(text.strip()) < MIN_CHARS:
            continue
        qualified += 1
        if qualified <= skip:
            continue
        prompts.append(text[:MAX_CHARS])
        if len(prompts) == n_prompts:
            break
    if len(prompts) < n_prompts:
        raise RuntimeError(
            f"WikiText yielded only {len(prompts)} prompts after skipping {skip}; "
            "lower --skip or --n-prompts"
        )
    return prompts


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args)
    set_global_seeds(config.seed)
    env_file.load_env_file()
    cache_dir = paths.hf_cache_dir()

    print(RULE)
    print(f"REFIT LENS -- extend the published fit   run '{config.run_name}'")
    print(RULE)

    artifact = jlens_lens.resolve_lens_artifact(
        config.model_name, config.lens_repo, config.lens_revision, config.lens_subfolder
    )
    lens_path = jlens_lens.download_lens(
        artifact, config.lens_repo, config.lens_revision, cache_dir
    )
    desc = jlens_lens.describe_lens_checkpoint(lens_path)
    print(f"published lens  : {artifact.subfolder} ({desc.checkpoint_format})")
    print(f"  prompts       : {desc.n_prompts}")
    print(f"  blocks        : {desc.source_layers[0]}..{desc.source_layers[-1]} "
          f"({len(desc.source_layers)})")
    print(f"  d_model       : {desc.d_model}")

    total_after = desc.n_prompts + args.n_prompts
    print(f"\nplan: fit {args.n_prompts} fresh prompts (skipping {args.skip} records "
          f"for disjointness),\n      then merge -> {total_after} prompts total.")
    print(f"  dim_batch     : {args.dim_batch} "
          f"({-(-desc.d_model // args.dim_batch)} backward passes/prompt)")
    print(f"  checkpoint    : every {args.checkpoint_every} prompts")

    # 4 * params * d_model * seq_len, independent of dim_batch (see fitting.py).
    flops = 4 * 32.8e9 * desc.d_model * FIT_MAX_SEQ_LEN
    for label, tflops in (("optimistic", 400e12), ("likely", 170e12)):
        per_prompt = flops / tflops
        print(f"  {label:11}: ~{per_prompt / 60:.1f} min/prompt -> "
              f"~{per_prompt * args.n_prompts / 3600:.1f} h total")
    print("\n  This is an overnight job. The pipeline runs fine on the published")
    print("  lens; this only produces a confirmatory re-run with less lens noise.")

    out_path = config.phase_dir / f"lens_merged_n{total_after}.pt"
    ckpt_path = config.phase_dir / f"refit_checkpoint_n{args.n_prompts}.pt"
    print(f"\noutput  : {out_path}")
    print(f"resume  : {ckpt_path} (delete to start the added prompts over)")

    if args.dry_run:
        print("\n--dry-run: nothing fitted, no model loaded.")
        return 0

    print("\ncollecting prompts ...")
    prompts = wikitext_slice(args.n_prompts, args.skip)
    print(f"  {len(prompts)} prompts, "
          f"{sum(len(p) for p in prompts) / len(prompts):.0f} chars mean")

    print(f"\nloading {config.model_name} ...")
    t0 = time.time()
    tokenizer = model_utils.load_tokenizer(
        config.model_name, config.model_revision, cache_dir,
        trust_remote_code=config.trust_remote_code,
    )
    hf_model = model_utils.load_model(
        config.model_name, revision=config.model_revision, cache_dir=cache_dir,
        dtype=config.dtype, device_map=config.device_map,
        attn_implementation=config.attn_implementation,
        trust_remote_code=config.trust_remote_code,
    )
    print(f"  loaded in {time.time() - t0:.0f}s")

    import jlens

    model = jlens.from_hf(hf_model, tokenizer)
    config.phase_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nfitting {len(prompts)} prompts (resumable; safe to kill and re-run) ...")
    t_fit = time.time()
    fresh = jlens.fit(
        model,
        prompts,
        # Must match the published lens exactly: JacobianLens.merge refuses
        # lenses that disagree on source_layers or d_model.
        source_layers=list(desc.source_layers),
        dim_batch=args.dim_batch,
        max_seq_len=FIT_MAX_SEQ_LEN,
        checkpoint_path=str(ckpt_path),
        checkpoint_every=args.checkpoint_every,
        resume=True,
    )
    print(f"  fitted {fresh.n_prompts} prompts in {(time.time() - t_fit) / 3600:.2f} h")

    published = jlens_lens.load_lens(lens_path)
    merged = jlens.JacobianLens.merge([published, fresh])
    print(f"\nmerged: {published.n_prompts} + {fresh.n_prompts} = {merged.n_prompts}")

    before = jlens_lens.identity_distances(published)
    after = jlens_lens.identity_distances(merged)
    blocks = sorted(after)
    print("\n||J - I|| before -> after (a real change means the added prompts moved it):")
    for block in blocks[:: max(1, len(blocks) // 10)] + [blocks[-1]]:
        print(f"  block {block:>3}: {before[block]:.4f} -> {after[block]:.4f}")

    merged.save(str(out_path))
    print(f"\nsaved {out_path} ({out_path.stat().st_size / 1024**3:.2f} GiB)")

    provenance.write_run_record(
        config.phase_dir,
        title=f"LENS REFIT -- {config.run_name}",
        sections={
            "run": {"stage": "refit_lens", "n_added": args.n_prompts,
                    "skip": args.skip, "dim_batch": args.dim_batch},
            "published": {"path": str(lens_path), "n_prompts": desc.n_prompts,
                          "format": desc.checkpoint_format},
            "merged": {"n_prompts": merged.n_prompts, "path": str(out_path)},
            "identity_before": {str(k): v for k, v in before.items()},
            "identity_after": {str(k): v for k, v in after.items()},
        },
        txt_name="refit_lens.txt", json_name="refit_lens.json",
    )

    print(f"\nUse it by re-running the gate against the merged lens:\n"
          f"  python run.py phase0 --set lens_local_path={out_path}\n"
          "Compare GATE A's emotion ranks against the 80-prompt run: a large\n"
          "improvement means the weak readouts were lens noise; little change means\n"
          "emotion is genuinely less verbalizable here, which is a finding.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
