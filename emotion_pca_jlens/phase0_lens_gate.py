"""Phase 0 (GATE): load the J-lens, verify the readout, stop.

Nothing downstream is valid if the lens is loaded wrong, so this stage proves
four things and then stops:

1. **What is in the lens file** -- tensor names, shapes, dtypes, fitted layer
   range -- cross-checked against the model's own config.
2. **What the readout formula actually is** -- read off the loaded model rather
   than assumed, including the scale-invariance that makes a bare ``+/-PC``
   readout well defined.
3. **GATE A, concept readout** -- on prompts with published ground truth, an
   implied concept the prompt never names should surface at mid layers.
4. **GATE B, late-layer identity** -- at the highest fitted block the transport
   spans one residual block, so the lens should collapse towards the plain logit
   lens and towards the model's real next token.

Usage::

    python run.py phase0 --dry-run     # lens facts only; no 65 GiB of weights
    python run.py phase0               # full gate
    python run.py phase0 --set topk=20 --set gate_blocks=evenly_spaced:16

``--dry-run`` is a gate before the gate: it resolves, downloads and describes the
lens and compares it to the model config without loading weights, so a wrong
lens is caught before an hour of GPU time.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from core import env_file, jlens_lens, model_utils, paths, provenance
from core.seeds import set_global_seeds
from emotion_pca_jlens import gate_prompts
from emotion_pca_jlens.pca_jlens_config import (
    PCAJLensConfig,
    load_config,
    resolve_block_spec,
)

RULE = "=" * 78
THIN = "-" * 78


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 0 gate: load and verify the Jacobian lens.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="describe the lens and cross-check the model config; never load weights",
    )
    p.add_argument(
        "--config-json", type=Path, default=None, help="JSON file of config overrides"
    )
    p.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE",
        help="override a config field; repeatable",
    )
    return p


# --------------------------------------------------------------------------- #
# Step 0: prove the activation store works before spending GPU time
# --------------------------------------------------------------------------- #

def preflight_r2(config: PCAJLensConfig) -> dict:
    """Verify the R2 destination with a real round-trip.

    Phase 0 itself writes no activations -- it reads out 12 short prompts and
    stores nothing but a run record. Phase 2 does, and on an ephemeral pod a
    broken R2 destination discovered *after* extraction means the activations are
    gone with the pod. So the round-trip is done here, at the cheapest possible
    moment: upload a small marker, confirm its size remotely, download it back,
    compare bytes.

    An env-var check alone would not catch a wrong bucket, a bad endpoint, or a
    key without write permission -- all of which look fine until the first PUT.
    """
    import json
    import tempfile

    from core.r2 import R2Client, r2_available

    print(RULE)
    print("STEP 0  Activation store (Cloudflare R2) preflight")
    print(RULE)

    prefix = config.resolved_r2_prefix()
    available, reason = r2_available()
    report: dict = {
        "r2_sync": config.r2_sync,
        "prefix": prefix,
        "delete_local_after_sync": config.delete_local_after_sync,
        "available": available,
        "reason": reason,
    }

    print(f"r2_sync            : {config.r2_sync}  (activations always go to R2)")
    print(f"bucket folder      : {prefix}")
    print(f"delete local after : {config.delete_local_after_sync} (only after a verified upload)")

    if not available:
        report["round_trip"] = False
        print(f"\n  NOT USABLE: {reason}")
        if config.r2_sync is True:
            print("\n  Phase 0 does not need R2 and will continue, but Phase 2 will abort")
            print("  on this before loading the model. Fix it now, not later.")
        return report

    client = R2Client.from_env()
    report["bucket"] = client.bucket
    report["endpoint"] = client.endpoint_url
    print(f"bucket             : {client.bucket}")
    print(f"endpoint           : {client.endpoint_url}")

    key = f"{prefix}/_preflight.json"
    payload = {
        "written_by": "phase0_lens_gate",
        "run_name": config.run_name,
        "model_name": config.model_name,
        "utc": provenance.utc_timestamp(),
        "note": "marker proving this prefix is writable; safe to delete",
    }
    body = json.dumps(payload, indent=2).encode("utf-8")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            up = Path(tmp) / "_preflight.json"
            up.write_bytes(body)
            client.upload_file(up, key)
            remote_size = client.head_size(key)
            if remote_size != len(body):
                raise IOError(
                    f"size mismatch: wrote {len(body)} bytes, remote reports {remote_size}"
                )
            down = Path(tmp) / "roundtrip.json"
            client.download_file(key, down)
            if down.read_bytes() != body:
                raise IOError("downloaded bytes differ from what was uploaded")
        existing = client.list_objects(prefix)
        report.update({"round_trip": True, "objects_at_prefix": len(existing)})
        print(f"\n  round trip OK: wrote, verified size, and read back {key}")
        print(f"  objects already at this prefix: {len(existing)}")
        if len(existing) > 1:
            print("  NOTE this prefix is not empty. A Phase 2 run with this run_name")
            print("       would resume into it rather than start clean.")
    except Exception as exc:
        report.update({"round_trip": False, "error": str(exc)})
        print(f"\n  ROUND TRIP FAILED: {exc}")
        print("  Credentials resolve but the bucket is not usable for writing.")
        print("  Phase 2 would lose its activations. Fix before extracting.")
    return report


# --------------------------------------------------------------------------- #
# Step 1-2: the lens file and the model config
# --------------------------------------------------------------------------- #

def inspect_lens(config: PCAJLensConfig, cache_dir: Path) -> tuple[dict, object]:
    """Resolve, download and describe the lens. Returns (report, description)."""
    print(RULE)
    print("STEP 1  Locate and describe the pre-fitted lens")
    print(RULE)

    if config.lens_local_path:
        # A locally-produced lens (e.g. from `run.py refit_lens`). No repo
        # resolution and therefore no hf_model_name guard -- say so, rather than
        # letting the absence of that check pass unnoticed.
        lens_path = Path(config.lens_local_path)
        if not lens_path.exists():
            raise SystemExit(f"lens_local_path does not exist: {lens_path}")
        print(f"lens         : LOCAL FILE {lens_path}")
        print("               (lens_local_path is set, so the published lens is not")
        print("                used and the hf_model_name match is NOT verified;")
        print("                the d_model / block-range check in STEP 2 still runs)")
        desc = jlens_lens.describe_lens_checkpoint(lens_path)
        print(f"\n  format        : {desc.checkpoint_format}")
        print(f"  d_model       : {desc.d_model}")
        print(f"  n_prompts     : {desc.n_prompts}")
        print(f"  fitted blocks : {desc.source_layers[0]}..{desc.source_layers[-1]} "
              f"({len(desc.source_layers)})")
        return {
            "source": "local",
            "local_path": str(lens_path),
            "checkpoint_format": desc.checkpoint_format,
            "d_model": desc.d_model,
            "n_prompts": desc.n_prompts,
            "n_fitted_blocks": len(desc.source_layers),
            "fitted_block_range": [desc.source_layers[0], desc.source_layers[-1]],
            "j_dtype_on_disk": desc.uniform_dtype,
            "file_bytes": desc.file_bytes,
        }, desc

    artifact = jlens_lens.resolve_lens_artifact(
        config.model_name,
        lens_repo=config.lens_repo,
        revision=config.lens_revision,
        subfolder=config.lens_subfolder,
    )
    fit = artifact.fit_config
    print(f"lens repo    : {config.lens_repo}")
    print(f"subfolder    : {artifact.subfolder}   (resolved from {config.model_name})")
    print(f"fitted on    : {fit.get('hf_model_name', '<no config.yaml>')}")
    if fit.get("hf_model_name") == config.model_name:
        print("               ^ matches the model we will load")
    print(f"download only: allow_patterns=['{artifact.allow_pattern()}']")
    print("\nfiles in that subfolder:")
    print(f"  {artifact.lens_file}   <- the lens")
    if artifact.config_file:
        print(f"  {artifact.config_file}")
    for extra in artifact.extra_files:
        print(f"  {extra}")

    if fit:
        results = fit.get("results", {}) if isinstance(fit.get("results"), dict) else {}
        fit_params = fit.get("fit", {}) if isinstance(fit.get("fit"), dict) else {}
        print("\nhow this lens was fitted (from its own config.yaml):")
        dataset = fit.get("dataset", {})
        if isinstance(dataset, dict):
            print(f"  corpus                 : {dataset.get('name')} / {dataset.get('config')}")
        print(f"  prompts requested      : {fit_params.get('n_prompts')}")
        print(f"  prompts actually fitted: {results.get('prompts_fitted')} "
              "(early-stopped on convergence)")
        print(f"  max_seq_len            : {fit_params.get('max_seq_len')}")
        print(f"  fit dtype              : {fit_params.get('dtype')}")
        print(f"  final identity distance: {results.get('final_identity_distance')}")
        print(f"  final mean rel change  : {results.get('final_mean_rel_change')}")

    print("\ndownloading (cached after the first run) ...")
    t0 = time.time()
    lens_path = jlens_lens.download_lens(
        artifact, config.lens_repo, config.lens_revision, cache_dir
    )
    print(f"  -> {lens_path}")
    print(f"  {lens_path.stat().st_size / 1024**3:.2f} GiB in {time.time() - t0:.0f}s")

    print("\nreading the checkpoint (loads it into host RAM) ...")
    desc = jlens_lens.describe_lens_checkpoint(lens_path)
    print(f"  checkpoint keys : {list(desc.checkpoint_keys)}")
    print(f"  on-disk format  : {desc.checkpoint_format}")
    if desc.checkpoint_format == "fit_checkpoint":
        print("                    ^ a resumable fit() checkpoint, not a saved lens:")
        print(f"                      'jacobian_sum' is the running SUM over "
              f"{desc.n_prompts} prompts,")
        print(f"                      so J_l = jacobian_sum / {desc.n_prompts}. "
              "Applied on load.")
        print("                      (Matters because RMSNorm is scale-invariant: an")
        print("                       un-normalised J gives identical top-k tokens but")
        print("                       wrong ||J - I|| and wrong variance splits.)")
    print(f"  d_model         : {desc.d_model}")
    print(f"  n_prompts       : {desc.n_prompts}")
    print(f"  fitted layers   : {len(desc.source_layers)} blocks, "
          f"{desc.source_layers[0]}..{desc.source_layers[-1]}")
    print(f"  J_l shape       : {desc.uniform_shape} (uniform across layers: "
          f"{desc.uniform_shape is not None})")
    print(f"  J_l dtype       : {desc.uniform_dtype} as stored on disk")
    print(f"  tensor bytes    : {desc.stored_bytes / 1024**3:.2f} GiB "
          f"(file {desc.file_bytes / 1024**3:.2f} GiB)")
    # Independent check on the divisor. config.yaml is written by the fit script
    # and records prompts_fitted separately from the tensor file, so agreement
    # confirms J_l = jacobian_sum / n_done used the right n -- the one thing about
    # a fit checkpoint that no readout-based gate can catch.
    results = fit.get("results", {}) if isinstance(fit.get("results"), dict) else {}
    claimed_prompts = results.get("prompts_fitted")
    if claimed_prompts is not None:
        agree = int(claimed_prompts) == desc.n_prompts
        print(f"\n  divisor cross-check: config.yaml prompts_fitted={claimed_prompts} "
              f"vs checkpoint n={desc.n_prompts} -> {'AGREE' if agree else 'DISAGREE'}")
        if not agree:
            print("    DISAGREEMENT: the normalisation constant is not confirmed.")
            print("    Top-k readouts would still look fine (RMSNorm is scale-free),")
            print("    but ||J - I|| and any variance split would be wrong. Investigate")
            print("    before trusting GATE B or Phase 6.")

    print("\n  NOTE jlens.JacobianLens.load upcasts every J to float32 in host RAM.")
    print(f"       Budget ~{desc.d_model ** 2 * 4 * len(desc.source_layers) / 1024**3:.1f} "
          "GiB of RAM for the loaded lens.")

    report = {
        "lens_repo": config.lens_repo,
        "subfolder": artifact.subfolder,
        "lens_file": artifact.lens_file,
        "local_path": str(lens_path),
        "fitted_on": fit.get("hf_model_name"),
        "fit_config": fit,
        "checkpoint_format": desc.checkpoint_format,
        "d_model": desc.d_model,
        "n_prompts": desc.n_prompts,
        "n_fitted_blocks": len(desc.source_layers),
        "fitted_block_range": [desc.source_layers[0], desc.source_layers[-1]],
        "j_shape": list(desc.uniform_shape) if desc.uniform_shape else None,
        "j_dtype_on_disk": desc.uniform_dtype,
        "file_bytes": desc.file_bytes,
    }
    return report, desc


def crosscheck_model(config: PCAJLensConfig, desc, cache_dir: Path) -> tuple[dict, object]:
    """Compare the lens against the model's config. No weights are loaded."""
    print()
    print(RULE)
    print("STEP 2  Cross-check the lens against the model config (no weights)")
    print(RULE)

    arch = model_utils.load_architecture_info(
        config.model_name, config.model_revision, cache_dir, config.trust_remote_code
    )
    print(f"model        : {config.model_name}")
    print(f"architecture : {arch.architectures}")
    print(f"n_layers     : {arch.n_layers}  (blocks 0..{arch.n_layers - 1})")
    print(f"hidden_size  : {arch.hidden_size}")
    print(f"resolved sha : {arch.resolved_sha}")

    highest = jlens_lens.max_lens_block(arch.n_layers)
    print("\nlayer-index conventions in play (the easiest thing to get wrong):")
    print(f"  jlens block index l   : output of residual block l, 0..{arch.n_layers - 1}")
    print(f"  this repo's layer idx : output_hidden_states index, 0..{arch.n_layers} "
          "(0 = embeddings)")
    print("  conversion            : hidden_state_index = block_index + 1")
    print(f"  lens covers blocks    : 0..{highest} (= n_layers - 2), i.e. "
          f"hidden states 1..{jlens_lens.hidden_state_index(highest)}")
    print(f"  no J for block        : {arch.n_layers - 1} (the last block is the "
          "transport *target*, not a source)")

    problems = desc.problems(n_layers=arch.n_layers, d_model=arch.hidden_size)
    print()
    if problems:
        print("  MISMATCH:")
        for problem in problems:
            print(f"    - {problem}")
    else:
        print("  OK  lens d_model, J shape, and fitted block range all agree with the model.")

    return {
        "n_layers": arch.n_layers,
        "hidden_size": arch.hidden_size,
        "architectures": list(arch.architectures),
        "model_sha": arch.resolved_sha,
        "highest_fitted_block": highest,
        "problems": problems,
    }, arch


# --------------------------------------------------------------------------- #
# Step 3: what the readout actually does
# --------------------------------------------------------------------------- #

def verify_readout_convention(readout, config: PCAJLensConfig, arch) -> dict:
    print()
    print(RULE)
    print("STEP 3  Verify the readout formula against the loaded objects")
    print(RULE)

    import torch

    described = readout.unembed_description()
    print("unembed, as read off the wrapped model:")
    for key, value in described.items():
        print(f"  {key:20}: {value}")
    print("\n  So the full readout is:")
    print(f"    lens_l(h) = {described['formula']}")
    print("  The 'norm step' is the model's own final norm with its learned weight --")
    print("  not an ad-hoc normalisation. Softcapping, where present, is monotonic and")
    print("  so cannot reorder top-k.")

    tokenizer = readout.tokenizer
    ids = tokenizer.encode("Fact: the sky is", add_special_tokens=True)
    bos = getattr(tokenizer, "bos_token_id", None)
    print(f"\n  tokenizer prepends BOS : {bool(ids and bos is not None and ids[0] == bos)}"
          f"  (bos_token_id={bos})")

    # Scale invariance: RMSNorm makes a bare-direction readout magnitude-free, so
    # +PC / -PC are well defined without picking a step size. Verified, not assumed.
    block = jlens_lens.max_lens_block(arch.n_layers) // 2
    generator = torch.Generator().manual_seed(config.seed)
    direction = torch.randn(arch.hidden_size, generator=generator)
    direction = direction / direction.norm()
    small = readout.top_tokens(direction, block, k=5)
    large = readout.top_tokens(direction * 250.0, block, k=5)
    same = [t.token_id for t in small] == [t.token_id for t in large]
    print(f"\n  scale invariance at block {block} (random unit direction vs x250):")
    print(f"    |v|=1    top-5: {[t.token for t in small]}")
    print(f"    |v|=250  top-5: {[t.token for t in large]}")
    print(f"    identical top-5: {same}")
    if same:
        print("    -> a bare +/-PC readout does not depend on the PC's magnitude.")
    else:
        print("    -> WARNING: readout depends on magnitude. Investigate before Phase 4;")
        print("       an eps- or dtype-scale effect would make +/-PC readouts arbitrary.")

    return {
        "unembed": {k: str(v) for k, v in described.items()},
        "bos_prepended": bool(ids and bos is not None and ids[0] == bos),
        "scale_invariant_top5": same,
        "scale_test_block": block,
    }


# --------------------------------------------------------------------------- #
# GATE A: concept readout
# --------------------------------------------------------------------------- #

def gate_a_concepts(readout, config: PCAJLensConfig, blocks: list[int]) -> dict:
    print()
    print(RULE)
    print("GATE A  Does the lens surface concepts the prompt never names?")
    print(RULE)
    print("Prompts and expected intermediates come from Anthropic's published sets")
    print("(probe-swap.json, lens-eval-association.json) -- ground truth we did not")
    print(f"choose ourselves. Reading at the final token position, blocks {blocks}.")

    rows: list[dict] = []
    for group_name, prompts in (
        ("factual / sport", gate_prompts.CONCEPT_PROMPTS),
        ("emotion vignettes", gate_prompts.EMOTION_ASSOCIATION_PROMPTS),
    ):
        print()
        print(THIN)
        print(f"{group_name}")
        print(THIN)
        for item in prompts:
            print(f"\n[{item.name}] {item.note}")
            print(f"  prompt : {item.prompt[:150]}{'...' if len(item.prompt) > 150 else ''}")
            targets = ", ".join(item.expect) + (f"  (answer: {item.answer})" if item.answer else "")
            print(f"  expect : {targets}")

            untokenizable = [w for w in item.expect if not readout.single_token_variants(w)]
            if untokenizable:
                print(f"  NOTE   : {untokenizable} is not a single token in this vocabulary,")
                print("           so the lens cannot surface it directly (the single-token")
                print("           caveat). Rank reported as n/a, not as a failure.")

            lens_logits, model_logits, input_ids = readout.apply_to_prompt(
                item.prompt, blocks, position=-1, max_seq_len=config.gate_max_seq_len
            )
            actual = readout.decode_top(model_logits, 1)[0]
            print(f"  model's actual next token: {actual.token!r} (p={actual.prob:.3f}); "
                  f"{input_ids.shape[-1]} tokens")

            best: dict[str, tuple[int, int]] = {}
            answer_ranks: dict[int, int] = {}
            for block in blocks:
                logits = lens_logits[block]
                tops = readout.decode_top(logits, config.topk)
                marks = []
                for word in item.expect:
                    rank, variant, _ = readout.rank_of_word(logits, word)
                    if rank is None:
                        continue
                    if word not in best or rank < best[word][0]:
                        best[word] = (rank, block)
                    if rank < config.topk:
                        marks.append(f"{variant.strip()}@{rank}")
                # Track the final answer too. A correctly-loaded lens shows the
                # implied concept peaking mid-stack and the answer taking over
                # late; that crossover is harder to fake than either alone, so it
                # is the part of GATE A worth reading closely.
                if item.answer:
                    a_rank, a_variant, _ = readout.rank_of_word(logits, item.answer)
                    if a_rank is not None:
                        answer_ranks[block] = a_rank
                        if a_rank < config.topk:
                            marks.append(f"={a_variant.strip()}@{a_rank}")
                mark = f"   <- {', '.join(marks)}" if marks else ""
                tokens = " ".join(repr(t.token) for t in tops)
                print(f"    block {block:>3}: {tokens}{mark}")

            if answer_ranks and best:
                concept_block = min(best.values())[1]
                answer_block = min(answer_ranks, key=answer_ranks.get)
                order = "concept-before-answer" if concept_block < answer_block else (
                    "same block" if concept_block == answer_block else "answer-first"
                )
                print(f"  crossover: concept best at block {concept_block}, "
                      f"answer best at block {answer_block}  [{order}]")

            for word, (rank, block) in best.items():
                verdict = "HIT" if rank < config.topk else ("near" if rank < 100 else "MISS")
                print(f"  best rank for {word!r}: {rank} at block {block}  [{verdict}]")
                rows.append({
                    "group": group_name, "item": item.name, "word": word,
                    "best_rank": rank, "best_block": block, "verdict": verdict,
                })
            for word in untokenizable:
                rows.append({
                    "group": group_name, "item": item.name, "word": word,
                    "best_rank": None, "best_block": None, "verdict": "not-single-token",
                })

    hits = sum(1 for r in rows if r["verdict"] == "HIT")
    scorable = [r for r in rows if r["verdict"] != "not-single-token"]
    print()
    print(THIN)
    print(f"GATE A summary: {hits}/{len(scorable)} expected concepts reached the top-"
          f"{config.topk} at some block "
          f"({len(rows) - len(scorable)} excluded as multi-token).")
    print(THIN)
    return {"rows": rows, "n_hits": hits, "n_scorable": len(scorable)}


# --------------------------------------------------------------------------- #
# GATE B: late-layer identity
# --------------------------------------------------------------------------- #

def gate_b_identity(readout, config: PCAJLensConfig) -> dict:
    print()
    print(RULE)
    print("GATE B  Does the lens collapse to the logit lens at the top of the stack?")
    print(RULE)
    print("At the highest fitted block the transport spans exactly one residual block,")
    print("so J should be near the identity and the three readouts -- J-lens, plain")
    print("logit lens, and the model's real output -- should converge.")

    distances = jlens_lens.identity_distances(readout.lens)
    blocks = sorted(distances)
    print("\nper-block ||J - I||_F / ||I||_F:")
    stride = max(1, len(blocks) // 12)
    for block in blocks[::stride]:
        bar = "#" * int(min(distances[block], 2.0) * 25)
        print(f"  block {block:>3}: {distances[block]:.4f}  {bar}")
    tail = blocks[-config.identity_check_blocks:]
    print(f"\n  highest {len(tail)} blocks in full:")
    for block in tail:
        print(f"    block {block:>3}: {distances[block]:.4f}")
    first, last = distances[blocks[0]], distances[blocks[-1]]
    print(f"\n  block {blocks[0]} -> {blocks[-1]}: {first:.4f} -> {last:.4f} "
          f"({'falls, as expected' if last < first else 'DOES NOT FALL -- investigate'})")

    top_block = blocks[-1]
    probe = gate_prompts.CONCEPT_PROMPTS[0]
    lens_logits, model_logits, _ = readout.apply_to_prompt(
        probe.prompt, [top_block], position=-1, max_seq_len=config.gate_max_seq_len
    )
    logit_only, _, _ = readout.apply_to_prompt(
        probe.prompt, [top_block], position=-1,
        max_seq_len=config.gate_max_seq_len, use_jacobian=False,
    )
    j_top = readout.decode_top(lens_logits[top_block], config.topk)
    l_top = readout.decode_top(logit_only[top_block], config.topk)
    m_top = readout.decode_top(model_logits, config.topk)

    print(f"\n  at block {top_block}, final position of [{probe.name}]:")
    print(f"    J-lens      : {[t.token for t in j_top[:8]]}")
    print(f"    logit lens  : {[t.token for t in l_top[:8]]}")
    print(f"    model output: {[t.token for t in m_top[:8]]}")
    j_ids, l_ids, m_ids = ({t.token_id for t in x} for x in (j_top, l_top, m_top))
    agree = {
        "top1_jlens_equals_model": j_top[0].token_id == m_top[0].token_id,
        "top1_jlens_equals_logitlens": j_top[0].token_id == l_top[0].token_id,
        f"top{config.topk}_overlap_jlens_model": len(j_ids & m_ids),
        f"top{config.topk}_overlap_jlens_logitlens": len(j_ids & l_ids),
    }
    print()
    for key, value in agree.items():
        print(f"    {key:38}: {value}")

    return {
        "identity_distances": {str(k): v for k, v in distances.items()},
        "lowest_block_distance": first,
        "highest_block_distance": last,
        "falls_with_depth": last < first,
        "top_block": top_block,
        "agreement": {k: (int(v) if isinstance(v, bool) else v) for k, v in agree.items()},
        "probe": probe.name,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args)
    set_global_seeds(config.seed)
    env_file.load_env_file()

    cache_dir = paths.hf_cache_dir()
    print(RULE)
    print(f"PHASE 0 GATE -- J-lens load + verification   run '{config.run_name}'")
    print(RULE)
    print(f"model    : {config.model_name}")
    print(f"lens     : {config.lens_repo}")
    print(f"hf cache : {cache_dir}")
    print(f"outputs  : {config.phase_dir}")
    print(f"r2 folder: {config.resolved_r2_prefix()}   (for Phase 2; this stage stores none)")
    print()
    print("This stage collects NO activations. It reads out 12 short gate prompts")
    print("and writes only a run record. Activation extraction is Phase 2.")
    print()

    r2_report = preflight_r2(config)
    print()

    lens_report, desc = inspect_lens(config, cache_dir)
    model_report, arch = crosscheck_model(config, desc, cache_dir)

    sections: dict = {
        "run": {"stage": "phase0_lens_gate", "run_name": config.run_name,
                "dry_run": args.dry_run},
        "config": config.to_dict(),
        "activation_store": r2_report,
        "lens": lens_report,
        "model": model_report,
        "jlens_reference": {
            "repo": jlens_lens.JLENS_REPO_URL,
            "verified_commit": jlens_lens.JLENS_VERIFIED_COMMIT,
        },
    }

    if args.dry_run:
        provenance.write_run_record(
            config.phase_dir, title=f"PHASE 0 DRY RUN -- {config.run_name}",
            sections=sections, txt_name="phase0_dry_run.txt",
            json_name="phase0_dry_run.json",
        )
        print()
        print(RULE)
        print("--dry-run complete: the lens matches the model config. No weights loaded.")
        print(f"Re-run without --dry-run to execute GATE A and GATE B "
              f"(loads {config.model_name}).")
        print(RULE)
        return 0 if not model_report["problems"] else 3

    if model_report["problems"]:
        print("\nABORTED: the lens does not match the model (see MISMATCH above).",
              file=sys.stderr)
        return 3

    # --- load the model ------------------------------------------------- #
    print()
    print(RULE)
    print(f"Loading {config.model_name} ({config.dtype}) ...")
    print(RULE)
    t0 = time.time()
    tokenizer = model_utils.load_tokenizer(
        config.model_name, config.model_revision, cache_dir,
        trust_remote_code=config.trust_remote_code,
    )
    hf_model = model_utils.load_model(
        config.model_name, revision=config.model_revision, cache_dir=cache_dir,
        dtype=config.dtype, device_map=config.device_map,
        quantization=config.quantization,
        attn_implementation=config.attn_implementation,
        trust_remote_code=config.trust_remote_code,
    )
    print(f"  weights loaded in {time.time() - t0:.0f}s")
    readout = jlens_lens.LensReadout.build(hf_model, tokenizer, lens_report["local_path"])
    print(f"  lens loaded: {readout.lens!r}")

    fitted = list(readout.source_layers)
    blocks = resolve_block_spec(config.gate_blocks, fitted)
    default_target = fitted[len(fitted) // 2]
    print(f"\n  gate blocks   : {blocks}")
    print(f"  middle of stack: {jlens_lens.describe_block(default_target, arch.n_layers)}")
    print("                   ^ the natural Phase 2 target_block unless you set one")

    sections["convention"] = verify_readout_convention(readout, config, arch)
    sections["gate_a"] = gate_a_concepts(readout, config, blocks)
    sections["gate_b"] = gate_b_identity(readout, config)
    # write_run_record adds an `environment` section (packages, hardware, git),
    # so only what is specific to this stage goes here.
    sections["resolved"] = {
        "gate_blocks": blocks,
        "suggested_target_block": default_target,
        "suggested_target_hidden_state": jlens_lens.hidden_state_index(default_target),
    }

    txt_path, json_path = provenance.write_run_record(
        config.phase_dir, title=f"PHASE 0 GATE -- {config.run_name}",
        sections=sections, txt_name="phase0_gate.txt", json_name="phase0_gate.json",
    )

    # --- verdict --------------------------------------------------------- #
    gate_a, gate_b = sections["gate_a"], sections["gate_b"]
    a_pass = gate_a["n_scorable"] > 0 and gate_a["n_hits"] / gate_a["n_scorable"] >= 0.5
    b_pass = gate_b["falls_with_depth"] and gate_b["agreement"]["top1_jlens_equals_model"]

    print()
    print(RULE)
    print("PHASE 0 VERDICT")
    print(RULE)
    print(f"  GATE A concept readout : {'PASS' if a_pass else 'REVIEW'} "
          f"({gate_a['n_hits']}/{gate_a['n_scorable']} concepts in top-{config.topk})")
    print(f"  GATE B late identity   : {'PASS' if b_pass else 'REVIEW'} "
          f"(||J-I|| {gate_b['lowest_block_distance']:.3f} -> "
          f"{gate_b['highest_block_distance']:.3f}; top-1 matches model: "
          f"{bool(gate_b['agreement']['top1_jlens_equals_model'])})")
    print(f"  activation store (R2)  : "
          f"{'READY' if r2_report.get('round_trip') else 'NOT READY'} "
          f"-> {r2_report['prefix']}")
    if not r2_report.get("round_trip"):
        print("                           Phase 2 will abort on this. No activations were")
        print("                           written by this stage, so nothing is lost yet.")
    print(f"\n  records: {txt_path}")
    print(f"           {json_path}")
    print()
    print("  These thresholds are a summary, not the judgement. Read the per-block")
    print("  token lists above: a lens loaded with the wrong layer offset can still")
    print("  score well on aggregate while being shifted by one block.")
    print()
    print("STOPPING at the Phase 0 gate, as agreed. Nothing downstream has run.")
    print(RULE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
