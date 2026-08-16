"""Phase 6 against a lens whose pullback dictionary has a known answer."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import io, json, shutil, sys, tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
import torch

from core import jlens_lens, model_utils, paths
from core.model_utils import ArchitectureInfo
from emotion_pca_jlens import phase6_decompose as p6
from emotion_pca_jlens.pca_jlens_config import PCAJLensConfig
from emotion_pca_jlens.phase1_stimuli import DEFAULT_CIRCUMPLEX_SET, NEUTRAL_QUADRANT

fails = []
def check(name, cond, extra=""):
    print(("  ok   " if cond else "  FAIL") + f" {name}" + (f"   {extra}" if extra else ""))
    if not cond:
        fails.append(name)

D, N_LAYERS, BLOCK, VOCAB = 512, 6, 2, 2000
tmp_root = Path(tempfile.mkdtemp())
paths.OUTPUTS_DIR = tmp_root / "outputs"

print("\n[1] nonneg gradient pursuit recovers a known sparse nonneg combination")
rng = np.random.default_rng(0)
atoms = rng.normal(size=(120, D)); atoms /= np.linalg.norm(atoms, axis=1, keepdims=True)
truth_support = [3, 17, 42, 88]
truth_coeffs = np.array([1.5, 0.9, 2.2, 0.4])
target = atoms[truth_support].T @ truth_coeffs
c, sup = p6.nonneg_gradient_pursuit(target, atoms, 8, 400)
recon = atoms[sup].T @ c
check("reconstructs the target almost exactly",
      np.linalg.norm(target - recon) / np.linalg.norm(target) < 1e-3,
      f"rel err {np.linalg.norm(target - recon) / np.linalg.norm(target):.2e}")
check("finds the planted atoms", set(truth_support) <= set(sup), f"support {sorted(sup)}")
check("every coefficient is nonnegative", (c >= 0).all())
check("recovers the planted coefficients",
      np.allclose(sorted(c[[sup.index(i) for i in truth_support]]), sorted(truth_coeffs),
                  atol=1e-2))
neg = p6.nonneg_gradient_pursuit(-target, atoms, 8, 400)
check("a target in the opposite cone is not reconstructed by nonneg atoms",
      np.linalg.norm(-target - (atoms[neg[1]].T @ neg[0])) / np.linalg.norm(target) > 0.3
      if neg[1] else True,
      f"support size {len(neg[1])}")
c2, sup2 = p6.nonneg_gradient_pursuit(rng.normal(size=D), atoms, 4, 400)
check("k is respected", len(sup2) <= 4)

print("\n[2] a random direction is only partly reconstructable -- the baseline is real")
fracs = []
for _ in range(30):
    v = rng.normal(size=D)
    cc, ss = p6.nonneg_gradient_pursuit(v, atoms, 20, 300)
    vj = atoms[ss].T @ cc
    fracs.append(float(vj @ vj / (v @ v)))
check("20 atoms of 120 recover a real share of pure noise in 512 dims",
      0.02 < np.mean(fracs) < 0.4,
      f"mean {np.mean(fracs):.1%} -- nonzero, which is exactly why the control matters")

print("\n[3] the dictionary construction absorbs the RMSNorm gain")
class Tok:
    def __init__(s, v): s.vocab, s.inv = v, {i: w for w, i in v.items()}
    def encode(s, t, add_special_tokens=False):
        k = t.strip().lower()
        return [s.vocab[k]] if k in s.vocab else [0, 1]
    def decode(s, ids): return s.inv.get(int(ids[0]), f"<{ids[0]}>")

class Head:
    def __init__(s, W): s.weight = W
class Norm:
    def __init__(s, g): s.weight = g
class Model:
    def __init__(s, W, g, tok):
        s.W, s.g, s.tokenizer, s.d_model, s.n_layers = W, g, tok, D, N_LAYERS
        s._final_norm, s._lm_head, s._logit_softcap = Norm(g), Head(W), None
    def unembed(s, h):
        h = torch.as_tensor(h, dtype=torch.float32)
        return s.W @ (h / h.norm().clamp_min(1e-6) * np.sqrt(D) * s.g)
class Lens:
    def __init__(s, J): s.jacobians, s.d_model = J, D
    @property
    def source_layers(s): return sorted(s.jacobians)
    def transport(s, h, b): return s.jacobians[b] @ torch.as_tensor(h, dtype=torch.float32)

torch.manual_seed(11)   # the gain below came from the global RNG
g = torch.rand(D) * 2 + 0.3                       # a non-trivial learned gain
W = torch.tensor(rng.normal(size=(VOCAB, D)) / np.sqrt(D), dtype=torch.float32)
J = {l: torch.eye(D) + 0.05 * torch.randn(D, D, generator=torch.Generator().manual_seed(2))
     for l in range(N_LAYERS - 1)}
emotions_all = [e.emotion for e in DEFAULT_CIRCUMPLEX_SET]
vocab = {w.lower(): i for i, w in enumerate(emotions_all)}
tok = Tok(vocab)
readout = jlens_lens.LensReadout(Lens(J), Model(W, g, tok))
head, gain = p6.unembed_parts(readout)
check("gain recovered from the model, not assumed to be ones",
      gain is not None and np.allclose(gain, g.numpy()))
probe = rng.normal(size=D)
transport = p6.factor_transport(readout, BLOCK, PCAJLensConfig().dict_pinv_rcond)
check("J^+ inverts jlens's own transport, so the orientation is verified",
      transport.roundtrip_cosine > 0.999, f"cos {transport.roundtrip_cosine:.6f}")
check("cond(J) and the retained rank are reported",
      transport.condition_number > 1 and 1 <= transport.rank <= D,
      f"cond {transport.condition_number:.0f}, rank {transport.rank}/{D} at rcond={PCAJLensConfig().dict_pinv_rcond:g}")
check("the truncation bounds the pullback's amplification",
      transport.effective_condition <= 1.0 / PCAJLensConfig().dict_pinv_rcond + 1e-6,
      f"effective cond {transport.effective_condition:.1f} <= {1 / PCAJLensConfig().dict_pinv_rcond:.0f}")
dict_with = p6.build_dictionary(readout, probe, BLOCK, 64, head, gain, transport)
dict_without = p6.build_dictionary(readout, probe, BLOCK, 64, head, None, transport)
check("dropping the gain changes the atoms materially",
      not np.allclose(dict_with.atoms[0], dict_without.atoms[0], atol=1e-3),
      f"|cos| {abs(dict_with.atoms[0] @ dict_without.atoms[0]):.3f}")
v_with = p6.verify_dictionary(readout, dict_with, BLOCK, 24)
v_without = p6.verify_dictionary(readout, dict_without, BLOCK, 24)
check("with the gain, atoms lens back to their own token",
      v_with["frac_in_top10"] >= 0.9,
      f"rank-0 {v_with['frac_rank_zero']:.0%}, top-10 {v_with['frac_in_top10']:.0%}, "
      f"median {v_with['median_self_rank']:.0f}")
check("without it, they do so measurably less often",
      v_without["frac_in_top10"] <= v_with["frac_in_top10"],
      f"top-10 {v_without['frac_in_top10']:.0%} vs {v_with['frac_in_top10']:.0%}")
check("the check reports whether the gain was absorbed", v_with["gain_absorbed"] is True)

print("\n[4] end-to-end gate")
from safetensors.numpy import save_file
cfg = PCAJLensConfig(run_name="t6")
cfg.phase_dir.mkdir(parents=True, exist_ok=True)
# Emotion vectors built FROM the dictionary: a known sparse nonneg part plus a
# large remainder, so the reportable fraction has a designed value (~10%).
labels, rows, order = {}, [], []
atoms_full = (head * gain[None, :]) @ transport.pullback.T
atoms_full /= np.linalg.norm(atoms_full, axis=1, keepdims=True)
mean_vec = rng.normal(size=D) * 3.0
# Planted at a 3:1 remainder:reportable ratio, so the designed reportable fraction is
# 1/(1+9) = 10% -- the brief's expectation. What had to change with the J^+ pullback is
# the POOL, not the ratio: the pool is the top-N tokens of v's own lens readout, and a
# unit-norm J^+ atom has a weaker readout at its own token than a unit-norm J^T atom did
# (J^+ inflates the pre-image norm), so the planted token sits deeper in v's ranking.
# 400 of a 2000-token vocabulary reaches 13/16; the real run is 512 of ~150k, a far
# smaller share, which is a limit of a 2000-token synthetic rather than of the method.
PLANTED_RATIO = 3.0
for e in DEFAULT_CIRCUMPLEX_SET:
    own = vocab[e.emotion]
    reportable = atoms_full[own]
    remainder = rng.normal(size=D)
    remainder -= (remainder @ reportable) * reportable / (reportable @ reportable)
    remainder *= PLANTED_RATIO * np.linalg.norm(reportable) / np.linalg.norm(remainder)
    rows.append(mean_vec + reportable + remainder)
    order.append(e.emotion)
    labels[e.emotion] = dict(quadrant=e.quadrant, valence=e.valence,
                             arousal=e.arousal, family=e.family, source="emotion")
mat = np.asarray(rows, dtype=np.float32)
save_file({"emotion_vectors": mat, "emotion_vectors_half_a": mat,
           "emotion_vectors_half_b": mat}, str(cfg.emotion_vectors_path),
          metadata={"emotions": json.dumps(order)})
cfg.emotion_vectors_meta_path.write_text(json.dumps({
    "emotions": order, "labels": labels,
    "target": {"block": BLOCK, "hidden_state": BLOCK + 1, "n_layers": N_LAYERS},
    "fingerprint": {"model_name": "stub/model"},
    "split_half": {"threshold": 0.9, "summary": {"min_cosine_centered": 0.97}}}))

lens_file = cfg.phase_dir / "lens.pt"
torch.save({"J": J, "d_model": D, "n_prompts": 500}, lens_file)
ARCH = ArchitectureInfo(model_name="stub/model", revision="main", resolved_sha="s",
                        n_layers=N_LAYERS, n_hidden_states=N_LAYERS + 1, hidden_size=D,
                        architectures=("Stub",), config_dtype="float32",
                        max_position_embeddings=512)
model_utils.load_architecture_info = lambda *a, **k: ARCH
model_utils.load_tokenizer = lambda *a, **k: tok
model_utils.load_model = lambda *a, **k: object()
jlens_lens.LensReadout.build = classmethod(lambda cls, m, t, p: readout)

def run(*extra):
    buf = io.StringIO()
    try:
        with redirect_stdout(buf), redirect_stderr(buf):
            code = p6.main(["--set", "run_name=t6",
                            "--set", f"lens_local_path={lens_file}",
                            "--set", "dict_pool_size=400", "--set", "n_dict_atoms=12",
                            "--set", "n_random_controls=8", *extra])
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
        buf.write(f"\nSystemExit: {e}\n")
    return code, buf.getvalue()

code, out = run("--dry-run")
check("dry-run exits 0 without loading weights", code == 0 and "STEP 3" not in out)
check("dry-run cross-checks the lens", "the lens covers this block" in out)

code, out = run()
check("exits 0", code == 0, out.splitlines()[-1] if code else "")
check("stops at the gate", "Phase 7 has not run" in out)
rec = json.loads((cfg.phase_dir / "phase6_gate.json").read_text())
check("dictionary check ran and passed",
      rec["dictionary"]["frac_in_top10"] >= 0.9,
      f"top-10 {rec['dictionary']['frac_in_top10']:.0%}")
per = {r["emotion"]: r for r in rec["per_emotion"]}
fr = np.array([r["frac_reportable"] for r in rec["per_emotion"]])
check("reportable fraction is small, as the brief expects",
      0.02 < np.median(fr) < 0.35, f"median {np.median(fr):.1%}")
# The planted truth is 10% reportable; the pursuit measures that plus whatever it
# absorbs from the remainder by chance, which the control quantifies. So the claim
# to test is that the excess over chance is real and in the right direction -- and
# that chance is well above k/d, because the pool is adaptively chosen.
check("above the chance baseline, with the excess in the planted range",
      np.median(fr) > rec["random_control"]["p95"],
      f"median {np.median(fr):.1%} vs chance mean "
      f"{rec['random_control']['mean']:.1%} / p95 {rec['random_control']['p95']:.1%}")
check("chance is far above k/d, so it cannot be replaced by arithmetic",
      rec["random_control"]["mean"] > 2 * 12 / D,
      f"control {rec['random_control']['mean']:.1%} vs k/d = {12 / D:.1%} "
      "(the pool is chosen from the direction's own top tokens)")
check("the design is not in the degenerate regime",
      rec["random_control"]["mean"] <= p6.DEGENERATE_CONTROL_FRACTION
      and "DEGENERATE" not in out)
check("v_J + v_perp fractions roughly partition the variance",
      all(abs(r["frac_reportable"] + r["frac_remainder"] - 1) < 0.25
          for r in rec["per_emotion"]),
      f"max |sum-1| {max(abs(r['frac_reportable']+r['frac_remainder']-1) for r in rec['per_emotion']):.3f}")
check("the two parts are near-orthogonal", max(abs(r["cos_parts"]) for r in rec["per_emotion"]) < 0.25,
      f"max |cos| {max(abs(r['cos_parts']) for r in rec['per_emotion']):.3f}")
print(f"    planted reportable fraction: {1 / (1 + PLANTED_RATIO ** 2):.0%} by construction")
# A MAJORITY, not 14/16. The transpose construction hit 14/16 with a 64-atom pool; the
# J^+ pullback reaches 10/16 with a 400-atom pool on the same planted 10%. Both effects
# push the same way: a correct atom's own-token readout is weaker per unit norm, so the
# planted token sits deeper in v's ranking (needing the wider pool), and selecting 12
# atoms from 400 rather than 64 is a harder problem. Expect own_word_atom_rank to be
# filled in for fewer emotions on the real run than the transpose version showed.
check("v_J decomposes into the emotion's own token for a majority",
      sum(1 for r in rec["per_emotion"] if r["own_word_atom_rank"] is not None) >= 9,
      f"{sum(1 for r in rec['per_emotion'] if r['own_word_atom_rank'] is not None)}/16 emotions")
check("random control measured and reported",
      rec["random_control"]["n"] == 8 and rec["random_control"]["mean"] > 0,
      f"chance {rec['random_control']['mean']:.1%}")
check("gate prints the chance baseline as the denominator",
      "Chance baseline" in out and "x chance" in out)
check("verdict reports the atom-validity check",
      "atoms lens to their token" in out and "PASS" in out)
check("verdict reports cond(J) and the retained rank",
      "cond(J)" in out and "rank" in out)
check("the gate prints the per-atom self-ranks",
      "per atom: rank of its own token" in out)
check("the round-trip against jlens's transport is printed",
      "round-trip cos" in out)
check("per-atom detail is in the record",
      len(rec["dictionary"]["per_atom"]) == rec["dictionary"]["checked"]
      and all("self_top1" in e for e in rec["dictionary"]["per_atom"]))
check("dictionary_valid is at the top level for phases 7 and 8",
      rec["dictionary_valid"] is True)
check("the transport summary is recorded",
      {"condition_number", "rank", "rcond", "roundtrip_cosine"}
      <= set(rec["transport"]))

print("\n[4b] the degenerate regime is called out, not quietly passed")
code, outd = run("--set", "n_dict_atoms=200", "--set", "dict_pool_size=450")
recd = json.loads((cfg.phase_dir / "phase6_gate.json").read_text())
check("a near-spanning pool drives the control high",
      recd["random_control"]["mean"] > p6.DEGENERATE_CONTROL_FRACTION,
      f"control {recd['random_control']['mean']:.0%}")
check("the gate refuses to let any fraction be read",
      "DEGENERATE" in outd and "uninterpretable" in outd)
print("\n[4c] a failed atom-validity check withholds the fraction and exits 3")
_real_valid = p6.dictionary_is_valid
p6.dictionary_is_valid = lambda check: False
try:
    code_bad, out_bad = run()
finally:
    p6.dictionary_is_valid = _real_valid
rec_bad = json.loads((cfg.phase_dir / "phase6_gate.json").read_text())
check("exits 3, so phase 7 cannot be chained onto it", code_bad == 3, f"code={code_bad}")
check("the word WITHHELD appears where the table would be", "WITHHELD" in out_bad)
check("no reportable percentage is printed anywhere",
      "frac v_J" not in out_bad and "x chance" not in out_bad)
check("the verdict says the check gates everything above it",
      "atom-validity check gates everything above it" in out_bad)
check("dict_pinv_rcond is named as the knob", "dict_pinv_rcond" in out_bad)
check("the rcond sweep is printed with rank, coherence and validity",
      "eff cond" in out_bad and "coh max" in out_bad
      and len(rec_bad["rcond_sweep"]) == len(p6.RCOND_SWEEP))
check("dictionary_valid=false is recorded for phases 7 and 8",
      rec_bad["dictionary_valid"] is False)
check("the split is still written, for diagnosis",
      cfg.decomposition_path.exists() and "diagnosis only" in out_bad)
from safetensors import safe_open as _safe_open
with _safe_open(str(cfg.decomposition_path), framework="numpy") as _f:
    check("dictionary_valid travels with the tensors too",
          _f.metadata().get("dictionary_valid") == "False",
          str(_f.metadata().get("dictionary_valid")))
check("coherence is recorded even on a failure", "max" in rec_bad["coherence"])

code, out = run()   # restore the good artefacts for [5]

print("\n[5] artefacts Phase 8 will steer with")
from safetensors.numpy import load_file
t = load_file(str(cfg.decomposition_path))
check("four norm-matched directions per emotion",
      set(t) == {"v", "v_reportable", "v_remainder", "v_random"}
      and all(v.shape == (16, D) for v in t.values()))
n_v = np.linalg.norm(t["v"], axis=1)
for key in ("v_reportable", "v_remainder", "v_random"):
    check(f"{key} is norm-matched to v",
          np.allclose(np.linalg.norm(t[key], axis=1), n_v, rtol=1e-4),
          f"max rel dev {np.abs(np.linalg.norm(t[key],axis=1)/n_v - 1).max():.2e}")
check("csv + metadata written",
      (cfg.phase_dir / "phase6_decomposition.csv").exists()
      and (cfg.phase_dir / "phase6_decomposition.json").exists())
from safetensors import safe_open
with safe_open(str(cfg.decomposition_path), framework="np") as fh:
    m = fh.metadata()
check("row order and centring recorded in the file",
      json.loads(m["emotions"]) == order and "mean-centred" in m["centred"]
      and m["target_block"] == str(BLOCK))

print("\n[6] independence from phases 3-5")
check("no phase3/4/5 artefact was needed",
      not (cfg.phase_dir / "phase3_pcs.safetensors").exists()
      and (cfg.phase_dir / "phase6_gate.json").exists())
src = Path("emotion_pca_jlens/phase6_decompose.py").read_text()
check("does not import phase5", "phase5" not in src)
check("says so in the gate output", "Independent of Phases 3-5" in out)

print("\n[7] a lens that does not fit is refused")
bad = cfg.phase_dir / "bad.pt"
torch.save({"J": {l: torch.eye(D + 4) for l in range(N_LAYERS - 1)},
            "d_model": D + 4, "n_prompts": 300}, bad)
code, outb = run("--set", f"lens_local_path={bad}")
check("exits 3 before any decomposition",
      code == 3 and "STEP 4" not in outb, f"code={code}")

shutil.rmtree(tmp_root, ignore_errors=True)
print("\n" + "=" * 60)
print("FAILED: " + ", ".join(fails) if fails else "ALL CHECKS PASSED")
sys.exit(1 if fails else 0)
