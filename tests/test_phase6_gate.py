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
read = p6.build_dictionary(readout, probe, BLOCK, 64, head, gain)
read_no_gain = p6.build_dictionary(readout, probe, BLOCK, 64, head, None)
check("atoms are read directions, J^T (g * w_t)", read.mode == "read")
check("dropping the gain changes the atoms materially",
      not np.allclose(read.atoms[0], read_no_gain.atoms[0], atol=1e-3),
      f"|cos| {abs(read.atoms[0] @ read_no_gain.atoms[0]):.3f}")
identity = p6.verify_read_directions(readout, read, BLOCK, np.random.default_rng(5))
check("the read identity holds: atoms ARE the lens's measurement weights",
      identity["holds"], f"min corr {identity['min_correlation']:.8f}")
no_gain_identity = p6.verify_read_directions(
    readout, read_no_gain, BLOCK, np.random.default_rng(5))
check("dropping the gain breaks the identity, which is why g is absorbed",
      not no_gain_identity["holds"],
      f"min corr {no_gain_identity['min_correlation']:.4f} without g vs "
      f"{identity['min_correlation']:.4f} with it")

print("\n[3b] write_space: the J^+ ablation is available and labelled")
transport = p6.factor_transport(readout, BLOCK, PCAJLensConfig().dict_pinv_rcond)
check("J^+ inverts jlens's own transport", transport.roundtrip_cosine > 0.999,
      f"cos {transport.roundtrip_cosine:.6f}")
check("cond(J) and the retained rank are reported",
      transport.condition_number > 1 and 1 <= transport.rank <= D,
      f"cond {transport.condition_number:.0f}, rank {transport.rank}/{D} "
      f"at rcond={PCAJLensConfig().dict_pinv_rcond:g}")
check("the truncation bounds the pullback's amplification",
      transport.effective_condition <= 1.0 / PCAJLensConfig().dict_pinv_rcond + 1e-6,
      f"effective cond {transport.effective_condition:.1f} <= "
      f"{1 / PCAJLensConfig().dict_pinv_rcond:.0f}")
write = p6.build_dictionary(readout, probe, BLOCK, 64, head, gain, transport)
check("write atoms are labelled as such", write.mode == "write")
check("write atoms are a different dictionary",
      float(np.abs(np.sum(read.atoms * write.atoms, axis=1)).max()) < 0.99,
      f"max |cos(read_t, write_t)| "
      f"{np.abs(np.sum(read.atoms * write.atoms, axis=1)).max():.3f}")

print("\n[4] end-to-end gate")
from safetensors.numpy import save_file
cfg = PCAJLensConfig(run_name="t6")
cfg.phase_dir.mkdir(parents=True, exist_ok=True)
# Emotion vectors built FROM the dictionary: a known sparse nonneg part plus a
# large remainder, so the reportable fraction has a designed value (~10%).
labels, rows, order = {}, [], []
# READ atoms, matching what build_dictionary now constructs: the planted component has
# to be an element of the dictionary for the pursuit to have a recoverable ground truth.
atoms_full = (head * gain[None, :]) @ np.asarray(
    torch.as_tensor(J[BLOCK], dtype=torch.float32))
atoms_full /= np.linalg.norm(atoms_full, axis=1, keepdims=True)
mean_vec = rng.normal(size=D) * 3.0
# Planted at a 3:1 remainder:reportable ratio, so the designed reportable fraction is
# 1/(1+9) = 10% -- the brief's expectation. The planted component is a READ atom, i.e.
# exactly what the dictionary contains, so the pursuit has a recoverable ground truth.
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

K_SMALL, K_LARGE = 12, 20   # the paper's 16/25 scaled to this 512-dim synthetic
# 400 controls: the p-value floors at 1/(n+1), and the gate's alpha is Bonferroni-
# corrected over 16 emotions (0.0031), so fewer draws could not produce a significant
# result however strong the planted structure. [4d] checks that the gate SAYS so rather
# than reporting a null quietly.
N_CONTROLS = 400
UNDERPOWERED_CONTROLS = 60

def run(*extra):
    """One phase-6 invocation. Later `--set` wins, so callers can override the defaults."""
    buf = io.StringIO()
    try:
        with redirect_stdout(buf), redirect_stderr(buf):
            code = p6.main(["--set", "run_name=t6",
                            "--set", f"lens_local_path={lens_file}",
                            "--set", "dict_pool_size=400",
                            "--set", f"dict_atom_counts={K_SMALL},{K_LARGE}",
                            "--set", f"n_dict_atoms={K_SMALL}",
                            "--set", f"n_random_controls={N_CONTROLS}",
                            "--set", "pursuit_steps=60", *extra])
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
        buf.write(f"\nSystemExit: {e}\n")
    return code, buf.getvalue()

code, out = run("--dry-run")
check("dry-run exits 0 without loading weights", code == 0 and "STEP 3" not in out)
check("dry-run cross-checks the lens", "the lens covers this block" in out)
check("dry-run names the read construction, not a pullback",
      "J^T (g * w_t)" in out and "READS" in out)

code, out = run()
check("exits 0", code == 0, out.splitlines()[-1] if code else "")
check("stops at the gate", "Phase 7 has not run" in out)
rec = json.loads((cfg.phase_dir / "phase6_gate.json").read_text())

print("\n  the read identity, which replaced the self-token check")
check("the read identity is recorded and holds", rec["read_identity"]["holds"],
      f"min corr {rec['read_identity']['min_correlation']:.8f}")
check("atom mode is recorded as read", rec["atom_mode"] == "read")
check("the removed self-token check leaves no trace in the output",
      "lens back to their own token" not in out
      and "own token at rank 0" not in out)
check("the gate says why there is no such check",
      "no 'does lensing an atom return its own token' check" in out.lower()
      or "NO 'does lensing an atom return its own token' check" in out)

print("\n  both k are reported")
per = {r["emotion"]: r for r in rec["per_emotion"]}
check("every emotion has a result at every k",
      all(set(r["per_k"]) == {str(K_SMALL), str(K_LARGE)} for r in rec["per_emotion"]))
check("reported_k and saved_k are both recorded",
      rec["reported_k"] == [K_SMALL, K_LARGE] and rec["saved_k"] == K_SMALL)
small = np.array([r["per_k"][str(K_SMALL)]["frac_reportable"] for r in rec["per_emotion"]])
large = np.array([r["per_k"][str(K_LARGE)]["frac_reportable"] for r in rec["per_emotion"]])
check("the fraction rises with k, as a union of cones must",
      bool((large >= small - 1e-9).all()),
      f"median {np.median(small):.1%} at k={K_SMALL} -> {np.median(large):.1%} "
      f"at k={K_LARGE}")
check("both k appear in the printed table",
      f"frac k={K_SMALL}" in out and f"frac k={K_LARGE}" in out)
check("reportable fraction is small, as the brief expects",
      0.02 < np.median(small) < 0.35, f"median {np.median(small):.1%}")

print("\n  the gate is the random null")
controls = rec["random_control"]
check("a null is measured at every k",
      set(controls) == {str(K_SMALL), str(K_LARGE)})
check("the null draws are recorded as a count, not dumped as 120 floats",
      controls[str(K_SMALL)]["n"] == N_CONTROLS
      and "fractions" not in controls[str(K_SMALL)])
check("chance is far above k/d, so it cannot be replaced by arithmetic",
      controls[str(K_SMALL)]["mean"] > 2 * K_SMALL / D,
      f"control {controls[str(K_SMALL)]['mean']:.1%} vs k/d = {K_SMALL / D:.1%} "
      "(the pool is chosen from the direction's own top tokens)")
check("every emotion has a p-value at every k",
      all(r["per_k"][k].get("p_value") is not None
          for r in rec["per_emotion"] for k in (str(K_SMALL), str(K_LARGE))))
p_small = np.array([r["per_k"][str(K_SMALL)]["p_value"] for r in rec["per_emotion"]])
check("the planted structure beats the null for most emotions",
      float((p_small <= 1.0 / (1 + N_CONTROLS) + 1e-12).mean()) > 0.5,
      f"{int((p_small <= 1.0 / (1 + N_CONTROLS) + 1e-12).sum())}/{len(p_small)} at the "
      f"p floor {1 / (1 + N_CONTROLS):.4f}")
check("the corrected alpha and the p floor are both recorded",
      abs(rec["gate"]["alpha_bonferroni"] - p6.GATE_ALPHA / len(rec["per_emotion"])) < 1e-12
      and abs(rec["gate"]["p_value_floor"] - 1 / (1 + N_CONTROLS)) < 1e-12)
check("the p floor clears the corrected alpha at this control count",
      rec["gate"]["p_value_floor"] < rec["gate"]["alpha_bonferroni"],
      f"floor {rec['gate']['p_value_floor']:.4f} < alpha "
      f"{rec['gate']['alpha_bonferroni']:.4f}")
check("emotions are recorded as beating the null",
      sum(len(v) for v in rec["gate"]["beat_null"].values()) > 0,
      str({k: len(v) for k, v in rec["gate"]["beat_null"].items()}))
check("the printed table marks significance and names the threshold",
      "significance: p <" in out and "Bonferroni" in out)

print("\n  the cones caveat is printed, not left to be remembered")
check("the verdict states the union-of-cones fact",
      "UNION OF CONES" in out and "not a linear subspace" in out.lower())
check("it says what v_perp does NOT mean",
      "never means 'intrinsically unverbalizable'" in out
      or "NOT CAPTURED BY THIS SPARSE APPROXIMATION" in out)
check("the unsupported sentence is named explicitly",
      "cannot verbalise this component" in out)
check("the caveat travels in the record too",
      "unverbalizable" in rec.get("v_remainder_means", ""))

check("v_J + v_perp fractions roughly partition the variance",
      all(abs(r["per_k"][str(K_SMALL)]["frac_reportable"]
              + r["per_k"][str(K_SMALL)]["frac_remainder"] - 1) < 0.25
          for r in rec["per_emotion"]))
check("the two parts are near-orthogonal",
      max(abs(r["per_k"][str(K_SMALL)]["cos_parts"]) for r in rec["per_emotion"]) < 0.25)
print(f"    planted reportable fraction: {1 / (1 + PLANTED_RATIO ** 2):.0%} by construction")
check("v_J selects the emotion's own token for a majority",
      sum(1 for r in rec["per_emotion"] if r["own_word_atom_rank"] is not None) >= 9,
      f"{sum(1 for r in rec['per_emotion'] if r['own_word_atom_rank'] is not None)}/16 "
      "emotions (reported, not gated)")
check("coherence is recorded", "max" in rec["coherence"])
check("no J^+ factorisation happened on the read path", rec.get("transport") is None)

print("\n[4d] too few controls to resolve the corrected alpha is SAID, not hidden")
code_u, out_u = run("--set", f"n_random_controls={UNDERPOWERED_CONTROLS}")
rec_u = json.loads((cfg.phase_dir / "phase6_gate.json").read_text())
check("the p floor cannot reach the corrected alpha",
      rec_u["gate"]["p_value_floor"] > rec_u["gate"]["alpha_bonferroni"],
      f"floor {rec_u['gate']['p_value_floor']:.4f} > alpha "
      f"{rec_u['gate']['alpha_bonferroni']:.4f}")
check("the gate prints TOO COARSE rather than reporting a quiet null",
      "TOO COARSE" in out_u)
check("nothing is recorded as beating the null, because nothing could",
      sum(len(v) for v in rec_u["gate"]["beat_null"].values()) == 0)
check("and it exits 3 rather than passing an untestable run",
      code_u == 3, f"code={code_u}")

print("\n[4b] the degenerate regime is called out, not quietly passed")
code, outd = run("--set", "dict_atom_counts=200", "--set", "n_dict_atoms=200",
                 "--set", "dict_pool_size=450", "--set", "n_random_controls=20")
recd = json.loads((cfg.phase_dir / "phase6_gate.json").read_text())
check("a near-spanning pool drives the null high",
      recd["random_control"]["200"]["mean"] > p6.DEGENERATE_CONTROL_FRACTION,
      f"null {recd['random_control']['200']['mean']:.0%}")
check("the gate refuses to let any fraction be read",
      "DEGENERATE" in outd and "interpretable" in outd)

print("\n[4c] write_space is available, labelled, and refused by phases 7 and 8")
code_w, out_w = run("--set", "write_space=true",
                    "--set", "n_random_controls=20")
rec_w = json.loads((cfg.phase_dir / "phase6_gate.json").read_text())
check("it runs", code_w in (0, 3), f"code={code_w}")
check("atoms are the J^+ ones", rec_w["atom_mode"] == "write")
check("every screen of output says so",
      out_w.count("write_space") >= 2 and "ABLATION" in out_w)
check("the verdict disclaims reportability",
      "not a statement about reportability" in out_w
      or "nothing here is a statement about reportability" in out_w)
check("cond(J) and the retained rank are printed, since rcond now matters",
      "cond(J)" in out_w and "rank kept" in out_w)
check("write_space is flagged in the sidecar for downstream phases",
      rec_w["write_space"] is True)
import emotion_pca_jlens.phase7_channels as _p7
_refused = False
try:
    _p7.read_decomposition(cfg)
except SystemExit as _e:
    _refused, _msg = True, str(_e)
check("phase 7 refuses a write_space decomposition", _refused,
      _msg.splitlines()[0] if _refused else "")
check("the refusal explains that frac_reportable is not about reportability there",
      _refused and "reportability" in _msg)

code, out = run("--set", "n_random_controls=20")   # restore read-space artefacts for [5]

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
