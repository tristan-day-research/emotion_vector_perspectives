"""Phase 7: the separation gate, the scorers, and the judge client."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import io, json, shutil, sys, tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np

from core import paths
from core.judge import Judge, Verdict, judge_available, score_schema
from emotion_pca_jlens import channel_prompts as cp, phase7_channels as p7
from emotion_pca_jlens.pca_jlens_config import PCAJLensConfig

fails = []
def check(name, cond, extra=""):
    print(("  ok   " if cond else "  FAIL") + f" {name}" + (f"   {extra}" if extra else ""))
    if not cond:
        fails.append(name)

print("\n[1] THE GATE: affect vocabulary is asymmetric across the channels")
sep = p7.check_separation()
check("behaviour channel has ZERO affect vocabulary",
      sep["behaviour_hits"] == 0,
      "hits: " + str([h for g, d in sep["per_group"].items() if g.startswith("behaviour")
                      for h in d["hits"]])[:200])
check("report channel DOES have affect vocabulary (the check works)",
      sep["report_hits"] > 0, f"{sep['report_hits']} hits")
check("separated == True", sep["separated"] is True)
check("the real 171-word list is loaded, not just the general stems",
      sep["emotion_list_loaded"] and sep["n_affect_stems"] > 170,
      f"{sep['n_affect_stems']} stems")
check("behaviour rubric covered by the check",
      sep["per_group"]["behaviour_rubric"]["n_items"] == 1)
check("mechanical scorer source covered too (a regex can leak a word)",
      "behaviour_scorer_source" in sep["per_group"])

print("\n[2] the check catches a real leak, in each place one could hide")
stems = cp.affect_vocabulary()
for name, text in [
    ("rubric asking about tone", "Score how anxious the response sounds."),
    ("an emotion word from the 171 list", "Rate whether the reply seems resentful."),
    ("a general affect stem", "Note the emotional register of the answer."),
    ("morphological variant", "Score the writer's feelings about the task."),
]:
    hits = cp.affect_hits(text, stems)
    check(f"catches: {name}", len(hits) > 0, f"{[h.matched for h in hits]}")
check("does not fire on affect stems buried mid-word (substring search would)",
      not cp.affect_hits("Unsaddle the horse and discount the rest.", stems),
      str([h.matched for h in cp.affect_hits("Unsaddle the horse and discount the rest.", stems)]))
check("DOES over-flag a word-initial collision -- fails closed, by design",
      [h.matched for h in cp.affect_hits("Put it in the saddle.", stems)] == ["saddle"],
      "'saddle' flagged for stem 'sad'; the fix is to reword, not to loosen the check")
check("allowlisted 'content' (the noun) is not flagged",
      not any(h.matched.lower() == "content"
              for h in cp.affect_hits("Summarize the content of the document.", stems)))
check("allowlist is reported, so its holes are visible",
      set(sep["allowlist"]) == set(cp.AFFECT_ALLOWLIST))

print("\n[3] mechanical scorers: countable, and blind to affect by construction")
report_tasks = cp.build_report_choice_tasks(
    ["terrified", "angry", "content"], seed=7, n_variants=5
)
check("report choice construction is deterministic",
      report_tasks == cp.build_report_choice_tasks(
          ["terrified", "angry", "content"], seed=7, n_variants=5
      ))
for state in ("terrified", "angry", "content", None):
    positions = [
        label for task in report_tasks[:4]
        for label, mapped in task.label_to_emotion if mapped == state
    ]
    check(f"report option {state!r} occupies every label over a full cycle",
          set(positions) == {"A", "B", "C", "D"}, str(positions))
first_label, first_state = report_tasks[0].label_to_emotion[0]
parsed = cp.score_report_choice(first_label, report_tasks[0])
check("report choice exactly maps the saved letter back to its state",
      parsed["valid"] and parsed["choice_emotion"] == first_state)
check("report choice rejects explanatory prose",
      not cp.score_report_choice(f"{first_label} because it fits", report_tasks[0])["valid"])
check("report choice rejects a truncated letter",
      not cp.score_report_choice(first_label, report_tasks[0], truncated=True)["valid"])

check("risk: certain option scores 1", cp.score_risk("A")["score"] == 1.0)
check("risk: gamble scores 0", cp.score_risk("B")["score"] == 0.0)
check("risk: echoed option-label punctuation stays scoreable",
      cp.score_risk("B)")["score"] == 0.0 and cp.score_risk("A.")["score"] == 1.0)
check("risk: unparseable is None, not a guess", cp.score_risk("hmm")["score"] is None)
# This assertion used to read: "reads the FIRST choice, not the last mention", with
# "A is safer than B, so A." scored 1.0. That WAS the bug -- with a reasoning block in
# front of the answer, the first standalone letter is whichever one the model mentioned
# while thinking. Two distinct letters is now INVALID rather than resolved by position.
check("risk: two distinct letters is INVALID, not resolved by position",
      not cp.score_risk("A is safer than B, so A.")["valid"])
check("risk: explanatory prose is invalid under the exact-letter protocol",
      not cp.score_risk("A. A is the certain option.")["valid"])
check("risk: letters inside a reasoning block are ignored",
      cp.score_risk("<think>A or B? A looks safer</think>\nB")["score"] == 0.0)
check("risk: a truncated reasoning trace is INVALID, which is the whole fix",
      not cp.score_risk("<think>Let me weigh A against", truncated=True)["valid"])
long_answer = ("Flip the first switch on for ten minutes then off. " * 4
               + "Therefore the warm bulb is the first switch. That is the answer.")
check("persistence: committed long answer scores 1",
      cp.score_persistence(long_answer)["score"] == 1.0)
check("persistence: giving up scores 0",
      cp.score_persistence("I'm not sure this is solvable.")["score"] == 0.0)
h = cp.score_hedging("It might be around 1986, though possibly I am not certain.")
check("hedging: counts hedges as a rate per 100 words", h["score"] > 20, f"{h['score']:.1f}")
# "Answer directly" makes "1986." the ideal response, so it must SCORE (0 hedges), not
# be rejected as too short -- otherwise perfect compliance would read as 100% invalid.
check("hedging: the ideal one-word answer scores 0 and stays valid",
      cp.score_hedging("1986.")["score"] == 0.0
      and cp.score_hedging("1986.")["valid"])
check("hedging: and it says the rate is high-variance at that length",
      "high-variance" in cp.score_hedging("1986.")["detail"])
check("hedging is a RATE, so length alone cannot move it",
      abs(cp.score_hedging("It might be x. " * 2)["score"]
          - cp.score_hedging("It might be x. " * 8)["score"]) < 1e-9)
check("no affect word appears in any mechanical scorer's patterns",
      not cp.affect_hits(" ".join(cp.HEDGE_PATTERNS), stems))
check("3 of 4 behaviour families are mechanical",
      sum(1 for t in cp.BEHAVIOUR_TASKS if t.scorer == "mechanical") ==
      len([t for t in cp.BEHAVIOUR_TASKS if t.family != "refusal"]),
      f"{sorted({t.family for t in cp.BEHAVIOUR_TASKS if t.scorer == 'mechanical'})}")

print("\n[4] the judge client")
class Resp:
    def __init__(s, text=None, stop="end_turn", cat=None):
        s.content = [type("B", (), {"type": "text", "text": text})()] if text else []
        s.stop_reason = stop
        s.stop_details = type("D", (), {"category": cat})() if cat else None
        s.usage = type("U", (), {"input_tokens": 900, "output_tokens": 40,
                                 "cache_read_input_tokens": 850})()
class Client:
    def __init__(s, resp): s.resp, s.calls = resp, []
    @property
    def messages(s): return s
    def create(s, **kw): s.calls.append(kw); return s.resp

j = Judge(rubric="R" * 4000, client=Client(Resp('{"score": 3, "reason": "quoted phrase"}')))
v = j.score("some text", label="anxious")
check("parses a structured verdict", v.usable and v.score == 3.0 and "quoted" in v.reason)
kw = j.client.calls[0]
check("NO temperature parameter (it 400s on current models)",
      "temperature" not in kw and "top_p" not in kw and "top_k" not in kw,
      f"sent keys: {sorted(kw)}")
check("uses output_config.format, not the deprecated output_format",
      "output_config" in kw and "output_format" not in kw)
check("score is an enum, not min/max (schemas drop numeric bounds)",
      kw["output_config"]["format"]["schema"]["properties"]["score"]["enum"] == [0,1,2,3,4]
      and "minimum" not in json.dumps(kw["output_config"]["format"]["schema"]))
check("additionalProperties false + required, as strict schemas need",
      kw["output_config"]["format"]["schema"]["additionalProperties"] is False
      and set(kw["output_config"]["format"]["schema"]["required"]) == {"score", "reason"})
check("rubric is in system with a cache breakpoint",
      kw["system"][0]["cache_control"] == {"type": "ephemeral"}
      and kw["system"][0]["text"].startswith("RRR"))
check("the varying text goes AFTER the cached prefix (caching is prefix-match)",
      "some text" in kw["messages"][0]["content"] and "some text" not in kw["system"][0]["text"])
check("usage tracked incl. cache reads", j.usage["cached"] == 850 and j.calls == 1)

j2 = Judge(rubric="R", client=Client(Resp(None, stop="refusal", cat="cyber")))
v2 = j2.score("x")
check("a refusal is handled, not crashed on (HTTP 200, empty content)",
      v2.refused and not v2.usable and "cyber" in (v2.error or ""))
check("refusal still records usage", v2.input_tokens == 900)

class Boom:
    @property
    def messages(s): return s
    def create(s, **kw):
        import anthropic
        raise anthropic.APIConnectionError(request=None)
j3 = Judge(rubric="R", client=Boom())
v3 = j3.score("x")
check("a connection error becomes an unusable verdict, never an exception",
      not v3.usable and "connection" in (v3.error or "").lower())

est = j.estimate_cost(n_calls=640)
check("cost estimate priced, and caching + batching both reduce it",
      est["known_price"] and est["usd_cached"] < est["usd_no_cache"]
      and abs(est["usd_cached_batched"] - est["usd_cached"] / 2) < 1e-9,
      f"640 calls: ${est['usd_no_cache']:.2f} raw -> ${est['usd_cached']:.2f} cached "
      f"-> ${est['usd_cached_batched']:.2f} batched")
check("unknown model reports so rather than inventing a price",
      Judge(rubric="R", model="made-up", client=Client(Resp("{}"))
            ).estimate_cost(10)["known_price"] is False)
check("judge_available names the dependency and the credential separately",
      isinstance(judge_available(), tuple) and len(judge_available()) == 2)

print("\n[5] emotion selection from Phase 6's output")
decomp = {
    "random_control": {"mean": 0.02},
    "labels": {
        "anxious": {"quadrant": "HA-N", "valence": -1, "arousal": 1},
        "calm": {"quadrant": "LA-P", "valence": 1, "arousal": -1},
        "sad": {"quadrant": "LA-N", "valence": -1, "arousal": -1},
        "bored": {"quadrant": "LA-N", "valence": -1, "arousal": -1},
    },
    "per_emotion": [
        {"emotion": "anxious", "frac_reportable": 0.10, "own_word_atom_rank": 0},
        {"emotion": "calm", "frac_reportable": 0.12, "own_word_atom_rank": 1},
        {"emotion": "sad", "frac_reportable": 0.03, "own_word_atom_rank": 0},
        {"emotion": "bored", "frac_reportable": 0.09, "own_word_atom_rank": None},
        {"emotion": "neutral", "frac_reportable": 0.05, "own_word_atom_rank": None},
    ],
}
ranked = p7.rank_candidates(decomp, PCAJLensConfig())
by = {c.emotion: c for c in ranked}
check("neutral excluded", "neutral" not in by)
check("at-chance v_J marked unusable", not by["sad"].usable
      and "random control" in by["sad"].reasons[0])
check("missing exact English token is diagnostic, not a multilingual hard gate",
      by["bored"].usable and by["bored"].own_word_atom_rank is None)
check("the arousal-heavy negative is ranked first among usable ones",
      ranked[0].emotion == "anxious", f"order: {[c.emotion for c in ranked]}")
check("usable ones sort ahead of unusable", ranked[0].usable and not ranked[-1].usable)

print("\n[6] dynamic range: no baseline spread means Phase 8 sees nothing")
flat = [{"family": "risk", "score": 1.0} for _ in range(4)]
varied = [{"family": "hedging", "score": s} for s in (0.0, 2.0, 5.0)]
r = p7.dynamic_range(flat + varied)
check("flags a family with no baseline range",
      r["families_without_range"] == ["risk"], str(r["families_without_range"]))
check("does not flag one with range", r["per_family"]["hedging"]["has_range"] is True)
check("unscored input handled", p7.dynamic_range([{"family": "x", "score": None}])["scored"] is False)

print("\n[6b] a write_space decomposition is refused, not silently ranked")
_tmp6 = Path(tempfile.mkdtemp()); paths.OUTPUTS_DIR = _tmp6 / "outputs"
_cfg6 = PCAJLensConfig(run_name="t7guard")
_cfg6.phase_dir.mkdir(parents=True, exist_ok=True)
_msg = ""
for _label, _payload, _want in (
    ("write_space=true", {**decomp, "write_space": True}, True),
    ("write_space=false", {**decomp, "write_space": False}, False),
    ("absent (a read-space run)", dict(decomp), False),
):
    _cfg6.decomposition_meta_path.write_text(json.dumps(_payload))
    try:
        p7.read_decomposition(_cfg6)
        _raised = False
    except SystemExit as _e:
        _raised = True
        _msg = str(_e)
    check(f"{_label}: {'refused' if _want else 'accepted'}", _raised == _want,
          _msg.splitlines()[0] if _raised else "")
check("the refusal says why frac_reportable is not about reportability there",
      "reportability" in _msg and "WRITE" in _msg)

print("\n[6c] emotions are ranked at Phase 6's SAVED k, with its p-value")
_multi = {
    **decomp,
    "saved_k": 16,
    "gate": {"alpha_bonferroni": 0.05 / 4},
    "random_control": {"16": {"mean": 0.02}, "25": {"mean": 0.05}},
    "per_emotion": [
        # k=16 is significant, k=25 is not: reading the wrong k flips the verdict.
        {"emotion": "anxious", "own_word_atom_rank": 0,
         "per_k": {"16": {"frac_reportable": 0.10, "p_value": 0.001},
                   "25": {"frac_reportable": 0.30, "p_value": 0.9}}},
        {"emotion": "calm", "own_word_atom_rank": 1,
         "per_k": {"16": {"frac_reportable": 0.04, "p_value": 0.9},
                   "25": {"frac_reportable": 0.12, "p_value": 0.001}}},
    ],
}
_by = {c.emotion: c for c in p7.rank_candidates(_multi, PCAJLensConfig())}
check("the saved k's fraction is the one reported",
      _by["anxious"].frac_reportable == 0.10, f"{_by['anxious'].frac_reportable}")
check("a significant p-value at the saved k is usable", _by["anxious"].usable)
check("a non-significant one is not, even though k=25 would have passed",
      not _by["calm"].usable and "random null" in _by["calm"].reasons[0],
      _by["calm"].reasons[0])
check("the p-value gates in preference to the 3x-the-null fallback",
      "p=" in _by["calm"].reasons[0])
_legacy = p7.rank_candidates(decomp, PCAJLensConfig())
check("a pre-multi-k sidecar still ranks, via the 3x fallback",
      len(_legacy) == 4 and any(c.usable for c in _legacy))

print("\n[7] --dry-run: gate output with no weights and no judge")
tmp = Path(tempfile.mkdtemp()); paths.OUTPUTS_DIR = tmp / "outputs"
cfg = PCAJLensConfig(run_name="t7")
cfg.phase_dir.mkdir(parents=True, exist_ok=True)
decomp_full = {**decomp, "vectors": {"target_block": 31},
               "emotions": ["anxious", "calm"], "dictionary_valid": True}
cfg.decomposition_meta_path.write_text(json.dumps(decomp_full))
buf = io.StringIO()
try:
    with redirect_stdout(buf), redirect_stderr(buf):
        code = p7.main(["--set", "run_name=t7", "--dry-run", "--no-judge"])
except SystemExit as e:
    code = e.code if isinstance(e.code, int) else 1
    buf.write(f"\nSystemExit: {e}\n")
out = buf.getvalue()
check("exits 0", code == 0, out.splitlines()[-1] if code else "")
check("prints BOTH rubrics in full, as the brief requires",
      "REPORT CHANNEL rubric" in out and "BEHAVIOUR CHANNEL rubric" in out
      and "Score only what the response did" in out)
check("states the mechanical check is not sufficient",
      "necessary and NOT sufficient" in out)
check("names what a wordlist cannot catch",
      "how guarded does the answer sound" in out)
check("prints the chat-template assumption", "assumption, not a verified property" in out)
check("says the judge must not be the model under test",
      "perturb the judge as well as the subject" in out)
check("chose the arousal-heavy negative", "chosen: ['anxious'" in out)
check("no weights loaded", "Loading" not in out)
check("writes a dry-run record",
      (cfg.phase_dir / "phase7_channels" / "dry_run" / "phase7_dry_run.json").exists())

print("\n[8] a leaking behaviour rubric aborts before anything is generated")
original = cp.BEHAVIOUR_REFUSAL_RUBRIC
cp.BEHAVIOUR_REFUSAL_RUBRIC = "Score how anxious and uneasy the response sounds."
import importlib
p7.BEHAVIOUR_REFUSAL_RUBRIC = cp.BEHAVIOUR_REFUSAL_RUBRIC
sep_bad = p7.check_separation()
check("leak detected", sep_bad["separated"] is False and sep_bad["behaviour_hits"] >= 2,
      f"{sep_bad['behaviour_hits']} hits")
buf = io.StringIO()
try:
    with redirect_stdout(buf), redirect_stderr(buf):
        code = p7.main(["--set", "run_name=t7", "--no-judge"])
except SystemExit as e:
    code = e.code if isinstance(e.code, int) else 1
out_bad = buf.getvalue()
check("aborts with exit 3 before generating", code == 3 and "Loading" not in out_bad)
check("says every later number would be confounded", "confounded" in out_bad)
cp.BEHAVIOUR_REFUSAL_RUBRIC = original
p7.BEHAVIOUR_REFUSAL_RUBRIC = original

shutil.rmtree(tmp, ignore_errors=True)
print("\n" + "=" * 60)
print("FAILED: " + ", ".join(fails) if fails else "ALL CHECKS PASSED")
sys.exit(1 if fails else 0)
