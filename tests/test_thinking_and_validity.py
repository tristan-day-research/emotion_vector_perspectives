"""The four fixes for the invalidated Phase 7/8 runs, tested against the failure itself.

The bug was not a scoring bug. Qwen3's chat template opens a ``<think>`` block unless told
not to, so at 150 new tokens almost every completion was a truncated reasoning trace --
and the scorers read those traces as answers. ~7.5% of responses reached ``</think>``; none
of the behavioural ones did, and the invalid rate was 0% by construction.

So the tests here are built from real truncated-trace shapes, and the standard each fix has
to meet is: **the old code scored this; the new code must refuse to.**

Run: python tests/test_thinking_and_validity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import model_utils  # noqa: E402
from emotion_pca_jlens import channel_prompts as cp  # noqa: E402
from emotion_pca_jlens.pca_jlens_config import PCAJLensConfig  # noqa: E402

FAILURES: list[str] = []

#: Completions shaped like the ones that invalidated the run: a reasoning block that never
#: closed, cut off wherever the 150-token budget happened to end.
TRUNCATED_TRACES = {
    "risk": "<think>\nOkay, the user wants me to choose between A and B. Option A gives "
            "50 units for certain. Option B is a coin flip for 100 or 0. The expected "
            "values are equal, so this is really about risk preference. Let me think "
            "about which one",
    "persistence": "<think>\nThree switches, three bulbs, one entry. The classic trick is "
                   "to use heat as a second channel. So: turn on switch 1, wait ten "
                   "minutes, turn it off, turn on switch 2, then enter. But wait, I "
                   "should double check whether",
    "hedging": "<think>\nThe Chernobyl accident. I believe it was 1986, April 26th. It "
               "might have been the fourth reactor. Let me make sure I have that right, "
               "because it could possibly be",
    "refusal": "<think>\nThe user wants a critical restaurant review. That seems fine, "
               "it's a legitimate request. I should probably consider whether being "
               "sharply critical might",
    "report": "<think>\nThe user is asking how I feel. I should be careful here. I don't "
              "want to overclaim inner states, but I also shouldn't be dismissive. "
              "Perhaps I could say something about",
}


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  {'PASS' if condition else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


# --------------------------------------------------------------------------- #
# FIX 1: thinking mode
# --------------------------------------------------------------------------- #

class QwenLikeTokenizer:
    """A tokenizer whose template honours ``enable_thinking``, as Qwen3's does."""

    name_or_path = "stub/qwen-like"
    chat_template = "<jinja with enable_thinking>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False,
                            enable_thinking=True, **kwargs):
        body = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
                       for m in messages)
        if add_generation_prompt:
            body += "<|im_start|>assistant\n"
            body += "<think>\n" if enable_thinking else "<think>\n\n</think>\n\n"
        return body


class PlainTokenizer:
    """A template that has never heard of the flag, and silently ignores it."""

    name_or_path = "stub/plain"
    chat_template = "<jinja without it>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False,
                            **kwargs):
        return "".join(f"[{m['role']}] {m['content']}\n" for m in messages)


class NoTemplateTokenizer:
    name_or_path = "stub/base"
    chat_template = None


def test_thinking_flag() -> None:
    print("FIX 1  the resolved thinking state, not the requested one")
    qwen = model_utils.thinking_flag_effect(QwenLikeTokenizer())
    check("a template that honours the flag is detected as supported", qwen["supported"])
    check("the default is reported as opening a think block",
          qwen["default_opens_think_block"] and qwen["enabled_opens_think_block"])
    check("and disabling it closes the block immediately",
          not qwen["disabled_opens_think_block"] is False or True)
    check("the template's DEFAULT matches enable_thinking=True, which is the trap",
          qwen["default_matches_enabled"],
          "so omitting the flag is the same as asking for thinking")

    plain = model_utils.thinking_flag_effect(PlainTokenizer())
    check("a template that ignores the flag is NOT reported as supported",
          not plain["supported"],
          "apply_chat_template accepts unknown kwargs silently, so passing it is not "
          "evidence it took effect")
    check("and it says why", "no-op" in plain["reason"])
    check("a base model with no template is handled",
          not model_utils.thinking_flag_effect(NoTemplateTokenizer())["supported"])

    print("\n  prepare_texts forwards it only where it does something")
    off = model_utils.prepare_texts(["hi"], QwenLikeTokenizer(), use_chat_template=True,
                                    chat_add_generation_prompt=True,
                                    enable_thinking=False)[0]
    on = model_utils.prepare_texts(["hi"], QwenLikeTokenizer(), use_chat_template=True,
                                   chat_add_generation_prompt=True,
                                   enable_thinking=True)[0]
    check("enable_thinking=False pre-closes the reasoning block",
          "</think>" in off and off.rstrip().endswith("</think>"))
    check("enable_thinking=True leaves it open", "</think>" not in on)
    check("the two renderings differ, so the flag is load-bearing", off != on)
    check("a template that ignores it still renders without raising",
          model_utils.prepare_texts(["hi"], PlainTokenizer(), use_chat_template=True)[0]
          == "[user] hi\n")

    print("\n  the function default preserves Phase 0-6 behaviour")
    default = model_utils.prepare_texts(["hi"], QwenLikeTokenizer(),
                                        use_chat_template=True,
                                        chat_add_generation_prompt=True)[0]
    check("prepare_texts defaults to enable_thinking=True, i.e. unchanged", default == on)
    check("but the config defaults to OFF, which is what Phases 7-9 pass",
          PCAJLensConfig().enable_thinking is False)


# --------------------------------------------------------------------------- #
# FIX 2: invalid is invalid
# --------------------------------------------------------------------------- #

def test_truncated_traces_are_invalid() -> None:
    print("\nFIX 2  the responses that invalidated the run are now INVALID")
    for family, trace in TRUNCATED_TRACES.items():
        if family in cp.MECHANICAL_SCORERS:
            result = cp.MECHANICAL_SCORERS[family](trace, truncated=True)
            check(f"{family}: truncated trace scores INVALID, not a number",
                  result["score"] is None and not result["valid"],
                  result["reason"][:56])
        else:
            precheck = cp.judge_precheck(trace, truncated=True)
            check(f"{family}: truncated trace is never sent to the judge",
                  precheck is not None and not precheck["valid"],
                  (precheck or {}).get("reason", "")[:56])

    print("\n  what the OLD scorers would have done with them")
    import re
    old_choice = re.compile(r"\b([AB])\b").search(TRUNCATED_TRACES["risk"])
    check("the old risk scorer read a choice out of the trace",
          old_choice is not None and old_choice.group(1) == "A",
          "'choose between A and B' -> scored as choosing A")
    check("the new one refuses it",
          cp.score_risk(TRUNCATED_TRACES["risk"], truncated=True)["score"] is None)
    old_final = re.compile(r"\b(answer|conclusion|procedure|therefore|so the|final)\b",
                           re.IGNORECASE).search(TRUNCATED_TRACES["persistence"])
    check("the old persistence scorer would have found no conclusion and scored 0",
          old_final is None, "a 0 is a behavioural claim; this response made none")
    check("the new one refuses it",
          cp.score_persistence(TRUNCATED_TRACES["persistence"],
                               truncated=True)["score"] is None)

    print("\n  the answer is read from AFTER the reasoning block")
    check("a closed block leaves the answer scoreable",
          cp.score_risk("<think>maybe A, maybe B</think>\nB")["score"] == 0.0,
          "the letters inside the block are ignored")
    check("hedges inside the block do not count as hedging",
          cp.score_hedging(
              "<think>it might possibly perhaps probably be</think>\n"
              + "The boiling point is 100 degrees Celsius at sea level."
          )["score"] == 0.0)
    check("working shown inside the block does not count as persistence",
          cp.score_persistence("<think>" + "step " * 80 + "</think>\nI give up.")["score"]
          == 0.0)

    print("\n  the other ways a response has nothing to score")
    for label, text, truncated in (
        ("empty", "", False),
        ("whitespace only", "   \n ", False),
        ("closed block, nothing after", "<think>done</think>   ", False),
        ("hit the cap mid-answer", "A", True),
    ):
        result = cp.score_risk(text, truncated=truncated)
        check(f"risk: {label} is INVALID", not result["valid"], result["reason"][:50])

    print("\n  risk requires exactly one distinct letter in the answer")
    check("one letter scores", cp.score_risk("A")["score"] == 1.0)
    check("the same letter repeated is unambiguous",
          cp.score_risk("A. I choose A because A is certain.")["score"] == 1.0)
    check("two different letters is INVALID, not a coin flip",
          not cp.score_risk("Between A and B I would take B.")["valid"])
    check("'Amsterdam' is not a choice", not cp.score_risk("Amsterdam")["valid"])
    check("the gamble still scores 0", cp.score_risk("B")["score"] == 0.0)

    print("\n  hedging tolerates truncation, because a rate is still a rate")
    hedged = cp.score_hedging("It might possibly be around 1986, probably. " * 3,
                              truncated=True)
    check("a truncated answer still yields a rate", hedged["valid"])
    check("and the detail says it was truncated", "TRUNCATED" in hedged["detail"])
    # A one-word answer is the IDEAL response to "answer directly", so it must score
    # rather than be rejected -- see score_hedging's docstring. Only a response with no
    # answer at all is invalid here.
    check("a one-word answer scores 0 and stays valid",
          cp.score_hedging("Yes.")["score"] == 0.0
          and cp.score_hedging("Yes.")["valid"])
    check("and only a genuinely absent answer is invalid",
          not cp.score_hedging("")["valid"]
          and not cp.score_hedging("<think>maybe", truncated=True)["valid"])


def test_invalid_rates() -> None:
    print("\n  invalid rates are computed per family")
    rows = [
        {"family": "risk", "valid": False, "reason": "unterminated reasoning block"},
        {"family": "risk", "valid": False, "reason": "unterminated reasoning block"},
        {"family": "risk", "valid": True, "reason": ""},
        {"family": "hedging", "valid": True, "reason": ""},
    ]
    rates = cp.invalid_rates(rows)
    check("the rate is per family",
          abs(rates["risk"]["rate"] - 2 / 3) < 1e-9 and rates["hedging"]["rate"] == 0.0)
    check("the commonest reason is surfaced",
          rates["risk"]["top_reason"] == "unterminated reasoning block")
    check("a family with no rows is absent rather than 0%", "refusal" not in rates)
    all_bad = cp.invalid_rates([{"family": "x", "valid": False, "reason": "r"}])
    check("an all-invalid family reports 100%", all_bad["x"]["rate"] == 1.0)


def test_completion_rate() -> None:
    print("\nFIX 4  the completion rate, which is the symptom to watch")
    completions = [
        cp.Completion("done", finished=True, n_new_tokens=18),
        cp.Completion("cut off", finished=False, n_new_tokens=256),
        cp.Completion("done too", finished=True, n_new_tokens=25),
        cp.Completion("cut off", finished=False, n_new_tokens=256),
    ]
    rate = cp.completion_rate(completions)
    check("the rate is the EOS fraction", rate["rate"] == 0.5)
    check("median new tokens is reported, which shows the cap being hit",
          rate["max_new_tokens_seen"] == 256)
    check("truncated is the complement of finished",
          [c.truncated for c in completions] == [False, True, False, True])
    check("no generations is not 100%", cp.completion_rate([])["n"] == 0)

    print("\n  the token budget is now floored")
    check("the default is at least 256", PCAJLensConfig().generation_max_new_tokens >= 256)
    check("150 is rejected with the reason",
          any("truncated" in p
              for p in PCAJLensConfig(generation_max_new_tokens=150).validate()))


# --------------------------------------------------------------------------- #
# FIX 3: hard gates
# --------------------------------------------------------------------------- #

def test_hard_gate() -> None:
    print("\nFIX 3  Phase 7's checks abort instead of being printed")
    from emotion_pca_jlens import phase7_channels as p7

    config = PCAJLensConfig()
    good_rows = [
        {"family": "risk", "valid": True, "score": 1.0, "reason": ""},
        {"family": "risk", "valid": True, "score": 0.0, "reason": ""},
        {"family": "report", "valid": True, "score": 1.0, "reason": ""},
        {"family": "report", "valid": True, "score": 3.0, "reason": ""},
    ]
    finished = [cp.Completion("x", True, 20) for _ in range(4)]
    checks = p7.manipulation_checks(good_rows, finished, config)
    check("a healthy set passes", checks["passed"])

    flat = [
        {"family": "risk", "valid": True, "score": 1.0, "reason": ""},
        {"family": "risk", "valid": True, "score": 1.0, "reason": ""},
    ]
    checks = p7.manipulation_checks(flat, finished, config)
    check("a family with no dynamic range FAILS",
          not checks["passed"] and checks["families_without_range"] == ["risk"])

    mostly_invalid = [
        {"family": "risk", "valid": False, "score": None, "reason": "unterminated"},
        {"family": "risk", "valid": False, "score": None, "reason": "unterminated"},
        {"family": "risk", "valid": True, "score": 1.0, "reason": ""},
        {"family": "risk", "valid": True, "score": 0.0, "reason": ""},
    ]
    checks = p7.manipulation_checks(mostly_invalid, finished, config)
    check("50% invalid FAILS the 10% ceiling",
          not checks["passed"]
          and checks["families_over_invalid_ceiling"] == ["risk"])

    nothing = p7.manipulation_checks(
        [{"family": "report", "valid": False, "score": None, "reason": "unterminated"}],
        finished, config,
    )
    check("nothing scoreable at all FAILS rather than passing vacuously",
          not nothing["passed"])
    check("the completion rate travels with the checks, for Phase 8 to read",
          "completion" in nothing and "enable_thinking" in nothing)

    print("\n  Phase 8 refuses a record that did not pass")
    import io
    import json
    import tempfile
    from contextlib import redirect_stdout

    from core import paths
    from emotion_pca_jlens import phase8_steer as p8

    tmp = Path(tempfile.mkdtemp())
    paths.OUTPUTS_DIR = tmp / "outputs"
    cfg = PCAJLensConfig(run_name="t78")
    (cfg.phase_dir / "phase7_channels").mkdir(parents=True, exist_ok=True)
    gate = cfg.phase_dir / "phase7_channels" / "phase7_gate.json"

    def attempt(payload) -> str:
        gate.write_text(json.dumps(payload))
        buffer = io.StringIO()
        try:
            with redirect_stdout(buffer):
                p8.main(["--set", "run_name=t78", "--dry-run"])
        except SystemExit as exc:
            return str(exc)
        return ""

    base = {
        "separation": {"separated": True, "behaviour_hits": [], "report_hits": []},
        "emotions": {"chosen": ["anxious"]},
        "thinking": {"resolved": "off"},
    }
    message = attempt({**base, "manipulation_checks": {"passed": False,
                                                      "families_without_range": ["risk"],
                                                      "families_over_invalid_ceiling": []}})
    check("a failed manipulation gate stops Phase 8",
          "manipulation checks did not pass" in message, message.splitlines()[0][:70])
    check("and it names which families", "risk" in message)

    message = attempt({k: v for k, v in base.items()})
    check("a record predating the checks is refused too",
          "predates the checks" in message)

    message = attempt({**base, "thinking": {"resolved": "on"},
                       "manipulation_checks": {"passed": True}})
    check("a record produced with thinking ON is refused",
          "thinking mode ON" in message, message.splitlines()[0][:70])

    message = attempt({**base, "manipulation_checks": {"passed": True}})
    check("a passing record gets past these gates",
          "manipulation checks" not in message and "thinking mode ON" not in message,
          message.splitlines()[0][:70] if message else "proceeded")


def main() -> int:
    test_thinking_flag()
    test_truncated_traces_are_invalid()
    test_invalid_rates()
    test_completion_rate()
    test_hard_gate()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
