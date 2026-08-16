"""Gate prompts for the Phase 0 J-lens sanity check.

Provenance
----------
These prompts are **not** invented here. They are taken from the published
prompt sets of ``anthropics/jacobian-lens`` (Apache-2.0, Anthropic PBC), commit
``581d398613e5602a5af361e1c34d3a92ea82ba8e``:

* :data:`CONCEPT_PROMPTS` -- from ``data/experiments/probe-swap.json``, plus the
  two-hop example from the repo's own ``README.md``.
* :data:`EMOTION_ASSOCIATION_PROMPTS` -- from
  ``data/evaluations/lens-eval-association.json``.

They are vendored rather than fetched because ``pip install jlens`` packages only
``jlens/data/*``, not the repo-root ``data/`` directory, so the installed package
does not carry them.

Why these, rather than a hand-written prompt
--------------------------------------------
A gate is only a gate if we know what the right answer looks like *independently*
of the code under test. Each item below ships with the intermediate concept the
lens is documented to surface -- a concept the prompt never names. A prompt we
made up would have no such ground truth, so a plausible-looking readout could not
be distinguished from a correctly-loaded lens.

:data:`EMOTION_ASSOCIATION_PROMPTS` earns its place for a second reason: it is
this project's experiment in miniature. Each vignette implies an emotion without
naming it, and the lens is documented to read that emotion out. If the lens
cannot recover "grief" from a vignette about a dead man's coffee mug, there is no
point asking it what a principal component of emotion space encodes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GatePrompt:
    """A prompt with a documented expected lens readout.

    Attributes:
        name: Identifier from the source dataset.
        prompt: Raw text, verbatim from the source.
        expect: Words the lens should surface at the read position. For
            ``probe-swap`` items this is the *intermediate* concept, which the
            prompt deliberately never states.
        answer: The final answer the model should actually emit, where the source
            records one. Expected to dominate at late layers, while ``expect``
            peaks in the middle -- that contrast is itself informative.
        note: What this item is testing.
    """

    name: str
    prompt: str
    expect: tuple[str, ...]
    answer: str | None = None
    note: str = ""


#: Two-hop and single-hop factual prompts whose *intermediate* concept is never
#: written down. The brief's "think of a sport" check is the two sport items:
#: the lens should surface ``basketball`` / ``tennis`` at mid layers at the
#: position right before the model names the answer.
CONCEPT_PROMPTS: tuple[GatePrompt, ...] = (
    GatePrompt(
        name="basketball-players",
        prompt=(
            "Fact: The number of players per side in the sport invented in "
            "Springfield, Massachusetts is "
        ),
        expect=("basketball",),
        answer="5",
        note="the sport gate: 'basketball' is never written, only implied",
    ),
    GatePrompt(
        name="sport-equip-tennis",
        prompt=(
            "Fact: The piece of equipment used to strike the ball in the sport "
            "played at Wimbledon is a"
        ),
        expect=("tennis",),
        answer="racket",
        note="second sport gate, different implied sport",
    ),
    GatePrompt(
        name="multihop-currency",
        prompt=(
            "Fact: The capital of Japan is Tokyo.\n"
            "Fact: The currency used in the country shaped like a boot is"
        ),
        expect=("Italy", "Italian"),
        answer="euro",
        note="the jlens README's own worked example",
    ),
    GatePrompt(
        name="animal-nose-elephant",
        prompt=(
            "Fact: The long flexible body part that the largest land animal uses "
            "to pick things up is called a"
        ),
        expect=("elephant",),
        answer="trunk",
        note="non-sport control from the same set",
    ),
    GatePrompt(
        name="amazon-language",
        prompt="Fact: The language spoken in the country where the Amazon River ends is ",
        expect=("Brazil",),
        answer="Portuguese",
        note="geography intermediate",
    ),
)


#: Vignettes that imply an emotion without naming it. The domain-relevant gate:
#: same task shape as this project's stimuli, with published ground truth.
EMOTION_ASSOCIATION_PROMPTS: tuple[GatePrompt, ...] = (
    GatePrompt(
        name="grief",
        prompt=(
            "She kept his coffee mug on the counter, washed it every morning and "
            "put it back, though no one had used it in eleven months."
        ),
        expect=("grief",),
        note="high-arousal-negative-adjacent; the target word never appears",
    ),
    GatePrompt(
        name="anger",
        prompt='"It\'s fine," she said, and set the cup down so carefully it didn\'t make a sound.',
        expect=("anger", "angry"),
        note="suppressed anger: the surface text says the opposite",
    ),
    GatePrompt(
        name="relief",
        prompt=(
            "She read the line twice before she let herself breathe, and the paper "
            "shook a little as she set it down."
        ),
        expect=("relief",),
        note="positive valence, mixed arousal",
    ),
    GatePrompt(
        name="lonely",
        prompt=(
            "He cooked for two out of habit and scraped the second plate into the "
            "bin before sitting down."
        ),
        expect=("lonely", "loneliness"),
        note="low-arousal negative",
    ),
    GatePrompt(
        name="shame",
        prompt=(
            "She paid in cash and didn't meet the clerk's eyes, the small bag "
            "tucked inside her coat before she reached the door."
        ),
        expect=("shame",),
        note="self-focused negative",
    ),
    GatePrompt(
        name="frustrated",
        prompt="He explained it a third time, slower, and watched her eyes go to the clock again.",
        expect=("frustrated", "frustration"),
        note="mid-arousal negative",
    ),
    GatePrompt(
        name="chess",
        prompt=(
            "Neither of them had spoken in forty minutes when she finally moved the "
            "knight, and he tapped the clock without looking up."
        ),
        expect=("chess",),
        note="non-emotion control: a readout here should be a topic, not an affect word",
    ),
)


def all_prompts() -> tuple[GatePrompt, ...]:
    return CONCEPT_PROMPTS + EMOTION_ASSOCIATION_PROMPTS
