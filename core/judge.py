"""External LLM judge for scoring generated text against a rubric.

Why the judge is external
-------------------------
Phases 7-9 score the steered model's own output. Using that same model as the judge
would confound the measurement: the steering vector perturbs the judge as well as
the subject, so a shift in scores could be the judge's disposition moving rather
than the behaviour it is meant to measure. The judge therefore runs on a different
model, through the Anthropic API -- a real dependency and a real cost line, not a
free step.

What this costs
---------------
One judge call per (generation, criterion). Phase 7 makes a few dozen while
validating the channels; Phase 8's grid is 4 conditions x 4 strengths x 2 channels x
N prompts, which is where the bill lands. Two things keep it down, both on by
default:

* **The rubric is cached.** It is identical across every call in a run and sits in
  the system prompt behind a cache breakpoint, so it is written once and read at
  roughly a tenth of input price thereafter. The scored text goes *after* the
  breakpoint -- prompt caching is a prefix match, so a rubric placed after the
  varying text would never be reused.
* **Bulk scoring should go through the Batches API** at half price. Phase 7 scores
  interactively because it is validating rubrics and wants the answer now; Phase 8
  should not. :meth:`Judge.estimate_cost` prints the arithmetic so that decision is
  made with numbers.

Three API facts this file exists to get right
---------------------------------------------
1. **No ``temperature``.** The standard LLM-judge recipe sets ``temperature=0`` for
   determinism. On current models that parameter is *removed* and sending it returns
   a 400. Determinism is not available by that route -- and never truly was -- so
   the judge asks for a bounded score and reports disagreement rather than
   pretending to be deterministic.
2. **Scores are an ``enum``, not a range.** Structured-output schemas do not support
   ``minimum``/``maximum``; those keywords are silently dropped. An enum of the
   permitted integers constrains the value exactly.
3. **A refusal is an HTTP 200.** The judge can decline, returning
   ``stop_reason == "refusal"`` with empty or partial content. Code that reads
   ``content[0].text`` unconditionally crashes on it, so the check comes first and
   the verdict carries the refusal rather than a fabricated score.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

#: Default judge model. Deliberately not the model under test -- see the module
#: docstring. Overridable per run, because which model judges is a methods choice
#: that belongs in the writeup.
DEFAULT_JUDGE_MODEL = "claude-opus-5"

#: Published list price per million tokens, for the cost estimate only. Recorded
#: here so an estimate printed at a gate can be checked against the invoice, and so
#: a stale number is visible rather than buried in a formula.
PRICE_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


def judge_available() -> tuple[bool, str]:
    """Whether a judge can run, plus a reason if not.

    Checks the dependency before the credential: an uninstalled SDK and a missing
    key are different problems with different fixes, and reporting "no API key" for
    a missing package sends people to the wrong place.

    Note an unset ``ANTHROPIC_API_KEY`` does not by itself mean there are no
    credentials -- the SDK also resolves ``ANTHROPIC_AUTH_TOKEN`` and an ``ant auth
    login`` profile on disk. So this reports the key as *absent* rather than as
    fatal, and construction is what actually decides.
    """
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False, (
            "the anthropic SDK is not installed in this interpreter.\n"
            "  Fix: python -m pip install anthropic\n"
            "  (it is in requirements.txt as the judge dependency)"
        )
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        return False, (
            "no ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN in the environment.\n"
            "  Put ANTHROPIC_API_KEY in r2.env alongside the R2 keys and HF_TOKEN, or\n"
            "  run `ant auth login` -- the SDK also reads a profile from disk, in which\n"
            "  case this check is over-strict and the judge will work anyway."
        )
    return True, "ok"


@dataclass(frozen=True)
class Verdict:
    """One judge decision, or the reason there isn't one."""

    score: float | None
    reason: str
    model: str
    refused: bool = False
    error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0

    @property
    def usable(self) -> bool:
        return self.score is not None and not self.refused and self.error is None

    def as_dict(self) -> dict:
        return {
            "score": self.score, "reason": self.reason, "model": self.model,
            "refused": self.refused, "error": self.error,
            "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
            "cached_tokens": self.cached_tokens,
        }


def score_schema(allowed: list[int]) -> dict:
    """JSON schema for a bounded integer score plus a one-sentence justification.

    ``enum`` rather than ``minimum``/``maximum``: structured-output schemas do not
    support numeric constraints and drop them silently, which would leave the score
    unbounded while looking constrained. An enum is enforced.
    """
    return {
        "type": "object",
        "properties": {
            "score": {"type": "integer", "enum": list(allowed)},
            "reason": {"type": "string"},
        },
        "required": ["score", "reason"],
        "additionalProperties": False,
    }


@dataclass
class Judge:
    """Scores text against a rubric with an external model.

    The rubric is passed once at construction and cached: it is the stable prefix of
    every call, and putting it in the system prompt behind a cache breakpoint is
    what makes a few hundred scoring calls affordable. Callers vary only the text.
    """

    rubric: str
    model: str = DEFAULT_JUDGE_MODEL
    allowed_scores: tuple[int, ...] = (0, 1, 2, 3, 4)
    max_tokens: int = 512
    client: object | None = None
    calls: int = field(default=0, init=False)
    usage: dict = field(default_factory=lambda: {"input": 0, "output": 0, "cached": 0},
                        init=False)

    def __post_init__(self) -> None:
        if self.client is None:
            import anthropic

            self.client = anthropic.Anthropic()

    def score(self, text: str, label: str = "") -> Verdict:
        """Score one passage. Never raises for an API-level failure.

        A judge that raises mid-grid would lose the generations already scored, and
        Phase 8's grid is hours of GPU time. Failures come back as an unusable
        verdict carrying the reason, so the caller decides whether a partial grid is
        worth keeping.
        """
        import anthropic

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                # The rubric is the cache prefix; the scored text goes after it.
                # Caching is a prefix match, so the reverse order would write a new
                # entry per call and never read one.
                system=[{
                    "type": "text",
                    "text": self.rubric,
                    "cache_control": {"type": "ephemeral"},
                }],
                output_config={
                    "format": {
                        "type": "json_schema",
                        "schema": score_schema(list(self.allowed_scores)),
                    }
                },
                messages=[{
                    "role": "user",
                    "content": f"Passage to score{f' ({label})' if label else ''}:\n\n"
                               f"<passage>\n{text}\n</passage>",
                }],
            )
        except anthropic.APIStatusError as exc:
            return Verdict(None, "", self.model, error=f"{exc.status_code}: {exc.message}")
        except anthropic.APIConnectionError as exc:
            return Verdict(None, "", self.model, error=f"connection error: {exc}")

        self.calls += 1
        return self._parse(response)

    def score_many(
        self,
        items: list[tuple[str, str]],
        use_batches: bool = True,
        poll_seconds: float = 20.0,
        timeout_seconds: float = 7200.0,
        progress=None,
    ) -> dict[str, Verdict]:
        """Score many passages, keyed by the caller's id. Batched by default.

        ``items`` is ``[(id, text), ...]``; the return is ``{id: Verdict}``.

        Phase 8's grid is hundreds of scoring calls and is not latency-sensitive --
        it runs after hours of generation and nobody is waiting on any individual
        score -- so it goes through the Batches API at half price. Phase 7 scores
        interactively instead, because there the point is to look at a rubric's
        output now.

        **Results come back in any order**, so they are keyed by ``custom_id`` and
        never by position; a positional read would silently mis-attribute every
        score. Any id missing from the results gets an unusable verdict rather than
        being dropped, so a partial batch is visible as gaps rather than as a
        shorter list.

        Falls back to the interactive loop when ``use_batches`` is false or the
        batch cannot be created -- a judge that refused to run at all would waste
        the GPU time already spent.
        """
        import time as _time

        if not items:
            return {}
        if not use_batches:
            return {key: self.score(text, label=key) for key, text in items}

        import anthropic
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request

        schema = score_schema(list(self.allowed_scores))
        requests = [
            Request(
                custom_id=key,
                params=MessageCreateParamsNonStreaming(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=[{
                        "type": "text", "text": self.rubric,
                        "cache_control": {"type": "ephemeral"},
                    }],
                    output_config={"format": {"type": "json_schema", "schema": schema}},
                    messages=[{
                        "role": "user",
                        "content": f"Passage to score ({key}):\n\n"
                                   f"<passage>\n{text}\n</passage>",
                    }],
                ),
            )
            for key, text in items
        ]
        try:
            batch = self.client.messages.batches.create(requests=requests)
        except Exception as exc:
            if progress:
                progress(f"batch creation failed ({exc}); falling back to interactive")
            return {key: self.score(text, label=key) for key, text in items}

        deadline = _time.monotonic() + timeout_seconds
        while True:
            state = self.client.messages.batches.retrieve(batch.id)
            if state.processing_status == "ended":
                break
            if _time.monotonic() > deadline:
                return {
                    key: Verdict(None, "", self.model,
                                 error=f"batch {batch.id} still running after "
                                       f"{timeout_seconds:.0f}s")
                    for key, _ in items
                }
            if progress:
                counts = getattr(state, "request_counts", None)
                progress(f"batch {batch.id}: {state.processing_status}"
                         + (f", {counts.processing} processing" if counts else ""))
            _time.sleep(poll_seconds)

        verdicts: dict[str, Verdict] = {}
        for result in self.client.messages.batches.results(batch.id):
            key = result.custom_id
            kind = result.result.type
            if kind != "succeeded":
                verdicts[key] = Verdict(None, "", self.model,
                                        error=f"batch result {kind}")
                continue
            verdicts[key] = self._parse(result.result.message)
        self.calls += len(items)
        for key, _ in items:
            verdicts.setdefault(
                key, Verdict(None, "", self.model, error="absent from batch results")
            )
        return verdicts

    def _parse(self, response) -> Verdict:
        """Turn one API message into a verdict. Shared by both scoring paths."""
        usage = getattr(response, "usage", None)
        counts = {
            "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
            "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
            "cached_tokens": int(getattr(usage, "cache_read_input_tokens", 0) or 0),
        }
        self.usage["input"] += counts["input_tokens"]
        self.usage["output"] += counts["output_tokens"]
        self.usage["cached"] += counts["cached_tokens"]
        if getattr(response, "stop_reason", None) == "refusal":
            category = getattr(getattr(response, "stop_details", None), "category", None)
            return Verdict(None, "", self.model, refused=True,
                           error=f"judge declined (category {category})", **counts)
        blocks = [
            b.text for b in response.content if getattr(b, "type", "") == "text"
        ]
        if not blocks:
            return Verdict(None, "", self.model, error="no text block", **counts)
        try:
            parsed = json.loads(blocks[0])
        except json.JSONDecodeError as exc:  # pragma: no cover - schema-constrained
            return Verdict(None, "", self.model,
                           error=f"unparseable despite json_schema: {exc}", **counts)
        return Verdict(score=float(parsed["score"]),
                       reason=str(parsed.get("reason", "")), model=self.model, **counts)

    def estimate_cost(self, n_calls: int, text_tokens: int = 400) -> dict:
        """List-price cost for ``n_calls`` scoring calls, with and without caching.

        Printed at a gate so the decision to run Phase 8's grid interactively rather
        than through the Batches API is made against a number. Batching halves the
        result; the estimate says so rather than leaving it implied.
        """
        rubric_tokens = max(len(self.rubric) // 4, 1)
        rates = PRICE_PER_MTOK.get(self.model)
        if rates is None:
            return {"known_price": False, "model": self.model, "n_calls": n_calls}
        per_in, per_out = rates
        uncached = n_calls * (rubric_tokens + text_tokens)
        # Cache write is ~1.25x on the first call; reads are ~0.1x thereafter.
        cached = (
            1.25 * rubric_tokens + n_calls * text_tokens
            + 0.1 * rubric_tokens * max(n_calls - 1, 0)
        )
        out = n_calls * self.max_tokens
        return {
            "known_price": True,
            "model": self.model,
            "n_calls": n_calls,
            "rubric_tokens": rubric_tokens,
            "usd_no_cache": (uncached * per_in + out * per_out) / 1e6,
            "usd_cached": (cached * per_in + out * per_out) / 1e6,
            "usd_cached_batched": (cached * per_in + out * per_out) / 1e6 * 0.5,
            "note": "list prices; output assumes max_tokens is fully used, so the "
                    "output term is an upper bound",
        }
