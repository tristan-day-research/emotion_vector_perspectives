"""Annotate the Phase 4 J-lens token readouts with script and English glosses.

**The originals are never touched.** For every ``phase4_readouts.csv`` this writes a
sibling ``phase4_readouts_translated.csv`` with three *added* columns -- ``script``,
``token_english``, ``gloss_confidence`` -- and the ``token`` column carried through
byte-for-byte. The multilingual character of these readouts *is* the finding; replacing
the tokens with glosses would erase it.

Two steps, deliberately separated by how much you can trust them:

STEP 1 -- script classification (no API, deterministic)
    Every token is bucketed by Unicode range into cjk / latin / cyrillic / thai /
    arabic / punctuation / whitespace / other, and ``analysis/readout_script_summary.csv``
    records the per-script fraction of the top-12 tokens for each
    ``(run, group, direction)``. This is the part that supports a claim in the writeup,
    so it must not depend on an LLM -- it is pure ``unicodedata`` and range tests, and it
    runs whether or not any credentials exist.

STEP 2 -- glossing (Claude API, best-effort reading aid)
    Only tokens that need it are sent: everything non-Latin, plus Latin tokens that
    are clearly not English. Unique tokens are translated once and mapped back over the
    (thousands of) rows, and every gloss is cached in
    ``analysis/translation_cache.json`` keyed by the *exact* token string -- leading
    space included -- so a re-run costs nothing.

    Glosses are a reading aid and carry no inferential weight, the same standing as the
    hand-written table in :mod:`analysis.zh_en_glossary`.

Fragments, not words
--------------------
These are BPE tokens: ``' нарко'`` is the prefix *narco-*, ``'刽'`` is one character of
刽子手 (executioner). The model is instructed to gloss a fragment *as* a fragment, to
mark its uncertainty in ``gloss_confidence`` (high/medium/low), and never to invent a
plausible whole word from a partial one -- a confident-looking wrong gloss is worse here
than an honest ``low``.

No API key
----------
``ANTHROPIC_API_KEY`` is read via :mod:`core.env_file` (it lives in ``r2.env`` next to
the R2 credentials). If it is absent, STEP 1 still runs and the script summary is still
written; only the glossing is skipped. The script analysis is never blocked by a
missing key.

Usage::

    python analysis/translate_readouts.py            # both steps
    python analysis/translate_readouts.py --no-api   # STEP 1 only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import unicodedata
from bisect import bisect_left
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import env_file, paths  # noqa: E402

#: Cheapest model that can hold the fragment-vs-word distinction over short glosses.
#: Verified against the current model list; Haiku 4.5 supports structured outputs.
MODEL = "claude-haiku-4-5"

#: One request per this many unique tokens. Small enough that a malformed reply costs
#: little, large enough that a few hundred tokens is a handful of calls.
BATCH_SIZE = 100

CACHE_PATH = paths.PROJECT_ROOT / "analysis" / "translation_cache.json"
SUMMARY_PATH = paths.PROJECT_ROOT / "analysis" / "readout_script_summary.csv"
READOUT_GLOB = "outputs/*/results/phases/phase4_readouts.csv"

#: Order used for the summary columns, so every run's table lines up.
SCRIPTS = ("cjk", "latin", "cyrillic", "thai", "arabic", "punctuation", "whitespace", "other")


# --------------------------------------------------------------------------------------
# STEP 1 -- script classification (no API)
# --------------------------------------------------------------------------------------

#: Inclusive codepoint ranges per script. ``cjk`` covers Han ideographs *and* kana and
#: Hangul -- the C, J and K of CJK -- because the question these readouts raise is "is
#: the model naming this concept in an East Asian script?", not "which of the three?".
_RANGES: tuple[tuple[int, int, str], ...] = (
    (0x3040, 0x309F, "cjk"),   # Hiragana
    (0x30A0, 0x30FF, "cjk"),   # Katakana
    (0x3400, 0x4DBF, "cjk"),   # CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF, "cjk"),   # CJK Unified Ideographs
    (0xAC00, 0xD7AF, "cjk"),   # Hangul Syllables
    (0xF900, 0xFAFF, "cjk"),   # CJK Compatibility Ideographs
    (0xFF00, 0xFFEF, "cjk"),   # Halfwidth and Fullwidth Forms
    (0x0400, 0x04FF, "cyrillic"),
    (0x0500, 0x052F, "cyrillic"),
    (0x0E00, 0x0E7F, "thai"),
    (0x0600, 0x06FF, "arabic"),
    (0x0750, 0x077F, "arabic"),
    (0xFB50, 0xFDFF, "arabic"),
    (0x0041, 0x005A, "latin"),  # A-Z
    (0x0061, 0x007A, "latin"),  # a-z
    (0x00C0, 0x024F, "latin"),  # Latin-1 Supplement letters + Extended-A/B
)


def _char_script(ch: str) -> str:
    """Bucket a single character. Digits count as ``latin``; marks follow their base."""
    if ch.isspace():
        return "whitespace"
    category = unicodedata.category(ch)
    # Punctuation is tested before the range table on purpose: CJK punctuation lives
    # inside the fullwidth block, so '､' and '！' would otherwise count as CJK *content*
    # and inflate the CJK fraction the summary reports. Han/kana/Hangul are all Lo.
    if category.startswith(("P", "S")):
        return "punctuation"
    code = ord(ch)
    for lo, hi, name in _RANGES:
        if lo <= code <= hi:
            return name
    if ch.isdigit():
        return "latin"
    if category.startswith("M"):  # combining mark -- attribute it to whatever it sits on
        return "other"
    return "other"


def classify_script(token: str) -> str:
    """Classify a whole token by the script its characters are drawn from.

    Whitespace-only tokens are ``whitespace``; tokens made entirely of punctuation and
    whitespace are ``punctuation``. Otherwise the most common *substantive* script wins,
    so ``' 骂人'`` is ``cjk`` and ``'.\\n'`` is ``punctuation``. Ties resolve by the
    :data:`SCRIPTS` order, which is stable across runs.
    """
    if not token:
        return "whitespace"
    counts: dict[str, int] = {}
    for ch in token:
        name = _char_script(ch)
        counts[name] = counts.get(name, 0) + 1
    substantive = {k: v for k, v in counts.items() if k not in ("whitespace", "punctuation")}
    if substantive:
        pool = substantive
    elif counts.get("punctuation"):
        return "punctuation"
    else:
        return "whitespace"
    return max(pool, key=lambda k: (pool[k], -SCRIPTS.index(k)))


def script_summary(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per ``(run, group, direction)``, the fraction of top-12 tokens in each script."""
    rows = []
    for run, df in frames.items():
        for (group, direction), block in df.groupby(["group", "direction"], sort=True):
            fractions = block["script"].value_counts(normalize=True)
            row = {"run": run, "group": group, "direction": direction, "n_tokens": len(block)}
            row.update({s: round(float(fractions.get(s, 0.0)), 4) for s in SCRIPTS})
            rows.append(row)
    return pd.DataFrame(rows, columns=["run", "group", "direction", "n_tokens", *SCRIPTS])


# --------------------------------------------------------------------------------------
# STEP 2 -- which tokens need a gloss
# --------------------------------------------------------------------------------------

_WORDLIST_PATHS = (Path("/usr/share/dict/words"), Path("/usr/dict/words"))


def _load_wordlist() -> list[str] | None:
    """Sorted lowercase English words from the system dictionary, or ``None``."""
    for path in _WORDLIST_PATHS:
        if path.is_file():
            words = sorted({w.strip().lower() for w in path.read_text(
                encoding="utf-8", errors="ignore").splitlines() if w.strip()})
            return words
    return None


def _looks_english(token: str, words: list[str] | None) -> bool:
    """Is this ASCII-Latin token plausibly English (a word, or the start of one)?

    BPE splits words, so ``' unhapp'`` must count as English even though it is not a
    dictionary entry -- the prefix test covers that. Without a system dictionary we
    return ``True`` for every ASCII token: over-skipping wastes no money and, more to the
    point, never puts a fabricated gloss in the output.
    """
    stripped = token.strip().lower()
    if not stripped:
        return True
    if any(ord(c) > 127 for c in stripped):
        return False  # diacritics -- not English
    if not stripped.isalpha():
        return True  # digits/punctuation fragments: nothing to translate
    if words is None:
        return True
    if len(stripped) <= 2:
        return True  # too short to judge; glossing it would be invention
    if _in_wordlist(stripped, words, exact=False):
        return True
    # Inflections: web2 lists lemmas, so "feelings" is neither an entry nor the prefix
    # of one. Strip the common English endings and retry against the stem -- but demand
    # an *exact* entry here. Allowing a prefix match on a stripped stem is what lets
    # " milfs" pass as English via "milfoil", and that cluster is part of the finding.
    for suffix, restore in (
        ("s", ""), ("es", ""), ("ed", ""), ("ed", "e"), ("ing", ""), ("ing", "e"),
        ("ly", ""), ("'s", ""), ("ies", "y"),
    ):
        if stripped.endswith(suffix) and len(stripped) - len(suffix) >= 3:
            if _in_wordlist(stripped[: -len(suffix)] + restore, words, exact=True):
                return True
    return False


def _in_wordlist(candidate: str, words: list[str], *, exact: bool) -> bool:
    """Dictionary lookup; ``exact=False`` also accepts a prefix of an entry.

    The prefix form is needed because BPE cuts words mid-stem (``' unhapp'``), but it is
    loose enough that it must not be applied twice -- see the caller.
    """
    index = bisect_left(words, candidate)
    if index >= len(words):
        return False
    return words[index] == candidate if exact else words[index].startswith(candidate)


def needs_gloss(token: str, script: str, words: list[str] | None) -> bool:
    if script in ("whitespace", "punctuation"):
        return False
    if script == "latin":
        return not _looks_english(token, words)
    return True


# --------------------------------------------------------------------------------------
# STEP 2 -- the API call
# --------------------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You gloss individual BPE vocabulary tokens from a multilingual language model into \
English, for a research table.

These are TOKENS, not words. Most are word FRAGMENTS -- prefixes, stems, or suffixes \
that only form a word when combined with neighbouring tokens.

Rules, in order of importance:

1. Never invent a plausible whole word from a partial one. If a token is a fragment, \
gloss it as a fragment and say so: " нарко" -> "narco- (prefix, as in narcotics)"; \
"刽" -> "single character of 刽子手 'executioner'; not a standalone word".
2. Mark your uncertainty honestly in gloss_confidence:
   - high   : you are confident of the meaning and of how the fragment is used.
   - medium : the meaning is clear but the fragment is ambiguous across several words, \
or the register/sense depends on context.
   - low    : you are guessing, the fragment is too short to pin down, or it could \
belong to several unrelated words.
   A wrong gloss marked "high" is far worse than a correct one marked "low".
3. Keep each gloss under about 12 words. Plain English, no quotes around the whole \
gloss, no trailing period needed.
4. If a token is already English, return it unchanged with confidence "high".
5. If you genuinely cannot tell, return "unclear" with confidence "low". Do not guess \
to fill the slot.

Note the tokens have had any leading space stripped; word-initial position is therefore \
not something you can infer from what you see."""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "glosses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "english": {"type": "string"},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                },
                "required": ["index", "english", "confidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["glosses"],
    "additionalProperties": False,
}


def gloss_batch(client, tokens: list[str]) -> dict[str, dict[str, str]]:
    """Gloss one batch of unique tokens. Keys of the result are the *original* tokens."""
    listing = "\n".join(f"{i}. {tok.strip()}" for i, tok in enumerate(tokens))
    response = client.messages.create(
        model=MODEL,
        max_tokens=8000,
        system=SYSTEM_PROMPT,
        output_config={"format": {"type": "json_schema", "schema": RESPONSE_SCHEMA}},
        messages=[{
            "role": "user",
            "content": (
                "Gloss each of these tokens. Return one entry per index, all "
                f"{len(tokens)} of them, in the same order.\n\n{listing}"
            ),
        }],
    )
    text = next(b.text for b in response.content if b.type == "text")
    out: dict[str, dict[str, str]] = {}
    for entry in json.loads(text)["glosses"]:
        index = entry["index"]
        if 0 <= index < len(tokens):
            out[tokens[index]] = {
                "english": entry["english"].strip(),
                "confidence": entry["confidence"],
            }
    return out


def translate(tokens: list[str], cache: dict[str, dict[str, str]]) -> tuple[int, str | None]:
    """Fill ``cache`` for every uncached token. Returns (n_new, error message or None)."""
    pending = [t for t in tokens if t not in cache]
    if not pending:
        return 0, None
    try:
        import anthropic
    except ImportError:
        return 0, "the `anthropic` package is not installed (pip install anthropic)"

    client = anthropic.Anthropic()
    added = 0
    for start in range(0, len(pending), BATCH_SIZE):
        batch = pending[start:start + BATCH_SIZE]
        print(f"[gloss] batch {start // BATCH_SIZE + 1}: {len(batch)} tokens", flush=True)
        try:
            result = gloss_batch(client, batch)
        except Exception as exc:  # one bad batch must not lose the batches before it
            _save_cache(cache)
            return added, f"{type(exc).__name__}: {exc}"
        missing = [t for t in batch if t not in result]
        if missing:
            print(f"[gloss] {len(missing)} tokens came back unglossed; leaving them uncached")
        cache.update(result)
        added += len(result)
        _save_cache(cache)  # checkpoint, so an interrupt never re-bills what is done
    return added, None


def seed_from_glossary(tokens: list[str], cache: dict[str, dict[str, str]]) -> int:
    """Fill the cache from the hand-written :mod:`analysis.zh_en_glossary` table.

    That module's keys are bare (``'仇恨'``); cache keys are exact tokens and may carry a
    leading space, so the match is on the stripped form. Existing cache entries win --
    seeding never overwrites a gloss that is already there.

    Its own docstring calls those glosses "indicative, not canonical", so they land as
    ``medium`` rather than ``high``. This runs with no credentials and no network.
    """
    try:
        from analysis.zh_en_glossary import TOKEN_GLOSS
    except ImportError:
        return 0
    try:
        from analysis._offline_glosses import OFFLINE_GLOSS
    except ImportError:
        OFFLINE_GLOSS = {}

    added = 0
    for token in tokens:
        if token in cache:
            continue
        bare = token.strip()
        if bare in TOKEN_GLOSS:
            cache[token] = {
                "english": TOKEN_GLOSS[bare],
                "confidence": "medium",
                "source": "zh_en_glossary",
            }
            added += 1
        elif bare in OFFLINE_GLOSS:
            english, confidence = OFFLINE_GLOSS[bare]
            cache[token] = {
                "english": english,
                "confidence": confidence,
                "source": "assistant-offline",
            }
            added += 1
    return added


def _load_cache() -> dict[str, dict[str, str]]:
    if CACHE_PATH.is_file():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def _save_cache(cache: dict[str, dict[str, str]]) -> None:
    CACHE_PATH.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------

def _cell(row: pd.Series) -> str:
    """``token`` plus its gloss, escaped for a markdown table cell."""
    token = row["token"].replace("\n", "\\n").replace("|", "\\|").replace("`", "'")
    gloss = str(row["token_english"]).replace("|", "\\|")
    if not gloss or gloss == row["token"].strip():
        return f"`{token}`"
    mark = {"low": "?", "medium": "~"}.get(row["gloss_confidence"], "")
    return f"`{token}` {gloss}{mark}"


def print_pc_table(frames: dict[str, pd.DataFrame]) -> None:
    """Markdown table of the top 3 PCs per run, both ends, tokens beside glosses."""
    print("\n### Phase 4 PC-end readouts, glossed\n")
    print("Glosses are a reading aid, not pipeline output. `~` = medium confidence, "
          "`?` = low.\n")
    for run, df in frames.items():
        pcs = df[df["group"] == "gate_b_pc"]
        if pcs.empty:
            continue
        print(f"**{run}**\n")
        print("| PC | end | top tokens |")
        print("| --- | --- | --- |")
        for pc in ("1", "2", "3"):
            for sign in ("+", "-"):
                direction = f"{sign}PC{pc}"
                block = pcs[pcs["direction"] == direction]
                if block.empty:
                    continue
                block = block.assign(_rank=block["rank"].astype(int)).sort_values("_rank")
                cells = " · ".join(_cell(r) for _, r in block.head(8).iterrows())
                print(f"| PC{pc} | {sign} | {cells} |")
        print()


# --------------------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-api", action="store_true",
                        help="run STEP 1 only; do not call the Claude API")
    args = parser.parse_args()

    readouts = sorted(paths.PROJECT_ROOT.glob(READOUT_GLOB))
    if not readouts:
        print(f"no readouts found under {READOUT_GLOB}", file=sys.stderr)
        return 1

    # `na_filter=False` keeps every field exactly as written -- an empty cell stays an
    # empty string, and a token that is literally " " or "\n" survives the round trip.
    frames: dict[str, pd.DataFrame] = {}
    for path in readouts:
        run = path.parents[2].name
        df = pd.read_csv(path, dtype=str, na_filter=False)
        df["script"] = df["token"].map(classify_script)
        frames[run] = df
        print(f"[read] {run}: {len(df)} rows, {df['token'].nunique()} unique tokens")

    # ---- STEP 1: the part that supports a claim -----------------------------------
    summary = script_summary(frames)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)
    print(f"[step1] wrote {SUMMARY_PATH.relative_to(paths.PROJECT_ROOT)} "
          f"({len(summary)} run/group/direction rows)")

    words = _load_wordlist()
    if words is None:
        print("[step2] no system word list found; ASCII tokens will all be treated as "
              "English and left unglossed")

    wanted = sorted({
        tok
        for df in frames.values()
        for tok, script in zip(df["token"], df["script"])
        if needs_gloss(tok, script, words)
    })
    print(f"[step2] {len(wanted)} unique tokens need a gloss")

    cache = _load_cache()
    seeded = seed_from_glossary(wanted, cache)
    if seeded:
        _save_cache(cache)
        print(f"[step2] seeded {seeded} glosses offline from analysis/zh_en_glossary.py "
              "and analysis/_offline_glosses.py (no API, no network)")
    skipped_reason: str | None = None

    if args.no_api:
        skipped_reason = "--no-api was passed"
    else:
        load = env_file.load_env_file()
        if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
            where = load.path.name if load.path else "no credentials file"
            skipped_reason = f"ANTHROPIC_API_KEY is not set ({where})"

    if skipped_reason is None:
        cached = sum(1 for t in wanted if t in cache)
        print(f"[step2] {cached} already cached, {len(wanted) - cached} to fetch")
        added, error = translate(wanted, cache)
        print(f"[step2] glossed {added} new tokens; cache now holds {len(cache)}")
        if error:
            print(f"[step2] stopped early: {error}", file=sys.stderr)
            skipped_reason = error
    else:
        print(f"[step2] translation skipped: {skipped_reason}")
        if cache:
            print(f"[step2] using {len(cache)} glosses already in the cache")

    # ---- write the annotated copies (never the originals) ---------------------------
    for path in readouts:
        run = path.parents[2].name
        df = frames[run].copy()
        df["token_english"] = [
            cache.get(t, {}).get("english", "") for t in df["token"]
        ]
        df["gloss_confidence"] = [
            cache[t]["confidence"] if t in cache
            else ("" if needs_gloss(t, s, words) else "n/a")
            for t, s in zip(df["token"], df["script"])
        ]
        out = path.with_name("phase4_readouts_translated.csv")
        df.to_csv(out, index=False)
        frames[run] = df
        print(f"[write] {out.relative_to(paths.PROJECT_ROOT)}")

    print_pc_table(frames)

    missing = sum(1 for df in frames.values() for c in df["gloss_confidence"] if c == "")
    if skipped_reason:
        state = "every token is glossed from the offline tables" if not missing else (
            f"{missing} rows are still unglossed")
        print(f"\nNo API call was made ({skipped_reason}); {state}. The script "
              f"classification in {SUMMARY_PATH.relative_to(paths.PROJECT_ROOT)} never "
              "depends on it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
