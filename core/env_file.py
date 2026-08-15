"""Load ``r2.env`` from the project root, so credentials need no shell setup.

Why this exists
---------------
R2 and Hugging Face credentials used to live in ``../scripts/r2.env`` -- a file
*outside* this repository, shared with other projects. A teammate who clones this
repo does not have it, and the sharing made ``R2_BUCKET`` a hazard (see below).

The single source of truth is now **``r2.env`` in the project root**: git-ignored,
templated by ``r2.env.example``, and read automatically by every entry point. No
``set -a; source ...`` step, and nothing to keep in sync between two files.

Search order -- the **first file that exists** is the one loaded::

    $R2_ENV_FILE          explicit path; set it to "none" to disable file loading
    <project root>/r2.env    <- the supported location
    <project root>/.env      legacy, still honoured
    /tmp/r2.env              where the shared scripts/ workflow lands creds on a pod
    <project root>/../scripts/r2.env    legacy shared file

Values from that file **override** variables already in the environment, which is
the opposite of the usual dotenv convention and is deliberate. On a RunPod pod,
``scripts/run_experiment.sh`` sources the *shared* ``r2.env`` into the environment
before our entry point runs, and that file sets ``R2_BUCKET`` to
``persona-activations`` for the other project that shares it. If the inherited
value won, this project would upload 8 GiB of activations into another project's
bucket. The project's own file is therefore authoritative, and every override is
reported by name (never by value) so it is not silent.

Set ``R2_ENV_FILE=none`` to opt out entirely and manage the environment yourself.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from core import paths

#: Candidate locations, in precedence order. ``$R2_ENV_FILE`` is consulted first.
DEFAULT_CANDIDATES: tuple[Path, ...] = (
    paths.PROJECT_ROOT / "r2.env",
    paths.PROJECT_ROOT / ".env",
    Path("/tmp/r2.env"),
    paths.PROJECT_ROOT.parent / "scripts" / "r2.env",
)

#: Sentinel value for ``$R2_ENV_FILE`` that disables file loading.
DISABLED = "none"

#: Prefixes whose overrides are worth printing. An ``AWS_*`` mirror differing from
#: your shell changes nothing about where this project reads or writes, so
#: announcing it on every command is noise; a differing ``R2_BUCKET`` is not.
NOTABLE_PREFIXES = ("R2_", "HF_")


@dataclass
class EnvFileLoad:
    """What :func:`load_env_file` did, for provenance and diagnostics."""

    path: Path | None = None
    """The file that was loaded, or ``None`` if none was found or loading was off."""

    keys: list[str] = field(default_factory=list)
    """Variable names set from the file."""

    overridden: list[str] = field(default_factory=list)
    """Names whose inherited value differed and was replaced."""

    reason: str = ""
    """Human-readable account of the outcome."""

    def describe(self) -> str:
        return self.reason


def parse_env_file(text: str) -> dict[str, str]:
    """Parse shell-style ``KEY=value`` assignments.

    Handles the ``export KEY=value`` form the shared ``r2.env`` uses, single and
    double quotes, blank lines, and ``#`` comments. Inline comments are stripped
    only from *unquoted* values, and only after whitespace -- quote the value if it
    legitimately contains ``" #"``.

    This is a deliberately small parser rather than a ``python-dotenv``
    dependency: it needs to read one flat file of credentials, and the file also
    has to stay sourceable by ``sh`` for the RunPod workflow.
    """
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("export ").lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key.replace("_", "").isalnum():
            continue  # not a plain variable assignment; ignore rather than guess
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        else:
            head, sep, _ = value.partition(" #")
            if sep:
                value = head.rstrip()
        out[key] = value
    return out


def _find_file() -> tuple[Path | None, str]:
    explicit = os.environ.get("R2_ENV_FILE", "").strip()
    if explicit.lower() == DISABLED:
        return None, "R2_ENV_FILE=none: file loading disabled, using the environment as-is"
    if explicit:
        path = Path(explicit).expanduser()
        if path.is_file():
            return path, f"loaded $R2_ENV_FILE ({path})"
        return None, f"$R2_ENV_FILE={explicit!r} does not exist; using the environment as-is"
    for candidate in DEFAULT_CANDIDATES:
        if candidate.is_file():
            return candidate, f"loaded {candidate}"
    return None, (
        f"no credentials file found (looked for {', '.join(str(c) for c in DEFAULT_CANDIDATES)}); "
        "using the environment as-is"
    )


_cache: EnvFileLoad | None = None


def load_env_file(verbose: bool = True, force: bool = False) -> EnvFileLoad:
    """Load the project's credentials file into ``os.environ``.

    Idempotent: the first call does the work and later calls return the same
    record, so every entry point can call it without worrying about ordering or
    repeated output.

    Args:
        verbose: print a line naming the file, and any variables whose inherited
            value was overridden. Values are never printed.
        force: re-read even if a previous call already loaded a file.
    """
    global _cache
    if _cache is not None and not force:
        return _cache

    path, reason = _find_file()
    load = EnvFileLoad(path=path, reason=reason)
    if path is not None:
        values = parse_env_file(path.read_text(encoding="utf-8"))
        for key, value in values.items():
            if key in os.environ and os.environ[key] != value:
                load.overridden.append(key)
            os.environ[key] = value
            load.keys.append(key)
        if load.overridden:
            load.reason += (
                f"; overrode inherited {', '.join(sorted(load.overridden))} "
                "(this project's file is authoritative)"
            )

    if verbose and path is not None:
        print(f"[env] {reason} ({len(load.keys)} variables)")
        notable = sorted(k for k in load.overridden if k.startswith(NOTABLE_PREFIXES))
        if notable:
            print(f"[env] {path.name} overrode inherited {', '.join(notable)} "
                  "(this project's file wins)")

    _cache = load
    return load


def loaded() -> EnvFileLoad | None:
    """The record from a previous :func:`load_env_file`, or ``None`` if never called."""
    return _cache
