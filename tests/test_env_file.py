"""Verify the r2.env loader, including the case that motivated its precedence.

The behaviour that matters and is easy to get wrong:

1. ``export KEY=value`` (what r2.env uses), quotes, comments, blank lines
2. the file OVERRIDES an inherited value -- on a pod the shared scripts/r2.env is
   sourced into the environment first and sets R2_BUCKET to another project's
   bucket, so losing this test means 8 GiB going to the wrong place
3. overrides are recorded by name, so `r2 check` and the run manifest can say so
4. ``R2_ENV_FILE`` selects a file explicitly, and ``=none`` disables loading
5. a missing file is not an error -- the environment is used as-is

No network or credentials needed.

    PYTHONPATH=. python tests/test_env_file.py
"""
import os
import tempfile
from pathlib import Path

import core.env_file as ef

# --- 1. parsing --------------------------------------------------------------- #
parsed = ef.parse_env_file(
    """
# a comment
export R2_BUCKET="quoted-bucket"
export R2_ACCESS_KEY_ID=bare_value
R2_SECRET_ACCESS_KEY='single quoted'
export R2_ENDPOINT=https://x.r2.cloudflarestorage.com   # trailing comment
export EMPTY=
not_an_assignment
export HF_TOKEN="hash#inside#quotes"
"""
)
assert parsed["R2_BUCKET"] == "quoted-bucket", parsed
assert parsed["R2_ACCESS_KEY_ID"] == "bare_value", parsed
assert parsed["R2_SECRET_ACCESS_KEY"] == "single quoted", parsed
assert parsed["R2_ENDPOINT"] == "https://x.r2.cloudflarestorage.com", parsed
assert parsed["EMPTY"] == "", parsed
assert parsed["HF_TOKEN"] == "hash#inside#quotes", parsed
assert "not_an_assignment" not in parsed
print(f"1. parsed {len(parsed)} assignments, quotes/comments/export handled")

# --- 2 & 3. the project's file beats an inherited value ----------------------- #
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "r2.env"
    path.write_text(
        'export R2_BUCKET="emotion-vector-perspectives"\n'
        'export R2_ACCESS_KEY_ID="from_file"\n'
    )

    # Simulate the pod: shared creds already exported, pointing at the other bucket.
    os.environ["R2_BUCKET"] = "persona-activations"
    os.environ.pop("R2_ACCESS_KEY_ID", None)
    os.environ["R2_ENV_FILE"] = str(path)

    load = ef.load_env_file(verbose=False, force=True)
    assert os.environ["R2_BUCKET"] == "emotion-vector-perspectives", os.environ["R2_BUCKET"]
    assert os.environ["R2_ACCESS_KEY_ID"] == "from_file"
    assert load.overridden == ["R2_BUCKET"], load.overridden
    assert load.path == path
    print(f"2. inherited R2_BUCKET=persona-activations overridden -> "
          f"{os.environ['R2_BUCKET']}")
    print(f"3. recorded overrides: {load.overridden}")

    # An unchanged value is not reported as an override.
    load = ef.load_env_file(verbose=False, force=True)
    assert load.overridden == [], load.overridden
    print("   re-load with identical values reports no override")

    # --- 4. disabling --------------------------------------------------------- #
    os.environ["R2_BUCKET"] = "untouched"
    os.environ["R2_ENV_FILE"] = "none"
    load = ef.load_env_file(verbose=False, force=True)
    assert load.path is None and os.environ["R2_BUCKET"] == "untouched"
    assert "disabled" in load.reason
    print(f"4. R2_ENV_FILE=none: {load.reason}")

    # --- 5. missing file is not fatal ---------------------------------------- #
    os.environ["R2_ENV_FILE"] = str(Path(tmp) / "does_not_exist.env")
    load = ef.load_env_file(verbose=False, force=True)
    assert load.path is None and os.environ["R2_BUCKET"] == "untouched"
    assert "does not exist" in load.reason
    print(f"5. missing file tolerated: {load.reason}")

# --- 6. idempotence ---------------------------------------------------------- #
os.environ.pop("R2_ENV_FILE", None)
first = ef.load_env_file(verbose=False, force=True)
second = ef.load_env_file(verbose=False)
assert first is second, "repeat calls must return the cached record"
print("6. idempotent: repeat calls reuse the cached record")

print("\nALL CHECKS PASSED")
