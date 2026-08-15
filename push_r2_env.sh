#!/usr/bin/env bash
# Copy this repo's r2.env to the pod, into the repo directory there.
#
#   ./push_r2_env.sh
#
# Why this is a separate step: sync_up.sh honours .gitignore, and r2.env is
# git-ignored (as it must be), so a normal `sync-up` deliberately will not carry
# credentials over the wire. This does it explicitly, once per pod.
#
# The pod also receives the *shared* ../scripts/r2.env at /tmp/r2.env on every
# `run-experiment`, and that file sets R2_BUCKET to another project's bucket.
# core/env_file.py prefers the repo-local r2.env over the inherited environment
# precisely so this one lands on top -- run this and the bucket is right without
# any RUN_ENTRYPOINT override.
#
# Safe to re-run; nothing else on the pod is touched.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -f r2.env ]; then
  echo "no r2.env in $SCRIPT_DIR" >&2
  echo "  fix: cp r2.env.example r2.env  and fill it in" >&2
  exit 1
fi

# Resolve REMOTE_HOST / REMOTE_DIR exactly the way the shared workflow does:
# the shared defaults first, then this project's overrides.
SHARED_ENV="$SCRIPT_DIR/../scripts/runpod.env"
if [ ! -f "$SHARED_ENV" ]; then
  echo "cannot find $SHARED_ENV — is the shared scripts/ checkout next to this repo?" >&2
  exit 1
fi
LOCAL_DIR="$SCRIPT_DIR"
# shellcheck disable=SC1090
source "$SHARED_ENV"
# shellcheck disable=SC1091
[ -f "$SCRIPT_DIR/.runpod.env" ] && source "$SCRIPT_DIR/.runpod.env"
[ -f "$SCRIPT_DIR/.runpod.local.env" ] && source "$SCRIPT_DIR/.runpod.local.env"

DEST="${REMOTE_DIR%/}/r2.env"
echo "→ $REMOTE_HOST:$DEST"
ssh "$REMOTE_HOST" "mkdir -p $(printf %q "${REMOTE_DIR%/}") && umask 077 && cat > $(printf %q "$DEST")" < r2.env
ssh "$REMOTE_HOST" "ls -l $(printf %q "$DEST")"
echo
echo "Confirm from the pod's point of view:"
echo "  run-experiment r2 check"
