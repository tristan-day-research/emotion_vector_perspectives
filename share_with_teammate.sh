#!/usr/bin/env bash
# Print copy-paste instructions for a teammate to download a run's activations.
#
#   ./share_with_teammate.sh                       # newest run in the bucket
#   ./share_with_teammate.sh <run_name>
#
# You still create the read-only R2 API token in the Cloudflare dashboard (this
# script cannot mint tokens -- the API for that needs account-level credentials we
# deliberately do not handle here). Everything else is generated for you.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

R2_ROOT="${R2_ROOT:-story-activations}"

# This project's bucket. Hardcoded as the default rather than inherited from the
# environment: scripts/r2.env (and any shell that sourced it) sets R2_BUCKET to the
# OTHER project's bucket, so trusting the ambient value silently looks in the wrong
# place. Override deliberately with SHARE_BUCKET=... if you ever need to.
BUCKET="${SHARE_BUCKET:-emotion-vector-perspectives}"

# Credentials from the shared r2.env, unless already exported.
if [ -z "${R2_ACCESS_KEY_ID:-}" ] && [ -f "$SCRIPT_DIR/../scripts/r2.env" ]; then
  set -a; . "$SCRIPT_DIR/../scripts/r2.env"; set +a
fi
export R2_BUCKET="$BUCKET"

PY="${PYTHON:-python}"
if ! "$PY" -c "import boto3" 2>/dev/null; then
  echo "boto3 is missing in $($PY -c 'import sys;print(sys.executable)')" >&2
  echo "  fix: $PY -m pip install boto3" >&2
  exit 1
fi

RUN="${1:-}"
if [ -z "$RUN" ]; then
  RUN=$("$PY" run.py r2 ls --prefix "$R2_ROOT/" \
        | sed -n "s|.*$R2_ROOT/\([^/]*\)/.*|\1|p" | sort -u | tail -1)
  if [ -z "$RUN" ]; then
    echo "No runs found under $R2_ROOT/ in bucket $R2_BUCKET." >&2
    exit 1
  fi
fi

PREFIX="$R2_ROOT/$RUN"
SUMMARY=$("$PY" run.py r2 ls --prefix "$PREFIX/" | tail -1)
ENDPOINT=$("$PY" run.py r2 check | sed -n 's/.*endpoint *: *//p')

cat <<EOF

================================================================================
 Share run: $RUN
 Bucket   : $BUCKET
 Contents : $SUMMARY
================================================================================

STEP 1 -- you, once per teammate (Cloudflare dashboard)

  R2 -> Manage R2 API Tokens -> Create API Token
    Permissions : Object Read only
    Scope       : ONLY the bucket "$R2_BUCKET"
  Copy the Access Key ID and Secret Access Key (secret is shown once).
  Use one token per person so you can revoke individually.

STEP 2 -- send them the block below, with <KEY>/<SECRET> filled in.
          Send the secret through a password manager, Bitwarden Send, or age
          (see README). The rest is not sensitive.

--------------------------- copy from here ---------------------------
# One-time setup
git clone <this-repo-url> && cd emotion_vector_perspectives
pip install -r requirements.txt

export R2_ENDPOINT="$ENDPOINT"
export R2_BUCKET="$R2_BUCKET"
export R2_ACCESS_KEY_ID="<KEY>"
export R2_SECRET_ACCESS_KEY="<SECRET>"

# Confirm access
python run.py r2 check

# Download the activations for this run (~$(echo "$SUMMARY" | grep -o '[0-9.]* GiB' || echo '8 GiB'))
python run.py r2 pull outputs/$RUN/activations --prefix $PREFIX

# Then fit directions and evaluate locally (no GPU needed)
python run.py compute_directions  --set run_name=$RUN
python run.py evaluate_directions --set run_name=$RUN
---------------------------- to here ---------------------------------

Note: the pull includes manifest.json, which records the model revision, layers
and pooling offset. Without it the activations cannot be interpreted, so always
share the whole run prefix rather than hand-picked chunks.

EOF
