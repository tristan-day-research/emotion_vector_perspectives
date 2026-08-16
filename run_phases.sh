#!/usr/bin/env bash
# Run the remaining Phase 3/4/6 gates back-to-back on a single GPU.
#
# Why a script rather than an ssh one-liner: the one-liner form needs three
# levels of quoting (ssh -> tmux -> bash) and long lines get mangled by terminal
# wrapping. This is also re-runnable, and each stage resumes or no-ops if its
# work is already done.
#
# Usage, from your Mac:
#     sync-up
#     ssh -p <port> root@<host> "tmux kill-session -t phases 2>/dev/null; \
#         tmux new-session -d -s phases \
#         'bash /workspace/emotion_vector_perspectives/run_phases.sh 2>&1 | tee /tmp/phases.log'"
#
# Then watch from your Mac (no login, no scrollback needed):
#     ssh -p <port> root@<host> "tail -60 /tmp/phases.log"

set -u

cd /workspace/emotion_vector_perspectives || exit 1

# Pin one GPU. Setting this on the Mac before `run-experiment` does nothing --
# the variable never crosses the ssh boundary -- so it has to be set here.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME=/workspace/hf_cache
export PYTHONUNBUFFERED=1

SIXTEEN=qwen3-32b_pca-jlens        # 16 balanced emotions, 400 stories each
ALL171=qwen3-32b_pca-jlens_171     # all 171 emotions, 200 stories each

# device_map=None loads the model whole onto one GPU. The "auto" default shards
# across every visible card, and when it cannot fit it silently leaves modules
# on the meta device -- which surfaces much later as
# "NotImplementedError: Cannot copy out of meta tensor; no data!".
GPU_OPTS="--set device_map=None"

run () {
    echo
    echo "=============================================================="
    echo ">>> run.py $*"
    echo "=============================================================="
    if ! python run.py "$@"; then
        # Keep going: these gates are independent apart from phase4 needing its
        # own run's phase3, and a failure late in the list should not cost the
        # results earlier in it.
        echo "!!! FAILED: run.py $*  (continuing to the next stage)"
    fi
}

run phase3 --set run_name="$SIXTEEN"
run phase4 --set run_name="$ALL171" $GPU_OPTS
run phase4 --set run_name="$SIXTEEN" $GPU_OPTS
run phase6 --set run_name="$SIXTEEN" $GPU_OPTS

echo
echo "=============================================================="
echo "ALL STAGES ATTEMPTED -- grep for 'FAILED' and 'VERDICT' above"
echo "=============================================================="
