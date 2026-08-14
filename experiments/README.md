# `experiments/`

Mechanistic-interpretability experiments. Intentionally empty for now — the baseline
in `extract_emotion_vectors/` comes first.

The planned sequence, and the interfaces each one builds on:

### 1. Experiencer binding

Is an emotion representation bound to *who* is feeling it — the default assistant,
an alternative first-person persona, or a third-person character?

The extraction config already carries the hooks: `use_chat_template`, `chat_role`,
`text_prefix`, `text_suffix`. All four are part of the activation compatibility
fingerprint, so each experiencer condition is forced into its own run rather than
being silently pooled with another. One run per condition, same `split_seed`, then
compare.

### 2. Trained linear probes

Logistic regression on the same pooled activations from `ActivationStore`. Keep
these strictly separate from the fixed mean-difference directions in `core/directions.py`
— different objects, different names, reported separately.

### 3. Cross-condition transfer

Fit in one experiencer condition, evaluate in another. `DirectionSet.score` already
accepts arbitrary activation matrices, so this is a loop over
`(fit_run, eval_run, layer)` rather than new machinery.

### 4. Causal steering

Add `α · direction` at a layer during generation and measure the behavioural effect.
`DirectionSet.direction(emotion, layer)` returns the unit vector; steering needs a
forward hook, which does not exist yet.

Because the same `split_seed` produces the same topic partition regardless of which
emotions or model a run used, held-out topics stay held out across all of the above.
