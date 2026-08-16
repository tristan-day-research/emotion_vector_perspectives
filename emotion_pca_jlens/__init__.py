"""Emotion-space PCA + Jacobian-lens interpretation.

Each phase stops at a gate for a human to read before the next one runs, because
every phase can silently invalidate the ones after it.

Structural half -- what the reportable geometry of emotion looks like:

0. ``phase0_lens_gate``   load the J-lens, verify the readout convention   [done]
1. ``phase1_stimuli``     emotions across the circumplex quadrants         [done]
2. emotion vectors        pooled residual activations, split-half gate
3. PCA                    mean-centre across emotions, PC1-PC2 scatter
4. J-lens the PCs         what each principal axis is disposed to verbalise
5. extensions             layer sweep / perspective axis / within-emotion PCA

Functional half -- whether that geometry is what actually moves the model. A lens
reads only the *verbalisable* component, which the workspace paper puts at ~6-10%
of a concept vector's variance, so Phases 0-5 characterise a small slice and
ignore the remainder:

6. decompose              v -> v_J (reportable) + v_perp (remainder)
7. two channels           self-report vs behaviour, rubrics kept strictly apart
8. steer                  v / v_J / v_perp / random x both channels
9. re-entry clamp         the decisive control (optional)

Phases 2-9 are not written. See the README section "Phases 6-9: from structure to
function" for the specification, including the open design question in Phase 6:
the lens exposes no dictionary, so one has to be constructed from the transport.
"""
