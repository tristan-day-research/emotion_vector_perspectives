<!-- Written to be adversarially checked. If you are an AI or a reviewer auditing this
     work, start here: every claim below names the artefact it came from and the command
     that reproduces it, and the last section lists the attacks most likely to succeed. -->

# Verification guide: the Chinese-readout analysis

## What this document is for

Sections 5e–5i of [results_notebook.ipynb](results_notebook.ipynb) added a new claim to
this project: **GATE A failed because it asked in the wrong language, not because the
emotion directions are unreadable.** That is a substantive reinterpretation of a
recorded failure, it rests partly on hand-written translation data, and it should not be
taken on trust.

This file exists so a second party can check it without re-reading the notebook. Each
claim is stated, sourced to a file, and paired with a command that re-derives it from
scratch. The final section is the honest list of where the analysis is weakest.

## Ground rules the analysis operated under

* **Read-only.** No model was loaded, no GPU used, no network call made, no R2 access.
  Every input is a file already under `outputs/*/results/phases/` or `data/`.
* **Nothing was re-run.** The pipeline artefacts are exactly as the original runs wrote
  them. All new numbers are derived from them by the notebook.
* **Environment.** conda `base` (`~/miniconda3/bin/python`): pandas 2.3.3, numpy 1.26.4,
  matplotlib 3.10.7, tokenizers 0.22.1. The system `python3` is Homebrew 3.14 with no
  packages and will not run the notebook.

Reproduce everything:

```bash
~/miniconda3/bin/python -m jupyter nbconvert --to notebook --execute \
    --inplace --ExecutePreprocessor.kernel_name=base analysis/results_notebook.ipynb
```

This rewrites `analysis/RESULTS.md` and `analysis/figures/`. The notebook is
deterministic: the only randomness is a permutation null seeded with
`np.random.default_rng(0)`.

## The two runs

| key | run directory | design |
| --- | --- | --- |
| `16` | `outputs/qwen3-32b_pca-jlens` | 16 balanced circumplex emotions, 400 stories each |
| `171` | `outputs/qwen3-32b_pca-jlens_171` | all 171 emotions, 200 stories each |

Both use Qwen3-32B at block 31 with the same `neuronpedia/jacobian-lens` `qwen3-32b`
artefact.

---

## Claim 1 — the readouts are majority CJK

**Stated as:** 43–61% of top-12 readout tokens are CJK script; overall 55.3% (16 run)
and 42.6% (171 run).

**Source:** `outputs/*/results/phases/phase4_readouts.csv`, rows with `rank < 12`.

**Method:** each token is classified by its first script-bearing character. CJK covers
U+4E00–9FFF, U+3400–4DBF, U+F900–FAFF, plus kana and hangul; a token containing any CJK
character is CJK regardless of what else it contains. Latin requires an ASCII (or
LATIN-named) letter. Anything with no letter is `punct / whitespace`.

**Check it:**

```bash
~/miniconda3/bin/python - <<'EOF'
import pandas as pd
def cjk(t): return any(0x4E00<=ord(c)<=0x9FFF or 0x3400<=ord(c)<=0x4DBF
                       or 0xF900<=ord(c)<=0xFAFF for c in str(t))
for r in ['qwen3-32b_pca-jlens','qwen3-32b_pca-jlens_171']:
    d=pd.read_csv(f'outputs/{r}/results/phases/phase4_readouts.csv')
    d=d[d['rank']<12]
    print(r, f"{d.token.map(cjk).mean():.1%} CJK of {len(d)} tokens")
EOF
```

Expected: `55.3% CJK of 360` and `42.4% CJK of 2220`. The 171-run figure is 42.4% here
against **42.6%** in RESULTS.md — the command above is a deliberately simplified
classifier, while the notebook's also counts kana, hangul and non-ASCII Latin. The 0.2 pt
gap is that difference and nothing else.

**Attack surface:** the classifier folds kana and hangul into "CJK". In this data that
changes nothing (there are no kana/hangul tokens), but the label is imprecise. Tokens
mixing scripts are counted as CJK, which biases the CJK share *upward*; the
`punct / whitespace` bucket is large (18–22%) and is counted as neither script.

---

## Claim 2 — "it's a Chinese model" is not a sufficient explanation

**Stated as:** 96–97% of the Latin tokens in these readouts are whole words, not subword
fragments, so the lens is not being pushed into Chinese by English tokenising badly.
(96% on the 16 run, 97% on the 171 run — quote the range, not the higher figure.)

**Source:** same CSVs. A "whole word" is defined as matching `^ [A-Za-z]+$` — leading
space, then letters only, which is the byte-level-BPE signature of a word-initial full
word.

**Check it:**

```bash
~/miniconda3/bin/python - <<'EOF'
import pandas as pd
for r in ['qwen3-32b_pca-jlens','qwen3-32b_pca-jlens_171']:
    d=pd.read_csv(f'outputs/{r}/results/phases/phase4_readouts.csv'); d=d[d['rank']<12]
    lat=d.token.astype(str)[d.token.astype(str).str.contains(r'[A-Za-z]', regex=True)]
    lat=lat[~lat.str.contains(r'[一-鿿]', regex=True)]
    print(r, f"{lat.str.match(r'^ [A-Za-z]+$').mean():.0%} whole-word of {len(lat)} Latin")
EOF
```

**Attack surface.** This is the weakest of the load-bearing claims and it is stated
as a negative ("not sufficient"), not as a mechanism.

* The regex counts a *token's form*, not its semantic density. ` pornost` matches
  `^ [A-Za-z]+$` and is a fragment of a word.
* Whole-word English tokens being available does not prove the model *could* have used
  them here — only that fragmentation is not the obstacle.
* The notebook does **not** claim to know why the readout prefers Chinese. Any reviewer
  finding text that says otherwise should flag it; the claims table records this as
  `FAILED as a complete explanation ... not further diagnosed here`.

---

## Claim 3 — the ordering signal survives total burial *(strongest claim here)*

**Stated as:** the `+valence` a-priori axis separates pleasant from unpleasant anchors
perfectly (AUROC 1.00) while ranking its best English probe word at 1,927 of 151,936,
with **zero** probes inside GATE A's top-12 window.

**Source:** `phase4_gate.json` → `controls_apriori_axes["+valence"]["probe_ranks"]`, a
field the pipeline already wrote. Nothing is recomputed; the notebook only groups the
ranks by the a-priori valence labels in `probes`.

**Check it:**

```bash
~/miniconda3/bin/python - <<'EOF'
import json
d=json.load(open('outputs/qwen3-32b_pca-jlens/results/phases/phase4_gate.json'))
ax=d['controls_apriori_axes']['+valence']
val=dict(zip(d['probes']['words'], d['probes']['valence']))
pr={k:v for k,v in ax['probe_ranks'].items() if v is not None}
pos=sorted(v for k,v in pr.items() if val[k]>0); neg=sorted(v for k,v in pr.items() if val[k]<0)
print('AUROC', ax['auroc_valence'], '| worst pleasant', max(pos), '| best unpleasant', min(neg))
print('best rank of any probe', min(pr.values()), 'of', d['gate_a']['vocab_size'])
print('separation is perfect:', max(pos) < min(neg))
EOF
```

Expected: `AUROC 1.0 | worst pleasant 71448 | best unpleasant 106432`, best rank 1927,
`separation is perfect: True`.

**Why it matters:** it explains, without any translation data, why GATE A and GATE B
disagreed. GATE A is an absolute containment test and is destroyed by the readout
preferring another language. AUROC is a *relative* test within a fixed probe set and is
invariant to burying every probe by a common factor.

**Attack surface:** n = 14 probes, so AUROC 1.00 is 7×7 with no ties — impressive but
low-resolution. The claim is about the +valence *a-priori axis*, which is constructed
from the anchor labels, so it is closer to a sanity check than an independent discovery;
the PC-based version (PC2, AUROC 1.00) is the one that is not circular.

---

## Claim 4 — GATE A re-scored in Chinese *(most consequential, most contestable)*

**Stated as:** applying GATE A's own top-12 containment rule to Chinese translations
gives 62.5% (16 run) and 25.1% (171 run), against permutation nulls of 10.9% and 1.7%,
p = 0.0005 with 2000 permutations.

**Sources:**
* readouts: `phase4_readouts.csv`, `group == "gate_a_emotion_vector"`, `rank < 12`
* translations: [`zh_en_glossary.py`](zh_en_glossary.py) → `EMOTION_ZH`
* T1 baseline: `phase4_gate.json` → `gate_a.rows[*].verdict == "HIT"` (the pipeline's
  own verdict, not recomputed)

**Method.** Three nested tiers, all asking the identical question — *is it in the top
12?* — and varying only what counts as the emotion's name:

| tier | what counts | who wrote it |
| --- | --- | --- |
| T1 | the exact English lemma | the pipeline |
| T2 | an English near-synonym | `EMOTION_EN_SYNONYMS` (hand-written) |
| T3 | a Chinese translation | `EMOTION_ZH` (hand-written) |

Matching is **exact set membership** after `.strip()` (and `.lower()` for Latin). No
substring matching, no fuzzy matching, no embedding similarity. This is deliberate: a
generous list is then visible in the list rather than hidden in a matcher.

**The null.** The obvious objection is that T3 gives each emotion 3–5 candidates where
T1 gives one lemma, so the hit rate must rise. The permutation null answers exactly
this: it keeps every list intact — same contents, same length, same generosity — and
only shuffles **which list is scored against which emotion's readout**. If the lists were
merely generous, the shuffled rate would match the observed rate. It does not.

**Check it** (this is the single most important command in this document):

```bash
~/miniconda3/bin/python - <<'EOF'
import sys, numpy as np, pandas as pd
sys.path.insert(0,'analysis'); import zh_en_glossary as G
rng=np.random.default_rng(0)
for r in ['qwen3-32b_pca-jlens','qwen3-32b_pca-jlens_171']:
    d=pd.read_csv(f'outputs/{r}/results/phases/phase4_readouts.csv')
    d=d[(d['rank']<12)&(d.group=='gate_a_emotion_vector')]
    tops={e:[str(t) for t in s.sort_values('rank').token] for e,s in d.groupby('direction')}
    E=sorted(tops)
    hit=lambda read,lst: any(t.strip() in set(G.EMOTION_ZH[lst]) for t in tops[read])
    obs=np.mean([hit(e,e) for e in E])
    mism=np.mean([hit(a,b) for a in E for b in E if a!=b])
    perm=np.array([np.mean([hit(a,b) for a,b in zip(E,rng.permutation(E))]) for _ in range(2000)])
    print(f'{r:26} obs {obs:.1%} | wrong-list {mism:.1%} | null {perm.mean():.1%} | '
          f'p={(np.sum(perm>=obs)+1)/2001:.4f}')
EOF
```

Expected: `obs 62.5% | wrong-list 7.9% | null 10.9% | p=0.0005` and
`obs 25.1% | wrong-list 1.6% | null 1.7% | p=0.0005`.

Note `p=0.0005` is the floor for 2000 permutations (`1/2001`), i.e. *no* permutation
reached the observed value. It is not a precise p-value and should not be quoted as one.

**Pre-commitment.** The translation lists were written from the 171 emotion words alone,
in one pass, and saved to `zh_en_glossary.py` **before** any matching code was run. The
module docstring records this and discloses the contamination that does exist: the
author had already seen the PC-end and axis-control readouts (~51 CJK tokens) and knew
which 5 emotions the pipeline scored HIT, but had *not* seen the per-emotion top-12
lists that T3 scores.

**How to attack this properly.** The permutation null defends against *generosity*, not
against *bias in list composition*. Two attacks that would actually bite:

1. **Regenerate the lists blind.** Ask an independent model for 3–5 Chinese translations
   of each of the 171 emotions, with no access to the readouts, and re-score. If the
   result holds, the pre-commitment claim is no longer load-bearing. This is the check
   worth running, and it needs no artefacts beyond the emotion list.
2. **Attack specific pairs.** The loosest entries are `serene → 柔和` (soft/gentle),
   `ecstatic → 狂欢` (revelry/carnival), `calm → 温和` (mild), and `thrilled → 狂欢`.
   Note that `狂欢` is shared by `ecstatic` and `thrilled` and `温和` by `calm`. Delete
   the four you find least defensible from `EMOTION_ZH` and re-run — the 16-run rate is
   only 10/16, so it is sensitive to a few entries. The 171-run rate (43 hits) is much
   more robust.

Every matched token is printed in RESULTS.md under *"every emotion scored as a hit in
the relaxed tiers, with the exact tokens that matched"*, and the full 171-row list is
printed under *"the full pre-committed translation table"*. Audit those, not this prose.

---

## Claim 5 — tokenizer-checked denominators

**Stated as:** only 59% of the Chinese candidates are single tokens; 152 of 171 emotions
have at least one single-token Chinese form; and **43 of the 57 emotions GATE A could
not score in English do have a single-token Chinese form.**

**Source:** the real Qwen3-32B tokenizer, cached locally at
`data/hf_cache/models--Qwen--Qwen3-32B/snapshots/*/tokenizer.json` (11 MB, tokenizer
only — **no model weights are present or needed**), plus
`phase4_gate.json → gate_a.untokenizable`.

**Check it:**

```bash
~/miniconda3/bin/python - <<'EOF'
import sys, json, glob
from tokenizers import Tokenizer
sys.path.insert(0,'analysis'); import zh_en_glossary as G
tok=Tokenizer.from_file(glob.glob('data/hf_cache/**/tokenizer.json', recursive=True)[0])
one=lambda s: len(tok.encode(s, add_special_tokens=False).ids)==1
tot=sum(len(v) for v in G.EMOTION_ZH.values())
sing=sum(one(c) for v in G.EMOTION_ZH.values() for c in v)
untok=set(json.load(open(
  'outputs/qwen3-32b_pca-jlens_171/results/phases/phase4_gate.json'))['gate_a']['untokenizable'])
print(f'single-token candidates {sing}/{tot} ({sing/tot:.0%})')
print('emotions with >=1:', sum(any(one(c) for c in v) for v in G.EMOTION_ZH.values()), '/171')
print('English-untokenizable rescued in Chinese:',
      sum(any(one(c) for c in G.EMOTION_ZH.get(e,[])) for e in untok), '/', len(untok))
EOF
```

**Why it is a correction and not a rescue:** a multi-token candidate can never appear in
a top-12 *token* list, so including it in the list can only deflate the measured rate.
Restricting to the Chinese-testable set moves the 171-run rate from 25.1% to 28.3% and
leaves the 16-run rate unchanged at 62.5%. The permutation-controlled conclusion does
not depend on which denominator is used.

**The independently important part** has nothing to do with translation: GATE A
structurally excluded 57 of 171 emotions (33%) because their English lemma is not a
single token. That is a fact about English orthography, not about their directions.

---

## Claim 6 — what a proper fix requires

**Stated as:** a rank-based cross-lingual GATE A needs `J[31]` (~105 MB) and `lm_head`
(~1.6 GB), is a CPU matrix-vector product plus a top-k, and does **not** need the GPU pod.

**Reasoning:** the lens readout is `logits = lm_head(final_norm(J_l @ h))` — recorded in
`phase0_gate.json → convention.unembed.formula` and verified there against the model.
The emotion vectors are already local in `phase2_emotion_vectors.safetensors`, and the
tokenizer is local (Claim 5), so the only missing inputs are those two tensors.

**Not yet done.** No rank-based cross-lingual result exists. Everything in Claims 4–5 is
top-12 containment, because `phase4_readouts.csv` persisted only the top 12 tokens per
direction. Anyone reading a rank claim into this work is reading something that is not
there.

---

## What did *not* change, and must not be reported as if it had

The Chinese analysis touches Phase 4 only. It is worth being explicit, because the
temptation to let a good result spread is exactly the failure mode this project's gates
exist to prevent.

| result | status after this analysis |
| --- | --- |
| Phases 1–3 (stimuli, emotion vectors, PCA, circumplex recovery) | **untouched** — computed from activations; the lens is not involved at any point |
| Split-half reliability, PC stability, cross-run PC correspondence | **untouched** |
| Phase 4 AUROC / GATE B verdicts | **untouched** — Claim 3 explains why they were immune, it does not alter them |
| The pipeline's recorded GATE A verdict | **still FAILED** — Claim 4 is a secondary re-scoring under hand-written data, reported alongside, never substituted |
| The under-converged 80-prompt lens | **untouched** — still the largest threat to everything magnitude-sensitive |
| Effective dimensionality 9.8; PC1 affect-presence contamination on the 171 run | **untouched** |
| Phase 6's dictionary decomposition | **changed by a re-run, not by this analysis** — see below |

### Phase 6 was re-run twice and the verdict reversed — read this before citing it

Phase 6 has had three states in this project. Only the third is current.

1. **`Jᵀ` atoms, judged by a self-lensing check → reported FAILED.** Superseded.
2. **`J⁺` (write-space) atoms → passed that check.** Now kept as an *ablation* under
   `results/phases/write_space_ablation/`, answering a different question.
3. **`Jᵀ(g ⊙ u_t)` read-space atoms (`atom_mode: read`) → the current result.**

**The methodological point is the finding.** The lens score for token `t` is `u_tᵀ J h`,
which regroups as `(Jᵀu_t)ᵀ h` — so the direction the *readout* is linear in is `Jᵀu_t`.
The write direction `J⁺u_t` answers the different question of what perturbation makes the
model emit `t`. The old gate asked "does lensing an atom return its own token?", which is
a **write**-direction property; a read dictionary has no reason to satisfy it. The
write-space ablation passes it 24/24, which is the tell. The read construction is
validated instead by a score-identity check (r > 0.9999 over 8 probes).

Current state, read-space, k = 16:

| | 16 run | 171 run |
| --- | --- | --- |
| mean frac_reportable | 3.0% | 2.3% |
| random-direction control (mean, n = 500) | 0.65% | 0.64% |
| ratio | 4.6× | 3.6× |
| beat null, Bonferroni | **17/17** | **0/172 — see below** |
| beat null, uncorrected 0.05 | 17/17 | 172/172 |
| atoms above 0.5 coherence | 30.9% | — |

**The 171 run's `0/172` is a resolution limit, not a null result**, and citing it as a
failure would be wrong. 500 permutations floor the p-value at 1/501 = 0.00200; Bonferroni
over 172 emotions demands 0.05/172 = 0.00029. No effect size can clear that. It needs
≈ 3,440 permutations. The 16-run test *is* resolvable (0.00294 > 0.00200) and passes 17/17.

**Check all of it:**

```bash
~/miniconda3/bin/python - <<'EOF'
import json, numpy as np, pandas as pd
for r in ['qwen3-32b_pca-jlens','qwen3-32b_pca-jlens_171']:
    g=json.load(open(f'outputs/{r}/results/phases/phase6_gate.json'))
    k=str(g['reported_k'][0]); gg=g['gate']; ctl=g['random_control'][k]
    e=[x for x in g['per_emotion'] if x['emotion']!='neutral']
    m=np.mean([x['per_k'][k]['frac_reportable'] for x in e])
    print(f"{r}\n  mode={g['atom_mode']} write_space={g['write_space']} "
          f"read_identity r>={g['read_identity']['min_correlation']:.6f} "
          f"holds={g['read_identity']['holds']}")
    print(f"  {m:.4f} vs ctl {ctl['mean']:.4f} (n={ctl['n']}) = {m/ctl['mean']:.2f}x")
    print(f"  beat_null {len(gg['beat_null'][k])}/{gg['n_emotions']}; "
          f"p_floor {gg['p_value_floor']:.5f} vs bonferroni {gg['alpha_bonferroni']:.5f} "
          f"-> resolvable {gg['p_value_floor'] < gg['alpha_bonferroni']} "
          f"(needs {int(np.ceil(1/gg['alpha_bonferroni']))} perms)")
    # k saturation
    d=[abs(x['per_k'][str(a)]['frac_reportable']-x['per_k'][str(b)]['frac_reportable'])
       for x in g['per_emotion'] for a,b in [tuple(g['reported_k'])]]
    print(f"  k={g['reported_k']} max |diff| {max(d):.2e}; agree to 3dp "
          f"{sum(1 for x in d if x<5e-4)}/{len(d)}")
EOF
```

**Attack surface.** The reportable *fraction* is well controlled; the *token attributions*
are not — 30.9% of atoms exceed the 0.5 interchangeability threshold, so which atom the
pursuit picked among near-duplicates is close to arbitrary. And "remainder" means outside
the k-sparse **nonnegative** span of this pool at this k — a union of cones, not a linear
subspace — so it is not evidence of anything being intrinsically unverbalizable.

## Priority order for a reviewer with limited time

1. **Claim 4's permutation null** — run the command. It is the load-bearing statistic.
2. **Regenerate `EMOTION_ZH` blind and re-score** — the one attack that would genuinely
   overturn or confirm Claim 4.
3. **Claim 3** — three lines of JSON, no hand-written data, and it carries most of the
   interpretive weight on its own.
4. Spot-check the four loose translation pairs named in Claim 4.
5. Confirm the "what did not change" table against RESULTS.md's claims table.

## Files added or changed in this analysis

| file | what it is |
| --- | --- |
| [results_notebook.ipynb](results_notebook.ipynb) | the analysis; sections 5e–5i are new |
| [zh_en_glossary.py](zh_en_glossary.py) | hand-written translation data; **the only non-derived input** |
| [nbtools.py](nbtools.py) | artefact loading, report accumulation, markdown tables, script classification, `tiered_gate_a` (the Claim 4 statistic) |
| [nbfigures.py](nbfigures.py) | every plot builder; each returns a bare matplotlib figure |
| [RESULTS.md](RESULTS.md) | generated export, every table and figure reference |
| `figures/phase4_gate_a_tiers.png` | Claim 4 |
| `figures/phase4_probe_rank_burial.png` | Claim 3 |
| this file | the audit trail |

The notebook is a document; the machinery is in `nbtools.py` and `nbfigures.py`. If you
are auditing the *statistic* behind Claim 4 rather than the prose around it, read
`nbtools.tiered_gate_a` — it is ~60 lines and contains the permutation null in full.
`Report.fig()` refuses to emit a figure without both a *How to read it* and a *What it
shows* note, and the notebook's last cell asserts that no figure went out unexplained.

No file under `outputs/` was modified. Verify with `git status --short outputs/`.
