# Round-4 plan

The pre-registered rules are in `PREREG_R4.md`. That file is pushed on its own, before any round-4 computation.

**Headline (fixed):** SING's cross-model ranking is indistinguishable from a null-energy ranking, and the data cannot separate null energy from real null content. "Energy is a confound" is removed throughout the paper. Statements are kept only if they hold under both translator families: SING's released translators and our retrained ridge translators.

## Paper Reviewer, round 3
| Fix | Action | Where |
|---|---|---|
| 1 A1 is AS/√ρ in disguise | Present A1 as "AS per unit null energy". Print its Spearman and τ against AS/√ρ. Stop citing A1/A4 as evidence of a confound. | J; §4.1, abstract |
| 2 A1 driven by two models | Report the jackknife and leave-two-out τ(A1). State that S11 without DinoViT/EVA02 reaches the KILL-R3 level. | J; §4.1 |
| 3 Leak-free decomposition, both halves | Report X with CIs, AS − AS_surr, τ(AS_surr, κ) and AS − AS_perm. Replace the old l.210 sentence with "the data cannot apportion the order". | S; §4.2 |
| 4 KILL-R3 wording for retrained translators | Evaluate the A rules on A1_ret and state the result in the rule's own terms. | J; abstract, §4.1, §5 |
| 5 Surrogate choice | Run the non-parametric and alternative surrogates. State that AS_surr depends on the surrogate. | S; §4.2, table |
| 6 C6-SD point outside its CI | Use a class-resampling bootstrap, with a variance-corrected SD as sensitivity. | C6 |
| 7 C10 borderline | Add a footnote with both intervals and the round-2 verdict; tally reads "one borderline". | §4.4 |
| 8 N10 | Close out: the accuracy criterion fails deterministically; state the shared box. | N10; §A, README |
| 9 Family-C table hand-typed | Generate it from code. | make_tables |
| 10 Planted sensitivity | Add the full own-ρ table and α*. Quantify or delete l.204–205. | P; §4.2 |
| 11 Sign convention | Report all correlations against the leakage order (d_null +0.46). | §3 |
| 12 Anonymity | Build the double-blind archive under `_local/anon/`. | repo |
| 13 Pinned requirements | Add `requirements-pinned.txt`, tested in a fresh venv. | repo |

## Paper Reviewer 2, round 2
| Item | Action |
|---|---|
| 1 Reframe headline | Use the fixed headline above. Present A1 as a per-energy statistic. |
| 2 Retrained translators | Report τ(A1_rel, A1_ret), τ(A1_ret, ρ), the paired CIs, A4/κ/γ under ridge, and the KILL-R3-terms sentence. Fix the §4.3 "artefact" contradiction. Note the possible in-sample advantage of the released translators. |
| 3 Content component | Report X, D_gauss and D_perm, partial Spearman given ρ, the within-class shuffle, and Q. |
| 4 Asymmetric rules; tempering | Use one symmetric bar in S. Acknowledge the round-3 asymmetry. Use point-estimate language. |
| 5 DINO | Run true DINO ViT-B/16 and ViT-S/16 plus the AugReg control (DN). |
| 6 Generality | Restrict the recommendation to AS-type scores; call NuSA an energy proxy. |
| 7 Planted table | Add the full table (P). PR2's "DINOv2 drops to 0.89×" reads the absolute AS (0.89° at α = 0.3, own ρ) as a ratio. The committed ratio is 1.17× (0.762° → 0.891°). The paper prints the committed ratio and the absolute values. |
| 8 C10 | Add the footnote. |
| 9 Timestamps | Push the prereg on its own; add a sentence in the paper. |
| 10 Anonymise | Build the archive. |
| 11 Table 7 | Generate it from code (family C). |
| 12 Citations | Cite Theiler et al. 1992, ESL §3.4, Bansal et al. 2021, Lenc & Vedaldi 2015. |

## Order
1. Push the prereg.
2. Write the `r4/` scripts and run them (S, J, A4_ret, DN, P, C6), with private outputs kept outside git.
3. Check the kills. If one fires, stop and report.
4. Edit the paper. Extend make_tables.
5. Build the archive.
6. Build, reproduce, check for a clean diff, commit and push.
