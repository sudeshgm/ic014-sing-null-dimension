# sanity_check

A cheap check for scores that compare models through the null space of their linear heads (or through residual feature energy): does the cross-model ranking survive centring, matched-energy content-free surrogates and a change of head? See Sec. 3 of the paper and the section "Use the sanity check on your own score" in the top-level `README.md`.

* `null_sanity_check.py`: `features` mode: per-model ρ = ρ_off + ρ_btw + ρ_wit (as `r8/posthoc/rho_decomp.py`) and, with `--as-variants`, SING's AS on the real features, offset only, offset removed, zero-mean Gaussian at the centred energy and centred null coordinates permuted across images (as `r8/posthoc/offset_surr.py`, via `r3/leakfree.py`). `rank` mode: Kendall τ_b of each column against a reference order with model-bootstrap percentile and BCa 95% CIs, and paired CIs against your score (`r3/common.py`, B = 2000, seed 0).
* `test_sanity_check.py`: `python -m pytest -q sanity_check`. The fast test reproduces `r8/posthoc/offset_tau.json` and `offset_surr_tau.json` exactly from the committed per-model JSON. The slow test (needs `IC014_FEATS`) reproduces ResNet50's committed ρ decomposition bit for bit and its AS variants to a relative 1e-9.

The surrogate analyses for SING are post hoc. The τ values the tool reports for SING's 13 models are the ones in the paper (Table 1). With few models, report a paired interval that covers zero as "not separated", not as equivalence.
