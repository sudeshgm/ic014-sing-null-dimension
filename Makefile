# make reproduce       : recompute every number and figure that needs only committed results (CPU, ~1 min)
# make reproduce-full  : rerun the feature-level analyses (needs private ImageNet features in $(IC014_FEATS))
PY ?= python3
IC014_FEATS ?= data/ic014
SING_ROOT ?= $(CURDIR)/_ext/sing
SING ?= $(SING_ROOT)

.PHONY: reproduce reproduce-full paper sing verify-feats

reproduce:
	PY=$(PY) scripts/reproduce.sh

paper:
	cd paper_workshop && tectonic -X compile main.tex

sing:
	scripts/fetch_sing.sh $(SING)

verify-feats:
	scripts/verify_feats.sh $(IC014_FEATS)

reproduce-full: sing verify-feats
	$(PY) r2/e2_controls.py --feats-root $(IC014_FEATS) --sing-root $(SING) --private $(IC014_FEATS)/r2_private
	$(PY) r2/e2_sweep.py
	$(PY) r2/e2_claims_ci.py --private $(IC014_FEATS)/r2_private
	$(PY) r2/pc_bands.py --feats-root $(IC014_FEATS) --sing-root $(SING)
	$(PY) r2/flatness_synth.py --feats-root $(IC014_FEATS) --sing-root $(SING)
	$(PY) r2/e9_holdout.py --feats-root $(IC014_FEATS) --sing-root $(SING)
	$(MAKE) reproduce
