# make reproduce       : regenerate every number, table and figure that needs only committed results (CPU, ~1 min)
# make reproduce-full  : rerun the feature-level analyses (needs the private ImageNet features: IC014_FEATS=/path/to/feats)
# make paper           : build paper_workshop/main.pdf (tectonic)
PY ?= python3
SING_ROOT ?= $(CURDIR)/_ext/sing
SING ?= $(SING_ROOT)
# Per-image private outputs (never committed, never written into the feature store)
IC014_PRIVATE ?= $(CURDIR)/_private
export IC014_PRIVATE

.PHONY: reproduce reproduce-full paper sing verify-feats check-feats holdout-fast

reproduce:
	PY=$(PY) scripts/reproduce.sh

paper:
	cd paper_workshop && tectonic -X compile main.tex

sing:
	scripts/fetch_sing.sh $(SING)

check-feats:
	@test -n "$(IC014_FEATS)" || { echo "IC014_FEATS is not set: point it at the private feature directory (see README)" >&2; exit 1; }
	@test -d "$(IC014_FEATS)" || { echo "IC014_FEATS=$(IC014_FEATS) does not exist" >&2; exit 1; }

verify-feats: check-feats
	scripts/verify_feats.sh $(IC014_FEATS)

reproduce-full: sing verify-feats
	$(PY) r2/e2_controls.py --feats-root $(IC014_FEATS) --sing-root $(SING) --private $(IC014_PRIVATE)
	$(PY) r2/e2_sweep.py
	$(PY) r2/e2_claims_ci.py --private $(IC014_PRIVATE)
	$(PY) r2/pc_bands.py --feats-root $(IC014_FEATS) --sing-root $(SING)
	$(PY) r2/flatness_synth.py --feats-root $(IC014_FEATS) --sing-root $(SING)
	$(PY) r2/e9_holdout.py --feats-root $(IC014_FEATS) --sing-root $(SING) --private $(IC014_PRIVATE)
	$(PY) r3/equal_energy.py --feats-root $(IC014_FEATS) --sing-root $(SING)
	$(PY) r3/leakfree.py --feats-root $(IC014_FEATS) --sing-root $(SING)
	$(PY) r3/pc_bands_r3.py --feats-root $(IC014_FEATS) --sing-root $(SING)
	$(PY) r3/retrain_translators.py --feats-root $(IC014_FEATS) --sing-root $(SING)
	$(PY) r3/retrain_lambda_posthoc.py --feats-root $(IC014_FEATS) --sing-root $(SING)
	$(PY) r3/fix14_floor.py --feats-root $(IC014_FEATS)
	$(MAKE) reproduce

# Reduced holdout (PREREG_R3.md N10): about 36 min on 8 cores (target was < 30), resumable cache in $(IC014_PRIVATE)
holdout-fast: sing verify-feats
	$(PY) r2/e9_holdout.py --feats-root $(IC014_FEATS) --sing-root $(SING) --private $(IC014_PRIVATE) --fast
