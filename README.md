# IC014: an audit of SING's null-space leakage score

This repo holds the code, committed result tables and the workshop paper (`paper_workshop/main.tex`, `main.pdf`) auditing SING (arXiv 2603.14610, CVPR 2026).
Main finding: the null-energy fraction ρ alone reproduces SING's 13-model order as well as SING's own score (Kendall τ = 0.82). Size-corrected rankings differ from SING's, but the 95% CIs are wide and the result is not stable across null tolerances.

## Layout
* `exp0b/` DinoViT identity (top-1 of five ViT-L candidates). `exp0/` Supp. Fig. 6b digitisation and null dims. `exp1/` within-model AS(d).
* `e1/` energy law; `e2/` 13-model ranking and claim audit; `e3/` planted-concept sensitivity; `e8/` null concept probe (pre-registered, KILL); `e9/` score screen (pre-registered, KILL).
* `r2/` round-2 analyses (`PREREG_R2.md`, `ROUND2_PLAN.md`); outputs in `r2/results/`. See `r2/README.md`.
* `gpu_jobs/` specs for work that needs ImageNet images or a GPU.
* `paper_workshop/archive/` earlier drafts.

## Reproduce
* `make reproduce` (or `scripts/reproduce.sh`) recomputes the E1/E3/E2 round-2 summaries and every figure from committed results. It runs on CPU in about 20 s (`pip install -r requirements-lock.txt`, Python 3.13).
* `make paper` builds the PDF with tectonic.
* `make reproduce-full` reruns the feature-level analyses. It needs the private ImageNet features in `$IC014_FEATS` (checked by `scripts/verify_feats.sh` against `scripts/checksums.sha256`) and SING at commit e47b37d (`scripts/fetch_sing.sh`).

## Data
We do not redistribute ImageNet images or features derived from them (ImageNet terms of access).
The image lists are in `scripts/subset_files_10k.txt` (10 per class, file name and label) and `scripts/subset_files_claims200.txt`. Re-extracting the features requires your own copy of ILSVRC2012 val.
