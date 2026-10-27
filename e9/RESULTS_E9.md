**VERDICT: KILL.** No candidate passed DEV, so there were no finalists and the holdout was never run. M1 had DEV tau 0.976 but was flat on 0/4 models; S10 was the only flat candidate, at tau 0.395. No post-hoc reruns.

# E9 score screen: results (DEV only, E9 = KILL)

Run: `e9/analyze_e9.py`, one automated pass, linear plant, DEV concepts red and striped, 8 models.
Selection rule (prereg step 2): DEV mean Kendall tau >= 0.8 AND flatness (max/min ratio <= 1.5) on >= 3/4 null-dim-1048 models at both alpha = 0.1 and alpha = 1.

Outcome: no candidate qualifies, so there are no finalists, the holdout was never touched, and E9 = KILL by the preregistered rule.
Top-3 singletons by DEV tau: M1, M2, S9 (combinations S11-S14 built from them).

| cand | parts | tau mean | 95% CI | tau red | tau striped | n_flat a=0.1 | n_flat a=1 | flat |
|---|---|---|---|---|---|---|---|---|
| S1 |  | -0.393 | [-0.446, -0.334] | -0.406 | -0.380 | 0/4 | 0/4 | False |
| S2 |  | -0.224 | [-0.295, -0.154] | -0.233 | -0.216 | 0/4 | 0/4 | False |
| S3 |  | 0.366 | [0.301, 0.429] | 0.387 | 0.346 | 2/4 | 4/4 | False |
| S4 |  | -0.051 | [-0.113, 0.009] | -0.070 | -0.031 | 0/4 | 0/4 | False |
| S5 |  | 0.311 | [0.249, 0.367] | 0.297 | 0.324 | 0/4 | 0/4 | False |
| S6 |  | 0.391 | [0.339, 0.444] | 0.387 | 0.396 | 0/4 | 0/4 | False |
| S7 |  | 0.209 | [0.139, 0.273] | 0.206 | 0.211 | 3/4 | 0/4 | False |
| S8 |  | -0.273 | [-0.366, -0.181] | -0.277 | -0.269 | 2/4 | 0/4 | False |
| S9 |  | 0.522 | [0.464, 0.578] | 0.520 | 0.524 | 0/4 | 1/4 | False |
| S10 |  | 0.395 | [0.330, 0.457] | 0.356 | 0.434 | 3/4 | 4/4 | True |
| M1 |  | 0.976 | [0.963, 0.988] | 0.976 | 0.977 | 0/4 | 0/4 | False |
| M2 |  | 0.540 | [0.495, 0.586] | 0.586 | 0.494 | 0/4 | 0/4 | False |
| M3 |  | 0.114 | [0.054, 0.179] | 0.106 | 0.121 | 1/4 | 4/4 | False |
| M4 |  | 0.234 | [0.153, 0.315] | 0.221 | 0.247 | 2/4 | 4/4 | False |
| M5 |  | 0.323 | [0.271, 0.372] | 0.281 | 0.364 | 0/4 | 0/4 | False |
| M6 |  | 0.359 | [0.310, 0.406] | 0.473 | 0.244 | 0/4 | 0/4 | False |
| S11 | M1+M2 | 0.792 | [0.764, 0.818] | 0.818 | 0.766 | 0/4 | 0/4 | False |
| S12 | M1+S9 | 0.781 | [0.748, 0.812] | 0.781 | 0.782 | 0/4 | 0/4 | False |
| S13 | M2+S9 | 0.618 | [0.570, 0.662] | 0.647 | 0.588 | 0/4 | 0/4 | False |
| S14 | M1+M2+S9 | 0.776 | [0.744, 0.806] | 0.793 | 0.759 | 0/4 | 0/4 | False |

Why nothing qualifies: the five candidates with tau >= 0.6 (M1 0.976, S11 0.792, S12 0.781, S14 0.776, S13 0.618) are flat on 0/4 models at both alphas; M1 is the only one above 0.8 and it fails flatness. The only candidate passing flatness is S10 (3/4 and 4/4), with tau 0.395.

Files: `e9/results/e9_dev.json` (full DEV output incl. raw flatness ratios), `e9/results/e9_dev_table.csv`. No features committed.
