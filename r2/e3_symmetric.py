"""Round-2 fix 7: E3 with the alpha=0 baseline subtracted from BOTH scores. Inputs: committed e3/results/e3_trials.csv, e3_table.csv."""
import csv, json, os, numpy as np
from scipy.stats import kendalltau
R = os.path.join(os.path.dirname(__file__), '..')
MODELS = ['resnet', 'resnext', 'dinov2', 'bitresnet', 'deit', 'eva02_large', 'convnext', 'densenet']
T = list(csv.DictReader(open(os.path.join(R, 'e3/results/e3_trials.csv'))))
base = {}
for r in csv.DictReader(open(os.path.join(R, 'e3/results/e3_table.csv'))): base[(r['model'], r['concept'])] = float(r['AS_fair_alpha0'])
concepts = sorted(set(r['concept'] for r in T), key=['red', 'striped', 'water'].index)
def run(models):
    per = {c: {'AS_SING': [], 'AS_fair_raw': [], 'AS_fair_sub': [], 'AS_fair_ratio': []} for c in concepts}
    for r in T:
        al = [float(r[f'alpha_{n}']) for n in models]; c = r['concept']
        s = [float(r[f'AS_SING_{n}']) for n in models]; f = [float(r[f'AS_fair_{n}']) for n in models]
        per[c]['AS_SING'].append(kendalltau(s, al).statistic); per[c]['AS_fair_raw'].append(kendalltau(f, al).statistic)
        per[c]['AS_fair_sub'].append(kendalltau([x - base[(n, c)] for x, n in zip(f, models)], al).statistic)
        per[c]['AS_fair_ratio'].append(kendalltau([x / base[(n, c)] for x, n in zip(f, models)], al).statistic)
    boot = np.random.default_rng(1); out = {}
    for k in ['AS_SING', 'AS_fair_raw', 'AS_fair_sub', 'AS_fair_ratio']:
        tm = np.mean([per[c][k] for c in concepts], 0); bs = boot.choice(tm, size=(10000, len(tm))).mean(1)
        out[k] = dict(mean=float(tm.mean()), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                      per_concept={c: float(np.mean(per[c][k])) for c in concepts})
    return out
res = dict(all8=run(MODELS), without_densenet=run([m for m in MODELS if m != 'densenet']),
           note='AS_fair_sub = AS_fair(alpha) - AS_fair(0) per (model, concept) [pre-registered]; AS_fair_ratio = AS_fair(alpha)/AS_fair(0) [post hoc, descriptive]')
json.dump(res, open(os.path.join(R, 'r2/results/fix7_e3_symmetric.json'), 'w'), indent=1); print(json.dumps(res, indent=1))
