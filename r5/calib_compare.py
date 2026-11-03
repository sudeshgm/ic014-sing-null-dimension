"""Round 5: compare the round-4 calibration worlds (in-sample residuals, r4/results/I2_calibration.json) with the corrected ones
(residuals permuted across images, r4/results/I2_calibration_v2.json). Aggregate only. Writes r5/results/R5_calibration_old_vs_new.json.
Usage: python r5/calib_compare.py"""
import os, json, math
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
J = lambda p: json.load(open(os.path.join(ROOT, p)))


def main():
    old, new = J('r4/results/I2_calibration.json'), J('r4/results/I2_calibration_v2.json'); rows = []
    for b in new['bases']:
        for W in 'CMD':
            to, tn = old['bases'][b]['worlds'][W]['tau'], new['bases'][b]['worlds'][W]['tau']
            for e in tn:
                o, n = to.get(e), tn[e]; same = (o is None and n is None) or (isinstance(o, float) and isinstance(n, float) and ((math.isnan(o) and math.isnan(n)) or abs(o - n) < 5e-4))
                rows.append(dict(base=b, world=W, estimator=e, old=o, new=n, changed=not same))
            if W != 'D':
                eo, en = old['bases'][b]['worlds'][W]['exponent'], new['bases'][b]['worlds'][W]['exponent']
                rows.append(dict(base=b, world=W, estimator='exponent_a', old=eo['a'], new=en['a'], changed=abs(eo['a'] - en['a']) >= 5e-4))
    out = dict(rows=rows, n_changed=sum(r['changed'] for r in rows), n_total=len(rows), content_diagnostic_old=old['content_diagnostic'], content_diagnostic_new=new['content_diagnostic'])
    os.makedirs(os.path.join(ROOT, 'r5', 'results'), exist_ok=True); json.dump(out, open(os.path.join(ROOT, 'r5', 'results', 'R5_calibration_old_vs_new.json'), 'w'), indent=1)
    for r in rows:
        if r['changed']: print(f"{r['base']:8s} {r['world']} {r['estimator']:9s} {r['old']:+.3f} -> {r['new']:+.3f}" if isinstance(r['old'], float) and not math.isnan(r['old']) else f"{r['base']:8s} {r['world']} {r['estimator']:9s} {r['old']} -> {r['new']:+.3f}")
    print('changed', out['n_changed'], 'of', out['n_total'], 'diagnostic old', old['content_diagnostic'], 'new', new['content_diagnostic'])


if __name__ == '__main__':
    main()
