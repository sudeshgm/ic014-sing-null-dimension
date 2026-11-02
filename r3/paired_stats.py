"""PREREG_R3 F: paired statistics on n=13 (N1). Output r3/results/F_paired_stats.json."""
import numpy as np
from scipy.stats import spearmanr, rankdata
from common import load_models, ktau, tau_ci, paired_ci, perm_p, dump, PRETTY, S11_DROP

ms, col, sing = load_models(); lk = -sing
rho, AS, IS = col('rho'), col('AS_SING'), col('IS')


def pk(x, y, z):
    txy, txz, tyz = ktau(x, y), ktau(x, z), ktau(y, z)
    return (txy - txz * tyz) / np.sqrt((1 - txz ** 2) * (1 - tyz ** 2))


def pspear(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    res = lambda a: a - np.polyval(np.polyfit(rz, a, 1), rz)
    return float(np.corrcoef(res(rx), res(ry))[0, 1])


lr, la = np.log(rho), np.log(AS)
resA = la - np.polyval(np.polyfit(lr, la, 1), lr); resR = lr - np.polyval(np.polyfit(la, lr, 1), la)
i_d, i_v = ms.index('densenet'), ms.index('dinovit1'); lk2 = lk.copy(); lk2[i_d], lk2[i_v] = lk[i_v], lk[i_d]
sc = np.array([m not in S11_DROP for m in ms])
ours = IS / AS
out = dict(
    n=len(ms), models=ms,
    tau_rho=tau_ci(rho, lk), tau_AS_SING=tau_ci(AS, lk),
    tau_rho_S11=tau_ci(rho[sc], lk[sc]), tau_AS_SING_S11=tau_ci(AS[sc], lk[sc]),
    paired_rho_minus_AS=paired_ci(rho, AS, lk), paired_rho_minus_AS_S11=paired_ci(rho[sc], AS[sc], lk[sc]),
    tau_rho_AS=ktau(rho, AS), spearman_rho_AS=float(spearmanr(rho, AS).statistic),
    partial_kendall=dict(rho_given_AS=float(pk(rho, lk, AS)), AS_given_rho=float(pk(AS, lk, rho))),
    partial_spearman=dict(rho_given_AS=pspear(rho, lk, AS), AS_given_rho=pspear(AS, lk, rho)),
    perm_p=dict(rho=perm_p(rho, lk), AS_SING=perm_p(AS, lk)),
    loglinear_residual_tau=dict(AS_resid_on_rho=ktau(resA, lk), rho_resid_on_AS=ktau(resR, lk)),
    near_tie_swap=dict(rho=ktau(rho, lk2), AS_SING=ktau(AS, lk2)),
    like_for_like=dict(tau_ours_vs_SING_ratio=tau_ci(ours, sing),
                       per_model={PRETTY[m]: dict(ours_IS_over_absAS=float(o), SING=float(s), SING_over_ours=float(s / o))
                                  for m, o, s in zip(ms, ours, sing)}),
)
pc = out['paired_rho_minus_AS']
out['reading_rule_N1'] = dict(paired_CI_covers_0=bool(pc['pct'][0] <= 0 <= pc['pct'][1] and pc['bca'][0] <= 0 <= pc['bca'][1]),
                              may_say='indistinguishable' if pc['pct'][0] <= 0 <= pc['pct'][1] else 'distinguishable', explains_allowed=False)
dump('F_paired_stats.json', out)
for k in ['tau_rho', 'tau_AS_SING', 'paired_rho_minus_AS', 'paired_rho_minus_AS_S11', 'partial_kendall', 'partial_spearman', 'perm_p',
          'loglinear_residual_tau', 'near_tie_swap', 'reading_rule_N1']:
    print(k, out[k])
print('tau_rho_AS', out['tau_rho_AS'], 'like-for-like', out['like_for_like']['tau_ours_vs_SING_ratio'])
