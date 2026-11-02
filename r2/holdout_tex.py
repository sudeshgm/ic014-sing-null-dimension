r"""Write paper_workshop/holdout_numbers.tex from results/fix10_e9_holdout.json.
Defines \HoldoutParagraph (text with the verdict) and \HoldoutTable (one row per candidate).
Usage: python r2/holdout_tex.py [--json PATH] [--out PATH]"""
import argparse, json, os
R = os.path.dirname(os.path.abspath(__file__))
NM = {'AS_SING': r'$\ASS$ (raw)', 'AS_fair': r'$\ASF$ (sub.)', 'M5': r'M5 ($\mathrm{AS}/\sqrt{\rho}$)', 'S4': r'S4 ($\mathrm{AS}/\rho$)'}
f2 = lambda x: f"{x:.2f}".replace('-', '$-$') if x < 0 else f"{x:.2f}"

def lin_ok(v): return v['kendall']['mean'] >= 0.8 and v['kendall']['ci95'][0] >= 0.7
def nl_ok(v): return v['kendall_nonlinear']['mean'] >= 0.6

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--json', default=os.path.join(R, 'results/fix10_e9_holdout.json'))
    ap.add_argument('--out', default=os.path.join(R, '..', 'paper_workshop', 'holdout_numbers.tex')); a = ap.parse_args()
    d = json.load(open(a.json)); C = d['candidates']; pm = d['M1_minus_ASfair_paired']
    rows = []
    for k, v in C.items():
        l, n, fo, fr = v['kendall'], v['kendall_nonlinear'], v['flatness'], v['flatness_revised']
        pas = 'yes' if v['pass_as_registered'] else ('rev.' if v['pass_with_revised_flatness'] else 'no')
        rows.append(f"{NM.get(k, k)} & {f2(l['mean'])} [{f2(l['ci95'][0])}, {f2(l['ci95'][1])}] & {f2(n['mean'])} & "
                    f"{fo['0.1']['n_flat']}/{fo['1.0']['n_flat']} & {fr['0.1']['n_flat']}/{fr['1.0']['n_flat']} & {pas} \\\\")
    table = (r"\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{2.5pt}" "\n"
             r"\begin{tabular}{@{}lccccc@{}}\toprule" "\n"
             r"Score & lin.\ $\tau$ [95\% CI] & nl.\ $\tau$ & flat & rev.\ flat & pass \\\midrule" "\n"
             + "\n".join(rows) + "\n" r"\bottomrule\end{tabular}" "\n"
             r"\caption{E9 holdout (post hoc; $\ASF$ is $\alpha{=}0$-subtracted; water, green, thin, knitted; 50 trials, seed 1; 8 models). Flat $=$ number of the 4 large-null models flat at $\alpha{=}0.1/1$ under the registered rule; rev.\ flat $=$ under the revised score/oracle rule. Pass: lin.\ $\tau\ge0.8$, CI lower $\ge0.7$, nl.\ $\tau\ge0.6$, flat on $\ge3/4$ at both $\alpha$ (yes $=$ registered rule, rev.\ $=$ only with the revised rule).}" "\n"
             r"\label{tab:holdout}\vspace{-0.8em}" "\n" r"\end{table}")
    reg = [k for k in C if C[k]['pass_as_registered']]; rev = [k for k in C if C[k]['pass_with_revised_flatness']]
    tau_ok = [k for k in C if lin_ok(C[k]) and nl_ok(C[k])]
    nm = lambda ks: ', '.join(NM.get(k, k).split(' (')[0] if k in ('M5', 'S4') else NM.get(k, k) for k in ks)
    s = (r"We ran the E9 holdout on four held-out concepts (water, green, thin, knitted; 50 trials at seed 1; linear and nonlinear plants) "
         r"for the six candidates declared in the round-2 pre-registration plus $\ASS$ and $\ASF$, with the E9 pass rule (Table~\ref{tab:holdout}). ")
    s += (f"Only {nm(tau_ok)} meet{'s' if len(tau_ok) == 1 else ''} both $\\tau$ thresholds. " if tau_ok else r"No candidate meets both $\tau$ thresholds. ")
    s += (f"Under the registered flatness rule, {nm(reg)} pass{'es' if len(reg) == 1 else ''}. " if reg else
          r"Under the registered flatness rule no candidate passes; that rule also fails for the oracle (above). ")
    s += (f"Under the revised rule, {nm(rev)} pass{'es' if len(rev) == 1 else ''}. " if rev else r"Under the revised rule no candidate passes either. ")
    lo, hi = pm['lin']['ci95']; nlo, nhi = pm['nl']['ci95']
    s += (f"Paired over trials (1000$\\times$, seed 0), $\\tau$(M1)$-\\tau$($\\ASF$) is {f2(pm['lin']['mean_diff'])} [{f2(lo)}, {f2(hi)}] for the linear plant "
          f"and {f2(pm['nl']['mean_diff'])} [{f2(nlo)}, {f2(nhi)}] for the nonlinear plant")
    beats = lo > 0 and nlo > 0
    s += (", so M1 tracks the planted leakage better than $\\ASF$ under both plants. " if beats else
          (", so M1 beats $\\ASF$ only for the linear plant. " if lo > 0 else ", so M1 does not reliably beat $\\ASF$. "))
    if 'S10' in C:
        fo, fr = C['S10']['flatness'], C['S10']['flatness_revised']
        s += (f"The revised rule also changes which candidate counts as flat: S10, the only candidate flat under the registered rule "
              f"({fo['0.1']['n_flat']}/4 and {fo['1.0']['n_flat']}/4 models at $\\alpha{{=}}0.1/1$), is flat on {fr['0.1']['n_flat']}/4 and {fr['1.0']['n_flat']}/4 under the revised one. ")
    s += (r"The plant is written along the null projection that M1 itself measures, and M1 is given the concept score that defines the plant, while the AS-family scores are not, so this comparison favours M1 by construction. "
          r"One of the four held-out concepts, water, was also an E3 plant concept whose results we had seen before choosing the holdout set; its inclusion was pre-registered, but the set is not fully unseen. "
          r"M1 was picked after the DEV results and the flatness rule was revised after the oracle test, so this holdout is post hoc. "
          r"E9 stays a KILL as registered, and M1 is a candidate for a future pre-registered test, not a validated score.")
    out = ("% generated by r2/holdout_tex.py from r2/results/fix10_e9_holdout.json\n"
           "\\newcommand{\\HoldoutParagraph}{" + s + "}\n\\newcommand{\\HoldoutTable}{" + table + "}\n")
    open(a.out, 'w').write(out); print(out)

if __name__ == '__main__': main()
