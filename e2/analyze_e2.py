"""IC014 E2 CPU analysis: SING's 13 models (whichever are present), no planting.
For every model, using the AS_fair definition registered for E3 (alpha = 0) and SING's released linear translator:
  AS(x, t)   = angle(T(f), t) - angle(T(f~), t), degrees;  AS_SING = mean_x |AS| for full null removal f~ = f - V_nV_n^T f
  AS_fair    = AS_SING / ( (1/R) sum_r mean_x |AS_{R_r}| ),  R = 20 energy-matched random subspaces of the full feature
               space (Haar basis, seed 10000+r, remove q_1..q_{k-1} fully and q_k by fraction beta so that
               E||delta_R||^2 = E||V_n^T f||^2 exactly).
Targets t: (i) the ground-truth class text (normalised mean of SING's 6 ImageNet prompt embeddings, as in Exp1/E1;
this is SING's Fig. 3/6b AS), (ii) every one of SING's 173 attribute concepts (attribute_embeddings..., entry 'mean').
Also IS = mean angle(T(f), T(f~)) and the IS/|AS| ratio (the Fig. 6b quantity).
Outputs (small): e2_models.csv, e2_concepts.csv (model x concept), e2_perclass.csv, e2_rankings.json, e2_claims.json/.md.
Inputs: one or more feature dirs holding <model>_full.npz (from e3/extract_e3_gpu.py or e2/extract_e2_gpu.py); all must
share an identical subset_files.txt."""
import argparse, os, sys, json, csv, math, numpy as np, torch
from scipy.stats import kendalltau, spearmanr

ALL = ['vgg16', 'vgg19', 'densenet', 'resnet', 'dinovit1', 'efficientnet', 'bitresnet', 'resnext', 'convnext',
       'swin_large', 'dinov2', 'deit', 'eva02_large']
FIG6B = {'densenet': 70.71, 'dinovit1': 70.16, 'eva02_large': 54.32, 'bitresnet': 47.21, 'convnext': 37.38,
         'efficientnet': 36.28, 'swin_large': 29.18, 'resnet': 24.81, 'vgg16': 24.26, 'vgg19': 19.89,
         'dinov2': 16.61, 'resnext': 13.88, 'deit': 9.51}   # exp0/fig6b_digitized.json, IS/AS, higher = better
TABLE1 = {'resnet': 12.04, 'efficientnet': 12.38, 'bitresnet': 9.19, 'dinovit1': 5.0, 'resnext': 11.15}  # |AS| toward target, lower = better
FIVE = ['dinovit1', 'resnet', 'resnext', 'efficientnet', 'bitresnet']
CLS = {'jellyfish': 107, 'porcupine': 334, 'arabian_camel': 354, 'sports_car': 817}

def ang(Z, T):  # Z (N,768) unnormalised, T (K,768) unit -> (N,K) degrees
    Zn = torch.nn.functional.normalize(Z, dim=-1)
    return torch.rad2deg(torch.acos(torch.clamp(Zn @ T.T, -1, 1)))

def ang_rows(Z, T):  # per-row target T (N,768)
    Zn = torch.nn.functional.normalize(Z, dim=-1)
    return torch.rad2deg(torch.acos(torch.clamp((Zn * T).sum(-1), -1, 1)))

def analyse_model(d, Vn, TA, TC, draws):
    F = torch.tensor(d['F']).float(); A = torch.tensor(d['A']).float(); Ab = torch.tensor(d['Ab']).float()
    y = torch.tensor(d['y']).long(); N, m = F.shape
    Z = F @ A.T + Ab; C = F @ Vn; Zt = Z - C @ (A @ Vn).T
    Enull = float((C ** 2).sum(1).mean()); rho = Enull / float((F ** 2).sum(1).mean())
    tc = TC[y]
    a0c = ang_rows(Z, tc); ASc = a0c - ang_rows(Zt, tc)               # class-text AS per image
    a0 = ang(Z, TA); ASa = (a0 - ang(Zt, TA)).abs().mean(0)            # per concept mean |AS|
    IS = float(ang_rows(Z, torch.nn.functional.normalize(Zt, dim=-1)).mean())
    rc, ra, ks = torch.zeros(N), torch.zeros(TA.shape[0]), []
    for r in range(draws):
        g = torch.Generator().manual_seed(10_000 + r)
        Q = torch.linalg.qr(torch.randn(m, m, generator=g))[0]
        P = F @ Q; cum = torch.cumsum((P ** 2).mean(0), 0)
        k = min(int(torch.searchsorted(cum, torch.tensor(Enull)).item()), m - 1)
        prev = float(cum[k - 1]) if k > 0 else 0.0
        beta = math.sqrt(max(Enull - prev, 0.0) / float((P[:, k] ** 2).mean()))
        coef = P[:, :k + 1].clone(); coef[:, k] *= beta
        Zr = Z - coef @ (A @ Q[:, :k + 1]).T
        rc += (a0c - ang_rows(Zr, tc)).abs() / draws; ra += (a0 - ang(Zr, TA)).abs().mean(0) / draws; ks.append(k + beta)
    return dict(N=N, m=m, null_dim=Vn.shape[1], rho=rho, k_rand=float(np.mean(ks)), y=y.numpy(),
                ASc_img=ASc.abs().numpy(), rand_img=rc.numpy(), absAS=float(ASc.abs().mean()), absAS_sd_img=float(ASc.abs().std()),
                IS=IS, absAS_rand=float(rc.mean()), AS_fair=float(ASc.abs().mean() / rc.mean()),
                concept_AS=ASa.numpy(), concept_rand=ra.numpy())

def rank(scores, high_good=False):
    return [k for k, _ in sorted(scores.items(), key=lambda kv: -kv[1] if high_good else kv[1])]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('--feats', nargs='+', required=True); ap.add_argument('--sing-root', required=True)
    ap.add_argument('--models', nargs='*', default=ALL); ap.add_argument('--draws', type=int, default=20)
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(a.sing_root, 'src'))
    from sing.core.projectors import compute_projectors_from_weight
    os.makedirs(a.out, exist_ok=True)
    att = np.load(os.path.join(a.sing_root, 'misc/embeddings/attribute_embeddings_broden_style_fp16.npz'), allow_pickle=True)
    cn = [str(c) for c in att['classes']]; pi = [str(p) for p in att['prompts']].index('mean')
    TA = torch.nn.functional.normalize(torch.tensor(att['embeddings'][:, pi].astype(np.float32)), dim=-1)
    emb = np.load(os.path.join(a.sing_root, 'misc/embeddings/imagenet_text_embeddings_fp16.npz'), allow_pickle=True)
    E = torch.nn.functional.normalize(torch.tensor(emb['embeddings'].astype(np.float32)), dim=-1)
    TC = torch.nn.functional.normalize(E.mean(1), dim=-1)
    subsets = {open(os.path.join(f, 'subset_files.txt')).read() for f in a.feats}
    assert len(subsets) == 1, 'feature dirs use different subsets'
    R, missing = {}, []
    for name in a.models:
        fn = next((os.path.join(f, f'{name}_full.npz') for f in a.feats if os.path.exists(os.path.join(f, f'{name}_full.npz'))), None)
        if fn is None: missing.append(name); continue
        d = np.load(fn); W = torch.tensor(d['W']).float()
        Vn = compute_projectors_from_weight(W.double()).v_null.float()
        R[name] = analyse_model(d, Vn, TA, TC, a.draws)
        print(json.dumps({name: {k: R[name][k] for k in ['null_dim', 'rho', 'absAS', 'IS', 'AS_fair', 'k_rand']}}), flush=True)
    M = list(R)
    with open(os.path.join(a.out, 'e2_models.csv'), 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['model', 'm', 'null_dim', 'rho_null', 'absAS_SING_class', 'absAS_sd_over_images', 'IS',
                                        'IS_over_absAS', 'absAS_rand_class', 'AS_fair_class', 'k_rand', 'AS_SING_concept_mean',
                                        'AS_fair_concept_mean', 'fig6b_ratio'])
        for n in M:
            r = R[n]; cf = r['concept_AS'] / r['concept_rand']
            w.writerow([n, r['m'], r['null_dim'], r['rho'], r['absAS'], r['absAS_sd_img'], r['IS'], r['IS'] / r['absAS'], r['absAS_rand'],
                        r['AS_fair'], r['k_rand'], float(r['concept_AS'].mean()), float(cf.mean()), FIG6B[n]])
    with open(os.path.join(a.out, 'e2_concepts.csv'), 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['model', 'concept', 'AS_SING', 'absAS_rand', 'AS_fair'])
        for n in M:
            for j, c in enumerate(cn):
                w.writerow([n, c, float(R[n]['concept_AS'][j]), float(R[n]['concept_rand'][j]), float(R[n]['concept_AS'][j] / R[n]['concept_rand'][j])])
    pc = {}
    with open(os.path.join(a.out, 'e2_perclass.csv'), 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['model', 'class', 'n', 'AS_SING', 'absAS_rand', 'AS_fair'])
        for n in M:
            y = R[n]['y']; s = np.bincount(y, R[n]['ASc_img'], 1000); q = np.bincount(y, R[n]['rand_img'], 1000); cnt = np.bincount(y, minlength=1000)
            ok = cnt > 0; pc[n] = (np.where(ok, s / np.maximum(cnt, 1), np.nan), np.where(ok, s / np.maximum(q, 1e-12), np.nan))
            for c in np.where(ok)[0]: w.writerow([n, int(c), int(cnt[c]), float(pc[n][0][c]), float(q[c] / cnt[c]), float(pc[n][1][c])])
    S = {'AS_SING': {n: R[n]['absAS'] for n in M}, 'AS_fair': {n: R[n]['AS_fair'] for n in M},
         'IS_over_absAS': {n: R[n]['IS'] / R[n]['absAS'] for n in M},
         'AS_SING_concept_mean': {n: float(R[n]['concept_AS'].mean()) for n in M},
         'AS_fair_concept_mean': {n: float((R[n]['concept_AS'] / R[n]['concept_rand']).mean()) for n in M}}
    good = lambda key, n: S[key][n] if key == 'IS_over_absAS' else -S[key][n]   # larger = better (less leakage)
    rk = dict(models=M, missing=missing, fig6b_ranking_best_first=rank({n: FIG6B[n] for n in M}, True), scores=S, rankings_best_first={}, kendall_vs_fig6b={})
    for key in S:
        rk['rankings_best_first'][key] = rank({n: good(key, n) for n in M}, True)
        if len(M) >= 3:
            t = kendalltau([good(key, n) for n in M], [FIG6B[n] for n in M]); rk['kendall_vs_fig6b'][key] = dict(tau=float(t.statistic), p=float(t.pvalue), n=len(M))
    json.dump(rk, open(os.path.join(a.out, 'e2_rankings.json'), 'w'), indent=1)
    # ---- SING comparative claims: holds / reverses for our AS_SING and AS_fair (lower = less leakage) ----
    claims = []
    def add(cid, text, src, sing, fn, need):
        miss = [n for n in need if n not in R]
        row = dict(id=cid, claim=text, source=src, sing=sing)
        if miss: row.update(status='not testable yet', missing=miss)
        else:
            for key in ['AS_SING', 'AS_fair']:
                v, ok = fn(key); row[key] = v; row[f'{key}_verdict'] = 'holds' if ok else 'reverses'
        claims.append(row)
    sc = lambda key: S[key]
    def best_of(target, pool, key):
        v = {n: sc(key)[n] for n in pool}; return v, min(v, key=v.get) == target
    def worst_of(target, pool, key):
        v = {n: sc(key)[n] for n in pool}; return v, max(v, key=v.get) == target
    f5 = {n: FIG6B[n] for n in FIVE}
    add('C1', 'DinoViT has the best IS/AS trade-off (highest ratio) of the 5 models', 'Fig. 3b / Sec. 4.2 / abstract', f5,
        lambda k: best_of('dinovit1', FIVE, k), FIVE)
    add('C2', 'ResNeXt101 has the lowest IS/AS ratio of the 5 models', 'Fig. 3b caption', f5, lambda k: worst_of('resnext', FIVE, k), FIVE)
    add('C3', 'ResNet50 leaks more class semantics into its null space than DinoViT', 'abstract / Fig. 1 / Fig. 4',
        {'resnet': FIG6B['resnet'], 'dinovit1': FIG6B['dinovit1']}, lambda k: ({n: sc(k)[n] for n in ['resnet', 'dinovit1']}, sc(k)['resnet'] > sc(k)['dinovit1']), ['resnet', 'dinovit1'])
    add('C4', 'DinoViT has the least class-relevant leakage among the examined networks', 'Introduction', f5, lambda k: best_of('dinovit1', FIVE, k), FIVE)
    add('C5', 'ResNeXt101 shows high AS (top-2 of the 5)', 'Sec. 4.2', f5,
        lambda k: ({n: sc(k)[n] for n in FIVE}, 'resnext' in sorted(FIVE, key=lambda n: -sc(k)[n])[:2]), FIVE)
    def perclass_spread(k):
        i = 0 if k == 'AS_SING' else 1; v = {n: dict(mean=float(np.nanmean(pc[n][i])), sd=float(np.nanstd(pc[n][i]))) for n in ['resnet', 'dinovit1']}
        return v, v['resnet']['mean'] > v['dinovit1']['mean'] and v['resnet']['sd'] > v['dinovit1']['sd']
    add('C6', 'ResNet50 has larger and more variable per-class AS than DinoViT', 'Fig. 4 / Sec. 4.3', 'qualitative', perclass_spread, ['resnet', 'dinovit1'])
    def under1(k):
        i = 0 if k == 'AS_SING' else 1; fr = float(np.nanmean(pc['dinovit1'][i] < 1)); return dict(frac_classes_below_1=fr), fr > 0.5
    add('C7', 'DinoViT per-class |AS| is typically < 1', 'Sec. 4.3', '|AS| < 1 typical', under1, ['dinovit1'])
    def pct(k):
        i = 0 if k == 'AS_SING' else 1; v = pc['resnet'][i]; o = {}
        for c in ['porcupine', 'sports_car']: o[c] = float(np.nanmean(v <= v[CLS[c]]))
        return dict(percentile_in_resnet=o), all(x >= 0.9 for x in o.values())
    add('C8', 'ResNet50 classes Porcupine and Sports-Car leak substantially more than other classes (>= 90th percentile)', 'Fig. 4 caption', 'highlighted', pct, ['resnet'])
    def nocorr(k):
        i = 0 if k == 'AS_SING' else 1; s = spearmanr(pc['resnet'][i], pc['dinovit1'][i], nan_policy='omit')
        return dict(spearman=float(s.statistic), p=float(s.pvalue)), s.pvalue >= 0.05
    add('C9', 'No significant correlation between ResNet50 and DinoViT per-class AS orderings', 'Sec. 4.3', 'no significant correlation', nocorr, ['resnet', 'dinovit1'])
    def jelly(k):
        i = 0 if k == 'AS_SING' else 1; v = {c: float(pc['dinovit1'][i][CLS[c]]) for c in ['jellyfish', 'arabian_camel']}
        return v, v['jellyfish'] > v['arabian_camel']
    add('C10', 'DinoViT: Jellyfish has substantially larger AS than Arabian Camel (proxy: class-text AS, 10 imgs/class; SING used 30 concepts, ~1300 imgs)', 'Fig. 5 / Sec. 4.3', 'qualitative', jelly, ['dinovit1'])
    def t1(k):
        v = {n: sc(k)[n] for n in TABLE1}; t = kendalltau([v[n] for n in TABLE1], list(TABLE1.values()))
        return dict(values=v, kendall_vs_table1=float(t.statistic)), min(v, key=v.get) == 'dinovit1'
    add('C11', 'Table 1: DinoViT least steerable; ResNet50/EfficientNet/ResNeXt101 susceptible (proxy: null-removal AS, not IS=40 deg text-gradient steps)', 'Table 1 / Sec. 4.4', TABLE1, t1, list(TABLE1))
    def full(k):
        t = kendalltau([-sc(k)[n] for n in M], [FIG6B[n] for n in M]); return dict(tau=float(t.statistic), p=float(t.pvalue), n=len(M)), t.statistic > 0 and t.pvalue < 0.05
    add('C12', 'Fig. 6b / supp. Fig. 12b 13-model IS/AS ordering (holds = Kendall tau > 0 and p < 0.05)', '13-model IS/AS bar plot (digitized in exp0 as Fig. 6b; supp. Fig. 12b)', FIG6B, full, ALL)
    claims.append(dict(id='C13', claim='EVA02 as target space preserves the 5-model ordering', source='Sec. 4.2 / supp. Fig. 12a', sing='same ordering', status='not testable (no EVA02-space translators released)'))
    claims.append(dict(id='C14', claim='Translator robustness: Pearson 0.972 between principal and translated-principal head accuracy', source='Sec. 4.2 / supp. Fig. 12c', sing=0.972, status='not a comparative leakage claim; not tested'))
    json.dump(claims, open(os.path.join(a.out, 'e2_claims.json'), 'w'), indent=1, default=float)
    with open(os.path.join(a.out, 'e2_claims.md'), 'w') as fh:
        fh.write('| id | claim | source | AS_SING | AS_fair |\n|---|---|---|---|---|\n')
        for c in claims:
            fh.write(f"| {c['id']} | {c['claim']} | {c['source']} | {c.get('AS_SING_verdict', c.get('status'))} | {c.get('AS_fair_verdict', c.get('status'))} |\n")
    print(json.dumps(dict(models=M, missing=missing, kendall=rk['kendall_vs_fig6b'], rankings=rk['rankings_best_first']), indent=1))

if __name__ == '__main__':
    main()
