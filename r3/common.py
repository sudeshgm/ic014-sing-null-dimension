"""Shared helpers for round-3 analyses (PREREG_R3.md). CPU only, numpy/scipy."""
import csv, os, json
import numpy as np
from scipy.stats import kendalltau, norm

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(R, 'r3', 'results')
PRIV_DEFAULT = os.environ.get('IC014_PRIVATE', os.path.join(R, '_private'))
DIGI = {'DenseNet': 'densenet', 'DinoViT': 'dinovit1', 'EVA02Large': 'eva02_large', 'BitResNet': 'bitresnet',
        'ConvNext': 'convnext', 'EfficientNet': 'efficientnet', 'SwinLarge': 'swin_large', 'ResNet50': 'resnet',
        'VGG16': 'vgg16', 'VGG19': 'vgg19', 'DinoV2': 'dinov2', 'ResNext101': 'resnext', 'DeiT': 'deit'}
PRETTY = {v: k for k, v in DIGI.items()}
RHO0, RHO0_SENS = 0.005, 0.001
S11_DROP = ('densenet', 'bitresnet')


def load_models():
    """Model table (fix2_models.csv) in its row order + SING digitised IS/AS ratio."""
    M = {r['model']: r for r in csv.DictReader(open(os.path.join(R, 'r2/results/fix2_models.csv')))}
    sing = {DIGI[r['model']]: float(r['ratio_IS_over_AS']) for r in csv.DictReader(open(os.path.join(R, 'exp0/fig6b_digitized.csv')))}
    ms = list(M)
    col = lambda k: np.array([float(M[m][k]) for m in ms])
    return ms, col, np.array([sing[m] for m in ms])


def ktau(x, y):
    return kendalltau(x, y).statistic   # tau-b


def boot_idx(n, B=2000, seed=0, min_distinct=4):
    r = np.random.default_rng(seed); out = []
    for _ in range(B):
        i = r.integers(0, n, n)
        if len(set(i.tolist())) >= min_distinct: out.append(i)
    return out


def bca(theta, boots, jack, alpha=0.05):
    """BCa interval from bootstrap replicates and jackknife (delete-one) values."""
    boots = np.asarray([b for b in boots if np.isfinite(b)]); jack = np.asarray(jack)
    p = np.mean(boots < theta) + 0.5 * np.mean(boots == theta)
    p = min(max(p, 1.0 / (len(boots) + 1)), 1 - 1.0 / (len(boots) + 1))
    z0 = norm.ppf(p); jm = jack.mean(); num = np.sum((jm - jack) ** 3); den = 6 * np.sum((jm - jack) ** 2) ** 1.5
    a = num / den if den > 0 else 0.0
    qs = []
    for z in (norm.ppf(alpha / 2), norm.ppf(1 - alpha / 2)):
        qs.append(norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z))))
    return [float(np.percentile(boots, 100 * q)) for q in qs], dict(z0=float(z0), a=float(a))


def tau_ci(x, y, B=2000, seed=0):
    """Kendall tau-b with model-bootstrap percentile and BCa CIs (delete-one-model jackknife)."""
    x, y = np.asarray(x, float), np.asarray(y, float); n = len(x); t = ktau(x, y)
    bs = [ktau(x[i], y[i]) for i in boot_idx(n, B, seed)]; bs = [b for b in bs if np.isfinite(b)]
    jk = [ktau(np.delete(x, j), np.delete(y, j)) for j in range(n)]
    ci_b, _ = bca(t, bs, jk)
    return dict(tau=float(t), pct=[float(v) for v in np.percentile(bs, [2.5, 97.5])], bca=ci_b, n=n, B_used=len(bs))


def paired_ci(x1, x2, y, B=2000, seed=0):
    """CI for tau(x1,y) - tau(x2,y) under the same model resamples."""
    x1, x2, y = (np.asarray(v, float) for v in (x1, x2, y)); n = len(y)
    d0 = ktau(x1, y) - ktau(x2, y); D = []
    for i in boot_idx(n, B, seed):
        a, b = ktau(x1[i], y[i]), ktau(x2[i], y[i])
        if np.isfinite(a) and np.isfinite(b): D.append(a - b)
    jk = [ktau(np.delete(x1, j), np.delete(y, j)) - ktau(np.delete(x2, j), np.delete(y, j)) for j in range(n)]
    ci_b, _ = bca(d0, D, jk)
    return dict(diff=float(d0), pct=[float(v) for v in np.percentile(D, [2.5, 97.5])], bca=ci_b, B_used=len(D))


def perm_p(x, y, n=20000, seed=1):
    r = np.random.default_rng(seed); t = ktau(x, y)
    null = np.array([ktau(r.permutation(np.asarray(x)), y) for _ in range(n)])
    return float((np.sum(null >= t - 1e-12) + 1) / (n + 1))


def dump(name, obj):
    os.makedirs(RES, exist_ok=True)
    json.dump(obj, open(os.path.join(RES, name), 'w'), indent=1, default=float)
