"""PREREG_R7 image side: IN-9 Backgrounds Challenge pairs, the pixel-shuffle artefact control and the nuisance views.

Conditions per foreground image i (all 224x224, same foreground pixels at the same place):
  X  = mixed_next/<fg>_bg_<synset>  (background from an image of the NEXT IN-9 class; its fine ImageNet synset is in the file name)
  S  = mixed_same/<same file name>  (background from another image of the SAME IN-9 class; the file name's bg tag does not apply)
  Xs, Ss = X, S with the background pixels (fg_mask == False) randomly permuted among themselves (same pixel multiset,
           foreground untouched; seed PIX_SEED + i) -- the same-pixels artefact control.
Nuisance view (level L, view v, image i): one parameter draw theta(L, i, v) from rng(VIEW_SEED, L, i, v), applied identically to
X, S, Xs, Ss: RandomResizedCrop -> flip -> colour jitter -> optional grayscale -> optional Gaussian blur -> JPEG re-encode.
Level 0 = clean (no augmentation, one view)."""
import io, os, re, json, numpy as np
from PIL import Image, ImageFilter, ImageEnhance

VIEW_SEED, PIX_SEED, SUBSET_SEED = 100700, 100710, 100701
CONDS = ('X', 'S', 'Xs', 'Ss')
LEVELS = {  # crop scale, aspect, flip p, brightness/contrast/saturation, hue, gray p, blur p, blur sigma max, jpeg q range
    1: dict(scale=(0.85, 1.0), ratio=(0.9, 1.11), flip=0.5, bcs=0.10, hue=0.00, gray=0.0, blur=0.0, sig=0.0, q=(85, 95)),
    2: dict(scale=(0.60, 1.0), ratio=(0.8, 1.25), flip=0.5, bcs=0.25, hue=0.02, gray=0.05, blur=0.3, sig=1.5, q=(60, 95)),
    3: dict(scale=(0.35, 1.0), ratio=(0.75, 1.33), flip=0.5, bcs=0.40, hue=0.05, gray=0.10, blur=0.5, sig=2.5, q=(30, 95)),
}
NV = 4


def items(root, per_class=100, seed=SUBSET_SEED, synsets=None):
    """sorted foreground files per IN-9 class, then a seeded subset of per_class each. Returns list of dicts."""
    out = []; rng = np.random.default_rng(seed); idx = {s: i for i, s in enumerate(synsets)} if synsets else None
    for c in sorted(os.listdir(os.path.join(root, 'mixed_next', 'val'))):
        fs = sorted(os.listdir(os.path.join(root, 'mixed_next', 'val', c)))
        pick = sorted(rng.choice(len(fs), per_class, replace=False)) if per_class else range(len(fs))
        for j in pick:
            f = fs[j]; m = re.match(r'fg_(n\d+_\d+)_bg_(n\d+)_(\d+)\.JPEG', f)
            out.append(dict(cls=c, g=int(c[:2]), file=f, fg=m.group(1), bg_syn=m.group(2), bg_idx=idx[m.group(2)] if idx else None))
    return out


def load_pair(root, it):
    X = np.asarray(Image.open(os.path.join(root, 'mixed_next', 'val', it['cls'], it['file'])).convert('RGB'))
    S = np.asarray(Image.open(os.path.join(root, 'mixed_same', 'val', it['cls'], it['file'])).convert('RGB'))
    M = np.load(os.path.join(root, 'fg_mask', 'val', it['cls'], it['fg'] + '.npy')).astype(bool)
    return X, S, M


def pixel_shuffle(img, mask, rng_or_perm):
    """permute the background pixels (mask False) among themselves; foreground unchanged"""
    out = img.copy(); bg = np.where(~mask.reshape(-1))[0]
    perm = rng_or_perm if isinstance(rng_or_perm, np.ndarray) else rng_or_perm.permutation(len(bg))
    flat = out.reshape(-1, 3); flat[bg] = img.reshape(-1, 3)[bg[perm]]; return out


def conditions(root, it, i):
    X, S, M = load_pair(root, it); rng = np.random.default_rng([PIX_SEED, i]); nb = int((~M).sum())
    p = rng.permutation(nb)  # the same permutation for X and S (same mask)
    return dict(X=X, S=S, Xs=pixel_shuffle(X, M, p), Ss=pixel_shuffle(S, M, p))


def draw(level, i, v, size=224):
    """theta(L, i, v): one nuisance parameter draw (shared by all conditions of image i in view v)"""
    P = LEVELS[level]; r = np.random.default_rng([VIEW_SEED, level, i, v]); area = size * size
    for _ in range(10):
        a = area * r.uniform(*P['scale']); lr = np.exp(r.uniform(np.log(P['ratio'][0]), np.log(P['ratio'][1])))
        w = int(round(np.sqrt(a * lr))); h = int(round(np.sqrt(a / lr)))
        if 0 < w <= size and 0 < h <= size: break
    else: w = h = size
    x0 = int(r.integers(0, size - w + 1)); y0 = int(r.integers(0, size - h + 1))
    return dict(box=(x0, y0, x0 + w, y0 + h), flip=bool(r.random() < P['flip']),
                b=float(r.uniform(1 - P['bcs'], 1 + P['bcs'])), c=float(r.uniform(1 - P['bcs'], 1 + P['bcs'])),
                s=float(r.uniform(1 - P['bcs'], 1 + P['bcs'])), h=float(r.uniform(-P['hue'], P['hue'])),
                gray=bool(r.random() < P['gray']), blur=float(r.uniform(0.1, P['sig'])) if r.random() < P['blur'] else 0.0,
                q=int(r.integers(P['q'][0], P['q'][1] + 1)))


def _hue(im, h):
    if h == 0: return im
    H, S, V = im.convert('HSV').split(); Hn = np.asarray(H, np.uint8).astype(np.int16)
    Hn = ((Hn + int(round(h * 255))) % 256).astype(np.uint8); return Image.merge('HSV', (Image.fromarray(Hn), S, V)).convert('RGB')


def apply(arr, th, size=224):
    im = Image.fromarray(arr).crop(th['box']).resize((size, size), Image.BILINEAR)
    if th['flip']: im = im.transpose(Image.FLIP_LEFT_RIGHT)
    im = ImageEnhance.Brightness(im).enhance(th['b']); im = ImageEnhance.Contrast(im).enhance(th['c'])
    im = ImageEnhance.Color(im).enhance(th['s']); im = _hue(im, th['h'])
    if th['gray']: im = im.convert('L').convert('RGB')
    if th['blur'] > 0: im = im.filter(ImageFilter.GaussianBlur(th['blur']))
    b = io.BytesIO(); im.save(b, format='JPEG', quality=th['q']); b.seek(0); return Image.open(b).convert('RGB')


def view_images(root, it, i, level, v, conds=None):
    """PIL images of every condition for (level, i, v); level 0 = clean"""
    C = conds or conditions(root, it, i)
    if level == 0: return {k: Image.fromarray(a) for k, a in C.items()}
    th = draw(level, i, v); return {k: apply(a, th) for k, a in C.items()}
