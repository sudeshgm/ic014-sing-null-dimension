"""IC014 Exp 1 (GPU pass): features on a 10k ImageNet-val subset (10/class) for SING models, then AS/IS after
removing random d-dim subspaces of each model's own null space, using SING's released linear translators and
SING's released CLIP text embeddings (misc/embeddings/imagenet_text_embeddings_fp16.npz, 6 prompts/class).

Outputs (no images, no raw features unless --save-features):
  exp1_per_seed.json   one row per (model, d, seed): mean|AS|, meanAS, meanIS, rho_d, per-prompt mean|AS|
  exp1_full_null_per_image.npz  per-image AS/IS (float16) at the full null space for every model (for bootstraps)
  subset_files.txt     image ids used (ImageNet file names only)
AS uses the unit-normalized mean of the 6 prompt embeddings of the ground-truth class (prompt texts are not
shipped by SING; per-prompt |AS| is also recorded). AS/IS formulas = sing.core.metrics. Null basis =
sing.core.projectors.compute_projectors_from_weight (tol 1e-5).
"""
import argparse, csv, json, os, sys, time
import numpy as np, torch
from PIL import Image

EXP1 = ['resnet', 'resnext', 'bitresnet', 'dinov2']
D_LIST = [24, 48, 100, 280, 536, 1048]
MATCHED = ['deit', 'eva02_large', 'convnext', 'densenet']
OTHERS = ['vgg16', 'vgg19', 'efficientnet', 'swin_large', 'dinovit1']

def subset(val, labels, synsets, per_class, limit):
    if labels:
        syn = sorted(l.split()[0] for l in open(synsets)); idx = {k: i for i, k in enumerate(syn)}
        rows = sorted((r[0], idx[r[1].split()[0]]) for r in list(csv.reader(open(labels)))[1:])
        items = [(os.path.join(val, f + '.JPEG'), f, y) for f, y in rows]
    else:  # class-folder layout
        cls = sorted(d for d in os.listdir(val) if os.path.isdir(os.path.join(val, d)))
        items = [(os.path.join(val, c, f), f, i) for i, c in enumerate(cls) for f in sorted(os.listdir(os.path.join(val, c)))]
    cnt, out = {}, []
    for p, f, y in items:
        if cnt.get(y, 0) < per_class: out.append((p, f, y)); cnt[y] = cnt.get(y, 0) + 1
    return out[:limit] if limit else out

class DS(torch.utils.data.Dataset):
    def __init__(s, items, tf): s.items, s.tf = items, tf
    def __len__(s): return len(s.items)
    def __getitem__(s, i): p, _, y = s.items[i]; return s.tf(Image.open(p).convert('RGB')), y

def angle(a, b):
    a = torch.nn.functional.normalize(a, dim=-1); b = torch.nn.functional.normalize(b, dim=-1)
    return torch.rad2deg(torch.acos(torch.clamp((a * b).sum(-1), -1, 1)))

def scores(T, f, ft, txt_mean, txt_all):
    zo, zp = T(f), T(ft)
    IS = angle(zo, zp)
    AS = angle(zo, txt_mean) - angle(zp, txt_mean)
    ASp = torch.stack([angle(zo, txt_all[:, k]) - angle(zp, txt_all[:, k]) for k in range(txt_all.shape[1])], 1)
    return AS, IS, ASp

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sing-root', required=True); ap.add_argument('--val', required=True)
    ap.add_argument('--labels'); ap.add_argument('--synsets'); ap.add_argument('--out', default='exp1_out')
    ap.add_argument('--models', nargs='*', default=EXP1 + MATCHED + OTHERS)
    ap.add_argument('--per-class', type=int, default=10); ap.add_argument('--limit', type=int)
    ap.add_argument('--seeds', type=int, default=20); ap.add_argument('--bs', type=int, default=100)
    ap.add_argument('--workers', type=int, default=8); ap.add_argument('--save-features', action='store_true')
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(a.sing_root, 'src'))
    from sing.models.registry import load_model
    from sing.translators.registry import load_default_translator
    from sing.core.projectors import compute_projectors_from_weight
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(a.out, exist_ok=True)
    items = subset(a.val, a.labels, a.synsets, a.per_class, a.limit)
    open(os.path.join(a.out, 'subset_files.txt'), 'w').write('\n'.join(f'{f}\t{y}' for _, f, y in items))
    emb = np.load(os.path.join(a.sing_root, 'misc/embeddings/imagenet_text_embeddings_fp16.npz'), allow_pickle=True)
    E = torch.tensor(emb['embeddings'].astype(np.float32))  # 1000 x 6 x 768
    E = torch.nn.functional.normalize(E, dim=-1); Em = torch.nn.functional.normalize(E.mean(1), dim=-1)
    pj = os.path.join(a.out, 'exp1_per_seed.json')
    rows = json.load(open(pj)) if os.path.exists(pj) else []
    done = {r['model'] for r in rows}
    per_image = dict(np.load(os.path.join(a.out, 'exp1_full_null_per_image.npz'))) if os.path.exists(os.path.join(a.out, 'exp1_full_null_per_image.npz')) else {}
    for name in a.models:
        if name in done: continue
        t0 = time.time()
        lm = load_model(name, dev)
        tr = load_default_translator(translators_root=__import__('pathlib').Path(a.sing_root) / 'translators',
                                     registry_path=__import__('pathlib').Path(a.sing_root) / 'translators/registry.yaml',
                                     model_name=name, device=dev)
        T = tr.model.eval().to(dev).float()
        dl = torch.utils.data.DataLoader(DS(items, lm.preprocess), batch_size=a.bs, num_workers=a.workers)
        F, Y = [], []
        with torch.inference_mode():
            for x, y in dl:
                F.append(lm.wrapper.extract_features(x.to(dev)).float()); Y.append(y)
        F = torch.cat(F); Y = torch.cat(Y)
        if a.save_features: np.save(os.path.join(a.out, f'feats_{name}.npy'), F.cpu().half().numpy())
        W = lm.wrapper.classifier_weight.detach().float()
        P = compute_projectors_from_weight(W.to(dev)); Vn = P.v_null.float()
        nd = Vn.shape[1]
        ty, tya = Em[Y].to(dev), E[Y].to(dev)
        fe = (F ** 2).sum(1).mean()
        conds = []
        if name in EXP1: conds += [(d, s) for d in D_LIST if d < nd for s in range(a.seeds)]
        if name == 'densenet': conds += [(24, s) for s in range(a.seeds)]
        conds += [(nd, -1)]  # full null space, deterministic
        with torch.inference_mode():
            logit_o = lm.wrapper.logits_from_features(F[:256].to(dev))
            for d, s in conds:
                if s < 0: Q = Vn
                else:
                    g = torch.Generator().manual_seed(1000 * d + s)
                    R = torch.randn(nd, d, generator=g).to(dev)
                    Q = Vn @ torch.linalg.qr(R)[0]
                proj = (F @ Q) @ Q.T
                Ft = F - proj
                AS, IS, ASp = scores(T, F, Ft, ty, tya)
                rho = float((proj ** 2).sum(1).mean() / fe)
                dlog = float((lm.wrapper.logits_from_features(Ft[:256]) - logit_o).abs().max())
                rows.append(dict(model=name, null_dim=nd, d=d, seed=s, full=(s < 0), n=len(F),
                                 abs_AS=float(AS.abs().mean()), AS=float(AS.mean()), IS=float(IS.mean()),
                                 IS_over_absAS=float(IS.mean() / AS.abs().mean()), rho=rho,
                                 abs_AS_per_prompt=[float(v) for v in ASp.abs().mean(0)], max_logit_change=dlog))
                if s < 0:
                    per_image[f'{name}_AS'] = AS.cpu().half().numpy(); per_image[f'{name}_IS'] = IS.cpu().half().numpy()
        per_image['labels'] = Y.numpy().astype(np.int16)
        print(json.dumps(dict(model=name, null_dim=nd, n=len(F), conds=len(conds), secs=round(time.time() - t0, 1),
                              full=[r for r in rows if r['model'] == name and r['full']][0])), flush=True)
        json.dump(rows, open(pj, 'w')); np.savez_compressed(os.path.join(a.out, 'exp1_full_null_per_image.npz'), **per_image)
        del lm, F; torch.cuda.empty_cache() if dev.type == 'cuda' else None
    print('DONE', pj)

if __name__ == '__main__':
    main()
