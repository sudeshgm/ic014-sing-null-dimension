"""IC014 E1 GPU pass: for each model, save per-image null-space coordinates and translated features for the
10k ImageNet-val subset (same subset rule as exp1/run_exp1_gpu.py: first 10 sorted file names per class).
Per model writes <out>/<model>_nullfeat.npz (PRIVATE: ImageNet-derived; never commit/publish):
  c      (N, nd) float32  null coordinates  f @ V_null
  z      (N, 768) float32 translator output T(f)
  fsq    (N,)    float32  ||f||^2
  y      (N,)    int16    ground-truth label
  Vn     (m, nd) float32  null basis (SING projectors, tol 1e-5)  [weights-derived]
  AVn    (768, nd) float32 translator weight @ V_null              [weights-derived]
Since SING's default translators are linear, T(f - V_null U U^T c) = z - (c U)(AVn U)^T exactly; all subspace
selection, AS and IS then run on CPU (e1/analyze_e1.py)."""
import argparse, os, sys, time, json, numpy as np, torch
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'exp1'))
from run_exp1_gpu import subset, DS

MODELS = ['resnet', 'resnext', 'dinov2', 'eva02_large', 'vgg16']

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sing-root', required=True); ap.add_argument('--val', required=True)
    ap.add_argument('--labels'); ap.add_argument('--synsets'); ap.add_argument('--out', default='e1_feats')
    ap.add_argument('--models', nargs='*', default=MODELS); ap.add_argument('--per-class', type=int, default=10)
    ap.add_argument('--limit', type=int); ap.add_argument('--bs', type=int, default=100); ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(a.sing_root, 'src'))
    from sing.models.registry import load_model
    from sing.translators.registry import load_default_translator
    from sing.core.projectors import compute_projectors_from_weight
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(a.out, exist_ok=True)
    items = subset(a.val, a.labels, a.synsets, a.per_class, a.limit)
    open(os.path.join(a.out, 'subset_files.txt'), 'w').write(''.join(f'{f}\t{y}\n' for _, f, y in items))
    meta = {}
    for name in a.models:
        fn = os.path.join(a.out, f'{name}_nullfeat.npz')
        if os.path.exists(fn): continue
        t0 = time.time()
        lm = load_model(name, dev)
        T = load_default_translator(translators_root=Path(a.sing_root) / 'translators', registry_path=Path(a.sing_root) / 'translators/registry.yaml',
                                    model_name=name, device=dev).model.eval().to(dev).float()
        assert hasattr(T, 'weight') and T.weight.ndim == 2, 'E1 assumes the linear translator'
        P = compute_projectors_from_weight(lm.wrapper.classifier_weight.detach().float().to(dev)); Vn = P.v_null.float()
        dl = torch.utils.data.DataLoader(DS(items, lm.preprocess), batch_size=a.bs, num_workers=a.workers)
        C, Z, FS, Y = [], [], [], []
        with torch.inference_mode():
            for x, y in dl:
                f = lm.wrapper.extract_features(x.to(dev)).float()
                C.append((f @ Vn).cpu()); Z.append(T(f).cpu()); FS.append((f ** 2).sum(1).cpu()); Y.append(y)
        AVn = (T.weight.detach().float() @ Vn).cpu()
        np.savez(fn, c=torch.cat(C).numpy(), z=torch.cat(Z).numpy(), fsq=torch.cat(FS).numpy(), y=torch.cat(Y).numpy().astype(np.int16),
                 Vn=Vn.cpu().numpy(), AVn=AVn.numpy())
        meta[name] = dict(null_dim=Vn.shape[1], n=len(items), secs=round(time.time() - t0, 1))
        print(json.dumps({name: meta[name]}), flush=True)
        del lm; torch.cuda.empty_cache() if dev.type == 'cuda' else None
    json.dump(meta, open(os.path.join(a.out, 'meta.json'), 'a'))
    print('DONE')

if __name__ == '__main__':
    main()
