"""PREREG_R7 feature extraction (CPU or GPU; identical code). For each model and each foreground image i, the penultimate
features (SING's classifier-input hook, the same definition as the cached 10k features) of every condition in CONDS under
the clean view (level 0) and NV = 4 independent nuisance views at levels 1-3. Output (PRIVATE, ImageNet-derived; never commit):
  <out>/<model>_r7.npy   float32 (52, N, d): slot = index of (level, view, cond) in SLOTS
  <out>/<model>_r7_head.npz  W (1000, d), bW (1000,)
  <out>/items_r7.json  the N foreground items (order of the N axis) and SLOTS
Resumable: progress in <out>/<model>_r7.progress (number of finished items).
Usage: python r7/extract_r7.py --sing-root _ext/sing --in9 _ext/in9/bg_challenge --out _ext/r7_feats --models resnet efficientnet"""
import argparse, os, sys, json, time, glob, numpy as np, torch
from pathlib import Path
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H)
import views_r7 as vw  # noqa: E402

SLOTS = [(0, 0, c) for c in vw.CONDS] + [(L, v, c) for L in (1, 2, 3) for v in range(vw.NV) for c in vw.CONDS]


def synsets():
    import timm
    f = glob.glob(os.path.join(os.path.dirname(timm.__file__), 'data', '_info', 'imagenet_synsets.txt'))[0]
    return [l.strip() for l in open(f)]


class DS(torch.utils.data.Dataset):
    def __init__(s, root, its, pre): s.root, s.its, s.pre = root, its, pre
    def __len__(s): return len(s.its)
    def __getitem__(s, i):
        C = vw.conditions(s.root, s.its[i], i); out = []; cache = {}
        for L, v, c in SLOTS:
            if (L, v) not in cache: cache[(L, v)] = vw.view_images(s.root, s.its[i], i, L, v, C)
            out.append(s.pre(cache[(L, v)][c]))
        return torch.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sing-root', required=True); ap.add_argument('--in9', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--models', nargs='+', required=True); ap.add_argument('--per-class', type=int, default=100)
    ap.add_argument('--threads', type=int, default=6); ap.add_argument('--workers', type=int, default=2); ap.add_argument('--items-per-batch', type=int, default=2)
    ap.add_argument('--limit', type=int, help='timing runs only: stop after this many items (output marked partial)')
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    sys.path.insert(0, os.path.join(a.sing_root, 'src')); from sing.models.registry import load_model
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); os.makedirs(a.out, exist_ok=True)
    its = vw.items(a.in9, a.per_class, synsets=synsets()); N = len(its)
    json.dump(dict(items=its, slots=SLOTS, per_class=a.per_class), open(os.path.join(a.out, 'items_r7.json'), 'w'))
    for name in a.models:
        fn = os.path.join(a.out, f'{name}_r7.npy'); pf = os.path.join(a.out, f'{name}_r7.progress')
        done = int(open(pf).read()) if os.path.exists(pf) else 0
        if done >= N: print(json.dumps({name: 'done'}), flush=True); continue
        lm = load_model(name, dev); d = lm.wrapper.feature_dim
        with torch.inference_mode():
            W = lm.wrapper.classifier_weight.detach().float()
            bW = lm.wrapper.logits_from_features(torch.zeros(1, W.shape[1], device=W.device, dtype=W.dtype)).float()[0]
        np.savez(os.path.join(a.out, f'{name}_r7_head.npz'), W=W.cpu().numpy(), bW=bW.cpu().numpy())
        F = np.lib.format.open_memmap(fn, mode='r+' if os.path.exists(fn) else 'w+', dtype=np.float32, shape=(len(SLOTS), N, d))
        sub = list(range(done, N if a.limit is None else min(N, done + a.limit)))
        dl = torch.utils.data.DataLoader(torch.utils.data.Subset(DS(a.in9, its, lm.preprocess), sub), batch_size=a.items_per_batch, num_workers=a.workers)
        t0 = time.time(); k = done
        with torch.inference_mode():
            for x in dl:
                b = x.shape[0]; f = lm.wrapper.extract_features(x.reshape(-1, *x.shape[2:]).to(dev)).float().cpu().numpy()
                F[:, k:k + b] = f.reshape(b, len(SLOTS), d).transpose(1, 0, 2); k += b
                if k % 20 == 0 or k == sub[-1] + 1:
                    F.flush(); open(pf, 'w').write(str(k))
                    print(json.dumps({name: k, 'of': N, 'img_per_s': round((k - done) * len(SLOTS) / (time.time() - t0), 1)}), flush=True)
        F.flush(); open(pf, 'w').write(str(k)); del lm
        print(json.dumps({name: 'finished' if k >= N else 'partial', 'items': k, 'secs': round(time.time() - t0, 1)}), flush=True)


if __name__ == '__main__':
    main()
