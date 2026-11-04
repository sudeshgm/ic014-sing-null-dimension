"""Round 9 GPU arm: classifier-input features of timm ImageNet-1k models (no translators, no CLIP).

Protocol (same as the cached features of the 13 SING models):
  images   val: the first 10 files per class in sorted file-name order of ILSVRC2012 val (10,000 images); the listing
           (file \\t label) must have sha256 75cdd9a7c10807c3d565ad83102dd3865c6398220707de249d6dcec96a65a42f
           train (optional, --train): the first 10 sorted files in each sorted synset folder of ILSVRC2012 train
           labels: index of the synset in the sorted list of the 1000 synsets
  model    timm.create_model(id, pretrained=True).eval(); preprocessing timm.data.resolve_model_data_config(model) +
           timm.data.create_transform(**cfg, is_training=False)
  feature  input of model.get_classifier(), captured by a forward pre-hook and flattened (B, d); float32, no autocast
  head     W = classifier weight (1x1 conv flattened), b = classifier bias (zeros if none); float32
  check    top-1 of argmax(F W^T + b) and of the model's own logits must agree to 1e-4 (stored)
Output (PRIVATE, ImageNet-derived; never publish): <out>/{val,train}/<id with '.' replaced by '__'>.npz with F (N, d) f32,
y (N,) int16, W (1000, d) f32, b (1000,) f32; <out>/manifest.json with sha256 of every npz, shapes, top-1, versions, times.

  python extract_r9_gpu.py --val $IMNET/ILSVRC/Data/CLS-LOC/val --labels $IMNET/LOC_val_solution.csv \\
      --synsets $IMNET/LOC_synset_mapping.txt --train $IMNET/ILSVRC/Data/CLS-LOC/train --models-file gpu_models.txt --out r9_gpu_feats
"""
import argparse, csv, hashlib, json, os, platform, time
import numpy as np
import torch
from PIL import Image

VAL_SHA = '75cdd9a7c10807c3d565ad83102dd3865c6398220707de249d6dcec96a65a42f'


def val_items(val, labels, synsets, per_class=10):
    syn = sorted(l.split()[0] for l in open(synsets)); idx = {k: i for i, k in enumerate(syn)}
    rows = sorted((r[0], idx[r[1].split()[0]]) for r in list(csv.reader(open(labels)))[1:])
    cnt, out = {}, []
    for f, y in rows:
        if cnt.get(y, 0) < per_class:
            out.append((os.path.join(val, f + '.JPEG'), f, y)); cnt[y] = cnt.get(y, 0) + 1
    return out, syn


def train_items(train, syn, per_class=10):
    out = []
    for i, s in enumerate(syn):
        fs = sorted(os.listdir(os.path.join(train, s)))[:per_class]
        out += [(os.path.join(train, s, f), f, i) for f in fs]
    return out


class DS(torch.utils.data.Dataset):
    def __init__(s, items, tf): s.items, s.tf = items, tf
    def __len__(s): return len(s.items)
    def __getitem__(s, i): p, _, y = s.items[i]; return s.tf(Image.open(p).convert('RGB')), y


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def extract(model, items, tf, dev, bs, workers):
    head = model.get_classifier(); cap = {}
    def hook(_m, args):
        x = args[0]; cap['f'] = torch.flatten(x, 1) if x.ndim > 2 else x
    h = head.register_forward_pre_hook(hook)
    dl = torch.utils.data.DataLoader(DS(items, tf), batch_size=bs, num_workers=workers)
    F, Y, top_model = [], [], []
    try:
        with torch.inference_mode():
            for x, y in dl:
                out = model(x.to(dev)).float(); F.append(cap['f'].float().cpu()); Y.append(y); top_model.append(out.argmax(1).cpu())
    finally:
        h.remove()
    return torch.cat(F).numpy(), torch.cat(Y).numpy().astype(np.int16), torch.cat(top_model).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--val', required=True); ap.add_argument('--labels', required=True); ap.add_argument('--synsets', required=True)
    ap.add_argument('--train'); ap.add_argument('--models-file', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--bs', type=int, default=100); ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--only', nargs='*'); ap.add_argument('--smoke', action='store_true', help='skip the listing check (fake data only)')
    a = ap.parse_args()
    import timm
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vi, syn = val_items(a.val, a.labels, a.synsets)
    listing = ''.join(f'{f}\t{y}\n' for _, f, y in vi)
    got = hashlib.sha256(listing.encode()).hexdigest()
    assert a.smoke or got == VAL_SHA, f'val subset differs from the registered 10k listing ({got})'
    splits = {'val': vi}
    if a.train:
        splits['train'] = train_items(a.train, syn)
    for s in splits: os.makedirs(os.path.join(a.out, s), exist_ok=True)
    open(os.path.join(a.out, 'val_subset_files.txt'), 'w').write(listing)
    if 'train' in splits:
        open(os.path.join(a.out, 'train_subset_files.txt'), 'w').write(''.join(f'{f}\t{y}\n' for _, f, y in splits['train']))
    mp = os.path.join(a.out, 'manifest.json'); man = json.load(open(mp)) if os.path.exists(mp) else {}
    man['_env'] = dict(timm=timm.__version__, torch=torch.__version__, python=platform.python_version(),
                       gpu=torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu', val_listing_sha256=got, smoke=a.smoke)
    ids = [l.split()[0] for l in open(a.models_file) if l.strip() and not l.startswith('#')]
    if a.only: ids = [i for i in ids if i in a.only]
    for mid in ids:
        tag = mid.replace('.', '__'); rec = man.get(mid, {})
        todo = [s for s in splits if not os.path.exists(os.path.join(a.out, s, tag + '.npz'))]
        if not todo: continue
        t0 = time.time()
        try:
            model = timm.create_model(mid, pretrained=True).to(dev).eval()
            cfg = timm.data.resolve_model_data_config(model); tf = timm.data.create_transform(**cfg, is_training=False)
            head = model.get_classifier(); W = head.weight.detach().float().cpu(); W = W.reshape(W.shape[0], -1)
            b = head.bias.detach().float().cpu() if getattr(head, 'bias', None) is not None else torch.zeros(W.shape[0])
            rec.update(model_id=mid, data_config={k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
                       head_type=type(head).__name__, d=int(W.shape[1]))
            for s in todo:
                F, y, top_m = extract(model, splits[s], tf, dev, a.bs, a.workers)
                L = F @ W.numpy().T + b.numpy(); top_f = L.argmax(1)
                p = os.path.join(a.out, s, tag + '.npz')
                np.savez(p, F=F.astype(np.float32), y=y, W=W.numpy().astype(np.float32), b=b.numpy().astype(np.float32))
                rec[s] = dict(n=int(len(y)), d=int(F.shape[1]), top1_from_features=float(np.mean(top_f == y)),
                              top1_model=float(np.mean(top_m == y)), argmax_agreement=float(np.mean(top_f == top_m)),
                              sha256=sha256(p), secs=round(time.time() - t0, 1))
                print(json.dumps({mid: {s: rec[s]}}), flush=True)
            del model; torch.cuda.empty_cache() if dev.type == 'cuda' else None
        except Exception as e:  # record and continue; a failed model is reported, not dropped silently
            rec['error'] = f'{type(e).__name__}: {e}'[:500]; print(json.dumps({mid: rec['error']}), flush=True)
        man[mid] = rec; json.dump(man, open(mp, 'w'), indent=1)
    print('DONE')


if __name__ == '__main__':
    main()
