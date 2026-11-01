"""Round-2 fix 14 (GPU/ImageNet job): pre-head CLS features of the ViT-L candidate checkpoints on the first 1000 images
of the IC014 10k subset (subset_files.txt order), for scoring with SING's released dinovit1 linear translator on CPU
(r2/fix14_fidelity.py). Each candidate uses its own published eval transform ('native', as in exp0b/eval_top1.py).
The feature is the input to the classifier head (timm: forward_head(pre_logits=True); torchvision: encoder output,
class token; HF: final layernorm output, class token). Outputs are ImageNet-derived: keep private, do not publish.
Usage: python gpu_jobs/fix14_extract_vitl.py --val $IMNET/ILSVRC/Data/CLS-LOC/val --subset <feats>/e3/ret/e3_feats/subset_files.txt --out fix14_feats"""
import argparse, os, sys, json, time, numpy as np, torch
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'exp0b'))
from eval_top1 import build, CANDS   # same checkpoints and native transforms as Exp 0b

def prehead(name, m):
    if name.startswith('timm_'):
        return lambda x: m.forward_head(m.forward_features(x), pre_logits=True)
    if name.startswith('tv_'):
        def f(x):
            x = m._process_input(x); n = x.shape[0]
            x = torch.cat([m.class_token.expand(n, -1, -1), x], dim=1); return m.encoder(x)[:, 0]
        return f
    if name.startswith('hf_'):
        return lambda x: m.h.vit(pixel_values=x).last_hidden_state[:, 0]
    raise ValueError(name)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--val', required=True); ap.add_argument('--subset', required=True)
    ap.add_argument('--out', default='fix14_feats'); ap.add_argument('--n', type=int, default=1000); ap.add_argument('--bs', type=int, default=50)
    ap.add_argument('--candidates', nargs='*', default=CANDS); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    files = [l.split('\t')[0] for l in open(a.subset).read().splitlines()[:a.n]]
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; meta = {}
    for c in a.candidates:
        fn = os.path.join(a.out, c.replace('/', '_') + '.npz')
        if os.path.exists(fn): continue
        t0 = time.time(); m, tf = build(c, 'native'); m = m.eval().to(dev); fwd = prehead(c, m); F = []; L = []
        with torch.inference_mode():
            for i in range(0, len(files), a.bs):
                x = torch.stack([tf(Image.open(os.path.join(a.val, f + '.JPEG')).convert('RGB')) for f in files[i:i + a.bs]]).to(dev)
                F.append(fwd(x).float().cpu()); L.append(m(x).float().argmax(1).cpu())
        np.savez(fn, F=torch.cat(F).numpy(), pred=torch.cat(L).numpy(), files=np.array(files))
        meta[c] = dict(n=len(files), dim=int(F[0].shape[1]), secs=round(time.time() - t0, 1), device=dev); print(json.dumps({c: meta[c]}), flush=True)
        del m; torch.cuda.empty_cache() if dev == 'cuda' else None
    json.dump(meta, open(os.path.join(a.out, 'meta.json'), 'w'), indent=1)

if __name__ == '__main__': main()
