"""IC014 E3 GPU pass (feature extraction only; all planting/metrics run on CPU in e3/analyze_e3.py).
Same 10k ImageNet-val subset as Exp1/E1 (first 10 sorted files per class).
Writes to --out (PRIVATE: ImageNet-derived; never commit/publish):
  <model>_full.npz : F (N,m) f32 penultimate features f; y (N,) int16; W (1000,m) head weight; bW (1000,) head bias;
                     A (768,m), Ab (768,) SING default linear translator; z16 (N,768) f16 T(f) (check only)
  clip_img.npz     : e (N,768) f32 L2-normalised CLIP image embeddings from the Karlo-v1-alpha image encoder
                     (CLIP ViT-L/14, the encoder of the UnCLIP model SING uses; same space as SING's text embeddings)
  subset_files.txt, meta.json"""
import argparse, os, sys, time, json, numpy as np, torch
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'exp1'))
from run_exp1_gpu import subset, DS

MODELS = ['resnet', 'resnext', 'dinov2', 'bitresnet', 'deit', 'eva02_large', 'convnext', 'densenet']
KARLO = 'kakaobrain/karlo-v1-alpha-image-variations'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sing-root', required=True); ap.add_argument('--val', required=True)
    ap.add_argument('--labels'); ap.add_argument('--synsets'); ap.add_argument('--out', default='e3_feats')
    ap.add_argument('--models', nargs='*', default=MODELS); ap.add_argument('--per-class', type=int, default=10)
    ap.add_argument('--limit', type=int); ap.add_argument('--bs', type=int, default=100); ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--no-clip', action='store_true')
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(a.sing_root, 'src'))
    from sing.models.registry import load_model
    from sing.translators.registry import load_default_translator
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(a.out, exist_ok=True)
    items = subset(a.val, a.labels, a.synsets, a.per_class, a.limit)
    open(os.path.join(a.out, 'subset_files.txt'), 'w').write(''.join(f'{f}\t{y}\n' for _, f, y in items))
    mp = os.path.join(a.out, 'meta.json'); meta = json.load(open(mp)) if os.path.exists(mp) else {}
    if not a.no_clip and not os.path.exists(os.path.join(a.out, 'clip_img.npz')):
        from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
        t0 = time.time()
        enc = CLIPVisionModelWithProjection.from_pretrained(KARLO, subfolder='image_encoder').to(dev).eval()
        proc = CLIPImageProcessor.from_pretrained(KARLO, subfolder='feature_extractor')
        tf = lambda im: proc(images=im, return_tensors='pt')['pixel_values'][0]
        dl = torch.utils.data.DataLoader(DS(items, tf), batch_size=a.bs, num_workers=a.workers)
        E = []
        with torch.inference_mode():
            for x, _ in dl: E.append(torch.nn.functional.normalize(enc(pixel_values=x.to(dev)).image_embeds.float(), dim=-1).cpu())
        np.savez(os.path.join(a.out, 'clip_img.npz'), e=torch.cat(E).numpy())
        meta['clip'] = dict(model=KARLO + '/image_encoder', n=len(items), secs=round(time.time() - t0, 1))
        print(json.dumps({'clip': meta['clip']}), flush=True); del enc
        json.dump(meta, open(mp, 'w'))
    for name in a.models:
        fn = os.path.join(a.out, f'{name}_full.npz')
        if os.path.exists(fn): continue
        t0 = time.time()
        lm = load_model(name, dev)
        T = load_default_translator(translators_root=Path(a.sing_root) / 'translators', registry_path=Path(a.sing_root) / 'translators/registry.yaml',
                                    model_name=name, device=dev).model.eval().to(dev).float()
        assert hasattr(T, 'weight') and T.weight.ndim == 2, 'E3 assumes the linear translator'
        dl = torch.utils.data.DataLoader(DS(items, lm.preprocess), batch_size=a.bs, num_workers=a.workers)
        F, Z, Y = [], [], []
        with torch.inference_mode():
            for x, y in dl:
                f = lm.wrapper.extract_features(x.to(dev)).float()
                F.append(f.cpu()); Z.append(T(f).half().cpu()); Y.append(y)
            W = lm.wrapper.classifier_weight.detach().float()
            bW = lm.wrapper.logits_from_features(torch.zeros(1, W.shape[1], device=W.device, dtype=W.dtype)).float()[0]
        Ab = T.bias.detach().float() if T.bias is not None else torch.zeros(T.weight.shape[0])
        np.savez(fn, F=torch.cat(F).numpy(), y=torch.cat(Y).numpy().astype(np.int16), W=W.cpu().numpy(), bW=bW.cpu().numpy(),
                 A=T.weight.detach().float().cpu().numpy(), Ab=Ab.cpu().numpy(), z16=torch.cat(Z).numpy())
        meta[name] = dict(m=W.shape[1], n=len(items), secs=round(time.time() - t0, 1))
        print(json.dumps({name: meta[name]}), flush=True)
        json.dump(meta, open(mp, 'w'))
        del lm; torch.cuda.empty_cache() if dev.type == 'cuda' else None
    print('DONE')

if __name__ == '__main__':
    main()
