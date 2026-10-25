"""IC014 Exp 0b: ImageNet-val top-1 of DinoViT candidate checkpoints.
Usage: python eval_top1.py --val /path/to/ILSVRC/Data/CLS-LOC/val --labels /path/to/LOC_val_solution.csv \
       --synsets /path/to/LOC_synset_mapping.txt --out results_top1.json [--candidates ...] [--limit N]
Also accepts --val pointing at a class-folder layout (val/nXXXX/*.JPEG); then --labels/--synsets are not needed.
Two settings per candidate:
  native : the candidate's own published eval transform at its native resolution
           (torchvision weights.transforms(); timm resolve_model_data_config -> create_transform, as SING's registry.py;
            HF AutoImageProcessor)
  r224   : Resize(256, candidate interpolation) + CenterCrop(224), candidate mean/std; model run at 224
           (position embeddings interpolated when native != 224).
For candidates whose native resolution is 224 the two settings differ only in resize/crop policy.
No ImageNet data is stored or redistributed by this script."""
import argparse, json, os, time, csv
import torch, torchvision
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode as IM
from PIL import Image

CANDS = ['tv_vit_l_16_IMAGENET1K_V1', 'timm_vit_large_patch16_224.augreg_in21k_ft_in1k',
         'tv_vit_l_16_IMAGENET1K_SWAG_LINEAR_V1', 'tv_vit_l_16_IMAGENET1K_SWAG_E2E_V1', 'hf_google/vit-large-patch16-224']
DOC = {'tv_vit_l_16_IMAGENET1K_V1': 79.662, 'timm_vit_large_patch16_224.augreg_in21k_ft_in1k': 85.856,
       'tv_vit_l_16_IMAGENET1K_SWAG_LINEAR_V1': 85.146, 'tv_vit_l_16_IMAGENET1K_SWAG_E2E_V1': 88.064, 'hf_google/vit-large-patch16-224': None}

def r224(interp, mean, std):
    return T.Compose([T.Resize(256, interpolation=interp), T.CenterCrop(224), T.ToTensor(), T.Normalize(mean, std)])

def build(name, setting):
    """returns (model taking BCHW tensor -> logits, transform PIL->tensor)"""
    if name.startswith('tv_'):
        from torchvision.models import vit_l_16, ViT_L_16_Weights
        from torchvision.models.vision_transformer import interpolate_embeddings
        w = getattr(ViT_L_16_Weights, name.split('vit_l_16_')[1])
        m = vit_l_16(weights=w)
        tf_native = w.transforms()
        native = tf_native.crop_size[0]
        if setting == 'native':
            return m, tf_native
        if native != 224:
            sd = interpolate_embeddings(224, 16, m.state_dict(), interpolation_mode='bicubic')
            m = vit_l_16(image_size=224); m.load_state_dict(sd)
        return m, r224(tf_native.interpolation, tf_native.mean, tf_native.std)
    if name.startswith('timm_'):
        import timm
        m = timm.create_model(name[5:], pretrained=True)
        cfg = timm.data.resolve_model_data_config(m)
        if setting == 'native':
            return m, timm.data.create_transform(**cfg, is_training=False)
        interp = IM.BICUBIC if cfg['interpolation'] == 'bicubic' else IM.BILINEAR
        return m, r224(interp, cfg['mean'], cfg['std'])
    if name.startswith('hf_'):
        from transformers import AutoModelForImageClassification, AutoImageProcessor
        mid = name[3:]
        hf = AutoModelForImageClassification.from_pretrained(mid)
        proc = AutoImageProcessor.from_pretrained(mid)
        class Wrap(torch.nn.Module):
            def __init__(s, h): super().__init__(); s.h = h
            def forward(s, x): return s.h(pixel_values=x).logits
        if setting == 'native':
            return Wrap(hf), (lambda im: proc(images=im, return_tensors='pt')['pixel_values'][0])
        return Wrap(hf), r224(IM.BILINEAR, proc.image_mean, proc.image_std)
    raise ValueError(name)

class Val(torch.utils.data.Dataset):
    def __init__(s, root, labels, synsets, tf, limit=None):
        s.tf = tf
        if labels:
            syn = [l.split()[0] for l in open(synsets)]
            idx = {k: i for i, k in enumerate(sorted(syn))}
            rows = list(csv.reader(open(labels)))[1:]
            s.items = sorted((os.path.join(root, r[0] + '.JPEG'), idx[r[1].split()[0]]) for r in rows)
        else:
            cls = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
            s.items = [(os.path.join(root, c, f), i) for i, c in enumerate(cls) for f in sorted(os.listdir(os.path.join(root, c)))]
        if limit: s.items = s.items[::max(1, len(s.items) // limit)][:limit]
    def __len__(s): return len(s.items)
    def __getitem__(s, i):
        p, y = s.items[i]; return s.tf(Image.open(p).convert('RGB')), y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--val', required=True); ap.add_argument('--labels'); ap.add_argument('--synsets')
    ap.add_argument('--out', default='results_top1.json'); ap.add_argument('--candidates', nargs='*', default=CANDS)
    ap.add_argument('--settings', nargs='*', default=['native', 'r224'])
    ap.add_argument('--bs', type=int, default=64); ap.add_argument('--workers', type=int, default=8); ap.add_argument('--limit', type=int)
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for c in a.candidates:
        for st in a.settings:
            key = f'{c}|{st}'
            if key in res and res[key].get('n') == (a.limit or 50000): continue
            m, tf = build(c, st); m = m.eval().to(dev)
            ds = Val(a.val, a.labels, a.synsets, tf, a.limit)
            dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, num_workers=a.workers, pin_memory=True)
            corr = n = 0; t = time.time(); shape = None
            with torch.inference_mode(), torch.autocast(dev, dtype=torch.float16, enabled=(dev == 'cuda')):
                for x, y in dl:
                    shape = list(x.shape[1:])
                    p = m(x.to(dev, non_blocking=True)).argmax(1).cpu()
                    corr += (p == y).sum().item(); n += len(y)
            res[key] = dict(candidate=c, setting=st, top1=round(100 * corr / n, 3), n=n, input_shape=shape,
                            documented_top1=DOC.get(c), secs=round(time.time() - t, 1), device=dev,
                            torch=torch.__version__, torchvision=torchvision.__version__)
            print(json.dumps(res[key]), flush=True)
            json.dump(res, open(a.out, 'w'), indent=1)
            del m; torch.cuda.empty_cache() if dev == 'cuda' else None
    match = [k for k, v in res.items() if abs(v['top1'] - 84.0) <= 0.3]
    res['_summary'] = dict(target=84.0, tol=0.3, matching=match)
    json.dump(res, open(a.out, 'w'), indent=1); print('MATCH 84.0+-0.3:', match)

if __name__ == '__main__':
    main()
