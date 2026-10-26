"""IC014 E2 GPU pass: the 5 SING models not cached by E3 (vgg16, vgg19, efficientnet, swin_large, dinovit1), with
exactly the E3 extractor (e3/extract_e3_gpu.py): same 10k subset rule/subset_files.txt, same <model>_full.npz format
(F, y, W, bW, A, Ab, z16). CLIP image embeddings are NOT recomputed (E3's clip_img.npz covers the same subset);
pass --with-clip to compute them anyway. dinovit1 = SING's registry entry (timm vit_large_patch16_224, default tag
augreg_in21k_ft_in1k), i.e. exactly the model SING's released translator was trained for.
Outputs are PRIVATE (ImageNet-derived): never commit/publish."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'e3'))
import extract_e3_gpu as e3

E2_MODELS = ['vgg16', 'vgg19', 'efficientnet', 'swin_large', 'dinovit1']

if __name__ == '__main__':
    e3.MODELS = E2_MODELS
    if '--with-clip' in sys.argv: sys.argv.remove('--with-clip')
    elif '--no-clip' not in sys.argv: sys.argv.append('--no-clip')
    if '--out' not in sys.argv: sys.argv += ['--out', 'e2_feats']
    e3.main()
