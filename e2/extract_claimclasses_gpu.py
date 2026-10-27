"""IC014 E2 claim-class GPU pass: ALL 50 ImageNet-val images of each class named in a SING claim
(jellyfish 107, porcupine 334, Arabian camel 354, sports car 817), for all 13 SING models plus CLIP, using exactly
the E3 extractor (same <model>_full.npz format, clip_img.npz, subset_files.txt = file\\tlabel). 200 images total.
Outputs are ImageNet-derived and private; do not commit."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'e3'))
import extract_e3_gpu as e3

CLAIM_CLASSES = {107: 'jellyfish', 334: 'porcupine', 354: 'Arabian camel', 817: 'sports car'}
ALL13 = ['vgg16', 'vgg19', 'densenet', 'resnet', 'dinovit1', 'efficientnet', 'bitresnet', 'resnext', 'convnext',
         'swin_large', 'dinov2', 'deit', 'eva02_large']
_orig = e3.subset

def claim_subset(val, labels, synsets, per_class, limit):
    items = [it for it in _orig(val, labels, synsets, 10 ** 9, None) if it[2] in CLAIM_CLASSES]
    return items[:limit] if limit else items

if __name__ == '__main__':
    e3.MODELS = ALL13; e3.subset = claim_subset
    if '--out' not in sys.argv: sys.argv += ['--out', 'e2_claim_feats']
    e3.main()
