# Exp 0b: DinoViT candidate top-1 (GPU)
Run with GPU Bot's own Kaggle copy of ImageNet (competition `imagenet-object-localization-challenge`). The script never stores or redistributes ImageNet.
```
pip install -r exp0b/requirements.txt
python exp0b/eval_top1.py --val $IMNET/ILSVRC/Data/CLS-LOC/val --labels $IMNET/LOC_val_solution.csv \
  --synsets $IMNET/LOC_synset_mapping.txt --out exp0b/results_top1.json
```
Ten configs: 5 candidates × {native, r224}. Smoke-tested on CPU with 2 fake images; every config builds and runs (SWAG-E2E native input is 512).

Licences checked against primary sources on 2026-10-25:
* SWAG_LINEAR_V1/E2E_V1: torchvision metadata points to github.com/facebookresearch/SWAG/blob/main/LICENSE, which is CC BY-NC 4.0. Non-commercial evaluation is allowed; the weights are not redistributed.
* google/vit-large-patch16-224: the HF model card metadata says license apache-2.0.
