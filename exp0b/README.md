# Exp 0b: DinoViT candidate top-1 (GPU)
This was run on GPU Bot's own Kaggle copy of ImageNet (competition `imagenet-object-localization-challenge`). The script does not store or redistribute ImageNet.
```
pip install -r exp0b/requirements.txt
python exp0b/eval_top1.py --val $IMNET/ILSVRC/Data/CLS-LOC/val --labels $IMNET/LOC_val_solution.csv \
  --synsets $IMNET/LOC_synset_mapping.txt --out exp0b/results_top1.json
```
There are ten configs: 5 candidates × {native, r224}. A CPU smoke test on 2 fake images showed that every config builds and runs (SWAG-E2E's native input is 512).

Licences, checked against primary sources on 2026-10-25:
* SWAG_LINEAR_V1/E2E_V1: the torchvision metadata points to github.com/facebookresearch/SWAG/blob/main/LICENSE, which is CC BY-NC 4.0. Non-commercial evaluation is allowed, and we do not redistribute the weights.
* google/vit-large-patch16-224: the license field in the HF model card metadata is apache-2.0.
