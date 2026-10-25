"""Null dimension m - rank(W) of each SING classifier head (SVD, tol 1e-5, as in sing/core/projectors.py).
Models follow /tmp/sing/src/sing/configs/models.yaml. CPU only."""
import torch, timm, torchvision, json, csv, gc, yaml, functools
SPECS=yaml.safe_load(open('/tmp/sing/src/sing/configs/models.yaml'))['models']
PLAN={'vgg16':3096,'vgg19':3096,'resnet':1048,'resnext':1048,'bitresnet':1048,'dinov2':1048,'swin_large':536,
      'efficientnet':280,'densenet':24,'convnext':24,'deit':24,'eva02_large':24,'dinovit1':24}
TV={'resnet50':lambda: torchvision.models.resnet50(weights='IMAGENET1K_V1'),'densenet121':lambda: torchvision.models.densenet121(weights='IMAGENET1K_V1'),
    'vgg16':lambda: torchvision.models.vgg16(weights='IMAGENET1K_V1'),'vgg19':lambda: torchvision.models.vgg19(weights='IMAGENET1K_V1')}
def get(root,path): return functools.reduce(lambda o,p: o[int(p)] if p.isdigit() else getattr(o,p), path.split('.'), root)
rows=[]
for name,s in SPECS.items():
    if s['source']=='torchvision': m=TV[s['model_id']]()
    elif s['source']=='timm': m=timm.create_model(s['model_id'],pretrained=True)
    else:
        from transformers import AutoModelForImageClassification
        m=AutoModelForImageClassification.from_pretrained(s['model_id'])
    W=get(m,s["classifier_layer"]).weight.detach().double(); W=W.reshape(W.shape[0],-1)  # BiT head.fc is a 1x1 conv
    sv=torch.linalg.svdvals(W); r=int((sv>1e-5).sum()); c,dim=W.shape
    rows.append(dict(model=name,model_id=s['model_id'],classes=c,m=dim,rank=r,null_dim=dim-r,plan_null_dim=PLAN[name],match=(dim-r)==PLAN[name],
                     sv_min=float(sv.min()),sv_at_rank=float(sv[r-1]),default_cfg_tag=getattr(m,'pretrained_cfg',{}).get('tag') if hasattr(m,'pretrained_cfg') else None))
    print(rows[-1],flush=True); del m; gc.collect()
json.dump(rows,open('null_dims.json','w'),indent=1)
with open('null_dims.csv','w') as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
