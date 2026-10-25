"""Tolerance sensitivity of the two borderline heads (float32 as in SING vs float64)."""
import torch, torchvision, json
out={}
for n,f in [('densenet121',torchvision.models.densenet121),('resnet50',torchvision.models.resnet50)]:
    m=f(weights='IMAGENET1K_V1'); W=(m.classifier if n.startswith('dense') else m.fc).weight.detach()
    s32=torch.linalg.svdvals(W.float()); s64=torch.linalg.svdvals(W.double())
    out[n]=dict(smallest6_f64=s64[-6:].tolist(),smallest6_f32=s32[-6:].tolist(),
       rank_f32={t:int((s32>t).sum()) for t in [1e-7,1e-6,1e-5,1e-4]},rank_f64={t:int((s64>t).sum()) for t in [1e-7,1e-6,1e-5,1e-4]},
       zero_rows=int((W.abs().sum(1)==0).sum()), dup_rows=int(len(W)-torch.unique(W,dim=0).shape[0]))
    print(n,out[n])
json.dump(out,open('sv_tolerance_check.json','w'),indent=1,default=str)
