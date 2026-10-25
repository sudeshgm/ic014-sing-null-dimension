"""Exp 0 stats: rank correlations of Fig. 6b IS/AS ratio vs null dimension (seed 0, 10k permutations).
Fig. 6b reports only the IS/AS ratio, so AS and IS individually cannot be read from it (see README)."""
import json, csv, numpy as np
from scipy.stats import spearmanr, kendalltau
MAP={'DenseNet':'densenet','DinoViT':'dinovit1','EVA02Large':'eva02_large','BitResNet':'bitresnet','ConvNext':'convnext',
     'EfficientNet':'efficientnet','SwinLarge':'swin_large','ResNet50':'resnet','VGG16':'vgg16','VGG19':'vgg19',
     'DinoV2':'dinov2','ResNext101':'resnext','DeiT':'deit'}
vals={MAP[r['model']]:r['ratio_IS_over_AS'] for r in json.load(open('fig6b_digitized.json'))['values']}
nd={r['model']:r['null_dim'] for r in json.load(open('null_dims.json'))}
import sys
SUFFIX=''
if '--densenet24' in sys.argv: nd['densenet']=24; SUFFIX='_densenet24'
NP=10000
def perm_p(x,y,f,rng):
    obs=f(x,y)[0]; c=sum(abs(f(x,rng.permutation(y))[0])>=abs(obs)-1e-12 for _ in range(NP))
    return (c+1)/(NP+1)
def block(models,tag):
    x=np.array([nd[m] for m in models],float); y=np.array([vals[m] for m in models])
    out=dict(set=tag,n=len(models))
    for nm,f in [('spearman',spearmanr),('kendall',kendalltau)]:
        rng=np.random.default_rng(0); r,p=f(x,y)
        out[nm]=float(r); out[nm+'_p_asym']=float(p); out[nm+'_p_perm10k']=perm_p(x,y,f,rng)
    return out
allm=list(MAP.values()); res={'ratio_vs_nulldim':[]}
res['ratio_vs_nulldim'].append(block(allm,'all13'))
res['ratio_vs_nulldim'].append(block([m for m in allm if m!='dinovit1'],'no_DinoViT'))
loo=[]
for drop in allm:
    b=block([m for m in allm if m!=drop],'LOO_drop_'+drop); loo.append(b)
res['leave_one_out_all13']=loo
loo2=[]
base=[m for m in allm if m!='dinovit1']
for drop in base:
    loo2.append(block([m for m in base if m!=drop],'noDinoViT_LOO_drop_'+drop))
res['leave_one_out_noDinoViT']=loo2
# derived: AS-direction proxy. Ratio IS/AS is inversely related to AS only if IS were constant; not computed.
json.dump(res,open(f'stats{SUFFIX}.json','w'),indent=1)
with open(f'stats{SUFFIX}.csv','w') as f:
    w=csv.DictWriter(f,fieldnames=list(res['ratio_vs_nulldim'][0])); w.writeheader()
    for b in res['ratio_vs_nulldim']+loo+loo2: w.writerow(b)
for b in res['ratio_vs_nulldim']: print(b)
for L in (loo,loo2): print('LOO spearman range',min(b['spearman'] for b in L),max(b['spearman'] for b in L),'kendall range',min(b['kendall'] for b in L),max(b['kendall'] for b in L),'max perm p',max(b['spearman_p_perm10k'] for b in L))
