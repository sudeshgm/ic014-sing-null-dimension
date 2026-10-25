"""Pixel-level digitization of SING Supp. Fig. 6b (image-to-attribute ratio, IS/AS).
Input: fig6b_800dpi.png (page 5 of supplement rendered at 800 dpi with pdftoppm, cropped)."""
import numpy as np, json, csv
from PIL import Image
a=np.asarray(Image.open('fig6b_800dpi.png').convert('RGB')).astype(int)
names=['DenseNet','DinoViT','EVA02Large','BitResNet','ConvNext','EfficientNet','SwinLarge','ResNet50','VGG16','VGG19','DinoV2','ResNext101','DeiT']
bars=[(281,369),(389,481),(501,593),(613,705),(725,817),(837,929),(949,1041),(1061,1149),(1169,1261),(1285,1373),(1393,1485),(1505,1597),(1617,1709)]
# gridline centres (rows) found by profile of inter-bar gaps; lines are 4 px thick
grid_y=np.array([618.5,470.5,326.5,178.5]); grid_v=np.array([0,20,40,60])
k,b=np.polyfit(grid_y,grid_v,1); resid=grid_v-(k*grid_y+b)
px_per_unit=-1/k
out=[]
for n,(s,e) in zip(names,bars):
    tops=[]
    for x in range(s+10,e-10):
        ref=a[600,x]
        ys=[y for y in range(77,616) if np.abs(a[y,x]-ref).sum()<40]
        tops.append(min(ys))
    top=np.median(tops)
    v=k*(top-0.5)+b   # top edge of bar
    # error: half a 200dpi source pixel (=2 px here) + calibration residual + antialias edge (1 px)
    err=float(np.sqrt((2.5/px_per_unit)**2+np.abs(resid).max()**2))
    out.append(dict(model=n,ratio_IS_over_AS=round(float(v),2),read_err=round(err,2),top_px=float(top)))
    print(n,round(v,2),'+-',round(err,2))
json.dump(dict(calib=dict(slope=k,intercept=b,max_resid=float(np.abs(resid).max()),px_per_unit=px_per_unit),values=out),open('fig6b_digitized.json','w'),indent=1)
with open('fig6b_digitized.csv','w') as f:
    w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
