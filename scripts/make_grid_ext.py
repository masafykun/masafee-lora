import os, glob
from PIL import Image, ImageDraw
EXT="/home/user/masafee-lora/sweep/ext60"
GEN=f"{EXT}/gen"
os.makedirs(f"{EXT}/grids", exist_ok=True)
rows=["e30","e40","e50","e60"]
T=320; LBL=120
data=[]
for label in rows:
    pics=sorted(glob.glob(f"{GEN}/{label}/*.png"))
    if pics: data.append((label,pics))
if data:
    ncols=max(len(p) for _,p in data)
    W=LBL+ncols*T; H=30+len(data)*T
    grid=Image.new("RGB",(W,H),"white")
    d=ImageDraw.Draw(grid)
    d.text((10,8),"cf_lr1_d32  epoch 30-60 extension",fill="black")
    for r,(label,pics) in enumerate(data):
        y=30+r*T
        d.text((10,y+T//2),label,fill="black")
        for c,p in enumerate(pics):
            im=Image.open(p).convert("RGB").resize((T,T))
            grid.paste(im,(LBL+c*T,y))
    grid.save(f"{EXT}/grids/ext60.png")
    print("grid saved")
else:
    print("no data")
