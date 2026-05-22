import os, glob
from PIL import Image, ImageDraw
SWEEP="/home/user/masafee-lora/sweep"
GEN=f"{SWEEP}/gen"; GRIDS=f"{SWEEP}/grids"
os.makedirs(GRIDS, exist_ok=True)
configs=["sd15_lr1_d16","sd15_lr1_d32","sd15_lr2_d32","cf_lr1_d16","cf_lr1_d32","cf_lr2_d32"]
T=320; LBL=120
for cfg in configs:
    rows=[("e10",f"{cfg}-000010"),("e20",f"{cfg}-000020"),("e30",cfg)]
    data=[]
    for label,ckpt in rows:
        pics=sorted(glob.glob(f"{GEN}/{ckpt}/*.png"))
        if pics: data.append((label,pics))
    if not data:
        print("no images for",cfg); continue
    ncols=max(len(p) for _,p in data)
    W=LBL+ncols*T; H=30+len(data)*T
    grid=Image.new("RGB",(W,H),"white")
    d=ImageDraw.Draw(grid)
    d.text((10,8),cfg,fill="black")
    for r,(label,pics) in enumerate(data):
        y=30+r*T
        d.text((10,y+T//2),label,fill="black")
        for c,p in enumerate(pics):
            try:
                im=Image.open(p).convert("RGB").resize((T,T))
                grid.paste(im,(LBL+c*T,y))
            except Exception as e:
                print("err",p,e)
    grid.save(f"{GRIDS}/{cfg}.png")
    print("grid saved:",cfg)
print("ALL GRIDS DONE")
