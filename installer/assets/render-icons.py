import cairosvg, io
from PIL import Image, ImageDraw
S=[16,20,24,32,40,48,64,128,256]
def r(svg,n): return Image.open(io.BytesIO(cairosvg.svg2png(url=svg,output_width=n,output_height=n))).convert('RGBA')
for tag,big,sm in [(t,t+'.svg',t.replace('pyre','pyre-small')+'.svg') for t in ('pyre','pyre-starting','pyre-stopped')]:
    imgs=[r(sm if n<=24 else big,n) for n in S]
    for n,im in zip(S,imgs): im.save(f'png/{tag}-{n}.png')
    imgs[-1].save(f'{tag}.ico',sizes=[(n,n) for n in S],append_images=imgs[:-1])
# preview: 128px running/starting/stopped + size strips, on light and dark
St=[16,20,24,32,40,48,64]; T=['pyre','pyre-starting','pyre-stopped']
W=20+3*148+20+sum(St)+12*len(St)+20; RH=20+128+20+3*76; H=2*RH
P=Image.new('RGBA',(W,H),'white'); d=ImageDraw.Draw(P); d.rectangle([0,RH,W,H],fill='#1e1e1e')
for y in (20,RH+20):
    for k,t in enumerate(T): P.alpha_composite(Image.open(f'png/{t}-128.png'),(20+k*148,y))
    x=20
    for n in St:
        for k,t in enumerate(T): P.alpha_composite(Image.open(f'png/{t}-{n}.png'),(x,y+148+k*76))
        x+=n+12
    x0=20+3*148+20
P.save('/workspace/pyre-icon-preview.png')
