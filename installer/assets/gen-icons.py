# pyre icon generator: ASCII-glyph flame only, transparent background. Glyphs = DejaVu Sans Mono Bold outlines (no font dep).
from fontTools.ttLib import TTFont
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
F=TTFont('/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf'); gs=F.getGlyphSet(); cmap=F.getBestCmap(); upm=F['head'].unitsPerEm
STROKE='#3b1606'   # warm dark-brown outline: contrast on light taskbars, invisible-ish on dark
def glyph(ch,x,y,size):
    sp=SVGPathPen(gs); s=size/upm; gs[cmap[ord(ch)]].draw(TransformPen(sp,(s,0,0,-s,x,y))); return sp.getCommands()
def grad(t):
    stops=[(0,(255,208,64)),(0.5,(255,124,20)),(1,(222,40,24))]
    for (t0,c0),(t1,c1) in zip(stops,stops[1:]):
        if t<=t1: f=(t-t0)/(t1-t0); return '#%02x%02x%02x'%tuple(round(c0[i]+(c1[i]-c0[i])*f) for i in range(3))
HEAD='<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256" width="256" height="256">'
GLOW='<defs><radialGradient id="g" cx="50%" cy="50%" r="50%"><stop offset="0" stop-color="#ff8a1e" stop-opacity="0.75"/><stop offset="1" stop-color="#ff5a14" stop-opacity="0"/></radialGradient></defs><ellipse cx="128" cy="214" rx="120" ry="44" fill="url(#g)"/>'
ROWS=["^","(*)","(#@)","(%@#)","\\@##@/"]
def text_rows(rows, ts, size=62, lh=47, bottom=242, sw=6):
    adv=size*0.602; n=len(rows); out_s=[]; out_f=[]
    for i,row in enumerate(rows):
        y=bottom-(n-1-i)*lh; x0=128-len(row)*adv/2 + (5 if i%2 else -3)*(1 if n>3 else 0)  # slight flicker offset
        d=''.join(glyph(ch,x0+j*adv,y,size) for j,ch in enumerate(row) if ch!=' ')
        out_f.append(f'<path d="{d}" fill="{grad(ts[i])}"/>')
    return ''.join(out_s+out_f)
def full(): return HEAD+text_rows(ROWS,[i/4 for i in range(5)])+'</svg>'
def starting(): return HEAD+text_rows(ROWS[2:],[0.35,0.65,1.0])+'</svg>'
def stopped(): return HEAD+GLOW+text_rows(["*~*", "*,*,*"],[0.3,0.75],size=72,lh=48,sw=7)+'</svg>'
# 16-24 px: chunky block silhouette (tip + two "character blocks"), thick outline
SM=lambda body: HEAD+body+'</svg>'
TIP=f'<path d="M128 6 C150 40 184 62 184 100 H72 C72 64 106 40 128 6 Z" fill="{grad(0)}"/>'
B1=f'<rect x="50" y="104" width="156" height="66" rx="12" fill="{grad(0.5)}"/>'
B2=f'<rect x="22" y="176" width="212" height="72" rx="14" fill="{grad(1)}"/>'
def small(): return SM(TIP+B1+B2)
def small_starting(): return SM(f'<path d="M128 84 C146 104 168 116 172 140 H84 C88 116 110 104 128 84 Z" fill="{grad(0.4)}"/>'+f'<rect x="22" y="150" width="212" height="98" rx="14" fill="{grad(1)}"/>')
def small_stopped(): return HEAD+GLOW.replace('cy="214"','cy="206"').replace('ry="44"','ry="60"')+''.join(f'<rect x="{x}" y="{y}" width="76" height="80" rx="14" fill="{c}"/>' for x,y,c in ((6,160,grad(0.75)),(90,128,grad(0.3)),(174,160,grad(0.75))))+'</svg>'

# ---- one silhouette outline around the whole composition (glow excluded) ----
import cairosvg, io, re, numpy as np
from PIL import Image
from scipy import ndimage
from skimage import measure
def outline(svg, close=26, grow=7, width=9):
    body=re.sub(r'<defs>.*?</defs><ellipse[^>]*/>','',svg)  # ignore glow
    N=512; im=np.array(Image.open(io.BytesIO(cairosvg.svg2png(bytestring=body.encode(),output_width=N,output_height=N))))[...,3]>40
    yy,xx=np.ogrid[-close:close+1,-close:close+1]; disk=xx*xx+yy*yy<=close*close
    m=np.pad(im,close+2); m=ndimage.binary_closing(m,structure=disk)[close+2:-close-2,close+2:-close-2]
    m=ndimage.binary_fill_holes(m); m=ndimage.binary_dilation(m,iterations=grow)
    paths=[]
    for c in measure.find_contours(np.pad(m,1).astype(float),0.5):
        if len(c)<20: continue
        c=measure.approximate_polygon(c,tolerance=0.8)
        paths.append('M'+' L'.join(f'{(x-1)*256/N:.1f} {(y-1)*256/N:.1f}' for y,x in c)+' Z')
    ring=f'<path d="{" ".join(paths)}" fill="none" stroke="{STROKE}" stroke-width="{width}" stroke-linejoin="round"/>'
    # insert after glow (if any) so the ring sits behind the glyphs
    i=svg.index('</defs><ellipse')+len('</defs>') if '<defs>' in svg else len(HEAD)
    j=svg.index('/>',i)+2 if '<defs>' in svg else i
    return svg[:j]+ring+svg[j:]
for name,s in [('pyre.svg',outline(full())),('pyre-small.svg',outline(small(),close=20,grow=6,width=16)),
               ('pyre-starting.svg',outline(starting())),('pyre-small-starting.svg',outline(small_starting(),close=20,grow=6,width=16)),
               ('pyre-stopped.svg',outline(stopped(),close=40)),('pyre-small-stopped.svg',outline(small_stopped(),close=50,grow=6,width=16))]: open(name,'w').write(s)
