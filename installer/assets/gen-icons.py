# pyre icon generator: ASCII-glyph flame only, transparent background. Glyphs = DejaVu Sans Mono Bold outlines (no font dep).
from fontTools.ttLib import TTFont
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
F=TTFont('/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf'); gs=F.getGlyphSet(); cmap=F.getBestCmap(); upm=F['head'].unitsPerEm
STROKE='#000000'   # one pure-black outline around the whole silhouette
def glyph(ch,x,y,size):
    sp=SVGPathPen(gs); s=size/upm; gs[cmap[ord(ch)]].draw(TransformPen(sp,(s,0,0,-s,x,y))); return sp.getCommands()
def grad(t):
    stops=[(0,(255,208,64)),(0.5,(255,124,20)),(1,(222,40,24))]
    for (t0,c0),(t1,c1) in zip(stops,stops[1:]):
        if t<=t1: f=(t-t0)/(t1-t0); return '#%02x%02x%02x'%tuple(round(c0[i]+(c1[i]-c0[i])*f) for i in range(3))
HEAD='<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256" width="256" height="256">'
GLOW='<defs><radialGradient id="g" cx="50%" cy="50%" r="50%"><stop offset="0" stop-color="#ff8a1e" stop-opacity="0.75"/><stop offset="1" stop-color="#ff5a14" stop-opacity="0"/></radialGradient></defs><ellipse cx="128" cy="214" rx="120" ry="44" fill="url(#g)"/>'
ROWS=["^","(*)","(#@)","(%@#)","\\@##@/"]
WOOD_ROWS=["\\=X==X=/", "/=#==#=\\"]; WOOD_COLS=['#a0693c', '#7a4a2a']   # lighter browns so logs read against the black outline
def text_rows(rows, ts, size=50, lh=35, bottom=186, cols=None, flick=True):
    adv=size*0.602; n=len(rows); out=[]
    for i,row in enumerate(rows):
        y=bottom-(n-1-i)*lh; x0=128-len(row)*adv/2 + ((4 if i%2 else -3) if flick and n>2 else 0)
        d=''.join(glyph(ch,x0+j*adv,y,size) for j,ch in enumerate(row) if ch!=' ')
        out.append(f'<path d="{d}" fill="{cols[i] if cols else grad(ts[i])}"/>')
    return ''.join(out)
def wood(): return text_rows(WOOD_ROWS,None,size=44,lh=27,bottom=242,cols=WOOD_COLS,flick=False)
def full(): return HEAD+wood()+text_rows(ROWS,[i/4 for i in range(5)])+'</svg>'
def starting(): return HEAD+wood()+text_rows(ROWS[2:],[0.35,0.65,1.0])+'</svg>'
def stopped(): return HEAD+GLOW.replace('cy="214"','cy="176"').replace('ry="44"','ry="52"')+wood()+text_rows(["*~*", "*,*,*"],[0.2,0.6],size=66,lh=44,bottom=192)+'</svg>'
# 16-24 px: chunky blocks + two crossed log bars
SM=lambda body: HEAD+body+'</svg>'
LOGS=('<rect x="14" y="204" width="228" height="22" rx="8" fill="#a0693c" transform="rotate(-6 128 215)"/>'
      '<rect x="14" y="222" width="228" height="22" rx="8" fill="#7a4a2a" transform="rotate(6 128 233)"/>')
TIP=f'<path d="M128 8 C148 36 178 54 178 88 H78 C78 56 108 36 128 8 Z" fill="{grad(0)}"/>'
B1=f'<rect x="58" y="92" width="140" height="50" rx="10" fill="{grad(0.5)}"/>'
B2=f'<rect x="36" y="146" width="184" height="56" rx="12" fill="{grad(1)}"/>'
def small(): return SM(LOGS+TIP+B1+B2)
def small_starting(): return SM(LOGS+f'<path d="M128 72 C146 94 168 106 172 128 H84 C88 106 110 94 128 72 Z" fill="{grad(0.4)}"/>'+f'<rect x="36" y="134" width="184" height="68" rx="12" fill="{grad(1)}"/>')
def small_stopped(): return HEAD+GLOW.replace('cy="214"','cy="160"').replace('rx="120"','rx="128"').replace('ry="44"','ry="70"')+LOGS+''.join(f'<rect x="{x}" y="{y}" width="84" height="88" rx="16" fill="{c}"/>' for x,y,c in ((6,112,grad(0.45)),(86,84,grad(0.1)),(166,112,grad(0.45))))+'</svg>'
# ---- one silhouette outline around the whole composition (glow excluded) ----
import cairosvg, io, re, numpy as np
from PIL import Image
from scipy import ndimage
from skimage import measure
def outline(svg, close=26, grow=8, width=12):
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
for name,s in [('pyre.svg',outline(full())),('pyre-small.svg',outline(small(),close=20,grow=8,width=20)),
               ('pyre-starting.svg',outline(starting())),('pyre-small-starting.svg',outline(small_starting(),close=20,grow=8,width=20)),
               ('pyre-stopped.svg',outline(stopped(),close=40)),('pyre-small-stopped.svg',outline(small_stopped(),close=50,grow=8,width=20))]: open(name,'w').write(s)
