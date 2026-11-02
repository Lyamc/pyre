from fontTools.ttLib import TTFont
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
import sys
F=TTFont('/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf'); gs=F.getGlyphSet(); cmap=F.getBestCmap(); upm=F['head'].unitsPerEm
def glyph(ch,x,y,size):
    sp=SVGPathPen(gs); s=size/upm
    gs[cmap[ord(ch)]].draw(TransformPen(sp,(s,0,0,-s,x,y)))
    return sp.getCommands()
def lerp(a,b,t): return tuple(round(a[i]+(b[i]-a[i])*t) for i in range(3))
def grad(t, grey=False):
    stops=[(0,(255,214,92)),(0.45,(255,128,26)),(1,(206,36,30))]
    for (t0,c0),(t1,c1) in zip(stops,stops[1:]):
        if t<=t1: c=lerp(c0,c1,(t-t0)/(t1-t0)); break
    if grey: g=round(0.3*c[0]+0.59*c[1]+0.11*c[2]); g=round(g*0.55+60); c=(g,g,g)
    return '#%02x%02x%02x'%c
ROWS=["    '    ","   (^    ","  ,(*).  ","  (/#\\)' "," .(%@%)  ","((#@@@#))","(\\%@@@%/)"," ~#@@@#~ ","  \\|%|/  "]
DOC_FILL='#F7F3EC'; DOC_STROKE='#2A2622'; FOLD='#DDD5C8'
def doc(stroke_w=8):
    return (f'<path d="M44 14 H170 L216 60 V238 a6 6 0 0 1 -6 6 H50 a6 6 0 0 1 -6 -6 V20 a6 6 0 0 1 6 -6 Z" fill="{DOC_FILL}" stroke="{DOC_STROKE}" stroke-width="{stroke_w}" stroke-linejoin="round"/>'
            f'<path d="M170 14 V54 a6 6 0 0 0 6 6 H216 Z" fill="{FOLD}" stroke="{DOC_STROKE}" stroke-width="{stroke_w}" stroke-linejoin="round"/>')
EMBERS=["  . ' ,  "," ,~.,~. "]
def full(rows=None, tcol=None):
    rows=rows or ROWS; size=26; adv=size*0.602; lh=18.5; y0=86+(len(ROWS)-len(rows))*lh; out=[]
    for i,row in enumerate(rows):
        t=tcol[i] if tcol else i/(len(rows)-1); col=grad(t); x0=130-len(row)*adv/2
        d=''.join(glyph(ch,x0+j*adv,y0+i*lh,size) for j,ch in enumerate(row) if ch!=' ')
        out.append(f'<path d="{d}" fill="{col}"/>')
    return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256" width="256" height="256">'+doc()+''.join(out)+'</svg>'
def small(level=3):
    c=[grad(t) for t in (0.0,0.5,1.0)]
    parts=[f'<path d="M128 70 C140 92 158 104 158 128 H98 C98 108 116 96 128 70 Z" fill="{c[0]}"/>',
           f'<rect x="88" y="136" width="80" height="34" rx="6" fill="{c[1]}"/>',
           f'<rect x="78" y="178" width="100" height="40" rx="6" fill="{c[2]}"/>']
    if level==3: fl=''.join(parts)
    elif level==2: fl=parts[1]+parts[2]
    else: fl=''.join(f'<rect x="{x}" y="196" width="22" height="22" rx="4" fill="{c[2]}"/>' for x in (84,117,150))
    return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256" width="256" height="256">'+doc(14)+fl+'</svg>'
for name,s in [('pyre.svg',full()),('pyre-small.svg',small(3)),
               ('pyre-starting.svg',full(ROWS[4:],[0.3,0.5,0.7,0.85,1.0])),('pyre-small-starting.svg',small(2)),
               ('pyre-stopped.svg',full(EMBERS,[0.75,1.0])),('pyre-small-stopped.svg',small(1))]:
    open(name,'w').write(s)
