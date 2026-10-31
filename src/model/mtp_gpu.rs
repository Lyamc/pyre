//! PYRE_MTP_GPUSTEP: the MTP drafter's per-token step on the GPU (draft-only, lossy numerics allowed: verify
//! decides every emitted token). Routed experts stay on the CPU (one hop per draft token). Kernels are pyre's own.
use burn_cubecl::cubecl;
use cubecl::prelude::*;
use cubecl::server::Handle as H;
use super::qn_cube::Cube;

#[cube]
fn sg(x: f32) -> f32 { 1.0f32 / (1.0f32 + f32::exp(-x)) }

/// y[seg*stride + j] = x[...] * rms(x seg) * w[(seg % wsegs)*l + j], j < l; one 256-thread cube per segment
#[cube(launch_unchecked)]
fn k_rms_seg(x: &[f32], w: &[f32], y: &mut [f32], l: u32, stride: u32, wsegs: u32, eps: f32) {
    let mut sh = Shared::<[f32]>::new_slice(256usize);
    let seg = CUBE_POS_X; let base = seg * stride; let wb = (seg % wsegs) * l; let tid = UNIT_POS_X;
    let mut a = 0.0f32; let mut i = tid;
    while i < l { let v = x[(base + i) as usize]; a += v * v; i += 256u32; }
    sh[tid as usize] = a; sync_cube();
    let mut h = 128u32;
    while h > 0u32 { if tid < h { sh[tid as usize] = sh[tid as usize] + sh[(tid + h) as usize]; } sync_cube(); h = h / 2u32; }
    let s = 1.0f32 / f32::sqrt(sh[0] / f32::cast_from(l) + eps);
    let mut j = tid;
    while j < l { y[(base + j) as usize] = x[(base + j) as usize] * s * w[(wb + j) as usize]; j += 256u32; }
}
#[cube(launch_unchecked)]
fn k_rope_seg(v: &mut [f32], inv: &[f32], stride: u32, half: u32, pos: f32) {
    let base = CUBE_POS_X * stride; let i = UNIT_POS_X;
    if i < half { let a = pos * inv[i as usize]; let s = f32::sin(a); let c = f32::cos(a); let x0 = v[(base + i) as usize]; let x1 = v[(base + i + half) as usize];
        v[(base + i) as usize] = x0 * c - x1 * s; v[(base + i + half) as usize] = x1 * c + x0 * s; }
}
#[cube(launch_unchecked)]
fn k_add_b(a: &[f32], b: &[f32], y: &mut [f32], d: u32, n: u32) { let i = ABSOLUTE_POS_X; if i < n { y[i as usize] = a[i as usize] + b[(i % d) as usize]; } }
#[cube(launch_unchecked)]
fn k_hc_pre(o: &[f32], lo: &mut [f32], inj: &mut [f32], lr: u32, hc: u32, has: u32, hcf: f32) {
    let i = ABSOLUTE_POS_X;
    if i < lr { let v = o[i as usize] / hcf; lo[i as usize] = v * sg(v); }
    if i < hc { if has == 1u32 { inj[i as usize] = o[(lr + i) as usize] / hcf; } else { inj[i as usize] = 0.0f32; } }
}
#[cube(launch_unchecked)]
fn k_hc_mixed(xn: &[f32], gate: &[f32], out: &mut [f32], d: u32, hc: u32, hcf: f32) {
    let i = ABSOLUTE_POS_X;
    if i < d { let mut a = 0.0f32; let mut s = 0u32; while s < hc { let k = s * d + i; a += xn[k as usize] * sg(gate[k as usize]); s += 1u32; } out[i as usize] = a / hcf; }
}
#[cube(launch_unchecked)]
fn k_hc_comb(rr: &mut [f32], out: &[f32], inj: &[f32], d: u32, n: u32) {
    let i = ABSOLUTE_POS_X; if i < n { let s = i / d; rr[i as usize] = rr[i as usize] + out[(i % d) as usize] * 2.0f32 * sg(inj[s as usize]); }
}
#[cube(launch_unchecked)]
fn k_cp(src: &[f32], dst: &mut [f32], so: u32, dof: u32, n: u32) { let i = ABSOLUTE_POS_X; if i < n { dst[(dof + i) as usize] = src[(so + i) as usize]; } }
#[cube(launch_unchecked)]
fn k_silu2(r: &[f32], g: &mut [f32], off: u32, sf: u32) { let i = ABSOLUTE_POS_X; if i < sf { let a = r[(off + i) as usize]; g[i as usize] = a * sg(a) * r[(off + sf + i) as usize]; } }
/// one cube per head: softmax(q.k * scale) over n cells starting at cell c0, weighted v, times sigmoid(gate)
#[cube(launch_unchecked)]
fn k_attn1(qn: &[f32], qf: &[f32], kc: &[f32], vc: &[f32], out: &mut [f32], c0: u32, n: u32, row: u32, hd: u32, grp: u32, scale: f32, cap: u32) {
    let mut sc = Shared::<[f32]>::new_slice(512usize);
    let mut red = Shared::<[f32]>::new_slice(2usize);
    let h = CUBE_POS_X; let kvh = h / grp; let tid = UNIT_POS_X;
    let mut p = tid;
    while p < n { let mut a = 0.0f32; let kb = ((c0 + p) % cap) * row + kvh * hd; let mut i = 0u32; while i < hd { a += qn[(h * 2 * hd + i) as usize] * kc[(kb + i) as usize]; i += 1u32; } sc[p as usize] = a * scale; p += 256u32; }
    sync_cube();
    if tid == 0u32 { let mut mx = f32::new(-3.0e38); let mut q = 0u32; while q < n { mx = f32::max(mx, sc[q as usize]); q += 1u32; }
        let mut s = 0.0f32; let mut q2 = 0u32; while q2 < n { let e = f32::exp(sc[q2 as usize] - mx); sc[q2 as usize] = e; s += e; q2 += 1u32; } red[0] = s; }
    sync_cube();
    let mut i2 = tid;
    while i2 < hd { let mut o = 0.0f32; let mut q3 = 0u32; while q3 < n { o += sc[q3 as usize] * vc[(((c0 + q3) % cap) * row + kvh * hd + i2) as usize]; q3 += 1u32; }
        out[(h * hd + i2) as usize] = o / red[0] * sg(qf[(h * 2 * hd + hd + i2) as usize]); i2 += 256u32; }
}

// PYRE_MTP_GRAPH: variants reading the per-call scalars from pb = [pos, kv slot offset, c0, n] (f32) so a captured graph stays valid
#[cube(launch_unchecked)]
fn k_rope_p(v: &mut [f32], inv: &[f32], pb: &[f32], stride: u32, half: u32) {
    let pos = pb[0]; let base = CUBE_POS_X * stride; let i = UNIT_POS_X;
    if i < half { let a = pos * inv[i as usize]; let s = f32::sin(a); let c = f32::cos(a); let x0 = v[(base + i) as usize]; let x1 = v[(base + i + half) as usize];
        v[(base + i) as usize] = x0 * c - x1 * s; v[(base + i + half) as usize] = x1 * c + x0 * s; }
}
#[cube(launch_unchecked)]
fn k_cp_p(src: &[f32], dst: &mut [f32], pb: &[f32], n: u32) { let i = ABSOLUTE_POS_X; if i < n { let o = u32::cast_from(pb[1]); dst[(o + i) as usize] = src[i as usize]; } }
#[cube(launch_unchecked)]
fn k_attn1p(qn: &[f32], qf: &[f32], kc: &[f32], vc: &[f32], out: &mut [f32], pb: &[f32], row: u32, hd: u32, grp: u32, scale: f32, cap: u32) {
    let mut sc = Shared::<[f32]>::new_slice(512usize);
    let mut red = Shared::<[f32]>::new_slice(2usize);
    let c0 = u32::cast_from(pb[2]); let n = u32::cast_from(pb[3]);
    let h = CUBE_POS_X; let kvh = h / grp; let tid = UNIT_POS_X;
    let mut p = tid;
    while p < n { let mut a = 0.0f32; let kb = ((c0 + p) % cap) * row + kvh * hd; let mut i = 0u32; while i < hd { a += qn[(h * 2 * hd + i) as usize] * kc[(kb + i) as usize]; i += 1u32; } sc[p as usize] = a * scale; p += 256u32; }
    sync_cube();
    if tid == 0u32 { let mut mx = f32::new(-3.0e38); let mut q = 0u32; while q < n { mx = f32::max(mx, sc[q as usize]); q += 1u32; }
        let mut s = 0.0f32; let mut q2 = 0u32; while q2 < n { let e = f32::exp(sc[q2 as usize] - mx); sc[q2 as usize] = e; s += e; q2 += 1u32; } red[0] = s; }
    sync_cube();
    let mut i2 = tid;
    while i2 < hd { let mut o = 0.0f32; let mut q3 = 0u32; while q3 < n { o += sc[q3 as usize] * vc[(((c0 + q3) % cap) * row + kvh * hd + i2) as usize]; q3 += 1u32; }
        out[(h * hd + i2) as usize] = o / red[0] * sg(qf[(h * 2 * hd + hd + i2) as usize]); i2 += 256u32; }
}

fn g1(n: usize) -> CubeCount { CubeCount::Static((n as u32).div_ceil(256).max(1), 1, 1) }
macro_rules! ba { ($h:expr, $n:expr) => { BufferArg::from_raw_parts(($h).clone(), $n) } }
pub fn rms_seg(cb: &Cube, x: &H, xn: usize, w: &H, wn: usize, y: &H, l: usize, stride: usize, nseg: usize, wsegs: usize, eps: f32) {
    unsafe { k_rms_seg::launch_unchecked(&cb.client, CubeCount::Static(nseg as u32, 1, 1), CubeDim::new_1d(256), ba!(x, xn), ba!(w, wn), ba!(y, xn), l as u32, stride as u32, wsegs as u32, eps) }
}
pub fn rope_seg(cb: &Cube, v: &H, vn: usize, inv: &H, half: usize, stride: usize, nseg: usize, pos: usize) {
    unsafe { k_rope_seg::launch_unchecked(&cb.client, CubeCount::Static(nseg as u32, 1, 1), CubeDim::new_1d(half.next_power_of_two().max(32) as u32), ba!(v, vn), ba!(inv, half), stride as u32, half as u32, pos as f32) }
}
pub fn add_b(cb: &Cube, a: &H, b: &H, y: &H, d: usize, n: usize) { unsafe { k_add_b::launch_unchecked(&cb.client, g1(n), CubeDim::new_1d(256), ba!(a, n), ba!(b, d), ba!(y, n), d as u32, n as u32) } }
pub fn hc_pre(cb: &Cube, o: &H, on: usize, lo: &H, inj: &H, lr: usize, hc: usize, has: bool) { unsafe { k_hc_pre::launch_unchecked(&cb.client, g1(lr.max(hc)), CubeDim::new_1d(256), ba!(o, on), ba!(lo, lr), ba!(inj, hc), lr as u32, hc as u32, has as u32, hc as f32) } }
pub fn hc_mixed(cb: &Cube, xn: &H, gate: &H, out: &H, d: usize, hc: usize) { unsafe { k_hc_mixed::launch_unchecked(&cb.client, g1(d), CubeDim::new_1d(256), ba!(xn, hc * d), ba!(gate, hc * d), ba!(out, d), d as u32, hc as u32, hc as f32) } }
pub fn hc_comb(cb: &Cube, rr: &H, out: &H, inj: &H, d: usize, hc: usize) { unsafe { k_hc_comb::launch_unchecked(&cb.client, g1(hc * d), CubeDim::new_1d(256), ba!(rr, hc * d), ba!(out, d), ba!(inj, hc), d as u32, (hc * d) as u32) } }
pub fn cp(cb: &Cube, src: &H, sn: usize, dst: &H, dn: usize, so: usize, dof: usize, n: usize) { unsafe { k_cp::launch_unchecked(&cb.client, g1(n), CubeDim::new_1d(256), ba!(src, sn), ba!(dst, dn), so as u32, dof as u32, n as u32) } }
pub fn silu2(cb: &Cube, r: &H, rn: usize, g: &H, off: usize, sf: usize) { unsafe { k_silu2::launch_unchecked(&cb.client, g1(sf), CubeDim::new_1d(256), ba!(r, rn), ba!(g, sf), off as u32, sf as u32) } }
pub fn attn1(cb: &Cube, qn: &H, qf: &H, qlen: usize, kc: &H, vc: &H, kvn: usize, out: &H, nh: usize, c0: usize, n: usize, row: usize, hd: usize, grp: usize, scale: f32, cap: usize) {
    unsafe { k_attn1::launch_unchecked(&cb.client, CubeCount::Static(nh as u32, 1, 1), CubeDim::new_1d(256), ba!(qn, qlen), ba!(qf, qlen), ba!(kc, kvn), ba!(vc, kvn), ba!(out, nh * hd), c0 as u32, n as u32, row as u32, hd as u32, grp as u32, scale, cap as u32) }
}
pub fn rope_p(cb: &Cube, v: &H, vn: usize, inv: &H, half: usize, stride: usize, nseg: usize, pb: &H) {
    unsafe { k_rope_p::launch_unchecked(&cb.client, CubeCount::Static(nseg as u32, 1, 1), CubeDim::new_1d(half.next_power_of_two().max(32) as u32), ba!(v, vn), ba!(inv, half), ba!(pb, 4), stride as u32, half as u32) }
}
pub fn cp_p(cb: &Cube, src: &H, sn: usize, dst: &H, dn: usize, pb: &H, n: usize) { let _ = sn; unsafe { k_cp_p::launch_unchecked(&cb.client, g1(n), CubeDim::new_1d(256), ba!(src, sn), ba!(dst, dn), ba!(pb, 4), n as u32) } }
pub fn attn1p(cb: &Cube, qn: &H, qf: &H, qlen: usize, kc: &H, vc: &H, kvn: usize, out: &H, nh: usize, pb: &H, row: usize, hd: usize, grp: usize, scale: f32, cap: usize) {
    unsafe { k_attn1p::launch_unchecked(&cb.client, CubeCount::Static(nh as u32, 1, 1), CubeDim::new_1d(256), ba!(qn, qlen), ba!(qf, qlen), ba!(kc, kvn), ba!(vc, kvn), ba!(out, nh * hd), ba!(pb, 4), row as u32, hd as u32, grp as u32, scale, cap as u32) }
}

// PYRE_MTP_GX: the drafter's routed experts on the GPU from a resident pool of Q8 expert slots (LRU, filled from the
// drafts' own routing); experts not resident are skipped (weights renormalised over the resident picks; draft-only).
#[cube]
fn dq(word: u32, j: u32) -> f32 { let raw = ((word ^ 0x80808080u32) >> (j * 8)) & 255u32; f32::reinterpret(raw | 0x4B000000u32) - 8388736.0f32 }
/// one 1024-thread cube: top-k of the router logits pk[0..ne]; sel = [slot (-1 = not resident)] k, [weight] k, [expert id] k
#[cube(launch_unchecked)]
fn k_xsel(pk: &[f32], smap: &[f32], sel: &mut [f32], ne: u32, k: u32) {
    let mut vals = Shared::<[f32]>::new_slice(1024usize);
    let mut sv = Shared::<[f32]>::new_slice(1024usize);
    let mut si = Shared::<[u32]>::new_slice(1024usize);
    let mut tv = Shared::<[f32]>::new_slice(64usize);
    let mut ti = Shared::<[u32]>::new_slice(64usize);
    let tid = UNIT_POS_X;
    if tid < ne { vals[tid as usize] = pk[tid as usize]; } else { vals[tid as usize] = f32::new(-3.0e38); }
    sync_cube();
    let mut t = 0u32;
    while t < k {
        sv[tid as usize] = vals[tid as usize]; si[tid as usize] = tid; sync_cube();
        let mut h = 512u32;
        while h > 0u32 {
            if tid < h { let a = sv[tid as usize]; let b = sv[(tid + h) as usize];
                if b > a || (b == a && si[(tid + h) as usize] < si[tid as usize]) { sv[tid as usize] = b; si[tid as usize] = si[(tid + h) as usize]; } }
            sync_cube(); h = h / 2u32;
        }
        if tid == 0u32 { let b0 = u32::min(si[0], ne - 1u32); tv[t as usize] = sv[0]; ti[t as usize] = b0; vals[b0 as usize] = f32::new(-3.0e38); }
        sync_cube(); t += 1u32;
    }
    if tid == 0u32 {
        let mx = tv[0]; let mut s = 0.0f32; let mut q = 0u32;
        while q < k { if smap[ti[q as usize] as usize] >= 0.0f32 { s += f32::exp(tv[q as usize] - mx); } q += 1u32; }
        let mut q2 = 0u32;
        while q2 < k { let sl = smap[ti[q2 as usize] as usize]; sel[q2 as usize] = sl;
            if sl >= 0.0f32 && s > 0.0f32 { sel[(k + q2) as usize] = f32::exp(tv[q2 as usize] - mx) / s; } else { sel[(k + q2) as usize] = 0.0f32; }
            sel[(2u32 * k + q2) as usize] = f32::cast_from(ti[q2 as usize]); q2 += 1u32; }
    }
}
/// gathered Q8 GEMV: cube (x, j): rows of slot sel[j]'s matrix (wsl words / ssl scales per slot) times x[j*xst..]
#[cube(launch_unchecked)]
fn k_xgv(x: &[f32], w: &[u32], s: &[f32], sel: &[f32], out: &mut [f32], rows: u32, cols: u32, xst: u32, wsl: u32, ssl: u32) {
    let j = CUBE_POS_Y; let row = CUBE_POS_X * CUBE_DIM_Y + UNIT_POS_Y;
    if row < rows {
        let sl = sel[j as usize];
        if sl >= 0.0f32 {
            let slot = u32::cast_from(sl); let words = cols / 4; let wb = slot * wsl + row * words; let sb = slot * ssl + row * (cols / 32); let xo = j * xst;
            let mut acc = 0.0f32; let mut wi = UNIT_POS_X;
            while wi < words { let wd = w[(wb + wi) as usize]; let kk = wi * 4; let mut p = 0.0f32;
                #[unroll] for q in 0..4u32 { p += dq(wd, q) * x[(xo + kk + q) as usize]; }
                acc += p * s[(sb + kk / 32) as usize]; wi += 32u32; }
            let tt = plane_sum(acc);
            if UNIT_POS_X == 0u32 { out[(j * rows + row) as usize] = tt; }
        } else { if UNIT_POS_X == 0u32 { out[(j * rows + row) as usize] = 0.0f32; } }
    }
}
#[cube(launch_unchecked)]
fn k_xact(h: &[f32], a: &mut [f32], ff: u32, n: u32) { let i = ABSOLUTE_POS_X; if i < n { let j = i / ff; let q = i % ff; let g = h[(j * 2u32 * ff + q) as usize]; a[i as usize] = g * sg(g) * h[(j * 2u32 * ff + ff + q) as usize]; } }
/// y = sum_j sel weight_j * o[j*d..] + shared expert y (pk[ne+d..]) * sigmoid(sgate . m2); one 256-thread cube per 256 outputs, dot redone per cube
#[cube(launch_unchecked)]
fn k_xsum(o: &[f32], sel: &[f32], pk: &[f32], sgw: &[f32], y: &mut [f32], d: u32, k: u32, ne: u32) {
    let mut sh = Shared::<[f32]>::new_slice(256usize);
    let tid = UNIT_POS_X; let mut a = 0.0f32; let mut i = tid;
    while i < d { a += sgw[i as usize] * pk[(ne + i) as usize]; i += 256u32; }
    sh[tid as usize] = a; sync_cube();
    let mut h = 128u32;
    while h > 0u32 { if tid < h { sh[tid as usize] = sh[tid as usize] + sh[(tid + h) as usize]; } sync_cube(); h = h / 2u32; }
    let g = sg(sh[0]); let r = CUBE_POS_X * 256u32 + tid;
    if r < d { let mut v = pk[(ne + d + r) as usize] * g; let mut j = 0u32; while j < k { v += sel[(k + j) as usize] * o[(j * d + r) as usize]; j += 1u32; } y[r as usize] = v; }
}
/// chain step: pb <- next position; pb[4..4+d] <- Q8 embedding row of the draft head index in bo[0]
#[cube(launch_unchecked)]
fn k_next(pb: &mut [f32], bo: &[f32], ew: &[u32], es: &[f32], d: u32, cap: u32, row: u32, window: u32) {
    let i = u32::cast_from(bo[0]); let tid = UNIT_POS_X; let mut c = tid;
    while c < d { let wd = ew[((i * d + c) / 4u32) as usize]; pb[(4u32 + c) as usize] = dq(wd, c % 4u32) * es[((i * d + c) / 32u32) as usize]; c += 256u32; }
    if tid == 0u32 { let pos = u32::cast_from(pb[0]) + 1u32; let n = pos + 1u32; let mut c0 = u32::cast_from(pb[2]); if n > window { if n - window > c0 { c0 = n - window; } }
        pb[0] = f32::cast_from(pos); pb[1] = f32::cast_from((pos % cap) * row); pb[2] = f32::cast_from(c0); pb[3] = f32::cast_from(n - c0); }
}
#[cube(launch_unchecked)]
fn k_cpw(src: &[u32], dst: &mut [u32], dof: u32, n: u32) { let i = ABSOLUTE_POS_X; if i < n { dst[(dof + i) as usize] = src[i as usize]; } }

pub fn xsel(cb: &Cube, pk: &H, pn: usize, smap: &H, sel: &H, ne: usize, k: usize) { unsafe { k_xsel::launch_unchecked(&cb.client, CubeCount::Static(1, 1, 1), CubeDim::new_1d(1024), ba!(pk, pn), ba!(smap, ne), ba!(sel, 3 * k), ne as u32, k as u32) } }
#[allow(clippy::too_many_arguments)]
pub fn xgv(cb: &Cube, x: &H, xn: usize, w: &H, wn: usize, s: &H, sn: usize, sel: &H, k: usize, out: &H, rows: usize, cols: usize, xst: usize, wsl: usize, ssl: usize) {
    unsafe { k_xgv::launch_unchecked(&cb.client, CubeCount::Static((rows as u32).div_ceil(8), k as u32, 1), CubeDim::new_2d(32, 8), ba!(x, xn), ba!(w, wn), ba!(s, sn), ba!(sel, 3 * k), ba!(out, k * rows), rows as u32, cols as u32, xst as u32, wsl as u32, ssl as u32) }
}
pub fn xact(cb: &Cube, h: &H, a: &H, ff: usize, k: usize) { let n = ff * k; unsafe { k_xact::launch_unchecked(&cb.client, g1(n), CubeDim::new_1d(256), ba!(h, 2 * n), ba!(a, n), ff as u32, n as u32) } }
#[allow(clippy::too_many_arguments)]
pub fn xsum(cb: &Cube, o: &H, sel: &H, pk: &H, pn: usize, sgw: &H, y: &H, d: usize, k: usize, ne: usize) { unsafe { k_xsum::launch_unchecked(&cb.client, g1(d), CubeDim::new_1d(256), ba!(o, k * d), ba!(sel, 3 * k), ba!(pk, pn), ba!(sgw, d), ba!(y, d), d as u32, k as u32, ne as u32) } }
#[allow(clippy::too_many_arguments)]
pub fn next(cb: &Cube, pb: &H, bo: &H, ew: &H, ewn: usize, es: &H, esn: usize, d: usize, cap: usize, row: usize, window: usize) { unsafe { k_next::launch_unchecked(&cb.client, CubeCount::Static(1, 1, 1), CubeDim::new_1d(256), ba!(pb, 4 + d), ba!(bo, 2), ba!(ew, ewn), ba!(es, esn), d as u32, cap as u32, row as u32, window as u32) } }
pub fn cpw(cb: &Cube, src: &H, dst: &H, dn: usize, dof: usize, n: usize) { unsafe { k_cpw::launch_unchecked(&cb.client, g1(n), CubeDim::new_1d(256), ba!(src, n), ba!(dst, dn), dof as u32, n as u32) } }

// PYRE_MTP_GX=2 (resident drafter): all routed drafter experts resident in VRAM as low-bit codes (PYRE_MTP_GXR_BITS, 2 default
// or 4) with one (f16 scale, f16 min) pair per 64 weights. Idea after Strata (MIT, github.com/Niko1221/Strata, src/core/mtp.cpp:
// all 512 drafter experts resident as Q2_0 and run inside the draft round graph, no CPU expert step); kernel is pyre's own.
#[cube]
fn h2f(h: u32) -> f32 { let e = (h >> 10u32) & 31u32; let mut r = 0.0f32; if e != 0u32 { r = f32::reinterpret(((h & 0x8000u32) << 16u32) | ((e + 112u32) << 23u32) | ((h & 0x3ffu32) << 13u32)); } r }
/// gathered low-bit GEMV: like k_xgv, w = packed codes (32/bits per word, row-major), s = per-64 (scale | min << 16) f16 pairs
#[cube(launch_unchecked)]
fn k_xgvq(x: &[f32], w: &[u32], s: &[u32], sel: &[f32], out: &mut [f32], rows: u32, cols: u32, xst: u32, wsl: u32, ssl: u32, #[comptime] bits: u32) {
    let j = CUBE_POS_Y; let row = CUBE_POS_X * CUBE_DIM_Y + UNIT_POS_Y;
    if row < rows {
        let sl = sel[j as usize];
        if sl >= 0.0f32 {
            // codes interleaved in 8-word chunks (word li of chunk c holds columns c*8*per + q*8 + li, q < per) so the 8 lanes of a
            // chunk read 8 consecutive x values per code; one (scale, min) per 64 columns = per 8 codes of a word
            let slot = u32::cast_from(sl); let words = cols * bits / 32u32; let wb = slot * wsl + row * words; let sb = slot * ssl + row * (cols / 64u32); let xo = j * xst; let mask = (1u32 << bits) - 1u32;
            let mut acc = 0.0f32; let mut wi = UNIT_POS_X;
            while wi < words { let wd = w[(wb + wi) as usize]; let base = (wi / 8u32) * comptime!(8 * (32 / bits)) + wi % 8u32;
                #[unroll] for h in 0..comptime!(32 / bits / 8) { let mut pc = 0.0f32; let mut px = 0.0f32;
                    #[unroll] for r in 0..8u32 { let q = h * 8u32 + r; let xv = x[(xo + base + q * 8u32) as usize]; pc += f32::cast_from((wd >> (q * bits)) & mask) * xv; px += xv; }
                    let sm = s[(sb + ((wi / 8u32) * comptime!(8 * (32 / bits)) + h * 64u32) / 64u32) as usize]; acc += h2f(sm & 0xffffu32) * pc + h2f(sm >> 16u32) * px; }
                wi += 32u32; }
            let tt = plane_sum(acc);
            if UNIT_POS_X == 0u32 { out[(j * rows + row) as usize] = tt; }
        } else { if UNIT_POS_X == 0u32 { out[(j * rows + row) as usize] = 0.0f32; } }
    }
}
#[allow(clippy::too_many_arguments)]
pub fn xgvq(cb: &Cube, x: &H, xn: usize, w: &H, wn: usize, s: &H, sn: usize, sel: &H, k: usize, out: &H, rows: usize, cols: usize, xst: usize, wsl: usize, ssl: usize, bits: usize) {
    unsafe { k_xgvq::launch_unchecked(&cb.client, CubeCount::Static((rows as u32).div_ceil(8), k as u32, 1), CubeDim::new_2d(32, 8), ba!(x, xn), ba!(w, wn), ba!(s, sn), ba!(sel, 3 * k), ba!(out, k * rows), rows as u32, cols as u32, xst as u32, wsl as u32, ssl as u32, bits as u32) }
}

/// one 32-lane plane: top-k of pk[0..ne] (ne <= 1024) with plane max/min reductions instead of 1024-thread tree passes; same output as k_xsel
#[cube(launch_unchecked)]
fn k_xsel2(pk: &[f32], smap: &[f32], sel: &mut [f32], ne: u32, k: u32) {
    let lane = UNIT_POS_X; let mut v = Array::<f32>::new(32usize);
    #[unroll] for i in 0..32u32 { let e = i * 32u32 + lane; if e < ne { v[i as usize] = pk[e as usize]; } else { v[i as usize] = f32::new(-3.0e38); } }
    let mut t = 0u32;
    while t < k {
        let mut bv = f32::new(-3.0e38); let mut bi = 0u32;
        #[unroll] for q in 0..32u32 { if v[q as usize] > bv { bv = v[q as usize]; bi = q; } }
        let gm = plane_max(bv); let mut cand = 0xffffffffu32; if bv == gm { cand = bi * 32u32 + lane; }
        let ge = u32::min(plane_min(cand), ne - 1u32);
        if ge % 32u32 == lane { #[unroll] for q in 0..32u32 { if q == ge / 32u32 { v[q as usize] = f32::new(-3.0e38); } } }
        if lane == 0u32 { sel[(2u32 * k + t) as usize] = f32::cast_from(ge); sel[(k + t) as usize] = gm; }
        t += 1u32;
    }
    if lane == 0u32 {
        let mx = sel[k as usize]; let mut s = 0.0f32; let mut q = 0u32;
        while q < k { let e = u32::cast_from(sel[(2u32 * k + q) as usize]); if smap[e as usize] >= 0.0f32 { s += f32::exp(sel[(k + q) as usize] - mx); } q += 1u32; }
        let mut q2 = 0u32;
        while q2 < k { let e = u32::cast_from(sel[(2u32 * k + q2) as usize]); let sl = smap[e as usize]; let lv = sel[(k + q2) as usize]; sel[q2 as usize] = sl;
            if sl >= 0.0f32 && s > 0.0f32 { sel[(k + q2) as usize] = f32::exp(lv - mx) / s; } else { sel[(k + q2) as usize] = 0.0f32; } q2 += 1u32; }
    }
}
pub fn xsel2(cb: &Cube, pk: &H, pn: usize, smap: &H, sel: &H, ne: usize, k: usize) { unsafe { k_xsel2::launch_unchecked(&cb.client, CubeCount::Static(1, 1, 1), CubeDim::new_1d(32), ba!(pk, pn), ba!(smap, ne), ba!(sel, 3 * k), ne as u32, k as u32) } }
