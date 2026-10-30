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
