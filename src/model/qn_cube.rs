//! Hand-written CubeCL kernels for the qwen4exp decode step (one token): the dense half of every
//! layer runs as ~14 launches on one stream with two readbacks, instead of ~60 small Burn ops.
//! Idea (fused per-layer GPU step, routed experts on the CPU) from Strata (MIT,
//! github.com/Niko1221/Strata); the kernels and layout here are pyre's own.
//! Weights are Q8_0 (32 x i8 per f32 scale), packed 4 per u32 word, one row after another.
use super::gguf_q::QMat;
use burn_cubecl::cubecl;
use cubecl::prelude::*;
use cubecl::server::Handle;

const ROWS_PER_CUBE: u32 = 8;

#[cube]
fn dot4(word: u32, x: &[f32], base: u32) -> f32 {
    let mut part = 0.0f32;
    #[unroll]
    for j in 0..4u32 {
        let raw = (word >> (j * 8)) & 255u32;
        let v = f32::cast_from(raw) - f32::cast_from(((raw >> 7) & 1u32) << 8);
        part += v * x[(base + j) as usize];
    }
    part
}

#[cube(launch_unchecked)]
fn k_gemv(x: &[f32], w: &[u32], s: &[f32], out: &mut [f32], rows: u32, cols: u32, xoff: u32, ooff: u32) {
    let row = CUBE_POS_X * CUBE_DIM_Y + UNIT_POS_Y;
    if row < rows {
        let words = cols / 4;
        let wb = row * words;
        let sb = row * (cols / 32);
        let mut acc = 0.0f32;
        let mut wi = UNIT_POS_X;
        // four independent loads in flight per lane
        while wi + 96 < words {
            let w0 = w[(wb + wi) as usize];
            let w1 = w[(wb + wi + 32) as usize];
            let w2 = w[(wb + wi + 64) as usize];
            let w3 = w[(wb + wi + 96) as usize];
            let k = wi * 4;
            acc += dot4(w0, x, xoff + k) * s[(sb + k / 32) as usize];
            acc += dot4(w1, x, xoff + k + 128) * s[(sb + k / 32 + 4) as usize];
            acc += dot4(w2, x, xoff + k + 256) * s[(sb + k / 32 + 8) as usize];
            acc += dot4(w3, x, xoff + k + 384) * s[(sb + k / 32 + 12) as usize];
            wi += 128;
        }
        while wi < words {
            let k = wi * 4;
            acc += dot4(w[(wb + wi) as usize], x, xoff + k) * s[(sb + k / 32) as usize];
            wi += 32;
        }
        let t = plane_sum(acc);
        if UNIT_POS_X == 0 {
            out[(ooff + row) as usize] = t;
        }
    }
}

/// k_gemv for up to 4 activation columns (x at xoff + c*xst, out at ooff + c*ost): each weight word is
/// loaded once for all columns; every column accumulates in exactly k_gemv's order (same results).
#[cube(launch_unchecked)]
fn k_gemv4(x: &[f32], w: &[u32], s: &[f32], out: &mut [f32], rows: u32, cols: u32, xoff: u32, xst: u32, ooff: u32, ost: u32, nt: u32) {
    let row = CUBE_POS_X * CUBE_DIM_Y + UNIT_POS_Y;
    if row < rows {
        let words = cols / 4;
        let wb = row * words;
        let sb = row * (cols / 32);
        let mut a0 = 0.0f32; let mut a1 = 0.0f32; let mut a2 = 0.0f32; let mut a3 = 0.0f32;
        let (x1, x2, x3) = (xoff + xst, xoff + 2 * xst, xoff + 3 * xst);
        let mut wi = UNIT_POS_X;
        while wi + 96 < words {
            let w0 = w[(wb + wi) as usize];
            let w1 = w[(wb + wi + 32) as usize];
            let w2 = w[(wb + wi + 64) as usize];
            let w3 = w[(wb + wi + 96) as usize];
            let k = wi * 4;
            let s0 = s[(sb + k / 32) as usize]; let s1 = s[(sb + k / 32 + 4) as usize]; let s2 = s[(sb + k / 32 + 8) as usize]; let s3 = s[(sb + k / 32 + 12) as usize];
            a0 += dot4(w0, x, xoff + k) * s0; a0 += dot4(w1, x, xoff + k + 128) * s1; a0 += dot4(w2, x, xoff + k + 256) * s2; a0 += dot4(w3, x, xoff + k + 384) * s3;
            if nt > 1 { a1 += dot4(w0, x, x1 + k) * s0; a1 += dot4(w1, x, x1 + k + 128) * s1; a1 += dot4(w2, x, x1 + k + 256) * s2; a1 += dot4(w3, x, x1 + k + 384) * s3; }
            if nt > 2 { a2 += dot4(w0, x, x2 + k) * s0; a2 += dot4(w1, x, x2 + k + 128) * s1; a2 += dot4(w2, x, x2 + k + 256) * s2; a2 += dot4(w3, x, x2 + k + 384) * s3; }
            if nt > 3 { a3 += dot4(w0, x, x3 + k) * s0; a3 += dot4(w1, x, x3 + k + 128) * s1; a3 += dot4(w2, x, x3 + k + 256) * s2; a3 += dot4(w3, x, x3 + k + 384) * s3; }
            wi += 128;
        }
        while wi < words {
            let k = wi * 4;
            let w0 = w[(wb + wi) as usize]; let s0 = s[(sb + k / 32) as usize];
            a0 += dot4(w0, x, xoff + k) * s0;
            if nt > 1 { a1 += dot4(w0, x, x1 + k) * s0; }
            if nt > 2 { a2 += dot4(w0, x, x2 + k) * s0; }
            if nt > 3 { a3 += dot4(w0, x, x3 + k) * s0; }
            wi += 32;
        }
        let t0 = plane_sum(a0); let t1 = plane_sum(a1); let t2 = plane_sum(a2); let t3 = plane_sum(a3);
        if UNIT_POS_X == 0 {
            out[(ooff + row) as usize] = t0;
            if nt > 1 { out[(ooff + ost + row) as usize] = t1; }
            if nt > 2 { out[(ooff + 2 * ost + row) as usize] = t2; }
            if nt > 3 { out[(ooff + 3 * ost + row) as usize] = t3; }
        }
    }
}

/// GEMV straight from IQ4_XS blocks (repacked: f32 d, scales_h, scales_l, 32 words of nibbles =
/// 35 words per 256). Each lane takes one qs word: 4 low nibbles (x[32ib+4k..]) and 4 high
/// nibbles (x[32ib+16+4k..]), so adjacent lanes read adjacent activations. kv = the 16 IQ4_NL levels.
#[cube(launch_unchecked)]
fn k_gemv_iq4(x: &[f32], w: &[u32], kv: &[f32], out: &mut [f32], rows: u32, cols: u32, xoff: u32, ooff: u32) {
    // x staged in shared memory, padded by one float per 32 so the (sub-block, word) lane layout
    // below is (nearly) bank-conflict free; the 8 rows of a cube share it.
    let mut xs = Shared::<[f32]>::new_slice(6336usize);
    let mut tab = Shared::<[f32]>::new_slice(16usize);
    let mut i = UNIT_POS;
    while i < cols {
        xs[(i + i / 32) as usize] = x[(xoff + i) as usize];
        i += CUBE_DIM;
    }
    if UNIT_POS < 16 {
        tab[UNIT_POS as usize] = kv[UNIT_POS as usize];
    }
    sync_cube();
    let row = CUBE_POS_X * CUBE_DIM_Y + UNIT_POS_Y;
    if row < rows {
        let rw = (cols / 256) * 35;
        let nb = cols / 256;
        let ib = UNIT_POS_X / 4;
        let k = UNIT_POS_X % 4;
        let mut acc = 0.0f32;
        // two blocks per step: all eight global loads issued before any use
        let mut blk = 0u32;
        while blk < nb {
            let b0 = row * rw + blk * 35;
            let b1 = b0 + 35;
            let d0 = f32::reinterpret(w[b0 as usize]);
            let sh0 = w[(b0 + 1) as usize];
            let sl0 = w[(b0 + 2) as usize];
            let q0 = w[(b0 + 3 + UNIT_POS_X) as usize];
            let mut d1 = 0.0f32;
            let mut sh1 = 0u32;
            let mut sl1 = 0u32;
            let mut q1 = 0u32;
            if blk + 1 < nb {
                d1 = f32::reinterpret(w[b1 as usize]);
                sh1 = w[(b1 + 1) as usize];
                sl1 = w[(b1 + 2) as usize];
                q1 = w[(b1 + 3 + UNIT_POS_X) as usize];
            }
            let ls0 = ((sl0 >> (4 * ib)) & 15u32) | (((sh0 >> (2 * ib)) & 3u32) << 4);
            let ls1 = ((sl1 >> (4 * ib)) & 15u32) | (((sh1 >> (2 * ib)) & 3u32) << 4);
            let p0 = blk * 264 + ib * 33 + 4 * k;
            let p1 = p0 + 264;
            let mut part0 = 0.0f32;
            let mut part1 = 0.0f32;
            #[unroll]
            for j in 0..4u32 {
                part0 += tab[((q0 >> (8 * j)) & 15u32) as usize] * xs[(p0 + j) as usize];
                part0 += tab[((q0 >> (8 * j + 4)) & 15u32) as usize] * xs[(p0 + 16 + j) as usize];
            }
            acc += part0 * (d0 * (f32::cast_from(ls0) - 32.0f32));
            if blk + 1 < nb {
                #[unroll]
                for j in 0..4u32 {
                    part1 += tab[((q1 >> (8 * j)) & 15u32) as usize] * xs[(p1 + j) as usize];
                    part1 += tab[((q1 >> (8 * j + 4)) & 15u32) as usize] * xs[(p1 + 16 + j) as usize];
                }
                acc += part1 * (d1 * (f32::cast_from(ls1) - 32.0f32));
            }
            blk += 2u32;
        }        let t = plane_sum(acc);
        if UNIT_POS_X == 0 {
            out[(ooff + row) as usize] = t;
        }
    }
}
/// One cube (256 units) per stream: xn = x * rsqrt(mean(x^2) + eps) * norm.
#[cube(launch_unchecked)]
fn k_rms(x: &[f32], norm: &[f32], xn: &mut [f32], d: u32, eps: f32) {
    let base = CUBE_POS_X * d;
    let mut acc = 0.0f32;
    let mut i = UNIT_POS_X;
    while i < d {
        let v = x[(base + i) as usize];
        acc += v * v;
        i += CUBE_DIM_X;
    }
    let p = plane_sum(acc);
    let mut sh = Shared::<[f32]>::new_slice(8usize);
    if UNIT_POS_X % 32 == 0 {
        sh[(UNIT_POS_X / 32) as usize] = p;
    }
    sync_cube();
    let mut tot = 0.0f32;
    for j in 0..8u32 {
        tot += sh[j as usize];
    }
    let r = 1.0f32 / f32::sqrt(tot / f32::cast_from(d) + eps);
    let mut i2 = UNIT_POS_X;
    while i2 < d {
        let k = (base + i2) as usize;
        xn[k] = x[k] * r * norm[k];
        i2 += CUBE_DIM_X;
    }
}

/// mixed[i] = mean_s xn[s,i] * sigmoid(up[s*d+i] . silu(dn[0..lr]/hc)); one plane per output i.
#[cube(launch_unchecked)]
fn k_gate(xn: &[f32], dn: &[f32], w: &[u32], s: &[f32], out: &mut [f32], d: u32, ooff: u32, #[comptime] lr: usize, #[comptime] hc: u32) {
    let mut lo = Shared::<[f32]>::new_slice(lr);
    let hcf = f32::cast_from(hc);
    let tid = UNIT_POS_Y * 32 + UNIT_POS_X;
    let mut j = tid;
    while (j as usize) < lr {
        let v = dn[j as usize] / hcf;
        lo[j as usize] = v / (1.0f32 + f32::exp(-v));
        j += 256u32;
    }
    sync_cube();
    let i = CUBE_POS_X * 8 + UNIT_POS_Y;
    if i < d {
        let words = comptime!(lr as u32 / 4);
        let nb = comptime!(lr as u32 / 32);
        let mut acc = 0.0f32;
        #[unroll]
        for st in 0..hc {
            let row = st * d + i;
            let mut g = 0.0f32;
            let mut wi = UNIT_POS_X;
            while wi < words {
                let word = w[(row * words + wi) as usize];
                let mut part = 0.0f32;
                #[unroll]
                for jj in 0..4u32 {
                    let raw = (word >> (jj * 8)) & 255u32;
                    let v = f32::cast_from(raw) - f32::cast_from(((raw >> 7) & 1u32) << 8);
                    part += v * lo[(wi * 4 + jj) as usize];
                }
                g += part * s[(row * nb + wi / 8) as usize];
                wi += 32u32;
            }
            let gt = plane_sum(g);
            acc += xn[row as usize] / (1.0f32 + f32::exp(-gt));
        }
        if UNIT_POS_X == 0 {
            out[(ooff + i) as usize] = acc / hcf;
        }
    }
}

/// x[s,i] += (a[i] * (gate? sigmoid(gb[goff]) : 1) + (b? b[i] : 0)) * 2 sigmoid(dn[lr+s]/hc)
#[cube(launch_unchecked)]
fn k_comb(x: &mut [f32], a: &[f32], b: &[f32], c: &[f32], gb: &[f32], dn: &[f32], d: u32, goff: u32, lr: u32, #[comptime] hc: u32, #[comptime] gate: bool, #[comptime] addb: bool, #[comptime] addc: bool) {
    let k = ABSOLUTE_POS as u32;
    if k < d * hc {
        let st = k / d;
        let i = k % d;
        let mut o = a[i as usize];
        if comptime!(gate) {
            o = o / (1.0f32 + f32::exp(-gb[goff as usize]));
        }
        if comptime!(addb) {
            o += b[i as usize];
        }
        if comptime!(addc) {
            o += c[i as usize];
        }
        let inj = dn[(lr + st) as usize] / f32::cast_from(hc);
        x[k as usize] += o * 2.0f32 / (1.0f32 + f32::exp(-inj));
    }
}

#[cube(launch_unchecked)]
fn k_copy(src: &[f32], dst: &mut [f32], so: u32, dof: u32, n: u32) {
    let j = ABSOLUTE_POS as u32;
    if j < n {
        dst[(dof + j) as usize] = src[(so + j) as usize];
    }
}

#[cube(launch_unchecked)]
fn k_silu_mul(f: &[f32], h: &mut [f32], og: u32, ou: u32, n: u32) {
    let j = ABSOLUTE_POS as u32;
    if j < n {
        let g = f[(og + j) as usize];
        h[j as usize] = g / (1.0f32 + f32::exp(-g)) * f[(ou + j) as usize];
    }
}

/// Gated DeltaNet decode, step 1: causal depthwise conv + SiLU per channel, shifting the conv state.
#[cube(launch_unchecked)]
fn k_conv(proj: &[f32], w: &[f32], cs: &mut [f32], co: &mut [f32], ch: u32, #[comptime] kk: u32) {
    let c = ABSOLUTE_POS as u32;
    if c < ch {
        let cur = proj[c as usize];
        let mut o = w[(c * kk + kk - 1) as usize] * cur;
        #[unroll]
        for k in 0..kk - 1 {
            o += w[(c * kk + k) as usize] * cs[(k * ch + c) as usize];
        }
        co[c as usize] = o / (1.0f32 + f32::exp(-o));
        #[unroll]
        for k in 0..kk - 2 {
            cs[(k * ch + c) as usize] = cs[((k + 1) * ch + c) as usize];
        }
        cs[((kk - 2) * ch + c) as usize] = cur;
    }
}

#[cube]
fn cube_sum(v: f32, red: &mut Shared<[f32]>, #[comptime] planes: u32) -> f32 {
    let p = plane_sum(v);
    if UNIT_POS_X % 32 == 0 {
        red[(UNIT_POS_X / 32) as usize] = p;
    }
    sync_cube();
    let mut t = 0.0f32;
    #[unroll]
    for i in 0..planes {
        t += red[i as usize];
    }
    sync_cube();
    t
}

/// Gated DeltaNet decode, step 2: one cube per value head, one unit per value column j (dk = dv = CUBE_DIM).
/// q/k L2 norms, decay g = exp(softplus(alpha + dt) * a), delta rule, RMSNorm * sigmoid(z).
#[cube(launch_unchecked)]
fn k_delta(co: &[f32], proj: &[f32], a: &[f32], dt: &[f32], norm: &[f32], s: &mut [f32], out: &mut [f32],
    nk: u32, nv: u32, zoff: u32, boff: u32, aoff: u32, eps: f32, #[comptime] dk: u32) {
    let h = CUBE_POS_X;
    let j = UNIT_POS_X;
    let kh = h % nk;
    let mut qs = Shared::<[f32]>::new_slice(dk as usize);
    let mut ks = Shared::<[f32]>::new_slice(dk as usize);
    let mut red = Shared::<[f32]>::new_slice(comptime!((dk / 32) as usize));
    let qj = co[(kh * dk + j) as usize];
    let kj = co[(nk * dk + kh * dk + j) as usize];
    let sq = cube_sum(qj * qj, &mut red, comptime!(dk / 32));
    let sk = cube_sum(kj * kj, &mut red, comptime!(dk / 32));
    qs[j as usize] = qj / f32::sqrt(sq + 1e-6f32) / f32::sqrt(f32::cast_from(dk));
    ks[j as usize] = kj / f32::sqrt(sk + 1e-6f32);
    sync_cube();
    let vj = co[(2 * nk * dk + h * dk + j) as usize];
    let x = proj[(aoff + h) as usize] + dt[h as usize];
    let mut sp = x;
    if x <= 20.0f32 {
        sp = f32::ln(1.0f32 + f32::exp(x));
    }
    let g = f32::exp(sp * a[h as usize]);
    let b = 1.0f32 / (1.0f32 + f32::exp(-proj[(boff + h) as usize]));
    let base = h * dk * dk;
    let mut mem = 0.0f32;
    for i in 0..dk {
        mem += s[(base + i * dk + j) as usize] * ks[i as usize];
    }
    mem *= g;
    let delta = (vj - mem) * b;
    let mut o = 0.0f32;
    for i in 0..dk {
        let idx = (base + i * dk + j) as usize;
        let v = s[idx] * g + ks[i as usize] * delta;
        s[idx] = v;
        o += v * qs[i as usize];
    }
    let so = cube_sum(o * o, &mut red, comptime!(dk / 32));
    let r = 1.0f32 / f32::sqrt(so / f32::cast_from(dk) + eps);
    let z = proj[(zoff + h * dk + j) as usize];
    out[(h * dk + j) as usize] = o * r * norm[j as usize] / (1.0f32 + f32::exp(-z));
    let _ = nv;
}

// ---- routed experts cached on the GPU in their native GGUF formats (Strata's expert cache idea) ----
// type codes: 0 = IQ2_S, 1 = IQ2_XXS, 2 = IQ1_M, 3 = Q2_0

#[cube]
fn byte(w: &[u32], off: u32) -> u32 {
    (w[(off / 4) as usize] >> ((off % 4) * 8)) & 255u32
}

#[cube]
fn hf(h: u32) -> f32 {
    let e = (h >> 10) & 31u32;
    let m = f32::cast_from(h & 1023u32);
    let mut v = m * 0.000000059604645f32;
    if e != 0 {
        v = (1.0f32 + m / 1024.0f32) * f32::powf(2.0f32, f32::cast_from(e) - 15.0f32);
    }
    if (h >> 15) & 1u32 == 1u32 {
        v = -v;
    }
    v
}

/// Dot of the 32 weights of sub-block `sb` of a row (starting at word `rb`, repacked by `repack`)
/// with x[xo + sb*32 ..].
#[cube]
fn subdot(w: &[u32], rb: u32, sb: u32, x: &[f32], xo: u32, g2s: &[u32], g2x: &[u32], ks: &[u32], g1: &[u32], #[comptime] ty: u32) -> f32 {
    let mut acc = 0.0f32;
    let xb = xo + sb * 32;
    if comptime!(ty == 0) {
        let blk = rb + (sb / 8) * 21;
        let ib = sb % 8;
        let d = f32::reinterpret(w[blk as usize]);
        let qs = w[(blk + 1 + ib) as usize];
        let sg = w[(blk + 9 + ib) as usize];
        let qh = (w[(blk + 17 + ib / 4) as usize] >> ((ib % 4) * 8)) & 255u32;
        let sc = (w[(blk + 19 + ib / 4) as usize] >> ((ib % 4) * 8)) & 255u32;
        #[unroll]
        for l in 0..4u32 {
            let idx = ((qs >> (8 * l)) & 255u32) | ((qh << (8 - 2 * l)) & 768u32);
            let s8 = (sg >> (8 * l)) & 255u32;
            let mut scl = 0.5f32 + f32::cast_from(sc & 15u32);
            if l >= 2 {
                scl = 0.5f32 + f32::cast_from(sc >> 4);
            }
            let ga = g2s[(idx * 2) as usize];
            let gb = g2s[(idx * 2 + 1) as usize];
            let mut part = 0.0f32;
            #[unroll]
            for j in 0..8u32 {
                let mut gw = ga;
                if j >= 4 {
                    gw = gb;
                }
                let mut v = f32::cast_from((gw >> ((j % 4) * 8)) & 255u32) * x[(xb + 8 * l + j) as usize];
                if (s8 >> j) & 1u32 == 1u32 {
                    v = -v;
                }
                part += v;
            }
            acc += part * scl;
        }
        acc *= d;
    }
    if comptime!(ty == 1) {
        let blk = rb + (sb / 8) * 17;
        let ib = sb % 8;
        let d = f32::reinterpret(w[blk as usize]);
        let q = w[(blk + 1 + 2 * ib) as usize];
        let a1 = w[(blk + 2 + 2 * ib) as usize];
        let db = d * (0.5f32 + f32::cast_from(a1 >> 28));
        #[unroll]
        for l in 0..4u32 {
            let idx = (q >> (8 * l)) & 255u32;
            let sg = ks[((a1 >> (7 * l)) & 127u32) as usize];
            let ga = g2x[(idx * 2) as usize];
            let gb = g2x[(idx * 2 + 1) as usize];
            let mut part = 0.0f32;
            #[unroll]
            for j in 0..8u32 {
                let mut gw = ga;
                if j >= 4 {
                    gw = gb;
                }
                let mut v = f32::cast_from((gw >> ((j % 4) * 8)) & 255u32) * x[(xb + 8 * l + j) as usize];
                if (sg >> j) & 1u32 == 1u32 {
                    v = -v;
                }
                part += v;
            }
            acc += part;
        }
        acc *= db;
    }
    if comptime!(ty == 2) {
        let blk = rb + (sb / 8) * 15;
        let ib = sb % 8;
        let d = f32::reinterpret(w[blk as usize]);
        let qs = w[(blk + 1 + ib) as usize];
        let hw = (w[(blk + 9 + ib / 2) as usize] >> ((ib % 2) * 16)) & 65535u32;
        let scw = (w[(blk + 13 + ib / 4) as usize] >> (((ib / 2) % 2) * 16)) & 65535u32;
        let sh = 6 * (ib % 2);
        let dl1 = d * f32::cast_from(2 * ((scw >> sh) & 7u32) + 1);
        let dl2 = d * f32::cast_from(2 * ((scw >> (sh + 3)) & 7u32) + 1);
        #[unroll]
        for l in 0..4u32 {
            let qb = (qs >> (8 * l)) & 255u32;
            let hh = (hw >> (8 * (l / 2))) & 255u32;
            let mut idx = qb | ((hh << 8) & 1792u32);
            let mut dbit = hh & 8u32;
            if l % 2 == 1 {
                idx = qb | ((hh << 4) & 1792u32);
                dbit = hh & 128u32;
            }
            let mut delta = 0.125f32;
            if dbit != 0 {
                delta = -0.125f32;
            }
            let mut sl = dl1;
            if l >= 2 {
                sl = dl2;
            }
            let ga = g1[(idx * 2) as usize];
            let gb2 = g1[(idx * 2 + 1) as usize];
            let mut part = 0.0f32;
            #[unroll]
            for j in 0..8u32 {
                let mut gw = ga;
                if j >= 4 {
                    gw = gb2;
                }
                let gb = (gw >> ((j % 4) * 8)) & 255u32;
                let mut gv = f32::cast_from(gb);
                if gb >= 128 {
                    gv = gv - 256.0f32;
                }
                part += (gv + delta) * x[(xb + 8 * l + j) as usize];
            }
            acc += part * sl;
        }
    }
    if comptime!(ty == 3) {
        let blk = rb + (sb / 2) * 5;
        let d = f32::reinterpret(w[blk as usize]);
        #[unroll]
        for k2 in 0..2u32 {
            let qw = w[(blk + 1 + 2 * (sb % 2) + k2) as usize];
            #[unroll]
            for k in 0..4u32 {
                let b = (qw >> (8 * k)) & 255u32;
                #[unroll]
                for j in 0..4u32 {
                    acc += (f32::cast_from((b >> (2 * j)) & 3u32) - 1.0f32) * x[(xb + 16 * k2 + 4 * k + j) as usize];
                }
            }
        }
        acc *= d;
    }
    acc
}


/// Like `subdot` but for 8 values only (`q` = index of the 8-group: sub-block q/4, part q%4), so
/// neighbouring lanes read neighbouring activations (the 32-per-lane layout made every x load a
/// 32-way bank conflict / uncoalesced access).
#[cube]
fn subdot8(w: &[u32], rb: u32, q: u32, x: &[f32], xo: u32, g2s: &[u32], g2x: &[u32], ks: &[u32], g1: &[u32], #[comptime] ty: u32) -> f32 {
    let mut acc = 0.0f32;
    let sb = q / 4;
    let l = q % 4;
    let xb = xo + q * 8;
    if comptime!(ty == 0) {
        let blk = rb + (sb / 8) * 21;
        let ib = sb % 8;
        let d = f32::reinterpret(w[blk as usize]);
        let qs = w[(blk + 1 + ib) as usize];
        let sg = w[(blk + 9 + ib) as usize];
        let qh = (w[(blk + 17 + ib / 4) as usize] >> ((ib % 4) * 8)) & 255u32;
        let sc = (w[(blk + 19 + ib / 4) as usize] >> ((ib % 4) * 8)) & 255u32;
        let idx = ((qs >> (8 * l)) & 255u32) | ((qh << (8 - 2 * l)) & 768u32);
        let s8 = (sg >> (8 * l)) & 255u32;
        let mut scl = 0.5f32 + f32::cast_from(sc & 15u32);
        if l >= 2 {
            scl = 0.5f32 + f32::cast_from(sc >> 4);
        }
        let ga = g2s[(idx * 2) as usize];
        let gb = g2s[(idx * 2 + 1) as usize];
        #[unroll]
        for j in 0..8u32 {
            let mut gw = ga;
            if j >= 4 {
                gw = gb;
            }
            let mut v = f32::cast_from((gw >> ((j % 4) * 8)) & 255u32) * x[(xb + j) as usize];
            if (s8 >> j) & 1u32 == 1u32 {
                v = -v;
            }
            acc += v;
        }
        acc *= d * scl;
    }
    if comptime!(ty == 1) {
        let blk = rb + (sb / 8) * 17;
        let ib = sb % 8;
        let d = f32::reinterpret(w[blk as usize]);
        let qw = w[(blk + 1 + 2 * ib) as usize];
        let a1 = w[(blk + 2 + 2 * ib) as usize];
        let db = d * (0.5f32 + f32::cast_from(a1 >> 28));
        let idx = (qw >> (8 * l)) & 255u32;
        let sg = ks[((a1 >> (7 * l)) & 127u32) as usize];
        let ga = g2x[(idx * 2) as usize];
        let gb = g2x[(idx * 2 + 1) as usize];
        #[unroll]
        for j in 0..8u32 {
            let mut gw = ga;
            if j >= 4 {
                gw = gb;
            }
            let mut v = f32::cast_from((gw >> ((j % 4) * 8)) & 255u32) * x[(xb + j) as usize];
            if (sg >> j) & 1u32 == 1u32 {
                v = -v;
            }
            acc += v;
        }
        acc *= db;
    }
    if comptime!(ty == 2) {
        let blk = rb + (sb / 8) * 15;
        let ib = sb % 8;
        let d = f32::reinterpret(w[blk as usize]);
        let qs = w[(blk + 1 + ib) as usize];
        let hw = (w[(blk + 9 + ib / 2) as usize] >> ((ib % 2) * 16)) & 65535u32;
        let scw = (w[(blk + 13 + ib / 4) as usize] >> (((ib / 2) % 2) * 16)) & 65535u32;
        let mut sh = 6 * (ib % 2);
        if l >= 2 {
            sh += 3;
        }
        let sl = d * f32::cast_from(2 * ((scw >> sh) & 7u32) + 1);
        let qb = (qs >> (8 * l)) & 255u32;
        let hh = (hw >> (8 * (l / 2))) & 255u32;
        let mut idx = qb | ((hh << 8) & 1792u32);
        let mut dbit = hh & 8u32;
        if l % 2 == 1 {
            idx = qb | ((hh << 4) & 1792u32);
            dbit = hh & 128u32;
        }
        let mut delta = 0.125f32;
        if dbit != 0 {
            delta = -0.125f32;
        }
        let ga = g1[(idx * 2) as usize];
        let gb2 = g1[(idx * 2 + 1) as usize];
        #[unroll]
        for j in 0..8u32 {
            let mut gw = ga;
            if j >= 4 {
                gw = gb2;
            }
            let gb = (gw >> ((j % 4) * 8)) & 255u32;
            let mut gv = f32::cast_from(gb);
            if gb >= 128 {
                gv = gv - 256.0f32;
            }
            acc += (gv + delta) * x[(xb + j) as usize];
        }
        acc *= sl;
    }
    if comptime!(ty == 3) {
        let blk = rb + (sb / 2) * 5;
        let d = f32::reinterpret(w[blk as usize]);
        let qw = w[(blk + 1 + 2 * (sb % 2) + l / 2) as usize];
        let half = (qw >> (16 * (l % 2))) & 65535u32;
        #[unroll]
        for k in 0..2u32 {
            let b = (half >> (8 * k)) & 255u32;
            #[unroll]
            for j in 0..4u32 {
                acc += (f32::cast_from((b >> (2 * j)) & 3u32) - 1.0f32) * x[(xb + 4 * k + j) as usize];
            }
        }
        acc *= d;
    }
    acc
}
/// Repack GGUF blocks into word-aligned blocks with an f32 scale first (layouts in `subdot`).
pub fn repack(ty: u32, raw: &[u8]) -> Vec<u32> {
    let f = |b: &[u8], o: usize| half::f16::from_bits(u16::from_le_bytes([b[o], b[o + 1]])).to_f32();
    let wd = |b: &[u8], o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
    let mut out = vec![];
    match ty {
        0 => for b in raw.chunks_exact(82) { out.push((f(b, 0) * 0.25).to_bits()); for o in (2..82).step_by(4) { out.push(wd(b, o)); } },
        1 => for b in raw.chunks_exact(66) { out.push((f(b, 0) * 0.25).to_bits()); for o in (2..66).step_by(4) { out.push(wd(b, o)); } },
        2 => for b in raw.chunks_exact(56) { let s = |k: usize| u16::from_le_bytes([b[48 + 2 * k], b[49 + 2 * k]]);
            let d = half::f16::from_bits((s(0) >> 12) | ((s(1) >> 8) & 0x00f0) | ((s(2) >> 4) & 0x0f00) | (s(3) & 0xf000)).to_f32();
            out.push(d.to_bits()); for o in (0..56).step_by(4) { out.push(wd(b, o)); } },
        _ => for b in raw.chunks_exact(18) { out.push(f(b, 0).to_bits()); for o in (2..18).step_by(4) { out.push(wd(b, o)); } },
    }
    out
}
pub fn block_words(ty: u32) -> (usize, usize) { match ty { 0 => (256, 21), 1 => (256, 17), 2 => (256, 15), _ => (64, 5) } }

/// h[row] = silu(gate_row . x) * (up_row . x) for one cached expert; one plane per row.
#[cube(launch_unchecked)]
fn k_xgu(x: &[f32], wg: &[u32], wu: &[u32], g2s: &[u32], g2x: &[u32], ks: &[u32], g1: &[u32], h: &mut [f32], hoff: u32, rows: u32, cols: u32, rbg: u32, rbu: u32,
    #[comptime] tg: u32, #[comptime] tu: u32) {
    let row = CUBE_POS_X * CUBE_DIM_Y + UNIT_POS_Y;
    if row < rows {
        let nsb = cols / 8;
        let mut gs = 0.0f32;
        let mut us = 0.0f32;
        let mut sb = UNIT_POS_X;
        while sb < nsb {
            gs += subdot8(wg, row * rbg, sb, x, 0u32, g2s, g2x, ks, g1, tg);
            us += subdot8(wu, row * rbu, sb, x, 0u32, g2s, g2x, ks, g1, tu);
            sb += 32u32;
        }
        let g = plane_sum(gs);
        let u = plane_sum(us);
        if UNIT_POS_X == 0 {
            h[(hoff + row) as usize] = g / (1.0f32 + f32::exp(-g)) * u;
        }
    }
}

/// Round h to the CPU's Q8 activation grid (per 32: d = amax/127), so GPU and CPU experts agree.
#[cube(launch_unchecked)]
fn k_q8sim(h: &mut [f32], n: u32) {
    let b = CUBE_POS_X * CUBE_DIM_Y + UNIT_POS_Y;
    let i = b * 32 + UNIT_POS_X;
    if b * 32 < n {
        let v = h[i as usize];
        let amax = plane_max(f32::abs(v));
        let d = amax / 127.0f32;
        let mut q = 0.0f32;
        if d > 0.0f32 {
            q = f32::round(v * (1.0f32 / d));
        }
        h[i as usize] = q * d;
    }
}

/// dst[i] = src[soff + i] rounded to the Q8 grid (per 32); one plane per block. Also zeroes `z`.
#[cube(launch_unchecked)]
fn k_q8copy(src: &[f32], dst: &mut [f32], z: &mut [f32], soff: u32, n: u32) {
    let b = CUBE_POS_X * CUBE_DIM_Y + UNIT_POS_Y;
    let i = b * 32 + UNIT_POS_X;
    if b * 32 < n {
        let v = src[(soff + i) as usize];
        let amax = plane_max(f32::abs(v));
        let d = amax / 127.0f32;
        let mut q = 0.0f32;
        if d > 0.0f32 {
            q = f32::round(v * (1.0f32 / d));
        }
        dst[i as usize] = q * d;
        z[i as usize] = 0.0f32;
    }
}

/// y[row] += wgt * (down_row . h) for one cached expert.
#[cube(launch_unchecked)]
fn k_xdown(h: &[f32], wd: &[u32], g2s: &[u32], g2x: &[u32], ks: &[u32], g1: &[u32], y: &mut [f32], hoff: u32, rows: u32, cols: u32, rbd: u32, wgt: f32, #[comptime] td: u32) {
    let row = CUBE_POS_X * CUBE_DIM_Y + UNIT_POS_Y;
    if row < rows {
        let nsb = cols / 8;
        let mut s = 0.0f32;
        let mut sb = UNIT_POS_X;
        while sb < nsb {
            s += subdot8(wd, row * rbd, sb, h, hoff, g2s, g2x, ks, g1, td);
            sb += 32u32;
        }
        let t = plane_sum(s);
        if UNIT_POS_X == 0 {
            y[row as usize] += wgt * t;
        }
    }
}


/// Batched cached experts (all of one layer in one launch): expert = CUBE_POS_Y, its word offsets
/// in the arena are meta[4e], meta[4e+1] (gate, up); h is [expert][rows].
#[cube(launch_unchecked)]
fn k_xgu_b(x: &[f32], ar: &[u32], meta: &[u32], g2s: &[u32], g2x: &[u32], ks: &[u32], g1: &[u32], h: &mut [f32], rows: u32, cols: u32, rbg: u32, rbu: u32,
    #[comptime] tg: u32, #[comptime] tu: u32) {
    let e = CUBE_POS_Y;
    let row = CUBE_POS_X * CUBE_DIM_Y + UNIT_POS_Y;
    if row < rows {
        let go = meta[(e * 4) as usize] + row * rbg;
        let uo = meta[(e * 4 + 1) as usize] + row * rbu;
        let nsb = cols / 8;
        let mut gs = 0.0f32;
        let mut us = 0.0f32;
        let mut sb = UNIT_POS_X;
        while sb < nsb {
            gs += subdot8(ar, go, sb, x, 0u32, g2s, g2x, ks, g1, tg);
            us += subdot8(ar, uo, sb, x, 0u32, g2s, g2x, ks, g1, tu);
            sb += 32u32;
        }
        let g = plane_sum(gs);
        let u = plane_sum(us);
        if UNIT_POS_X == 0 {
            h[(e * rows + row) as usize] = g / (1.0f32 + f32::exp(-g)) * u;
        }
    }
}

/// y[row] += sum_e wgt_e * (down_e row . h_e); down offset meta[4e+2], weight bits meta[4e+3].
#[cube(launch_unchecked)]
fn k_xdown_b(h: &[f32], ar: &[u32], meta: &[u32], g2s: &[u32], g2x: &[u32], ks: &[u32], g1: &[u32], y: &mut [f32], ne: u32, rows: u32, cols: u32, rbd: u32, #[comptime] td: u32) {
    let row = CUBE_POS_X * CUBE_DIM_Y + UNIT_POS_Y;
    if row < rows {
        let nsb = cols / 8;
        let mut s = 0.0f32;
        for e in 0..ne {
            let dof = meta[(e * 4 + 2) as usize] + row * rbd;
            let w = f32::reinterpret(meta[(e * 4 + 3) as usize]);
            let mut p = 0.0f32;
            let mut sb = UNIT_POS_X;
            while sb < nsb {
                p += subdot8(ar, dof, sb, h, e * cols, g2s, g2x, ks, g1, td);
                sb += 32u32;
            }
            s += w * p;
        }
        let t = plane_sum(s);
        if UNIT_POS_X == 0 {
            y[row as usize] += t;
        }
    }
}
/// Full-attention prep for one token: q heads (cube < nh) get RMS norm + RoPE into q; k heads get
/// norm + RoPE and, with v, are written to the KV cache at `pos`. cs = [cos(half), sin(half)].
#[cube(launch_unchecked)]
fn k_attn_prep(p: &[f32], qn: &[f32], kn: &[f32], cs: &[f32], q: &mut [f32], kc: &mut [f32], vc: &mut [f32], ko: u32, vo: u32, nh: u32, nkv: u32, hd: u32, half: u32, pos: u32, eps: f32) {
    let h = CUBE_POS_X;
    let lane = UNIT_POS_X;
    let mut base = h * 2 * hd;
    let mut dst = h * hd;
    if h >= nh {
        base = ko + (h - nh) * hd;
        dst = (pos * nkv + h - nh) * hd;
    }
    let mut ss = 0.0f32;
    let mut i = lane;
    while i < hd {
        let v = p[(base + i) as usize];
        ss += v * v;
        i += 32u32;
    }
    let s = 1.0f32 / f32::sqrt(plane_sum(ss) / f32::cast_from(hd) + eps);
    i = lane;
    while i < hd {
        let mut wi = 0.0f32;
        let mut wj = 0.0f32;
        let mut j = i;
        if i < half {
            j = i + half;
        } else if i < 2 * half {
            j = i - half;
        }
        if h < nh {
            wi = qn[i as usize];
            wj = qn[j as usize];
        } else {
            wi = kn[i as usize];
            wj = kn[j as usize];
        }
        let ni = p[(base + i) as usize] * (s * wi);
        let nj = p[(base + j) as usize] * (s * wj);
        let mut o = ni;
        if i < half {
            o = ni * cs[i as usize] - nj * cs[(half + i) as usize];
        } else if i < 2 * half {
            o = ni * cs[j as usize] + nj * cs[(half + j) as usize];
        }
        if h < nh {
            q[(dst + i) as usize] = o;
        } else {
            kc[(dst + i) as usize] = o;
            vc[(dst + i) as usize] = p[(vo + (h - nh) * hd + i) as usize];
        }
        i += 32u32;
    }
}

/// One query head per cube (one plane): softmax(q.K^T * scale) V over n positions, times sigmoid(gate).
#[cube(launch_unchecked)]
fn k_attn(q: &[f32], kc: &[f32], vc: &[f32], p: &[f32], sc: &mut [f32], o: &mut [f32], nh: u32, nkv: u32, hd: u32, n: u32, cap: u32, scale: f32) {
    let h = CUBE_POS_X;
    let lane = UNIT_POS_X;
    let kvh = h / (nh / nkv);
    let mut mx = scale * -3.0e30f32;
    let mut pp = lane;
    while pp < n {
        let kb = (pp * nkv + kvh) * hd;
        let mut dd = 0.0f32;
        for i in 0..hd {
            dd += q[(h * hd + i) as usize] * kc[(kb + i) as usize];
        }
        let v = dd * scale;
        sc[(h * cap + pp) as usize] = v;
        mx = f32::max(mx, v);
        pp += 32u32;
    }
    let m = plane_max(mx);
    let mut sum = 0.0f32;
    pp = lane;
    while pp < n {
        let e = f32::exp(sc[(h * cap + pp) as usize] - m);
        sc[(h * cap + pp) as usize] = e;
        sum += e;
        pp += 32u32;
    }
    let inv = 1.0f32 / plane_sum(sum);
    sync_cube();
    let mut i = lane;
    while i < hd {
        let mut acc = 0.0f32;
        for t in 0..n {
            acc += sc[(h * cap + t) as usize] * inv * vc[((t * nkv + kvh) * hd + i) as usize];
        }
        let g = p[(h * 2 * hd + hd + i) as usize];
        o[(h * hd + i) as usize] = acc * (1.0f32 / (1.0f32 + f32::exp(-g)));
        i += 32u32;
    }
}
/// Grid tables for the expert kernels.
pub struct XTabs { g2s: Handle, g2x: Handle, ks: Handle, g1: Handle }
/// One cached expert: raw GGUF bytes of its gate, up and down rows.
pub struct XEnt { pub off: [u32; 3], pub arena: usize, pub slot: usize, pub bytes: usize }
/// Words of one repacked expert (gate, up, down).
pub fn xwords(tys: (u32, u32, u32), d: usize, ff: usize) -> (usize, usize, usize) { let rw = |ty: u32, cols: usize| { let (e, w) = block_words(ty); cols / e * w }; (ff * rw(tys.0, d), ff * rw(tys.1, d), d * rw(tys.2, ff)) }
/// Words of one repacked expert (gate, up, down). fn xwords(tys: (u32, u32, u32), d: usize, ff: usize) -> (usize, usize, usize) { let rw = |ty: u32, cols: usize| { let (e, w) = block_words(ty); cols / e * w }; (ff * rw(tys.0, d), ff * rw(tys.1, d), d * rw(tys.2, ff)) }
pub fn xtype(t: super::gguf_q::QType) -> Option<u32> {
    use super::gguf_q::QType::*;
    match t { IQ2S => Some(0), IQ2XXS => Some(1), IQ1M => Some(2), Q2_0 => Some(3), _ => None }
}

pub struct Cube { pub client: Client, pub budget: std::sync::atomic::AtomicI64 }

pub struct CMat { w: Handle, s: Handle, pub rows: usize, pub cols: usize, pub splits: Vec<usize>,
    /// native IQ4_XS row ranges (out row, rows, words) and Q8 row ranges (out row, rows, row in w/s); empty = all Q8
    nat: Vec<(usize, usize, Handle)>, qseg: Vec<(usize, usize, usize)>, kv: Option<Handle> }

impl Cube {
    pub fn new() -> anyhow::Result<Self> {
        let client = cubecl::Device::cuda(0).map_err(|e| anyhow::anyhow!("cuda: {e:?}"))?.client();
        let v = crate::limits::vram().ok_or_else(|| anyhow::anyhow!("no NVML VRAM reading"))?;
        let budget = v.free as i64 - crate::limits::reserve() as i64 - (1536i64 << 20);
        Ok(Self { client, budget: budget.into() })
    }
    pub fn take(&self, bytes: usize) -> bool {
        use std::sync::atomic::Ordering::Relaxed;
        let n = bytes as i64;
        if self.budget.fetch_sub(n, Relaxed) < n { self.budget.fetch_add(n, Relaxed); false } else { true }
    }
    /// Upload the rows of `ms` (then the f32 `extra` rows) as one Q8 matrix.
    pub fn upload(&self, ms: &[&QMat], extra: &[&[f32]]) -> Option<CMat> {
        let cols = ms[0].cols;
        let rows: usize = ms.iter().map(|m| m.rows).sum::<usize>() + extra.len();
        if !self.take(rows * cols + rows * cols / 8) { return None; }
        let q = super::qn_gpu::to_q8_0(ms);
        let nb = cols / 32;
        let mut words: Vec<u32> = Vec::with_capacity(rows * cols / 4);
        let mut sc: Vec<f32> = Vec::with_capacity(rows * nb);
        for blk in q.chunks_exact(34) {
            sc.push(half::f16::from_bits(u16::from_le_bytes([blk[0], blk[1]])).to_f32());
            for c in blk[2..].chunks_exact(4) { words.push(u32::from_le_bytes([c[0], c[1], c[2], c[3]])); }
        }
        for e in extra { for b in 0..nb { let x = &e[b * 32..(b + 1) * 32]; let amax = x.iter().fold(0f32, |a, v| a.max(v.abs())); let d = amax / 127.0; let id = if d > 0.0 { 1.0 / d } else { 0.0 };
            sc.push(d); for c in x.chunks_exact(4) { let qb: Vec<u8> = c.iter().map(|v| ((v * id).round().clamp(-127.0, 127.0) as i8) as u8).collect(); words.push(u32::from_le_bytes([qb[0], qb[1], qb[2], qb[3]])); } } }
        let mut splits = vec![0]; for m in ms { splits.push(splits.last().unwrap() + m.rows); } for _ in extra { splits.push(splits.last().unwrap() + 1); }
        Some(CMat { w: self.client.create_from_slice(u32::as_bytes(&words)), s: self.client.create_from_slice(f32::as_bytes(&sc)), rows, cols, splits, nat: vec![], qseg: vec![], kv: None })
    }
    /// Like `upload`, but IQ4_XS matrices stay in their own format (half the bytes of Q8_0);
    /// the rest are packed as Q8_0 as before. Opt-in (PYRE_QN_NATIVE=1): on the RTX 3060 this kernel is
    /// still ALU/LSU-bound (mout 2560x6144: 65 us vs 59 us for Q8_0), so it is not a win yet.
    pub fn upload_mix(&self, ms: &[&QMat], extra: &[&[f32]]) -> Option<CMat> {
        use super::gguf_q::QType;
        let cols = ms[0].cols;
        let native = |m: &QMat| m.ty == QType::IQ4XS && cols % 256 == 0 && cols <= 6144;
        if std::env::var("PYRE_QN_NATIVE").map(|v| v != "1").unwrap_or(true) || !ms.iter().any(|m| native(m)) { return self.upload(ms, extra); }
        let q8: Vec<&QMat> = ms.iter().copied().filter(|m| !native(m)).collect();
        let nq8 = q8.iter().map(|m| m.rows).sum::<usize>() + extra.len();
        let nnat: usize = ms.iter().filter(|m| native(m)).map(|m| m.rows).sum();
        if !self.take(nq8 * cols * 9 / 8 + nnat * cols / 256 * 140) { return None; }
        let (mut words, mut sc): (Vec<u32>, Vec<f32>) = (vec![], vec![]);
        if !q8.is_empty() { let q = super::qn_gpu::to_q8_0(&q8);
            for blk in q.chunks_exact(34) { sc.push(half::f16::from_bits(u16::from_le_bytes([blk[0], blk[1]])).to_f32()); for c in blk[2..].chunks_exact(4) { words.push(u32::from_le_bytes([c[0], c[1], c[2], c[3]])); } } }
        let nb = cols / 32;
        for e in extra { for b in 0..nb { let x = &e[b * 32..(b + 1) * 32]; let amax = x.iter().fold(0f32, |a, v| a.max(v.abs())); let d = amax / 127.0; let id = if d > 0.0 { 1.0 / d } else { 0.0 };
            sc.push(d); for c in x.chunks_exact(4) { let qb: Vec<u8> = c.iter().map(|v| ((v * id).round().clamp(-127.0, 127.0) as i8) as u8).collect(); words.push(u32::from_le_bytes([qb[0], qb[1], qb[2], qb[3]])); } } }
        if words.is_empty() { words.push(0); sc.push(0.0); }
        let (mut nat, mut qseg) = (vec![], vec![]); let (mut r0, mut qb) = (0usize, 0usize);
        for m in ms {
            if native(m) {
                let raw = m.raw_rows(0, m.rows); let mut v: Vec<u32> = Vec::with_capacity(raw.len() / 136 * 35);
                for b in raw.chunks_exact(136) { v.push(half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32().to_bits()); v.push(u16::from_le_bytes([b[2], b[3]]) as u32); v.push(u32::from_le_bytes([b[4], b[5], b[6], b[7]]));
                    for c in b[8..].chunks_exact(4) { v.push(u32::from_le_bytes([c[0], c[1], c[2], c[3]])); } }
                nat.push((r0, m.rows, self.client.create_from_slice(u32::as_bytes(&v))));
            } else { match qseg.last_mut() { Some((a, n, _)) if *a + *n == r0 => *n += m.rows, _ => qseg.push((r0, m.rows, qb)) } qb += m.rows; }
            r0 += m.rows;
        }
        for _ in extra { match qseg.last_mut() { Some((a, n, _)) if *a + *n == r0 => *n += 1, _ => qseg.push((r0, 1, qb)) } qb += 1; r0 += 1; }
        let mut splits = vec![0]; for m in ms { splits.push(splits.last().unwrap() + m.rows); } for _ in extra { splits.push(splits.last().unwrap() + 1); }
        let kv: Vec<f32> = super::iq_tables::KVALUES_IQ4NL.iter().map(|&v| v as f32).collect();
        Some(CMat { w: self.client.create_from_slice(u32::as_bytes(&words)), s: self.client.create_from_slice(f32::as_bytes(&sc)), rows: r0, cols, splits, nat, qseg, kv: Some(self.client.create_from_slice(f32::as_bytes(&kv))) })
    }    pub fn buf(&self, v: &[f32]) -> Handle { self.client.create_from_slice(f32::as_bytes(v)) }
    pub fn empty(&self, n: usize) -> Handle { self.client.empty(n * 4) }
    pub fn read(&self, h: &Handle) -> Vec<f32> {
        let t0 = std::time::Instant::now();
        let b = self.client.read_one(h.clone()).expect("gpu readback");
        let v = f32::from_bytes(&b).to_vec();
        super::qn_gpu::GPU_NS.fetch_add(t0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed); super::qn_gpu::GPU_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        v
    }
    /// out[ooff..ooff+rows] = W x[xoff..xoff+cols]
    pub fn gemv(&self, m: &CMat, x: &Handle, xn: usize, xoff: usize, out: &Handle, on: usize, ooff: usize) {
        if !m.nat.is_empty() {
            for (r0, n, w) in &m.nat { unsafe { k_gemv_iq4::launch_unchecked(&self.client, CubeCount::Static((*n as u32).div_ceil(ROWS_PER_CUBE), 1, 1), CubeDim::new_2d(32, ROWS_PER_CUBE),
                BufferArg::from_raw_parts(x.clone(), xn), BufferArg::from_raw_parts(w.clone(), n * m.cols / 256 * 35), BufferArg::from_raw_parts(m.kv.clone().unwrap(), 16),
                BufferArg::from_raw_parts(out.clone(), on), *n as u32, m.cols as u32, xoff as u32, (ooff + r0) as u32) } }
            for (r0, n, qb) in &m.qseg { let (wo, so) = ((qb * m.cols) as u64, (qb * m.cols / 32 * 4) as u64); unsafe { k_gemv::launch_unchecked(&self.client, CubeCount::Static((*n as u32).div_ceil(ROWS_PER_CUBE), 1, 1), CubeDim::new_2d(32, ROWS_PER_CUBE),
                BufferArg::from_raw_parts(x.clone(), xn), BufferArg::from_raw_parts(m.w.clone().offset_start(wo), n * m.cols / 4), BufferArg::from_raw_parts(m.s.clone().offset_start(so), n * m.cols / 32),
                BufferArg::from_raw_parts(out.clone(), on), *n as u32, m.cols as u32, xoff as u32, (ooff + r0) as u32) } }
            return;
        }
        unsafe { k_gemv::launch_unchecked(&self.client, CubeCount::Static((m.rows as u32).div_ceil(ROWS_PER_CUBE), 1, 1), CubeDim::new_2d(32, ROWS_PER_CUBE),
            BufferArg::from_raw_parts(x.clone(), xn), BufferArg::from_raw_parts(m.w.clone(), m.rows * m.cols / 4), BufferArg::from_raw_parts(m.s.clone(), m.rows * m.cols / 32),
            BufferArg::from_raw_parts(out.clone(), on), m.rows as u32, m.cols as u32, xoff as u32, ooff as u32) }
    }
    /// `gemv` for `nt` activation columns: x[c] at xoff + c*xst (xn = x buffer length), out[c] at ooff + c*ost.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_m(&self, m: &CMat, x: &Handle, xn: usize, xoff: usize, xst: usize, out: &Handle, on: usize, ooff: usize, ost: usize, nt: usize) {
        if !m.nat.is_empty() || nt == 1 || std::env::var("PYRE_QN_GEMV4").map(|v| v == "0").unwrap_or(false) {
            for c in 0..nt { self.gemv(m, x, xn, xoff + c * xst, out, on, ooff + c * ost); }
            return;
        }
        let mut c0 = 0;
        while c0 < nt { let n = (nt - c0).min(4);
            unsafe { k_gemv4::launch_unchecked(&self.client, CubeCount::Static((m.rows as u32).div_ceil(ROWS_PER_CUBE), 1, 1), CubeDim::new_2d(32, ROWS_PER_CUBE),
                BufferArg::from_raw_parts(x.clone(), xn), BufferArg::from_raw_parts(m.w.clone(), m.rows * m.cols / 4), BufferArg::from_raw_parts(m.s.clone(), m.rows * m.cols / 32),
                BufferArg::from_raw_parts(out.clone(), on), m.rows as u32, m.cols as u32, (xoff + c0 * xst) as u32, xst as u32, (ooff + c0 * ost) as u32, ost as u32, n as u32) }
            c0 += n; }
    }
    pub fn rms(&self, x: &Handle, norm: &Handle, xn: &Handle, hc: usize, d: usize, eps: f32) {
        unsafe { k_rms::launch_unchecked(&self.client, CubeCount::Static(hc as u32, 1, 1), CubeDim::new_1d(256),
            BufferArg::from_raw_parts(x.clone(), hc * d), BufferArg::from_raw_parts(norm.clone(), hc * d), BufferArg::from_raw_parts(xn.clone(), hc * d), d as u32, eps) }
    }
    pub fn gate(&self, xn: &Handle, dn: &Handle, up: &CMat, out: &Handle, on: usize, ooff: usize, hc: usize, d: usize) {
        let lr = up.cols;
        unsafe { k_gate::launch_unchecked(&self.client, CubeCount::Static((d as u32).div_ceil(8), 1, 1), CubeDim::new_2d(32, 8),
            BufferArg::from_raw_parts(xn.clone(), hc * d), BufferArg::from_raw_parts(dn.clone(), lr + hc), BufferArg::from_raw_parts(up.w.clone(), up.rows * lr / 4),
            BufferArg::from_raw_parts(up.s.clone(), up.rows * lr / 32), BufferArg::from_raw_parts(out.clone(), on), d as u32, ooff as u32, lr, hc as u32) }
    }
    #[allow(clippy::too_many_arguments)]
    pub fn comb(&self, x: &Handle, a: &Handle, b: Option<&Handle>, g: Option<(&Handle, usize, usize)>, dn: &Handle, lr: usize, hc: usize, d: usize) { self.comb3(x, a, b, None, g, dn, lr, hc, d) }
    #[allow(clippy::too_many_arguments)]
    pub fn comb3(&self, x: &Handle, a: &Handle, b: Option<&Handle>, c3: Option<&Handle>, g: Option<(&Handle, usize, usize)>, dn: &Handle, lr: usize, hc: usize, d: usize) {
        let (gb, gn, goff) = g.map(|(h, n, o)| (h.clone(), n, o)).unwrap_or((a.clone(), d, 0));
        let bb = b.cloned().unwrap_or_else(|| a.clone()); let cc = c3.cloned().unwrap_or_else(|| a.clone());
        unsafe { k_comb::launch_unchecked(&self.client, CubeCount::Static(((hc * d) as u32).div_ceil(256), 1, 1), CubeDim::new_1d(256),
            BufferArg::from_raw_parts(x.clone(), hc * d), BufferArg::from_raw_parts(a.clone(), d), BufferArg::from_raw_parts(bb, d), BufferArg::from_raw_parts(cc, d), BufferArg::from_raw_parts(gb, gn),
            BufferArg::from_raw_parts(dn.clone(), lr + hc), d as u32, goff as u32, lr as u32, hc as u32, g.is_some(), b.is_some(), c3.is_some()) }
    }
    pub fn conv(&self, proj: &Handle, pn: usize, w: &Handle, cs: &Handle, co: &Handle, ch: usize, kk: usize) {
        unsafe { k_conv::launch_unchecked(&self.client, CubeCount::Static((ch as u32).div_ceil(256), 1, 1), CubeDim::new_1d(256),
            BufferArg::from_raw_parts(proj.clone(), pn), BufferArg::from_raw_parts(w.clone(), ch * kk), BufferArg::from_raw_parts(cs.clone(), ch * (kk - 1)),
            BufferArg::from_raw_parts(co.clone(), ch), ch as u32, kk as u32) }
    }
    #[allow(clippy::too_many_arguments)]
    pub fn delta(&self, co: &Handle, ch: usize, proj: &Handle, pn: usize, w: &GdnW, s: &Handle, out: &Handle, nk: usize, nv: usize, dk: usize, zoff: usize, boff: usize, aoff: usize, eps: f32) {
        unsafe { k_delta::launch_unchecked(&self.client, CubeCount::Static(nv as u32, 1, 1), CubeDim::new_1d(dk as u32),
            BufferArg::from_raw_parts(co.clone(), ch), BufferArg::from_raw_parts(proj.clone(), pn), BufferArg::from_raw_parts(w.a.clone(), nv), BufferArg::from_raw_parts(w.dt.clone(), nv),
            BufferArg::from_raw_parts(w.norm.clone(), dk), BufferArg::from_raw_parts(s.clone(), nv * dk * dk), BufferArg::from_raw_parts(out.clone(), nv * dk),
            nk as u32, nv as u32, zoff as u32, boff as u32, aoff as u32, eps, dk as u32) }
    }
    /// Decode attention for one token (see k_attn_prep / k_attn); `p` = the fused q|k|v projection.
    #[allow(clippy::too_many_arguments)]
    pub fn attn(&self, p: &Handle, pn: usize, ko: usize, vo: usize, qn: &Handle, kn: &Handle, cs: &Handle, kc: &Handle, vc: &Handle, cap: usize, out: &Handle, nh: usize, nkv: usize, hd: usize, half: usize, pos: usize, eps: f32) {
        let a = |hd_: &Handle, n: usize| unsafe { BufferArg::from_raw_parts(hd_.clone(), n) };
        let q = self.empty(nh * hd); let sc = self.empty(nh * cap);
        unsafe {
            k_attn_prep::launch_unchecked(&self.client, CubeCount::Static((nh + nkv) as u32, 1, 1), CubeDim::new_1d(32),
                a(p, pn), a(qn, hd), a(kn, hd), a(cs, 2 * half), a(&q, nh * hd), a(kc, cap * nkv * hd), a(vc, cap * nkv * hd), ko as u32, vo as u32, nh as u32, nkv as u32, hd as u32, half as u32, pos as u32, eps);
            k_attn::launch_unchecked(&self.client, CubeCount::Static(nh as u32, 1, 1), CubeDim::new_1d(32),
                a(&q, nh * hd), a(kc, cap * nkv * hd), a(vc, cap * nkv * hd), a(p, pn), a(&sc, nh * cap), a(out, nh * hd), nh as u32, nkv as u32, hd as u32, (pos + 1) as u32, cap as u32, (hd as f32).sqrt().recip());
        }
    }    pub fn xtabs(&self) -> XTabs {
        use super::iq_tables::*;
        let w64 = |t: &[u64]| -> Vec<u32> { t.iter().flat_map(|v| [*v as u32, (*v >> 32) as u32]).collect() };
        let ks: Vec<u32> = KSIGNS_IQ2XS.iter().map(|&v| v as u32).collect();
        XTabs { g2s: self.client.create_from_slice(u32::as_bytes(&w64(&IQ2S_GRID))), g2x: self.client.create_from_slice(u32::as_bytes(&w64(&IQ2XXS_GRID))),
            ks: self.client.create_from_slice(u32::as_bytes(&ks)), g1: self.client.create_from_slice(u32::as_bytes(&w64(&IQ1S_GRID))) }
    }
    pub fn xarena(&self, bytes: usize) -> Handle { self.client.empty(bytes) }
    /// Repack one expert into arena slot `slot` (slot_bytes each; the arena is ar_bytes long).
    #[allow(clippy::too_many_arguments)]
    pub fn xupload(&self, ar: &Handle, ar_bytes: usize, slot: usize, slot_bytes: usize, g: &[u8], u: &[u8], d: &[u8], tys: (u32, u32, u32)) -> XEnt {
        let mut v = repack(tys.0, g); let uo = v.len(); v.extend(repack(tys.1, u)); let dof = v.len(); v.extend(repack(tys.2, d));
        let base = slot * slot_bytes; let len = v.len() * 4; assert!(len <= slot_bytes && base + len <= ar_bytes);
        let hd = ar.clone().offset_start(base as u64).offset_end((ar_bytes - base - len) as u64);
        self.client.write(&hd, cubecl::bytes::Bytes::from_elems(v));
        let b = (base / 4) as u32;
        XEnt { off: [b, b + uo as u32, b + dof as u32], arena: 0, slot, bytes: slot_bytes }
    }
    /// All cached experts of one layer: h[e] = silu(G_e x) * (U_e x) -> Q8 grid -> y += sum_e w_e D_e h[e].
    #[allow(clippy::too_many_arguments)]
    pub fn xrun_b(&self, tb: &XTabs, ar: &Handle, ar_bytes: usize, ents: &[(&XEnt, f32)], x: &Handle, d: usize, ff: usize, tys: (u32, u32, u32), h: &Handle, y: &Handle) {
        let a = |hd: &Handle, n: usize| unsafe { BufferArg::from_raw_parts(hd.clone(), n) };
        let rw = |ty: u32, cols: usize| { let (e, w) = block_words(ty); cols / e * w };
        let rbs = (rw(tys.0, d), rw(tys.1, d), rw(tys.2, ff));
        let ne = ents.len();
        let meta: Vec<u32> = ents.iter().flat_map(|(e, w)| [e.off[0], e.off[1], e.off[2], w.to_bits()]).collect();
        let mh = self.client.create_from_slice(u32::as_bytes(&meta));
        self.xrun_m(tb, ar, ar_bytes, &mh, ne, x, d, ff, tys, h, y, 0);
    }
    /// xrun_b with the per-expert meta already on the GPU; skip bits 1/2/4 drop gate-up / q8 / down (benchmarks).
    #[allow(clippy::too_many_arguments)]
    pub fn xrun_m(&self, tb: &XTabs, ar: &Handle, ar_bytes: usize, mh: &Handle, ne: usize, x: &Handle, d: usize, ff: usize, tys: (u32, u32, u32), h: &Handle, y: &Handle, skip: u32) {
        let a = |hd: &Handle, n: usize| unsafe { BufferArg::from_raw_parts(hd.clone(), n) };
        let rw = |ty: u32, cols: usize| { let (e, w) = block_words(ty); cols / e * w };
        let rbs = (rw(tys.0, d), rw(tys.1, d), rw(tys.2, ff)); let ml = ne * 4;
        let aw = ar_bytes / 4;
        static RPC: std::sync::OnceLock<u32> = std::sync::OnceLock::new(); let rpc = *RPC.get_or_init(|| std::env::var("PYRE_QN_XRPC").ok().and_then(|v| v.parse().ok()).unwrap_or(8));
        unsafe {
            if skip & 1 == 0 { k_xgu_b::launch_unchecked(&self.client, CubeCount::Static((ff as u32).div_ceil(rpc), ne as u32, 1), CubeDim::new_2d(32, rpc),
                a(x, d), a(ar, aw), a(mh, ml), a(&tb.g2s, 2048), a(&tb.g2x, 512), a(&tb.ks, 128), a(&tb.g1, 4096), a(h, ne * ff), ff as u32, d as u32, rbs.0 as u32, rbs.1 as u32, tys.0, tys.1); }
            if skip & 2 == 0 { k_q8sim::launch_unchecked(&self.client, CubeCount::Static(((ne * ff) as u32 / 32).div_ceil(8), 1, 1), CubeDim::new_2d(32, 8), a(h, ne * ff), (ne * ff) as u32); }
            if skip & 4 == 0 { k_xdown_b::launch_unchecked(&self.client, CubeCount::Static((d as u32).div_ceil(rpc), 1, 1), CubeDim::new_2d(32, rpc),
                a(h, ne * ff), a(ar, aw), a(mh, ml), a(&tb.g2s, 2048), a(&tb.g2x, 512), a(&tb.ks, 128), a(&tb.g1, 4096), a(y, d), ne as u32, d as u32, ff as u32, rbs.2 as u32, tys.2); }
        }
    }    /// dst = Q8-grid copy of src[soff..soff+n]; z[..n] = 0.
    pub fn q8copy(&self, src: &Handle, sn: usize, soff: usize, dst: &Handle, z: &Handle, n: usize) {
        unsafe { k_q8copy::launch_unchecked(&self.client, CubeCount::Static((n as u32 / 32).div_ceil(8), 1, 1), CubeDim::new_2d(32, 8),
            BufferArg::from_raw_parts(src.clone(), sn), BufferArg::from_raw_parts(dst.clone(), n), BufferArg::from_raw_parts(z.clone(), n), soff as u32, n as u32) }
    }
    /// dst[dof..dof+n] = src[so..so+n] (srcn / dstn: buffer lengths in floats)
    #[allow(clippy::too_many_arguments)]
    pub fn copy(&self, src: &Handle, srcn: usize, so: usize, dst: &Handle, dstn: usize, dof: usize, n: usize) {
        unsafe { k_copy::launch_unchecked(&self.client, CubeCount::Static((n as u32).div_ceil(256), 1, 1), CubeDim::new_1d(256),
            BufferArg::from_raw_parts(src.clone(), srcn), BufferArg::from_raw_parts(dst.clone(), dstn), so as u32, dof as u32, n as u32) }
    }
    pub fn silu_mul(&self, f: &Handle, fnn: usize, og: usize, ou: usize, h: &Handle, n: usize) {
        unsafe { k_silu_mul::launch_unchecked(&self.client, CubeCount::Static((n as u32).div_ceil(256), 1, 1), CubeDim::new_1d(256),
            BufferArg::from_raw_parts(f.clone(), fnn), BufferArg::from_raw_parts(h.clone(), n), og as u32, ou as u32, n as u32) }
    }
}

/// A hyper-connection mixer: norm weights, down(+inject) rows, up rows.
pub struct CHc { pub norm: Handle, pub di: CMat, pub up: CMat }
/// One layer's dense half (see `Model::layer_cube`). `fin` = router | shared gate | shared up | shared-gate row.
pub struct GdnW { pub conv: Handle, pub a: Handle, pub dt: Handle, pub norm: Handle }
pub struct CLayer { pub an: Option<(Handle, Handle)>, pub gdn: Option<GdnW>, pub hca: CHc, pub min: CMat, pub mout: CMat, pub hcf: CHc, pub fin: CMat, pub sdown: CMat }
