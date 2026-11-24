//! Geometric mesh tools for the cabin: stats, normals, smoothing, colour
//! projection, center/scale, decimate, remesh, repair, UVs, format conversion,
//! and mesh -> SDF.
//!
//! Pure Rust. No neural net, no GPU, no weights. GLB writing goes through
//! [`crate::triposr::write_glb`] so cabin keeps one glTF dialect.

use anyhow::{bail, Context, Result};
use serde::Serialize;
use serde_json::{json, Value};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

/// Refuse a mesh whose triangle list would make an operation allocate without
/// bound. 2M triangles is ~24 MB of indices plus the vertex buffers.
pub const MAX_TRIANGLES: usize = 2_000_000;
/// SDF grids default to 64^3; this is the hard cap.
pub const MAX_SDF_RESOLUTION: usize = 128;

/// In-memory mesh. Positions/triangles are required, everything else optional.
#[derive(Clone, Debug, Default)]
pub struct Mesh {
    pub pos: Vec<[f32; 3]>,
    pub tri: Vec<[u32; 3]>,
    pub nor: Option<Vec<[f32; 3]>>,
    /// RGBA, 0..1.
    pub col: Option<Vec<[f32; 4]>>,
    pub uv: Option<Vec<[f32; 2]>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeshFormat {
    Glb,
    Obj,
    Ply,
    Stl,
}

impl MeshFormat {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s.to_ascii_lowercase().as_str() {
            "glb" | "gltf" => MeshFormat::Glb,
            "obj" => MeshFormat::Obj,
            "ply" => MeshFormat::Ply,
            "stl" => MeshFormat::Stl,
            _ => return None,
        })
    }
    pub fn as_str(self) -> &'static str {
        match self {
            MeshFormat::Glb => "glb",
            MeshFormat::Obj => "obj",
            MeshFormat::Ply => "ply",
            MeshFormat::Stl => "stl",
        }
    }
    pub fn extension(self) -> &'static str {
        match self {
            MeshFormat::Glb => "glb",
            MeshFormat::Obj => "obj",
            MeshFormat::Ply => "ply",
            MeshFormat::Stl => "stl",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct EdgeCounts {
    /// Edges used by exactly one triangle.
    pub boundary: usize,
    /// Edges used by exactly two triangles.
    pub manifold: usize,
    /// Edges used by three or more triangles.
    pub non_manifold: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct MeshStats {
    pub vertices: usize,
    pub triangles: usize,
    pub bbox_min: [f32; 3],
    pub bbox_max: [f32; 3],
    pub area: f32,
    /// No boundary edges and no non-manifold edges.
    pub watertight: bool,
    pub edge_counts: EdgeCounts,
}

/// Result of a least-squares conformal map unfold.
#[derive(Clone, Debug, Serialize)]
pub struct LscmReport {
    pub charts: usize,
    pub iterations: usize,
    pub max_error: f32,
    pub seams_split: bool,
}

#[derive(Clone, Copy, Debug, Default)]
struct Complex {
    re: f32,
    im: f32,
}

impl Complex {
    fn add(self, o: Self) -> Self { Self { re: self.re + o.re, im: self.im + o.im } }
    fn sub(self, o: Self) -> Self { Self { re: self.re - o.re, im: self.im - o.im } }
    fn mul(self, o: Self) -> Self {
        Self { re: self.re * o.re - self.im * o.im, im: self.re * o.im + self.im * o.re }
    }
    fn conj(self) -> Self { Self { re: self.re, im: -self.im } }
    fn div_f32(self, d: f32) -> Self { Self { re: self.re / d, im: self.im / d } }
    fn abs2(self) -> f32 { self.re * self.re + self.im * self.im }
    fn div(self, o: Self) -> Self {
        let d = o.abs2();
        if d < 1e-20 { Self::default() } else { self.mul(o.conj()).div_f32(d) }
    }
    fn finite(self) -> bool { self.re.is_finite() && self.im.is_finite() }
}

#[derive(Clone, Debug, Serialize)]
pub struct RepairReport {
    pub welded_vertices: usize,
    pub removed_degenerate: usize,
    pub flipped_faces: usize,
    pub holes_filled: usize,
    pub boundary_edges: usize,
    pub non_manifold_edges: usize,
    /// Boundary loops larger than the fill limit are left open.
    pub unfixed_boundaries: usize,
}

impl Mesh {
    #[cfg(test)]
    pub fn unit_cube() -> Self {
        let pos = vec![
            [-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0],
        ];
        let tri = vec![
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [3, 7, 6], [3, 6, 2],
            [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
        ];
        Mesh { pos, tri, ..Mesh::default() }
    }

    #[cfg(test)]
    pub fn tetrahedron() -> Self {
        let pos = vec![[1.0, 1.0, 1.0], [-1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, -1.0]];
        let tri = vec![[0, 1, 2], [0, 3, 1], [3, 2, 1], [0, 2, 3]];
        Mesh { pos, tri, ..Mesh::default() }
    }

    pub fn validate(&self) -> Result<()> {
        if self.tri.len() > MAX_TRIANGLES {
            bail!("mesh has {} triangles; the limit is {}", self.tri.len(), MAX_TRIANGLES);
        }
        for (i, t) in self.tri.iter().enumerate() {
            for &v in t {
                if v as usize >= self.pos.len() {
                    bail!("triangle {i} references vertex {v}, but there are only {} vertices", self.pos.len());
                }
            }
        }
        Ok(())
    }

    pub fn bbox(&self) -> ([f32; 3], [f32; 3]) {
        if self.pos.is_empty() {
            return ([0.0; 3], [0.0; 3]);
        }
        let mut mn = [f32::MAX; 3];
        let mut mx = [f32::MIN; 3];
        for p in &self.pos {
            for q in 0..3 {
                mn[q] = mn[q].min(p[q]);
                mx[q] = mx[q].max(p[q]);
            }
        }
        (mn, mx)
    }

    /// Area-weighted vertex normals. Tries to make the winding consistent first
    /// when the mesh is orientable; returns the number of flipped faces.
    pub fn compute_normals(&mut self) -> usize {
        let flipped = self.orient_faces();
        let mut nor = vec![[0.0f32; 3]; self.pos.len()];
        for t in &self.tri {
            let (a, b, c) = (self.pos[t[0] as usize], self.pos[t[1] as usize], self.pos[t[2] as usize]);
            let f = cross(sub(b, a), sub(c, a));
            for &i in t {
                nor[i as usize][0] += f[0];
                nor[i as usize][1] += f[1];
                nor[i as usize][2] += f[2];
            }
        }
        for n in nor.iter_mut() {
            *n = normalize_or(*n, [0.0, 0.0, 1.0]);
        }
        self.nor = Some(nor);
        flipped
    }

    /// Propagate winding across manifold edges. Returns the number of triangles
    /// flipped. Non-orientable patches are left as-is.
    pub fn orient_faces(&mut self) -> usize {
        let n = self.tri.len();
        if n == 0 {
            return 0;
        }
        // For each undirected edge: (triangle, traversed low->high?).
        let mut edges: HashMap<(u32, u32), Vec<(usize, bool)>> = HashMap::new();
        for (ti, t) in self.tri.iter().enumerate() {
            for j in 0..3 {
                let a = t[j];
                let b = t[(j + 1) % 3];
                let key = if a < b { (a, b) } else { (b, a) };
                edges.entry(key).or_default().push((ti, a < b));
            }
        }
        let mut seen = vec![false; n];
        let mut flipped = 0usize;
        let mut queue = std::collections::VecDeque::new();
        for start in 0..n {
            if seen[start] { continue; }
            seen[start] = true;
            queue.push_back(start);
            while let Some(ti) = queue.pop_front() {
                let t = self.tri[ti];
                for j in 0..3 {
                    let a = t[j];
                    let b = t[(j + 1) % 3];
                    let key = if a < b { (a, b) } else { (b, a) };
                    let Some(inc) = edges.get(&key) else { continue };
                    if inc.len() != 2 { continue; }
                    let same_dir = a < b;
                    for &(other, other_dir) in inc {
                        if other == ti || seen[other] { continue; }
                        if other_dir == same_dir {
                            let o = &mut self.tri[other];
                            o.swap(1, 2);
                            flipped += 1;
                        }
                        seen[other] = true;
                        queue.push_back(other);
                    }
                }
            }
        }
        flipped
    }

    /// Move the bounding-box centre to the origin without scaling.
    pub fn center(&mut self) {
        if self.pos.is_empty() {
            return;
        }
        let (mn, mx) = self.bbox();
        let c = [(mn[0] + mx[0]) * 0.5, (mn[1] + mx[1]) * 0.5, (mn[2] + mx[2]) * 0.5];
        for p in self.pos.iter_mut() {
            for q in 0..3 {
                p[q] -= c[q];
            }
        }
    }

    /// Bounding-box centre to the origin, then uniform scale so the largest
    /// extent equals `target_extent`.
    pub fn center_and_scale(&mut self, target_extent: f32) {
        if self.pos.is_empty() {
            return;
        }
        let (mn, mx) = self.bbox();
        let center = [(mn[0] + mx[0]) * 0.5, (mn[1] + mx[1]) * 0.5, (mn[2] + mx[2]) * 0.5];
        let extent = (mx[0] - mn[0]).max(mx[1] - mn[1]).max(mx[2] - mn[2]);
        let s = if extent > 1e-12 { target_extent / extent } else { 1.0 };
        for p in self.pos.iter_mut() {
            for q in 0..3 {
                p[q] = (p[q] - center[q]) * s;
            }
        }
    }

    /// Taubin smoothing (lambda/mu passes) over the triangle adjacency. Geometry
    /// only; stored normals are dropped so the writer recomputes them. Boundary
    /// vertices are smoothed too, so open meshes may shrink slightly.
    pub fn smooth(&mut self, iterations: usize, lambda: f32, mu: f32) {
        let n = self.pos.len();
        if n < 3 || iterations == 0 || self.tri.is_empty() {
            return;
        }
        let mut sets: Vec<HashSet<u32>> = vec![HashSet::new(); n];
        for t in &self.tri {
            for j in 0..3 {
                let a = t[j];
                let b = t[(j + 1) % 3];
                sets[a as usize].insert(b);
                sets[b as usize].insert(a);
            }
        }
        let nb: Vec<Vec<u32>> = sets.into_iter().map(|s| s.into_iter().collect()).collect();
        for _ in 0..iterations {
            for factor in [lambda, mu] {
                let mut delta = vec![[0.0f32; 3]; n];
                for (i, ns) in nb.iter().enumerate() {
                    if ns.is_empty() {
                        continue;
                    }
                    let mut avg = [0.0f32; 3];
                    for &j in ns {
                        let p = self.pos[j as usize];
                        for q in 0..3 {
                            avg[q] += p[q];
                        }
                    }
                    for q in 0..3 {
                        avg[q] /= ns.len() as f32;
                        delta[i][q] = (avg[q] - self.pos[i][q]) * factor;
                    }
                }
                for (p, d) in self.pos.iter_mut().zip(&delta) {
                    for q in 0..3 {
                        p[q] += d[q];
                    }
                }
            }
        }
        self.nor = None;
    }

    /// Project vertex colours from `source` by nearest point on its surface.
    pub fn recolour_from_mesh(&mut self, source: &Mesh) -> Result<usize> {
        let scol = source
            .col
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("source mesh has no vertex colours (COLOR_0)"))?;
        if source.tri.is_empty() {
            bail!("source mesh has no triangles");
        }
        if scol.len() != source.pos.len() {
            bail!("source mesh has {} colours for {} vertices", scol.len(), source.pos.len());
        }
        let bvh = Bvh::build(source);
        let mut out = Vec::with_capacity(self.pos.len());
        for &p in &self.pos {
            let hit = bvh.closest(source, p);
            let t = source.tri[hit.tri];
            let (a, b, c) = (source.pos[t[0] as usize], source.pos[t[1] as usize], source.pos[t[2] as usize]);
            let w = barycentric(hit.p, a, b, c);
            let mut col = [0.0f32; 4];
            for k in 0..4 {
                col[k] = w[0] * scol[t[0] as usize][k] + w[1] * scol[t[1] as usize][k] + w[2] * scol[t[2] as usize][k];
            }
            out.push(col);
        }
        self.col = Some(out);
        Ok(self.pos.len())
    }

    /// Project colours from an image. With no camera, uses a front orthographic
    /// view along -Z covering the mesh bbox (x right, y up, image v flipped).
    /// With `{origin, target, up, fov_deg, aspect}`, uses a pinhole camera.
    /// Vertices outside the image keep their previous colour.
    pub fn recolour_from_image(&mut self, img: &image::RgbaImage, camera: Option<&Value>) -> Result<usize> {
        let (iw, ih) = img.dimensions();
        if iw == 0 || ih == 0 {
            bail!("image has no pixels");
        }
        let mut out = self.col.clone().unwrap_or_else(|| vec![[1.0, 1.0, 1.0, 1.0]; self.pos.len()]);
        if out.len() != self.pos.len() {
            out = vec![[1.0, 1.0, 1.0, 1.0]; self.pos.len()];
        }
        let mut changed = 0usize;
        let sample = |u: f32, v: f32| -> Option<[f32; 4]> {
            if !(0.0..=1.0).contains(&u) || !(0.0..=1.0).contains(&v) {
                return None;
            }
            let x = ((u * (iw - 1) as f32).round() as u32).min(iw - 1);
            let y = (((1.0 - v) * (ih - 1) as f32).round() as u32).min(ih - 1);
            let p = img.get_pixel(x, y);
            Some([p[0] as f32 / 255.0, p[1] as f32 / 255.0, p[2] as f32 / 255.0, p[3] as f32 / 255.0])
        };
        if let Some(cam) = camera {
            let vec3 = |k: &str, default: [f32; 3]| -> [f32; 3] {
                let a = &cam[k];
                [a[0].as_f64().unwrap_or(default[0] as f64) as f32,
                 a[1].as_f64().unwrap_or(default[1] as f64) as f32,
                 a[2].as_f64().unwrap_or(default[2] as f64) as f32]
            };
            let origin = vec3("origin", [0.0, 0.0, 3.0]);
            let target = vec3("target", [0.0, 0.0, 0.0]);
            let forward = normalize_or(sub(target, origin), [0.0, 0.0, -1.0]);
            let up0 = normalize_or(vec3("up", [0.0, 1.0, 0.0]), [0.0, 1.0, 0.0]);
            let right = normalize_or(cross(forward, up0), [1.0, 0.0, 0.0]);
            let up = cross(right, forward);
            let fov = cam["fov_deg"].as_f64().unwrap_or(40.0).clamp(5.0, 120.0).to_radians();
            let aspect = cam["aspect"].as_f64().unwrap_or(iw as f64 / ih as f64).max(1e-3) as f32;
            let tan = (fov as f32 * 0.5).tan();
            for (i, &p) in self.pos.iter().enumerate() {
                let d = sub(p, origin);
                let z = dot(d, forward);
                if z <= 1e-6 {
                    continue;
                }
                let u = (dot(d, right) / (z * tan * aspect)) * 0.5 + 0.5;
                let v = 0.5 - (dot(d, up) / (z * tan)) * 0.5;
                if let Some(c) = sample(u, v) {
                    out[i] = c;
                    changed += 1;
                }
            }
        } else {
            let (mn, mx) = self.bbox();
            let (w, h) = ((mx[0] - mn[0]).max(1e-9), (mx[1] - mn[1]).max(1e-9));
            for (i, &p) in self.pos.iter().enumerate() {
                let u = (p[0] - mn[0]) / w;
                let v = (p[1] - mn[1]) / h;
                if let Some(c) = sample(u, v) {
                    out[i] = c;
                    changed += 1;
                }
            }
        }
        self.col = Some(out);
        Ok(changed)
    }

    pub fn area(&self) -> f32 {
        self.tri.iter().map(|t| {
            let (a, b, c) = (self.pos[t[0] as usize], self.pos[t[1] as usize], self.pos[t[2] as usize]);
            0.5 * length(cross(sub(b, a), sub(c, a)))
        }).sum()
    }

    pub fn stats(&self) -> MeshStats {
        let (mn, mx) = self.bbox();
        let mut edges: HashMap<(u32, u32), usize> = HashMap::new();
        for t in &self.tri {
            for j in 0..3 {
                let a = t[j];
                let b = t[(j + 1) % 3];
                let key = if a < b { (a, b) } else { (b, a) };
                *edges.entry(key).or_default() += 1;
            }
        }
        let mut ec = EdgeCounts::default();
        for &c in edges.values() {
            match c {
                0 | 1 => ec.boundary += 1,
                2 => ec.manifold += 1,
                _ => ec.non_manifold += 1,
            }
        }
        MeshStats {
            vertices: self.pos.len(),
            triangles: self.tri.len(),
            bbox_min: mn,
            bbox_max: mx,
            area: self.area(),
            watertight: !self.tri.is_empty() && ec.boundary == 0 && ec.non_manifold == 0,
            edge_counts: ec,
        }
    }

    /// Quadric edge collapse to about `target_faces`. Boundary vertices are kept
    /// so open meshes do not lose their silhouette. Colours/UVs are copied from
    /// the kept endpoint (seams are not re-unwrapped).
    pub fn decimate(&mut self, target_faces: usize) -> usize {
        if self.tri.len() <= target_faces || self.pos.len() < 4 {
            return self.tri.len();
        }
        let target_faces = target_faces.max(4);
        let nv = self.pos.len();
        let mut quad = vec![[[0.0f32; 4]; 4]; nv];
        for t in &self.tri {
            let (a, b, c) = (self.pos[t[0] as usize], self.pos[t[1] as usize], self.pos[t[2] as usize]);
            let mut n = cross(sub(b, a), sub(c, a));
            let l = length(n);
            if l < 1e-20 { continue; }
            n = [n[0] / l, n[1] / l, n[2] / l];
            let d = -dot(n, a);
            let mut plane = [[0.0f32; 4]; 4];
            for i in 0..3 {
                for j in 0..3 { plane[i][j] = n[i] * n[j]; }
                plane[i][3] = n[i] * d;
                plane[3][i] = n[i] * d;
            }
            plane[3][3] = d * d;
            for &v in t {
                for i in 0..4 { for j in 0..4 { quad[v as usize][i][j] += plane[i][j]; } }
            }
        }
        // Boundary vertices stay put.
        let mut deg: HashMap<(u32, u32), usize> = HashMap::new();
        for t in &self.tri {
            for j in 0..3 {
                let a = t[j];
                let b = t[(j + 1) % 3];
                let key = if a < b { (a, b) } else { (b, a) };
                *deg.entry(key).or_default() += 1;
            }
        }
        let mut boundary = vec![false; nv];
        for (&(a, b), &c) in &deg {
            if c == 1 { boundary[a as usize] = true; boundary[b as usize] = true; }
        }
        let mut vfaces: Vec<Vec<usize>> = vec![Vec::new(); nv];
        for (fi, t) in self.tri.iter().enumerate() {
            for &v in t { vfaces[v as usize].push(fi); }
        }
        let mut alive_v = vec![true; nv];
        let mut alive_f = vec![true; self.tri.len()];
        let mut live_f = self.tri.len();
        let mut ver = vec![0u32; nv];
        let mut heap = BinaryHeap::new();
        let mut pushed: HashSet<(u32, u32)> = HashSet::new();
        for (&(a, b), _) in &deg {
            if boundary[a as usize] || boundary[b as usize] { continue; }
            if pushed.insert((a, b)) {
                heap.push(Edge { cost: edge_cost(&quad, &self.pos, a, b), a, b, va: 0, vb: 0 });
            }
        }
        while live_f > target_faces {
            // Drop stale entries.
            let Some(e) = heap.pop() else { break };
            if !alive_v[e.a as usize] || !alive_v[e.b as usize] { continue; }
            if ver[e.a as usize] != e.va || ver[e.b as usize] != e.vb { continue; }
            // Collapse e.b into e.a. Midpoint is stable and avoids fold-overs
            // from an ill-conditioned optimal point.
            let a = e.a as usize;
            let b = e.b as usize;
            self.pos[a] = [
                (self.pos[a][0] + self.pos[b][0]) * 0.5,
                (self.pos[a][1] + self.pos[b][1]) * 0.5,
                (self.pos[a][2] + self.pos[b][2]) * 0.5,
            ];
            for i in 0..4 { for j in 0..4 { quad[a][i][j] += quad[b][i][j]; } }
            alive_v[b] = false;
            ver[a] = ver[a].wrapping_add(1);
            let faces = vfaces[b].clone();
            for fi in faces {
                if !alive_f[fi] { continue; }
                let mut changed = false;
                for v in self.tri[fi].iter_mut() {
                    if *v == e.b { *v = e.a; changed = true; }
                }
                if changed {
                    let t = self.tri[fi];
                    if t[0] == t[1] || t[1] == t[2] || t[0] == t[2] {
                        alive_f[fi] = false;
                        live_f -= 1;
                    } else {
                        vfaces[a].push(fi);
                    }
                }
            }
            // Refresh edges incident to the kept vertex.
            let mut seen_neighbors: HashSet<u32> = HashSet::new();
            for &fi in &vfaces[a] {
                if !alive_f[fi] { continue; }
                for &v in &self.tri[fi] {
                    if v as usize != a && alive_v[v as usize] && seen_neighbors.insert(v) {
                        let (lo, hi) = if (a as u32) < v { (a as u32, v) } else { (v, a as u32) };
                        if boundary[lo as usize] || boundary[hi as usize] { continue; }
                        heap.push(Edge {
                            cost: edge_cost(&quad, &self.pos, lo, hi),
                            a: lo, b: hi,
                            va: ver[lo as usize], vb: ver[hi as usize],
                        });
                    }
                }
            }
        }
        // Compact vertices, copying kept attributes.
        let mut map = vec![u32::MAX; nv];
        let mut pos = Vec::new();
        for i in 0..nv {
            if alive_v[i] {
                map[i] = pos.len() as u32;
                pos.push(self.pos[i]);
            }
        }
        let old_nor = self.nor.take();
        let old_col = self.col.take();
        let old_uv = self.uv.take();
        let mut nor = old_nor.map(|a| Vec::with_capacity(pos.len()).tap(|v| {
            for i in 0..nv { if alive_v[i] { v.push(a[i]); } }
        }));
        let mut col = old_col.map(|a| Vec::with_capacity(pos.len()).tap(|v| {
            for i in 0..nv { if alive_v[i] { v.push(a[i]); } }
        }));
        let mut uv = old_uv.map(|a| Vec::with_capacity(pos.len()).tap(|v| {
            for i in 0..nv { if alive_v[i] { v.push(a[i]); } }
        }));
        let tri: Vec<[u32; 3]> = self.tri.iter().enumerate()
            .filter(|(i, _)| alive_f[*i])
            .map(|(_, t)| [map[t[0] as usize], map[t[1] as usize], map[t[2] as usize]])
            .collect();
        self.pos = pos;
        self.tri = tri;
        self.nor = nor.take();
        self.col = col.take();
        self.uv = uv.take();
        self.tri.len()
    }

    /// Voxel remesh: sample the signed distance field at `target_edge` spacing,
    /// then run the existing marching-cubes tables. Returns the new triangle count.
    pub fn remesh(&mut self, target_edge: f32, max_resolution: usize) -> Result<usize> {
        let (mn, mx) = self.bbox();
        let extent = (mx[0] - mn[0]).max(mx[1] - mn[1]).max(mx[2] - mn[2]);
        if !(extent > 0.0) || !target_edge.is_finite() || target_edge <= 0.0 {
            bail!("remesh needs a positive target edge length");
        }
        let res = ((extent / target_edge).ceil() as usize + 1)
            .clamp(4, max_resolution.clamp(4, MAX_SDF_RESOLUTION));
        let sdf = compute_sdf(self, res, SdfMethod::Pseudonormal)?;
        if let Some(m) = sdf.isosurface() {
            // Keep colours only when the input had them; remeshed vertices have
            // no meaningful correspondence, so start white.
            let col = self.col.as_ref().map(|_| vec![[1.0, 1.0, 1.0, 1.0]; m.pos.len()]);
            self.pos = m.pos;
            self.tri = m.tri;
            self.nor = None;
            self.col = col;
            self.uv = None;
        }
        Ok(self.tri.len())
    }

    /// Weld, drop degenerate faces, orient, and fill small boundary loops.
    pub fn repair(&mut self, weld_tolerance: f32, max_fill_loop: usize) -> RepairReport {
        let before_verts = self.pos.len();
        self.weld(weld_tolerance);
        let welded = before_verts.saturating_sub(self.pos.len());
        let before_faces = self.tri.len();
        let pos = &self.pos;
        self.tri.retain(|t| t[0] != t[1] && t[1] != t[2] && t[0] != t[2] && {
            let (a, b, c) = (pos[t[0] as usize], pos[t[1] as usize], pos[t[2] as usize]);
            length(cross(sub(b, a), sub(c, a))) > 1e-16
        });
        let removed_degenerate = before_faces - self.tri.len();
        let flipped = self.orient_faces();
        let (filled, unfixed) = self.fill_boundaries(max_fill_loop);
        let stats = self.stats();
        RepairReport {
            welded_vertices: welded,
            removed_degenerate,
            flipped_faces: flipped,
            holes_filled: filled,
            boundary_edges: stats.edge_counts.boundary,
            non_manifold_edges: stats.edge_counts.non_manifold,
            unfixed_boundaries: unfixed,
        }
    }

    fn weld(&mut self, tolerance: f32) {
        if self.pos.is_empty() { return; }
        let tol = tolerance.max(0.0);
        // Quantise so nearby vertices share a bucket; check the 27 neighbours.
        let mut buckets: HashMap<(i64, i64, i64), Vec<u32>> = HashMap::new();
        let mut map = vec![u32::MAX; self.pos.len()];
        let mut pos: Vec<[f32; 3]> = Vec::new();
        let mut nor = self.nor.as_ref().map(|_| Vec::new());
        let mut col = self.col.as_ref().map(|_| Vec::new());
        let mut uv = self.uv.as_ref().map(|_| Vec::new());
        for (i, p) in self.pos.iter().enumerate() {
            let k = ((p[0] / tol.max(1e-12)).round() as i64,
                     (p[1] / tol.max(1e-12)).round() as i64,
                     (p[2] / tol.max(1e-12)).round() as i64);
            let mut found = None;
            'search: for dx in -1..=1 { for dy in -1..=1 { for dz in -1..=1 {
                if let Some(v) = buckets.get(&(k.0 + dx, k.1 + dy, k.2 + dz)) {
                    for &j in v {
                        let q = pos[j as usize];
                        let d = ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt();
                        if d <= tol + 1e-12 { found = Some(j); break 'search; }
                    }
                }
            }}}
            let idx = if let Some(j) = found {
                j
            } else {
                let j = pos.len() as u32;
                pos.push(*p);
                if let Some(n) = self.nor.as_ref() { nor.as_mut().unwrap().push(n[i]); }
                if let Some(c) = self.col.as_ref() { col.as_mut().unwrap().push(c[i]); }
                if let Some(t) = self.uv.as_ref() { uv.as_mut().unwrap().push(t[i]); }
                buckets.entry(k).or_default().push(j);
                j
            };
            map[i] = idx;
        }
        for t in self.tri.iter_mut() {
            t[0] = map[t[0] as usize];
            t[1] = map[t[1] as usize];
            t[2] = map[t[2] as usize];
        }
        self.pos = pos;
        self.nor = nor;
        self.col = col;
        self.uv = uv;
    }

    /// Add a fan to boundary loops up to `max_loop` vertices. Returns
    /// (loops filled, loops left open).
    fn fill_boundaries(&mut self, max_loop: usize) -> (usize, usize) {
        let mut counts: HashMap<(u32, u32), usize> = HashMap::new();
        let mut directed: HashMap<u32, Vec<u32>> = HashMap::new();
        for t in &self.tri {
            for j in 0..3 {
                let a = t[j];
                let b = t[(j + 1) % 3];
                let key = if a < b { (a, b) } else { (b, a) };
                *counts.entry(key).or_default() += 1;
                directed.entry(a).or_default().push(b);
            }
        }
        let mut boundary_out: HashMap<u32, u32> = HashMap::new();
        for (&(a, b), &c) in &counts {
            if c == 1 {
                // Find the directed version.
                if directed.get(&a).map_or(false, |v| v.contains(&b)) { boundary_out.insert(a, b); }
                else if directed.get(&b).map_or(false, |v| v.contains(&a)) { boundary_out.insert(b, a); }
            }
        }
        let mut used: HashSet<u32> = HashSet::new();
        let mut filled = 0usize;
        let mut open = 0usize;
        let starts: Vec<u32> = boundary_out.keys().copied().collect();
        for start in starts {
            if used.contains(&start) { continue; }
            let mut loop_v = Vec::new();
            let mut cur = start;
            loop {
                if used.contains(&cur) { break; }
                used.insert(cur);
                loop_v.push(cur);
                match boundary_out.get(&cur) {
                    Some(&n) => cur = n,
                    None => break,
                }
                if cur == start { break; }
                if loop_v.len() > max_loop { break; }
            }
            if loop_v.len() < 3 || loop_v.len() > max_loop {
                open += 1;
                continue;
            }
            // Closed when the last vertex points back at the start.
            let closed = *loop_v.last().unwrap() == start
                || boundary_out.get(loop_v.last().unwrap()).map_or(false, |&n| n == start);
            if !closed {
                open += 1;
                continue;
            }
            // loop_v[0..n] is a closed loop; reverse-wind the fan so it is
            // consistent with the boundary edges.
            for i in 1..loop_v.len() - 1 {
                self.tri.push([loop_v[0], loop_v[i + 1], loop_v[i]]);
            }
            filled += 1;
        }
        (filled, open)
    }

    /// Box projection UVs from the vertex normals (or the bbox when normals are
    /// missing). One chart, seams are not split.
    pub fn add_box_uvs(&mut self) {
        if self.nor.is_none() {
            self.compute_normals();
        }
        let (mn, mx) = self.bbox();
        let d = [(mx[0] - mn[0]).max(1e-12), (mx[1] - mn[1]).max(1e-12), (mx[2] - mn[2]).max(1e-12)];
        let mut uv = Vec::with_capacity(self.pos.len());
        for (i, p) in self.pos.iter().enumerate() {
            let n = self.nor.as_ref().map(|a| a[i]).unwrap_or([0.0, 0.0, 1.0]);
            let axis = (0..3).max_by(|&a, &b| n[a].abs().partial_cmp(&n[b].abs()).unwrap_or(Ordering::Equal)).unwrap_or(2);
            let (u, v) = match axis {
                0 => (1, 2),
                1 => (0, 2),
                _ => (0, 1),
            };
            uv.push([(p[u] - mn[u]) / d[u], (p[v] - mn[v]) / d[v]]);
        }
        self.uv = Some(uv);
    }

    /// Least-squares conformal map (LSCM) UVs, in-tree and pure Rust.
    ///
    /// Each connected component is a chart with two pinned vertices. For every
    /// triangle we alternate between fitting a complex similarity `a*z + b`
    /// from the current UVs and averaging the fitted UVs back onto the vertices.
    /// This is the alternating least-squares form of the LSCM equations; no
    /// sparse linear solver or external chart library is needed. Charts are not
    /// packed or seam-split, so overlapping components share UV space.
    pub fn add_lscm_uvs(&mut self, iterations: usize) -> Result<LscmReport> {
        let n = self.pos.len();
        if n == 0 || self.tri.is_empty() {
            bail!("LSCM needs a mesh with vertices and triangles");
        }
        if n > 200_000 {
            bail!("LSCM vertex cap is 200000, got {n}");
        }
        let iterations = iterations.clamp(1, 2000);

        // vertex -> incident faces
        let mut vertex_faces = vec![Vec::<usize>::new(); n];
        for (fi, t) in self.tri.iter().enumerate() {
            for &v in t {
                let v = v as usize;
                if v >= n {
                    bail!("triangle references vertex {v} outside the {n}-vertex mesh");
                }
                vertex_faces[v].push(fi);
            }
        }

        // connected components (BFS over vertices)
        let mut seen = vec![false; n];
        let mut components = Vec::new();
        for seed in 0..n {
            if seen[seed] {
                continue;
            }
            let mut stack = vec![seed];
            let mut comp = Vec::new();
            seen[seed] = true;
            while let Some(v) = stack.pop() {
                comp.push(v);
                for &fi in &vertex_faces[v] {
                    for &w in &self.tri[fi] {
                        let w = w as usize;
                        if !seen[w] {
                            seen[w] = true;
                            stack.push(w);
                        }
                    }
                }
            }
            components.push(comp);
        }

        let mut uv = vec![Complex::default(); n];
        let mut max_error = 0.0f32;
        for comp in &components {
            let (mn, mx) = (
                [0usize, 1, 2].map(|a| comp.iter().map(|&v| self.pos[v][a]).fold(f32::INFINITY, f32::min)),
                [0usize, 1, 2].map(|a| comp.iter().map(|&v| self.pos[v][a]).fold(f32::NEG_INFINITY, f32::max)),
            );
            let extent = [0usize, 1, 2]
                .map(|a| (mx[a] - mn[a]).abs())
                .into_iter()
                .fold(0.0f32, f32::max)
                .max(1e-12);
            // Pin the corner with the smallest coordinate sum and the farthest vertex.
            let anchor = comp
                .iter()
                .copied()
                .min_by(|&a, &b| {
                    let sa: f32 = self.pos[a].iter().sum();
                    let sb: f32 = self.pos[b].iter().sum();
                    sa.partial_cmp(&sb).unwrap_or(Ordering::Equal)
                })
                .unwrap();
            let axis = [0usize, 1, 2]
                .into_iter()
                .max_by(|&a, &b| (mx[a] - mn[a]).partial_cmp(&(mx[b] - mn[b])).unwrap_or(Ordering::Equal))
                .unwrap();
            let far = comp
                .iter()
                .copied()
                .max_by(|&a, &b| self.pos[a][axis].partial_cmp(&self.pos[b][axis]).unwrap_or(Ordering::Equal))
                .unwrap();
            uv[anchor] = Complex::default();
            uv[far] = Complex { re: extent, im: 0.0 };

            // Per-face 2D coordinates in the face's own tangent plane.
            let mut faces: Vec<([usize; 3], [Complex; 3])> = Vec::new();
            let in_comp: HashSet<usize> = comp.iter().copied().collect();
            for &fi in &self.tri {
                let t = [fi[0] as usize, fi[1] as usize, fi[2] as usize];
                if !t.iter().all(|v| in_comp.contains(v)) {
                    continue;
                }
                let p0 = self.pos[t[0]];
                let e1 = [
                    self.pos[t[1]][0] - p0[0],
                    self.pos[t[1]][1] - p0[1],
                    self.pos[t[1]][2] - p0[2],
                ];
                let e1len = (e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]).sqrt();
                if e1len < 1e-12 {
                    continue;
                }
                let e1 = [e1[0] / e1len, e1[1] / e1len, e1[2] / e1len];
                // Gram-Schmidt against e1 using the world up unless it is parallel.
                let up = if e1[1].abs() < 0.9 { [0.0, 1.0, 0.0] } else { [1.0, 0.0, 0.0] };
                let dot = e1[0] * up[0] + e1[1] * up[1] + e1[2] * up[2];
                let mut e2 = [up[0] - dot * e1[0], up[1] - dot * e1[1], up[2] - dot * e1[2]];
                let e2len = (e2[0] * e2[0] + e2[1] * e2[1] + e2[2] * e2[2]).sqrt();
                if e2len < 1e-12 {
                    continue;
                }
                e2 = [e2[0] / e2len, e2[1] / e2len, e2[2] / e2len];
                let proj = |v: [f32; 3]| -> Complex {
                    let d = [v[0] - p0[0], v[1] - p0[1], v[2] - p0[2]];
                    Complex {
                        re: d[0] * e1[0] + d[1] * e1[1] + d[2] * e1[2],
                        im: d[0] * e2[0] + d[1] * e2[1] + d[2] * e2[2],
                    }
                };
                faces.push((t, [proj(self.pos[t[0]]), proj(self.pos[t[1]]), proj(self.pos[t[2]])]));
            }
            if faces.is_empty() {
                continue;
            }

            for _ in 0..iterations {
                let mut acc = vec![Complex::default(); n];
                let mut cnt = vec![0usize; n];
                for (t, z) in &faces {
                    let w = [uv[t[0]], uv[t[1]], uv[t[2]]];
                    let mean_z = z[0].add(z[1]).add(z[2]).div_f32(3.0);
                    let mean_w = w[0].add(w[1]).add(w[2]).div_f32(3.0);
                    let mut num = Complex::default();
                    let mut den = 0.0f32;
                    for i in 0..3 {
                        let dz = z[i].sub(mean_z);
                        let dw = w[i].sub(mean_w);
                        num = num.add(dz.conj().mul(dw));
                        den += dz.abs2();
                    }
                    let a = if den > 1e-20 { num.div(Complex { re: den, im: 0.0 }) } else { Complex::default() };
                    let b = mean_w.sub(a.mul(mean_z));
                    for i in 0..3 {
                        acc[t[i]] = acc[t[i]].add(a.mul(z[i]).add(b));
                        cnt[t[i]] += 1;
                    }
                }
                let mut delta = 0.0f32;
                for &v in comp {
                    if v == anchor || v == far || cnt[v] == 0 {
                        continue;
                    }
                    let target = acc[v].div_f32(cnt[v] as f32);
                    let next = uv[v].add(target.sub(uv[v]).div_f32(2.0)); // damped Jacobi
                    if next.finite() {
                        delta = delta.max(target.sub(uv[v]).abs2().sqrt());
                        uv[v] = next;
                    }
                }
                if delta < 1e-5 {
                    break;
                }
                max_error = max_error.max(delta);
            }
        }

        if uv.iter().any(|c| !c.finite()) {
            bail!("LSCM produced non-finite UVs");
        }
        self.uv = Some(uv.iter().map(|c| [c.re, c.im]).collect());
        Ok(LscmReport {
            charts: components.len(),
            iterations,
            max_error,
            seams_split: false,
        })
    }
}

// Small helper so the decimate compaction stays readable.
trait Tap: Sized {
    fn tap(mut self, f: impl FnOnce(&mut Self)) -> Self { f(&mut self); self }
}
impl<T> Tap for T {}

#[derive(Clone, Copy)]
struct Edge {
    cost: f32,
    a: u32,
    b: u32,
    va: u32,
    vb: u32,
}
impl PartialEq for Edge {
    fn eq(&self, other: &Self) -> bool { self.cost == other.cost }
}
impl Eq for Edge {}
impl PartialOrd for Edge {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}
impl Ord for Edge {
    // Reverse: BinaryHeap is a max-heap, we want the cheapest edge first.
    fn cmp(&self, other: &Self) -> Ordering {
        other.cost.partial_cmp(&self.cost).unwrap_or(Ordering::Equal)
    }
}

fn edge_cost(q: &[[[f32; 4]; 4]], pos: &[[f32; 3]], a: u32, b: u32) -> f32 {
    let pa = pos[a as usize];
    let pb = pos[b as usize];
    let p = [(pa[0] + pb[0]) * 0.5, (pa[1] + pb[1]) * 0.5, (pa[2] + pb[2]) * 0.5];
    let mut s = 0.0f32;
    for i in 0..4 {
        for j in 0..4 {
            let pi = if i < 3 { p[i] } else { 1.0 };
            let pj = if j < 3 { p[j] } else { 1.0 };
            s += (q[a as usize][i][j] + q[b as usize][i][j]) * pi * pj;
        }
    }
    s
}

// ---------- vector helpers ----------

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] { [a[0] - b[0], a[1] - b[1], a[2] - b[2]] }
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 { a[0] * b[0] + a[1] * b[1] + a[2] * b[2] }
fn length(a: [f32; 3]) -> f32 { dot(a, a).sqrt() }
fn normalize_or(a: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let l = length(a);
    if l > 1e-12 { [a[0] / l, a[1] / l, a[2] / l] } else { fallback }
}

// ---------- format detection / decode ----------

pub fn decode_mesh_data(bytes: &[u8]) -> Result<Mesh> {
    let mut mesh = if bytes.len() >= 12 && &bytes[0..4] == b"glTF" {
        decode_glb(bytes)?
    } else if bytes.starts_with(b"ply\n") || bytes.starts_with(b"ply\r\n") {
        decode_ply(bytes)?
    } else if let Some(m) = decode_stl_binary(bytes)? {
        m
    } else if let Ok(text) = std::str::from_utf8(bytes) {
        let t = text.trim_start();
        if t.starts_with("solid") && t.contains("facet normal") {
            decode_stl_ascii(text)?
        } else {
            parse_obj(text)?
        }
    } else {
        bail!("unsupported mesh format (expected GLB, OBJ, PLY, or STL)");
    };
    mesh.validate()?;
    if mesh.pos.is_empty() {
        bail!("mesh has no vertices");
    }
    // Malformed files can declare normals/UVs/colours for only some vertices;
    // drop attributes that do not line up rather than emit a broken GLB.
    let n = mesh.pos.len();
    if mesh.nor.as_ref().map_or(false, |a| a.len() != n) {
        mesh.nor = None;
    }
    if mesh.col.as_ref().map_or(false, |a| a.len() != n) {
        mesh.col = None;
    }
    if mesh.uv.as_ref().map_or(false, |a| a.len() != n) {
        mesh.uv = None;
    }
    Ok(mesh)
}

fn parse_obj(text: &str) -> Result<Mesh> {
    let mut pos: Vec<[f32; 3]> = Vec::new();
    let mut nor: Vec<[f32; 3]> = Vec::new();
    let mut uv: Vec<[f32; 2]> = Vec::new();
    let mut tri: Vec<[u32; 3]> = Vec::new();
    let mut map: HashMap<(i64, i64, i64), u32> = HashMap::new();
    let mut out_pos: Vec<[f32; 3]> = Vec::new();
    let mut out_nor: Vec<[f32; 3]> = Vec::new();
    let mut out_uv: Vec<[f32; 2]> = Vec::new();
    let mut any_nor = false;
    let mut any_uv = false;

    let resolve = |raw: i64, len: usize| -> Option<usize> {
        if raw == 0 { return None; }
        if raw > 0 { let i = (raw - 1) as usize; if i < len { Some(i) } else { None } }
        else { let i = len as i64 + raw; if i >= 0 && (i as usize) < len { Some(i as usize) } else { None } }
    };

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') { continue; }
        let mut it = line.split_whitespace();
        let tag = it.next().unwrap_or("");
        match tag {
            "v" => {
                let x = it.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
                let y = it.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
                let z = it.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
                pos.push([x, y, z]);
            }
            "vn" => {
                let x = it.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
                let y = it.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
                let z = it.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
                nor.push([x, y, z]);
                any_nor = true;
            }
            "vt" => {
                let u = it.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
                let v = it.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
                uv.push([u, v]);
                any_uv = true;
            }
            "f" => {
                let refs: Vec<(i64, i64, i64)> = it.filter_map(|tok| {
                    let mut parts = tok.split('/');
                    let vi = parts.next().unwrap_or("").parse::<i64>().ok()?;
                    let ti = parts.next().and_then(|s| if s.is_empty() { None } else { s.parse().ok() }).unwrap_or(0);
                    let ni = parts.next().and_then(|s| if s.is_empty() { None } else { s.parse().ok() }).unwrap_or(0);
                    Some((vi, ti, ni))
                }).collect();
                let ids: Vec<u32> = refs.iter().map(|&(vi, ti, ni)| {
                    *map.entry((vi, ti, ni)).or_insert_with(|| {
                        let id = out_pos.len() as u32;
                        let v = resolve(vi, pos.len()).map(|i| pos[i]).unwrap_or([0.0; 3]);
                        out_pos.push(v);
                        if any_nor { out_nor.push(resolve(ni, nor.len()).map(|i| nor[i]).unwrap_or([0.0, 0.0, 1.0])); }
                        if any_uv { out_uv.push(resolve(ti, uv.len()).map(|i| uv[i]).unwrap_or([0.0, 0.0])); }
                        id
                    })
                }).collect();
                for i in 1..ids.len().saturating_sub(1) {
                    tri.push([ids[0], ids[i], ids[i + 1]]);
                }
            }
            _ => {}
        }
    }
    if out_pos.is_empty() || tri.is_empty() {
        bail!("OBJ contains no faces");
    }
    Ok(Mesh {
        pos: out_pos,
        tri,
        nor: if any_nor { Some(out_nor) } else { None },
        col: None,
        uv: if any_uv { Some(out_uv) } else { None },
    })
}

fn decode_stl_binary(bytes: &[u8]) -> Result<Option<Mesh>> {
    if bytes.len() < 84 { return Ok(None); }
    let n = u32::from_le_bytes([bytes[80], bytes[81], bytes[82], bytes[83]]) as usize;
    if bytes.len() != 84 + n * 50 { return Ok(None); }
    let mut pos = Vec::with_capacity(n * 3);
    let mut tri = Vec::with_capacity(n);
    for f in 0..n {
        let base = 84 + f * 50 + 12; // skip the facet normal
        let start = pos.len() as u32;
        for v in 0..3 {
            let o = base + v * 12;
            let p = [
                f32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]),
                f32::from_le_bytes([bytes[o + 4], bytes[o + 5], bytes[o + 6], bytes[o + 7]]),
                f32::from_le_bytes([bytes[o + 8], bytes[o + 9], bytes[o + 10], bytes[o + 11]]),
            ];
            pos.push(p);
        }
        tri.push([start, start + 1, start + 2]);
    }
    Ok(Some(Mesh { pos, tri, ..Mesh::default() }))
}

fn decode_stl_ascii(text: &str) -> Result<Mesh> {
    let mut pos = Vec::new();
    let mut tri = Vec::new();
    let mut verts: Vec<[f32; 3]> = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("vertex ") {
            let mut it = rest.split_whitespace();
            let x = it.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
            let y = it.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
            let z = it.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
            verts.push([x, y, z]);
            if verts.len() == 3 {
                let s = pos.len() as u32;
                pos.extend_from_slice(&verts);
                tri.push([s, s + 1, s + 2]);
                verts.clear();
            }
        }
    }
    if tri.is_empty() { bail!("ASCII STL contains no facets"); }
    Ok(Mesh { pos, tri, ..Mesh::default() })
}

fn decode_glb(bytes: &[u8]) -> Result<Mesh> {
    if bytes.len() < 20 { bail!("truncated GLB"); }
    let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    if magic != 0x4654_6C67 { bail!("not a GLB file"); }
    let total = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
    if total > bytes.len() { bail!("truncated GLB: header says {} bytes, file has {}", total, bytes.len()); }
    let mut off = 12usize;
    let mut js: Option<serde_json::Value> = None;
    let mut bin: Option<&[u8]> = None;
    while off + 8 <= total {
        let len = u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]) as usize;
        let ty = u32::from_le_bytes([bytes[off + 4], bytes[off + 5], bytes[off + 6], bytes[off + 7]]);
        let start = off + 8;
        let end = start.checked_add(len).ok_or_else(|| anyhow::anyhow!("GLB chunk overflow"))?;
        if end > bytes.len() { bail!("truncated GLB chunk"); }
        match ty {
            0x4E4F_534A => js = Some(serde_json::from_slice(&bytes[start..end])?),
            0x004E_4942 => bin = Some(&bytes[start..end]),
            _ => {}
        }
        off = end;
    }
    let js = js.ok_or_else(|| anyhow::anyhow!("GLB has no JSON chunk"))?;
    let bin = bin.ok_or_else(|| anyhow::anyhow!("GLB has no BIN chunk"))?;
    let prim = &js["meshes"][0]["primitives"][0];
    let pos_idx = prim["attributes"]["POSITION"].as_u64()
        .ok_or_else(|| anyhow::anyhow!("GLB primitive has no POSITION"))? as usize;
    let (pos, _) = read_vec3_f32(&js, bin, pos_idx)?;
    let tri = if let Some(i) = prim["indices"].as_u64() {
        let idx = read_indices(&js, bin, i as usize)?;
        if idx.len() < 3 { bail!("GLB index buffer is empty"); }
        idx.chunks(3).filter(|c| c.len() == 3).map(|c| [c[0], c[1], c[2]]).collect()
    } else {
        (0..pos.len() as u32).collect::<Vec<_>>().chunks(3).filter(|c| c.len() == 3).map(|c| [c[0], c[1], c[2]]).collect()
    };
    let nor = prim["attributes"]["NORMAL"].as_u64()
        .and_then(|i| read_vec3_f32(&js, bin, i as usize).ok())
        .map(|(v, _)| v);
    let uv = prim["attributes"]["TEXCOORD_0"].as_u64()
        .and_then(|i| read_vec2_f32(&js, bin, i as usize).ok());
    let col = prim["attributes"]["COLOR_0"].as_u64()
        .and_then(|i| read_colors(&js, bin, i as usize).ok());
    Ok(Mesh { pos, tri, nor, col, uv })
}

fn accessor<'a>(js: &serde_json::Value, bin: &'a [u8], idx: usize) -> Result<Accessor<'a>> {
    let a = js["accessors"].get(idx).ok_or_else(|| anyhow::anyhow!("missing accessor {idx}"))?;
    let view_idx = a["bufferView"].as_u64().ok_or_else(|| anyhow::anyhow!("accessor {idx} has no bufferView"))? as usize;
    let bv = js["bufferViews"].get(view_idx).ok_or_else(|| anyhow::anyhow!("missing bufferView {view_idx}"))?;
    let comps = match a["type"].as_str().unwrap_or("SCALAR") {
        "SCALAR" => 1, "VEC2" => 2, "VEC3" => 3, "VEC4" => 4, "MAT4" => 16, _ => 1,
    };
    let ct = a["componentType"].as_u64().unwrap_or(5126) as u32;
    let count = a["count"].as_u64().unwrap_or(0) as usize;
    let elem = match ct { 5120 | 5121 => 1, 5122 | 5123 => 2, 5125 | 5126 => 4, _ => 1 };
    let stride = bv["byteStride"].as_u64().map(|x| x as usize).unwrap_or(elem * comps);
    let off = (bv["byteOffset"].as_u64().unwrap_or(0) + a["byteOffset"].as_u64().unwrap_or(0)) as usize;
    let need = if count == 0 { 0 } else { stride * (count - 1) + elem * comps };
    if off + need > bin.len() {
        bail!("accessor {idx} reads past the GLB binary chunk");
    }
    Ok(Accessor { bytes: &bin[off..], count, comps, ct, stride })
}

struct Accessor<'a> {
    bytes: &'a [u8],
    count: usize,
    comps: usize,
    ct: u32,
    stride: usize,
}

fn read_f32_at(b: &[u8], o: usize) -> f32 {
    f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}
fn read_u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}
fn read_u16_at(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}

fn read_vec3_f32(js: &serde_json::Value, bin: &[u8], idx: usize) -> Result<(Vec<[f32; 3]>, usize)> {
    let a = accessor(js, bin, idx)?;
    if a.ct != 5126 || a.comps < 3 { bail!("POSITION/NORMAL accessor is not f32 VEC3"); }
    let mut out = Vec::with_capacity(a.count);
    for i in 0..a.count {
        let o = i * a.stride;
        out.push([read_f32_at(a.bytes, o), read_f32_at(a.bytes, o + 4), read_f32_at(a.bytes, o + 8)]);
    }
    Ok((out, a.count))
}

fn read_vec2_f32(js: &serde_json::Value, bin: &[u8], idx: usize) -> Result<Vec<[f32; 2]>> {
    let a = accessor(js, bin, idx)?;
    if a.ct != 5126 || a.comps < 2 { bail!("TEXCOORD_0 accessor is not f32 VEC2"); }
    let mut out = Vec::with_capacity(a.count);
    for i in 0..a.count {
        let o = i * a.stride;
        out.push([read_f32_at(a.bytes, o), read_f32_at(a.bytes, o + 4)]);
    }
    Ok(out)
}

fn read_colors(js: &serde_json::Value, bin: &[u8], idx: usize) -> Result<Vec<[f32; 4]>> {
    let a = accessor(js, bin, idx)?;
    let scale = match a.ct { 5121 => 255.0, 5123 => 65535.0, _ => 1.0 };
    let mut out = Vec::with_capacity(a.count);
    for i in 0..a.count {
        let o = i * a.stride;
        let mut c = [1.0f32; 4];
        for k in 0..a.comps.min(4) {
            let v = match a.ct {
                5126 => read_f32_at(a.bytes, o + k * 4),
                5121 => a.bytes[o + k] as f32 / scale,
                5123 => read_u16_at(a.bytes, o + k * 2) as f32 / scale,
                _ => 1.0,
            };
            c[k] = v;
        }
        out.push(c);
    }
    Ok(out)
}

fn read_indices(js: &serde_json::Value, bin: &[u8], idx: usize) -> Result<Vec<u32>> {
    let a = accessor(js, bin, idx)?;
    let mut out = Vec::with_capacity(a.count);
    for i in 0..a.count {
        let o = i * a.stride;
        out.push(match a.ct {
            5121 => a.bytes[o] as u32,
            5123 => read_u16_at(a.bytes, o) as u32,
            5125 => read_u32_at(a.bytes, o),
            _ => bail!("unsupported GLB index component type {}", a.ct),
        });
    }
    Ok(out)
}

// ---------- PLY ----------

#[derive(Clone)]
enum PlyProp { Scalar(String, String), List(String, String, String) }
#[derive(Clone)]
struct PlyElement { name: String, count: usize, props: Vec<PlyProp> }

fn decode_ply(bytes: &[u8]) -> Result<Mesh> {
    let header_end;
    let mut off = 0usize;
    loop {
        let Some(rel) = bytes[off..].windows(1).position(|w| w[0] == b'\n') else {
            bail!("PLY header has no end_header");
        };
        let line = std::str::from_utf8(&bytes[off..off + rel]).unwrap_or("").trim_end_matches('\r');
        off += rel + 1;
        if line == "end_header" { header_end = off; break; }
    }
    let header = std::str::from_utf8(&bytes[..header_end])?;
    let mut format = "ascii";
    let mut elements: Vec<PlyElement> = Vec::new();
    for line in header.lines() {
        let line = line.trim_end_matches('\r').trim();
        if line.starts_with("format ") { format = line.split_whitespace().nth(1).unwrap_or("ascii"); }
        else if let Some(rest) = line.strip_prefix("element ") {
            let mut it = rest.split_whitespace();
            let name = it.next().unwrap_or("").to_string();
            let count = it.next().and_then(|s| s.parse().ok()).unwrap_or(0);
            elements.push(PlyElement { name, count, props: Vec::new() });
        } else if let Some(rest) = line.strip_prefix("property ") {
            let Some(el) = elements.last_mut() else { continue };
            let mut it = rest.split_whitespace();
            let first = it.next().unwrap_or("");
            if first == "list" {
                let ct = it.next().unwrap_or("uchar").to_string();
                let it_ty = it.next().unwrap_or("int").to_string();
                let name = it.next().unwrap_or("").to_string();
                el.props.push(PlyProp::List(name, ct, it_ty));
            } else {
                let ty = first.to_string();
                let name = it.next().unwrap_or("").to_string();
                el.props.push(PlyProp::Scalar(name, ty));
            }
        }
    }
    let binary = format != "ascii";
    if binary && format != "binary_little_endian" { bail!("unsupported PLY format '{format}'"); }

    let mut pos: Vec<[f32; 3]> = Vec::new();
    let mut col: Vec<[f32; 4]> = Vec::new();
    let mut nor: Vec<[f32; 3]> = Vec::new();
    let mut uv: Vec<[f32; 2]> = Vec::new();
    let mut tri: Vec<[u32; 3]> = Vec::new();
    let mut has_col = false;
    let mut has_nor = false;
    let mut has_uv = false;

    // ASCII token cursor.
    let text = if binary { "" } else { std::str::from_utf8(&bytes[header_end..])? };
    let mut tokens = text.split_whitespace();
    let mut boff = header_end;

    for el in &elements {
        for _ in 0..el.count {
            let mut xyz = [0.0f32; 3];
            let mut rgb = [1.0f32; 4];
            let mut nrm = [0.0f32; 3];
            let mut tex = [0.0f32; 2];
            let mut face: Vec<u32> = Vec::new();
            for prop in &el.props {
                match prop {
                    PlyProp::Scalar(name, ty) => {
                        let v = if binary { read_ply_scalar(bytes, &mut boff, ty)? } else { next_ply_token(&mut tokens)?.parse::<f64>().unwrap_or(0.0) };
                        match name.as_str() {
                            "x" => xyz[0] = v as f32,
                            "y" => xyz[1] = v as f32,
                            "z" => xyz[2] = v as f32,
                            "nx" => { nrm[0] = v as f32; has_nor = true; }
                            "ny" => { nrm[1] = v as f32; has_nor = true; }
                            "nz" => { nrm[2] = v as f32; has_nor = true; }
                            "red" | "r" => { rgb[0] = if v > 1.0 { v as f32 / 255.0 } else { v as f32 }; has_col = true; }
                            "green" | "g" => { rgb[1] = if v > 1.0 { v as f32 / 255.0 } else { v as f32 }; has_col = true; }
                            "blue" | "b" => { rgb[2] = if v > 1.0 { v as f32 / 255.0 } else { v as f32 }; has_col = true; }
                            "alpha" | "a" => rgb[3] = if v > 1.0 { v as f32 / 255.0 } else { v as f32 },
                            "u" | "s" | "texture_u" | "texture_s" => { tex[0] = v as f32; has_uv = true; }
                            "v" | "t" | "texture_v" | "texture_t" => { tex[1] = v as f32; has_uv = true; }
                            _ => {}
                        }
                    }
                    PlyProp::List(name, ct, it_ty) => {
                        let n = if binary {
                            read_ply_scalar(bytes, &mut boff, ct)? as usize
                        } else {
                            next_ply_token(&mut tokens)?.parse::<usize>().unwrap_or(0)
                        };
                        let mut vals = Vec::with_capacity(n);
                        for _ in 0..n {
                            vals.push(if binary {
                                read_ply_scalar(bytes, &mut boff, it_ty)? as u32
                            } else {
                                next_ply_token(&mut tokens)?.parse::<u32>().unwrap_or(0)
                            });
                        }
                        if name.contains("vertex_ind") || name == "vertex_index" || name == "vertex" {
                            face = vals;
                        }
                    }
                }
            }
            match el.name.as_str() {
                "vertex" => {
                    pos.push(xyz);
                    if has_col { col.push(rgb); }
                    if has_nor { nor.push(nrm); }
                    if has_uv { uv.push(tex); }
                }
                "face" if face.len() >= 3 => {
                    for i in 1..face.len() - 1 {
                        tri.push([face[0], face[i], face[i + 1]]);
                    }
                }
                _ => {}
            }
        }
    }
    if pos.is_empty() || tri.is_empty() { bail!("PLY contains no faces"); }
    Ok(Mesh {
        pos,
        tri,
        nor: if has_nor { Some(nor) } else { None },
        col: if has_col { Some(col) } else { None },
        uv: if has_uv { Some(uv) } else { None },
    })
}

fn next_ply_token<'a>(it: &mut impl Iterator<Item = &'a str>) -> Result<&'a str> {
    it.next().ok_or_else(|| anyhow::anyhow!("PLY body ended early"))
}

fn ply_size(ty: &str) -> usize {
    match ty {
        "char" | "uchar" | "int8" | "uint8" => 1,
        "short" | "ushort" | "int16" | "uint16" => 2,
        "int" | "uint" | "int32" | "uint32" | "float" | "float32" => 4,
        "double" | "float64" => 8,
        _ => 4,
    }
}

fn read_ply_scalar(b: &[u8], off: &mut usize, ty: &str) -> Result<f64> {
    let n = ply_size(ty);
    if *off + n > b.len() { bail!("PLY body ended early"); }
    let s = &b[*off..*off + n];
    *off += n;
    Ok(match ty {
        "char" | "int8" => s[0] as i8 as f64,
        "uchar" | "uint8" => s[0] as f64,
        "short" | "int16" => i16::from_le_bytes([s[0], s[1]]) as f64,
        "ushort" | "uint16" => u16::from_le_bytes([s[0], s[1]]) as f64,
        "int" | "int32" => i32::from_le_bytes([s[0], s[1], s[2], s[3]]) as f64,
        "uint" | "uint32" => u32::from_le_bytes([s[0], s[1], s[2], s[3]]) as f64,
        "double" | "float64" => f64::from_le_bytes([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]]),
        _ => f32::from_le_bytes([s[0], s[1], s[2], s[3]]) as f64,
    })
}

// ---------- encode ----------

pub fn encode_mesh_format(mesh: &Mesh, format: MeshFormat) -> Result<Vec<u8>> {
    match format {
        MeshFormat::Glb => Ok(encode_glb(mesh)),
        MeshFormat::Obj => Ok(encode_obj(mesh)),
        MeshFormat::Ply => Ok(encode_ply(mesh)),
        MeshFormat::Stl => Ok(encode_stl(mesh)),
    }
}

fn normals_of(mesh: &Mesh) -> Vec<[f32; 3]> {
    if let Some(n) = &mesh.nor { return n.clone(); }
    let mut nor = vec![[0.0f32; 3]; mesh.pos.len()];
    for t in &mesh.tri {
        let (a, b, c) = (mesh.pos[t[0] as usize], mesh.pos[t[1] as usize], mesh.pos[t[2] as usize]);
        let f = cross(sub(b, a), sub(c, a));
        for &i in t { for q in 0..3 { nor[i as usize][q] += f[q]; } }
    }
    for n in nor.iter_mut() { *n = normalize_or(*n, [0.0, 0.0, 1.0]); }
    nor
}

fn encode_glb(mesh: &Mesh) -> Vec<u8> {
    let nor = normals_of(mesh);
    let col: Option<Vec<[f32; 3]>> = mesh.col.as_ref().map(|c| c.iter().map(|x| [x[0], x[1], x[2]]).collect());
    crate::triposr::write_glb(&mesh.pos, &nor, col.as_deref(), mesh.uv.as_deref(), &mesh.tri)
}

fn encode_obj(mesh: &Mesh) -> Vec<u8> {
    let mut s = String::from("# pyre mesh\n");
    for p in &mesh.pos { s.push_str(&format!("v {} {} {}\n", p[0], p[1], p[2])); }
    if let Some(n) = &mesh.nor { for v in n { s.push_str(&format!("vn {} {} {}\n", v[0], v[1], v[2])); } }
    if let Some(t) = &mesh.uv { for v in t { s.push_str(&format!("vt {} {}\n", v[0], v[1])); } }
    let has_n = mesh.nor.is_some();
    let has_t = mesh.uv.is_some();
    for f in &mesh.tri {
        let idx = |i: u32| i + 1;
        match (has_t, has_n) {
            (true, true) => s.push_str(&format!("f {}/{}/{} {}/{}/{} {}/{}/{}\n", idx(f[0]), idx(f[0]), idx(f[0]), idx(f[1]), idx(f[1]), idx(f[1]), idx(f[2]), idx(f[2]), idx(f[2]))),
            (true, false) => s.push_str(&format!("f {}/{} {}/{} {}/{}\n", idx(f[0]), idx(f[0]), idx(f[1]), idx(f[1]), idx(f[2]), idx(f[2]))),
            (false, true) => s.push_str(&format!("f {}//{} {}//{} {}//{}\n", idx(f[0]), idx(f[0]), idx(f[1]), idx(f[1]), idx(f[2]), idx(f[2]))),
            (false, false) => s.push_str(&format!("f {} {} {}\n", idx(f[0]), idx(f[1]), idx(f[2]))),
        }
    }
    s.into_bytes()
}

fn encode_stl(mesh: &Mesh) -> Vec<u8> {
    let mut out = vec![0u8; 80];
    out[0..12].copy_from_slice(b"pyre binary ");
    out.extend((mesh.tri.len() as u32).to_le_bytes());
    for t in &mesh.tri {
        let (a, b, c) = (mesh.pos[t[0] as usize], mesh.pos[t[1] as usize], mesh.pos[t[2] as usize]);
        let n = normalize_or(cross(sub(b, a), sub(c, a)), [0.0, 0.0, 1.0]);
        for q in 0..3 { out.extend(n[q].to_le_bytes()); }
        for p in [a, b, c] { for q in 0..3 { out.extend(p[q].to_le_bytes()); } }
        out.extend([0u8; 2]);
    }
    out
}

fn encode_ply(mesh: &Mesh) -> Vec<u8> {
    let mut s = String::from("ply\nformat ascii 1.0\ncomment pyre mesh\n");
    s.push_str(&format!("element vertex {}\n", mesh.pos.len()));
    s.push_str("property float x\nproperty float y\nproperty float z\n");
    if mesh.nor.is_some() {
        s.push_str("property float nx\nproperty float ny\nproperty float nz\n");
    }
    if mesh.col.is_some() {
        s.push_str("property uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar alpha\n");
    }
    if mesh.uv.is_some() {
        s.push_str("property float u\nproperty float v\n");
    }
    s.push_str(&format!("element face {}\nproperty list uchar int vertex_indices\nend_header\n", mesh.tri.len()));
    for i in 0..mesh.pos.len() {
        let p = mesh.pos[i];
        s.push_str(&format!("{} {} {}", p[0], p[1], p[2]));
        if let Some(n) = &mesh.nor { let q = n[i]; s.push_str(&format!(" {} {} {}", q[0], q[1], q[2])); }
        if let Some(c) = &mesh.col {
            let q = c[i];
            s.push_str(&format!(" {} {} {} {}", (q[0] * 255.0).round() as u8, (q[1] * 255.0).round() as u8, (q[2] * 255.0).round() as u8, (q[3] * 255.0).round() as u8));
        }
        if let Some(t) = &mesh.uv { let q = t[i]; s.push_str(&format!(" {} {}", q[0], q[1])); }
        s.push('\n');
    }
    for t in &mesh.tri { s.push_str(&format!("3 {} {} {}\n", t[0], t[1], t[2])); }
    s.into_bytes()
}

// ---------- SDF ----------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SdfMethod {
    /// Solid-angle winding number (slow: O(triangles) per sample).
    WindingNumber,
    /// Sign from the closest triangle's outward normal.
    Pseudonormal,
}

impl SdfMethod {
    pub fn parse(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "winding" | "winding_number" | "winding-number" => SdfMethod::WindingNumber,
            _ => SdfMethod::Pseudonormal,
        }
    }
    pub fn as_str(self) -> &'static str {
        match self { SdfMethod::WindingNumber => "winding-number", SdfMethod::Pseudonormal => "pseudonormal" }
    }
}

#[derive(Clone, Debug)]
pub struct SdfResult {
    /// Negative inside. Indexed `i*res*res + j*res + k`.
    pub grid: Vec<f32>,
    pub origin: [f32; 3],
    pub spacing: f32,
    pub dimensions: [u32; 3],
    pub method: SdfMethod,
}

impl SdfResult {
    /// Marching cubes on the zero level set. `None` when the grid has no sign
    /// change (e.g. an empty or fully open surface).
    pub fn isosurface(&self) -> Option<Mesh> {
        let r = self.dimensions[0] as usize;
        // marching_cubes treats >= thr as inside; SDF is negative inside, so negate.
        let neg: Vec<f32> = self.grid.iter().map(|v| -v).collect();
        let (gv, tri) = crate::triposr::marching_cubes(&neg, r, 0.0);
        if gv.is_empty() || tri.is_empty() { return None; }
        let pos: Vec<[f32; 3]> = gv.iter().map(|v| [
            self.origin[0] + v[0] * self.spacing,
            self.origin[1] + v[1] * self.spacing,
            self.origin[2] + v[2] * self.spacing,
        ]).collect();
        let mut mesh = Mesh { pos, tri, ..Mesh::default() };
        // Make the closed surface outward-facing.
        let vol: f64 = mesh.tri.iter().map(|t| {
            let (a, b, c) = (mesh.pos[t[0] as usize], mesh.pos[t[1] as usize], mesh.pos[t[2] as usize]);
            (a[0] as f64) * ((b[1] * c[2] - b[2] * c[1]) as f64)
                - (a[1] as f64) * ((b[0] * c[2] - b[2] * c[0]) as f64)
                + (a[2] as f64) * ((b[0] * c[1] - b[1] * c[0]) as f64)
        }).sum();
        if vol < 0.0 { for t in mesh.tri.iter_mut() { t.swap(1, 2); } }
        mesh.nor = None;
        Some(mesh)
    }
}

pub fn compute_sdf(mesh: &Mesh, resolution: usize, method: SdfMethod) -> Result<SdfResult> {
    if mesh.tri.is_empty() { bail!("cannot build an SDF from an empty mesh"); }
    let res = resolution.clamp(4, MAX_SDF_RESOLUTION);
    if method == SdfMethod::WindingNumber {
        // Winding number is O(triangles) per sample; refuse a request that
        // would run for hours instead of hanging the HTTP worker.
        let work = (res * res * res).saturating_mul(mesh.tri.len());
        if work > 500_000_000 {
            bail!(
                "winding-number SDF is too slow for {res}^3 samples x {} triangles ({} sample-triangle pairs); use sign_method 'pseudonormal' or a lower resolution",
                mesh.tri.len(),
                work
            );
        }
    }
    let (mn, mx) = mesh.bbox();
    let extent = (mx[0] - mn[0]).max(mx[1] - mn[1]).max(mx[2] - mn[2]);
    if !(extent > 1e-12) { bail!("mesh has zero extent"); }
    // Pad the domain by about one cell so no sample sits exactly on the
    // surface (the bbox touches it) and the isosurface stays closed.
    let pad = extent / (res - 2) as f32;
    let spacing = (extent + 2.0 * pad) / (res - 1) as f32;
    let origin = [mn[0] - pad, mn[1] - pad, mn[2] - pad];
    let bvh = Bvh::build(mesh);
    let mut grid = vec![0.0f32; res * res * res];
    for i in 0..res {
        let x = origin[0] + i as f32 * spacing;
        for j in 0..res {
            let y = origin[1] + j as f32 * spacing;
            for k in 0..res {
                let z = origin[2] + k as f32 * spacing;
                let p = [x, y, z];
                let hit = bvh.closest(mesh, p);
                let dist = hit.d2.sqrt();
                let inside = match method {
                    SdfMethod::Pseudonormal => dot(sub(p, hit.p), hit.n) < 0.0,
                    SdfMethod::WindingNumber => winding_inside(mesh, p),
                };
                grid[i * res * res + j * res + k] = if inside { -dist } else { dist };
            }
        }
    }
    Ok(SdfResult { grid, origin, spacing, dimensions: [res as u32; 3], method })
}

/// Absolute solid-angle winding number > 0.5 means inside.
fn winding_inside(mesh: &Mesh, p: [f32; 3]) -> bool {
    let mut w = 0.0f64;
    for t in &mesh.tri {
        let a = sub(mesh.pos[t[0] as usize], p);
        let b = sub(mesh.pos[t[1] as usize], p);
        let c = sub(mesh.pos[t[2] as usize], p);
        let la = length(a) as f64;
        let lb = length(b) as f64;
        let lc = length(c) as f64;
        let num = dot(a, cross(b, c)) as f64;
        let den = la * lb * lc + dot(a, b) as f64 * lc + dot(b, c) as f64 * la + dot(c, a) as f64 * lb;
        w += 2.0 * num.atan2(den);
    }
    (w / (4.0 * std::f64::consts::PI)).abs() > 0.5
}

struct BvhNode {
    min: [f32; 3],
    max: [f32; 3],
    left: Option<usize>,
    right: Option<usize>,
    tris: Vec<usize>,
}

struct Bvh {
    nodes: Vec<BvhNode>,
    root: usize,
}

#[derive(Clone, Copy)]
struct Hit {
    d2: f32,
    p: [f32; 3],
    n: [f32; 3],
    tri: usize,
}

impl Bvh {
    fn build(mesh: &Mesh) -> Self {
        let idx: Vec<usize> = (0..mesh.tri.len()).collect();
        let mut nodes = Vec::new();
        let root = Self::build_node(mesh, idx, &mut nodes);
        Bvh { nodes, root }
    }

    fn build_node(mesh: &Mesh, mut idx: Vec<usize>, nodes: &mut Vec<BvhNode>) -> usize {
        let mut min = [f32::MAX; 3];
        let mut max = [f32::MIN; 3];
        for &t in &idx {
            for &v in &mesh.tri[t] {
                let p = mesh.pos[v as usize];
                for q in 0..3 { min[q] = min[q].min(p[q]); max[q] = max[q].max(p[q]); }
            }
        }
        let slot = nodes.len();
        nodes.push(BvhNode { min, max, left: None, right: None, tris: Vec::new() });
        if idx.len() <= 4 {
            nodes[slot].tris = idx;
            return slot;
        }
        let axis = (0..3).max_by(|&a, &b| (max[a] - min[a]).partial_cmp(&(max[b] - min[b])).unwrap_or(Ordering::Equal)).unwrap_or(0);
        idx.sort_by(|&a, &b| centroid(mesh, a)[axis].partial_cmp(&centroid(mesh, b)[axis]).unwrap_or(Ordering::Equal));
        let mid = idx.len() / 2;
        let right_idx = idx.split_off(mid);
        let left = Self::build_node(mesh, idx, nodes);
        let right = Self::build_node(mesh, right_idx, nodes);
        nodes[slot].left = Some(left);
        nodes[slot].right = Some(right);
        slot
    }

    fn closest(&self, mesh: &Mesh, p: [f32; 3]) -> Hit {
        let mut best = Hit { d2: f32::INFINITY, p, n: [0.0, 0.0, 1.0], tri: 0 };
        self.visit(mesh, self.root, p, &mut best);
        if !best.d2.is_finite() {
            best.d2 = 0.0;
        }
        best
    }

    fn visit(&self, mesh: &Mesh, node: usize, p: [f32; 3], best: &mut Hit) {
        let n = &self.nodes[node];
        if box_d2(n.min, n.max, p) >= best.d2 { return; }
        if !n.tris.is_empty() {
            for &ti in &n.tris {
                let t = mesh.tri[ti];
                let (a, b, c) = (mesh.pos[t[0] as usize], mesh.pos[t[1] as usize], mesh.pos[t[2] as usize]);
                if let Some((q, d2)) = closest_on_triangle(p, a, b, c) {
                    if d2 < best.d2 {
                        let nrm = normalize_or(cross(sub(b, a), sub(c, a)), [0.0, 0.0, 1.0]);
                        *best = Hit { d2, p: q, n: nrm, tri: ti };
                    }
                }
            }
            return;
        }
        if let Some(l) = n.left { self.visit(mesh, l, p, best); }
        if let Some(r) = n.right { self.visit(mesh, r, p, best); }
    }
}

fn centroid(mesh: &Mesh, t: usize) -> [f32; 3] {
    let tri = mesh.tri[t];
    let a = mesh.pos[tri[0] as usize];
    let b = mesh.pos[tri[1] as usize];
    let c = mesh.pos[tri[2] as usize];
    [(a[0] + b[0] + c[0]) / 3.0, (a[1] + b[1] + c[1]) / 3.0, (a[2] + b[2] + c[2]) / 3.0]
}

fn box_d2(min: [f32; 3], max: [f32; 3], p: [f32; 3]) -> f32 {
    let mut d = 0.0;
    for q in 0..3 {
        if p[q] < min[q] { d += (min[q] - p[q]).powi(2); }
        else if p[q] > max[q] { d += (p[q] - max[q]).powi(2); }
    }
    d
}

/// Barycentric weights of `p` with respect to triangle `a, b, c`. `p` is
/// expected to be on the triangle; degenerate triangles fall back to vertex a.
fn barycentric(p: [f32; 3], a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [f32; 3] {
    let v0 = sub(b, a);
    let v1 = sub(c, a);
    let v2 = sub(p, a);
    let d00 = dot(v0, v0);
    let d01 = dot(v0, v1);
    let d11 = dot(v1, v1);
    let d20 = dot(v2, v0);
    let d21 = dot(v2, v1);
    let den = d00 * d11 - d01 * d01;
    if den.abs() < 1e-20 {
        return [1.0, 0.0, 0.0];
    }
    let v = (d11 * d20 - d01 * d21) / den;
    let w = (d00 * d21 - d01 * d20) / den;
    [1.0 - v - w, v, w]
}

/// Closest point on triangle (Ericson, Real-Time Collision Detection).
fn closest_on_triangle(p: [f32; 3], a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> Option<([f32; 3], f32)> {
    let ab = sub(b, a);
    let ac = sub(c, a);
    let ap = sub(p, a);
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    let (q, dist);
    if d1 <= 0.0 && d2 <= 0.0 {
        q = a; dist = length(ap);
    } else {
        let bp = sub(p, b);
        let d3 = dot(ab, bp);
        let d4 = dot(ac, bp);
        if d3 >= 0.0 && d4 <= d3 {
            q = b; dist = length(bp);
        } else {
            let vc = d1 * d4 - d3 * d2;
            if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
                let v = d1 / (d1 - d3);
                q = [a[0] + ab[0] * v, a[1] + ab[1] * v, a[2] + ab[2] * v];
                dist = length(sub(p, q));
            } else {
                let cp = sub(p, c);
                let d5 = dot(ab, cp);
                let d6 = dot(ac, cp);
                if d6 >= 0.0 && d5 <= d6 {
                    q = c; dist = length(cp);
                } else {
                    let vb = d5 * d2 - d1 * d6;
                    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
                        let w = d2 / (d2 - d6);
                        q = [a[0] + ac[0] * w, a[1] + ac[1] * w, a[2] + ac[2] * w];
                        dist = length(sub(p, q));
                    } else {
                        let va = d3 * d6 - d5 * d4;
                        if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
                            let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
                            q = [b[0] + (c[0] - b[0]) * w, b[1] + (c[1] - b[1]) * w, b[2] + (c[2] - b[2]) * w];
                            dist = length(sub(p, q));
                        } else {
                            let denom = 1.0 / (va + vb + vc);
                            let v = vb * denom;
                            let w = vc * denom;
                            q = [a[0] + ab[0] * v + ac[0] * w, a[1] + ab[1] * v + ac[1] * w, a[2] + ab[2] * v + ac[2] * w];
                            dist = length(sub(p, q));
                        }
                    }
                }
            }
        }
    }
    if !dist.is_finite() { None } else { Some((q, dist * dist)) }
}

// ---------- HTTP request ----------

/// Decode one `POST /api/cabin/mesh` body and run the requested op.
///
/// Kept free of axum types so the whole request path is unit-testable. The
/// cabin handler only wraps this in `spawn_blocking` + `json_or_err`.
pub fn run_request(b: &Value) -> Result<Value> {
    use base64::Engine;
    const MAX_INPUT: usize = 64 << 20;
    let enc = base64::engine::general_purpose::STANDARD;
    let input = b.get("mesh").and_then(|v| {
        v.as_str().map(str::to_string)
            .or_else(|| v.get("glb").and_then(|g| g.as_str()).map(str::to_string))
    }).or_else(|| b.get("glb").and_then(|v| v.as_str()).map(str::to_string))
        .ok_or_else(|| anyhow::anyhow!("mesh missing: send base64 GLB/OBJ/PLY/STL in 'mesh' or {{'glb'}}"))?;
    let s = input.split_once("base64,").map(|x| x.1).unwrap_or(&input);
    let bytes = enc.decode(s.trim()).map_err(|e| anyhow::anyhow!("invalid base64: {e}"))?;
    if bytes.len() > MAX_INPUT { bail!("mesh input is {} bytes; the limit is {} bytes", bytes.len(), MAX_INPUT); }
    let op = b["op"].as_str().unwrap_or("stats").to_string();
    let p = b.get("params").cloned().unwrap_or_else(|| json!({}));
    let num = |k: &str, def: f64| p.get(k).and_then(|v| v.as_f64()).unwrap_or(def);
    let uint = |k: &str, def: usize| p.get(k).and_then(|v| v.as_u64()).map(|v| v as usize).unwrap_or(def);
    let mut m = decode_mesh_data(&bytes)?;
    let mut extra = json!({});
    match op.as_str() {
        "stats" => {}
        "normals" => {
            let flipped = m.compute_normals();
            extra["flipped_faces"] = json!(flipped);
        }
        "smooth" => {
            let iterations = uint("iterations", 2).clamp(1, 50);
            let lambda = (num("lambda", 0.5) as f32).clamp(0.0, 1.0);
            let mu = (num("mu", -0.53) as f32).clamp(-1.0, 0.0);
            m.smooth(iterations, lambda, mu);
            extra["smooth"] = json!({
                "iterations": iterations,
                "lambda": lambda,
                "mu": mu,
                "method": "Taubin (lambda/mu) over triangle adjacency"
            });
        }
        "recolour" => {
            let source = p.get("source").and_then(|v| v.as_str()).unwrap_or("mesh");
            match source {
                "image" => {
                    let s = p.get("image").and_then(|v| v.as_str())
                        .ok_or_else(|| anyhow::anyhow!("recolour from image needs params.image"))?;
                    let s = s.split_once("base64,").map(|x| x.1).unwrap_or(s);
                    let bytes = enc.decode(s.trim()).map_err(|e| anyhow::anyhow!("invalid image base64: {e}"))?;
                    if bytes.len() > MAX_INPUT {
                        bail!("recolour image is {} bytes; the limit is {} bytes", bytes.len(), MAX_INPUT);
                    }
                    let img = image::load_from_memory(&bytes).context("decoding recolour image")?;
                    if img.width() as u64 * img.height() as u64 > 4096 * 4096 {
                        bail!("recolour image is {}x{}; max 16 MP", img.width(), img.height());
                    }
                    let img = img.to_rgba8();
                    let cam = p.get("camera").filter(|c| c.is_object());
                    let changed = m.recolour_from_image(&img, cam)?;
                    extra["recolour"] = json!({
                        "source": "image",
                        "camera": if cam.is_some() { "pinhole (origin/target/up/fov_deg/aspect)" } else { "front orthographic along -Z over the mesh bbox" },
                        "vertices_changed": changed,
                        "image": [img.width(), img.height()]
                    });
                }
                "mesh" => {
                    let s = p.get("mesh").and_then(|v| v.as_str())
                        .ok_or_else(|| anyhow::anyhow!("recolour from mesh needs params.mesh"))?;
                    let s = s.split_once("base64,").map(|x| x.1).unwrap_or(s);
                    let bytes = enc.decode(s.trim()).map_err(|e| anyhow::anyhow!("invalid source mesh base64: {e}"))?;
                    if bytes.len() > MAX_INPUT {
                        bail!("source mesh is {} bytes; the limit is {} bytes", bytes.len(), MAX_INPUT);
                    }
                    let src = decode_mesh_data(&bytes)?;
                    let changed = m.recolour_from_mesh(&src)?;
                    extra["recolour"] = json!({
                        "source": "mesh",
                        "method": "nearest point on the source surface, barycentric colour interpolation",
                        "vertices_changed": changed
                    });
                }
                other => bail!("recolour source must be 'mesh' or 'image', not '{other}'"),
            }
        }
        "center" => m.center(),
        "scale" => m.center_and_scale(num("scale", 1.0) as f32),
        "decimate" => {
            let before = m.tri.len();
            let target = match p.get("target_vertices").and_then(|v| v.as_u64()) {
                // Euler: a closed triangle mesh has about twice as many
                // faces as vertices.
                Some(v) => (v as usize).saturating_mul(2).max(4),
                None => uint("target_faces", before / 2).max(4),
            };
            let after = m.decimate(target);
            extra["decimate"] = json!({"target_faces": target, "before": before, "after": after, "uv_seams_preserved": m.uv.is_none()});
        }
        "remesh" => {
            let mut edge = num("target_edge", 0.0) as f32;
            if !(edge > 0.0) {
                let (mn, mx) = m.bbox();
                edge = ((mx[0] - mn[0]).max(mx[1] - mn[1]).max(mx[2] - mn[2]) / 32.0).max(1e-4);
            }
            let n = m.remesh(edge, uint("resolution", 64))?;
            extra["remesh"] = json!({"target_edge": edge, "triangles": n, "method": "voxel-sdf + marching cubes"});
        }
        "repair" => {
            let report = m.repair(num("weld_tolerance", 1e-5) as f32, uint("max_fill_loop", 64));
            extra["repair"] = serde_json::to_value(report)?;
        }
        "uv" => {
            let mode = p.get("projection").and_then(|v| v.as_str()).unwrap_or("box").to_ascii_lowercase();
            match mode.as_str() {
                "box" => {
                    m.add_box_uvs();
                    extra["uv"] = json!({"projection": "box", "vertices": m.pos.len(), "seams_split": false});
                }
                "lscm" => {
                    let r = m.add_lscm_uvs(uint("iterations", 200))?;
                    let mut v = serde_json::to_value(r)?;
                    v["projection"] = json!("lscm");
                    v["vertices"] = json!(m.pos.len());
                    extra["uv"] = v;
                }
                other => bail!("unknown UV projection '{other}' (expected box or lscm)"),
            }
        }
        "convert" => {
            let name = p.get("format").and_then(|v| v.as_str()).unwrap_or("glb");
            let fmt = MeshFormat::parse(name).ok_or_else(|| anyhow::anyhow!("unknown output format '{name}'"))?;
            let data = encode_mesh_format(&m, fmt)?;
            extra["data"] = json!(enc.encode(&data));
            extra["format"] = json!(fmt.as_str());
            extra["bytes"] = json!(data.len());
            extra["filename"] = json!(format!("mesh.{}", fmt.extension()));
        }
        "sdf" => {
            let res = uint("resolution", 64).clamp(4, MAX_SDF_RESOLUTION);
            let method = SdfMethod::parse(p.get("sign_method").and_then(|v| v.as_str()).unwrap_or("pseudonormal"));
            let sdf = compute_sdf(&m, res, method)?;
            let raw: Vec<u8> = sdf.grid.iter().flat_map(|v| v.to_le_bytes()).collect();
            let iso = sdf.isosurface();
            let mut s = json!({
                "format": "raw-f32-le",
                "dimensions": sdf.dimensions,
                "origin": sdf.origin,
                "spacing": sdf.spacing,
                "method": sdf.method.as_str(),
                "sign_convention": "negative inside",
                "grid": enc.encode(&raw),
                "grid_bytes": raw.len(),
                "header": {
                    "dtype": "<f4",
                    "shape": [res, res, res],
                    "order": "C",
                    "origin": sdf.origin,
                    "spacing": sdf.spacing
                }
            });
            if let Some(iso) = iso {
                let glb = encode_mesh_format(&iso, MeshFormat::Glb)?;
                s["isosurface"] = json!({"glb": enc.encode(&glb), "bytes": glb.len(), "triangles": iso.tri.len()});
            }
            extra["sdf"] = s;
        }
        other => bail!("unknown mesh op '{other}'"),
    }
    let stats = m.stats();
    let glb = encode_mesh_format(&m, MeshFormat::Glb)?;
    let mut out = json!({
        "op": op,
        "stats": stats,
        "vertices": m.pos.len(),
        "triangles": m.tri.len(),
        "glb": enc.encode(&glb),
        "glb_bytes": glb.len(),
        "bytes": glb.len()
    });
    if let (Some(dst), Value::Object(src)) = (out.as_object_mut(), extra) {
        for (k, v) in src { dst.insert(k, v); }
    }
    Ok(out)
}

// ---------- tests ----------

#[cfg(test)]
mod tests {
    use super::*;

    fn grid_plane(n: usize) -> Mesh {
        let mut pos = Vec::new();
        for j in 0..=n { for i in 0..=n { pos.push([i as f32 / n as f32, j as f32 / n as f32, 0.0]); } }
        let mut tri = Vec::new();
        let w = n + 1;
        for j in 0..n { for i in 0..n {
            let a = (j * w + i) as u32;
            let b = a + 1;
            let c = a + w as u32;
            let d = c + 1;
            tri.push([a, b, d]);
            tri.push([a, d, c]);
        }}
        Mesh { pos, tri, ..Mesh::default() }
    }

    #[test]
    fn cube_and_tetrahedron_stats() {
        let c = Mesh::unit_cube().stats();
        assert_eq!(c.vertices, 8);
        assert_eq!(c.triangles, 12);
        assert!(c.watertight, "cube should be closed: {:?}", c.edge_counts);
        assert!(c.area > 0.0);
        let t = Mesh::tetrahedron().stats();
        assert_eq!(t.vertices, 4);
        assert_eq!(t.triangles, 4);
        assert!(t.watertight, "tetrahedron should be closed: {:?}", t.edge_counts);
    }

    #[test]
    fn format_round_trips() {
        let mesh = Mesh::unit_cube();
        for fmt in [MeshFormat::Glb, MeshFormat::Obj, MeshFormat::Ply, MeshFormat::Stl] {
            let bytes = encode_mesh_format(&mesh, fmt).unwrap();
            let back = decode_mesh_data(&bytes).unwrap();
            assert_eq!(back.tri.len(), 12, "{:?} triangle count", fmt);
            assert!(back.pos.len() >= 8, "{:?} vertex count", fmt);
            let (mn, mx) = back.bbox();
            for q in 0..3 {
                assert!((mn[q] + 1.0).abs() < 1e-4, "{:?} min {q}: {}", fmt, mn[q]);
                assert!((mx[q] - 1.0).abs() < 1e-4, "{:?} max {q}: {}", fmt, mx[q]);
            }
        }
    }

    #[test]
    fn normals_point_outward() {
        let mut mesh = Mesh::unit_cube();
        mesh.compute_normals();
        let nor = mesh.nor.as_ref().unwrap();
        for (p, n) in mesh.pos.iter().zip(nor) {
            assert!(dot(*p, *n) > 0.0, "normal {n:?} at {p:?} should point outward");
            assert!((length(*n) - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn center_and_scale_translated_box() {
        let mut mesh = Mesh::unit_cube();
        for p in mesh.pos.iter_mut() { p[0] += 10.0; p[1] -= 4.0; }
        mesh.center_and_scale(2.0);
        let (mn, mx) = mesh.bbox();
        for q in 0..3 {
            assert!(((mn[q] + mx[q]) * 0.5).abs() < 1e-5);
            assert!(((mx[q] - mn[q]) - 2.0).abs() < 1e-5 || q == 2);
        }
    }

    #[test]
    fn decimate_reduces_plane_and_keeps_corners() {
        let mesh = grid_plane(20);
        let before = mesh.tri.len();
        let mut m = mesh.clone();
        let after = m.decimate(before / 2);
        assert!(after < before, "decimate should reduce faces: {before} -> {after}");
        let (mn, mx) = m.bbox();
        assert!((mn[0] - 0.0).abs() < 1e-5 && (mx[0] - 1.0).abs() < 1e-5);
        assert!((mn[1] - 0.0).abs() < 1e-5 && (mx[1] - 1.0).abs() < 1e-5);
        // The four corners are boundary vertices and stay.
        let corners = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        for c in corners {
            assert!(m.pos.iter().any(|p| (p[0] - c[0]).abs() < 1e-5 && (p[1] - c[1]).abs() < 1e-5), "corner {c:?} kept");
        }
    }

    #[test]
    fn repair_welds_duplicate_vertex() {
        let mut mesh = Mesh::unit_cube();
        let v = mesh.pos[0];
        let dup = mesh.pos.len() as u32;
        mesh.pos.push(v);
        // Point one triangle at the duplicate.
        mesh.tri[0][0] = dup;
        let report = mesh.repair(1e-5, 64);
        assert!(report.welded_vertices >= 1, "report: {report:?}");
        assert_eq!(mesh.pos.len(), 8);
    }

    #[test]
    fn non_manifold_reports_without_panicking() {
        // Three triangles sharing edge 0-1.
        let mesh = Mesh {
            pos: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
            tri: vec![[0, 1, 2], [1, 0, 3], [0, 1, 4]],
            ..Mesh::default()
        };
        let mut m = mesh;
        let r = m.repair(1e-5, 64);
        assert!(r.non_manifold_edges >= 1, "report: {r:?}");
    }

    #[test]
    fn lscm_unfold_is_finite_and_spans_both_axes() {
        let mut m = Mesh::unit_cube();
        let r = m.add_lscm_uvs(60).unwrap();
        assert_eq!(r.charts, 1, "one connected cube: {r:?}");
        assert!(r.seams_split == false);
        let uv = m.uv.as_ref().expect("LSCM UVs");
        assert_eq!(uv.len(), m.pos.len());
        assert!(uv.iter().all(|p| p.iter().all(|x| x.is_finite())), "non-finite UVs");
        let us = uv.iter().map(|p| p[0]).fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), x| (a.min(x), b.max(x)));
        let vs = uv.iter().map(|p| p[1]).fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), x| (a.min(x), b.max(x)));
        assert!(us.1 - us.0 > 1e-3, "u span {:?}", us);
        assert!(vs.1 - vs.0 > 1e-3, "v span {:?}", vs);
        assert!(uv[..3].iter().any(|p| p[0].abs() < 1e-5 && p[1].abs() < 1e-5), "no pinned anchor");
    }

    #[test]
    fn lscm_request_selects_projection_and_rejects_unknown() {
        let base = |mode: &str| -> serde_json::Value {
            let obj = String::from_utf8(crate::mesh::encode_mesh_format(&Mesh::unit_cube(), MeshFormat::Obj).unwrap()).unwrap();
            use base64::Engine;
            let b64 = base64::engine::general_purpose::STANDARD.encode(obj.as_bytes());
            serde_json::json!({"mesh": b64, "op": "uv", "params": {"projection": mode, "iterations": 10}})
        };
        let r = crate::mesh::run_request(&base("lscm")).unwrap();
        assert_eq!(r["uv"]["projection"], "lscm");
        assert!(r["uv"]["charts"].as_u64().unwrap() >= 1);
        assert!(crate::mesh::run_request(&base("bad")).is_err());
    }

    #[test]
    fn sdf_cube_signs_and_isosurface() {
        let mesh = Mesh::unit_cube();
        let sdf = compute_sdf(&mesh, 24, SdfMethod::Pseudonormal).unwrap();
        let r = 24usize;
        let at = |i: usize, j: usize, k: usize| sdf.grid[i * r * r + j * r + k];
        // Centre of the cube: negative.
        let mid = r / 2;
        assert!(at(mid, mid, mid) < 0.0, "centre {}", at(mid, mid, mid));
        // The padded grid corners are outside the cube: positive.
        assert!(at(0, 0, 0) > 0.0, "corner {}", at(0, 0, 0));
        // The grid straddles the surface, so some sample sits within half a cell.
        let closest = sdf.grid.iter().fold(f32::INFINITY, |a, &v| a.min(v.abs()));
        assert!(closest < sdf.spacing * 0.75, "closest sample {closest}, spacing {}", sdf.spacing);
        let iso = sdf.isosurface().expect("marching cubes output");
        assert!(iso.tri.len() > 0);
        let (mn, mx) = iso.bbox();
        for q in 0..3 {
            assert!((mn[q] + 1.0).abs() < 0.2, "iso min {q}: {}", mn[q]);
            assert!((mx[q] - 1.0).abs() < 0.2, "iso max {q}: {}", mx[q]);
        }
        // The level set is a closed box: normals cover all six sides.
        let mut sides = [false; 6];
        for t in &iso.tri {
            let (a, b, c) = (iso.pos[t[0] as usize], iso.pos[t[1] as usize], iso.pos[t[2] as usize]);
            let n = normalize_or(cross(sub(b, a), sub(c, a)), [0.0, 0.0, 1.0]);
            for q in 0..3 {
                if n[q] > 0.9 { sides[q * 2] = true; }
                if n[q] < -0.9 { sides[q * 2 + 1] = true; }
            }
        }
        assert!(sides.iter().all(|&s| s), "all six cube sides present: {sides:?}");
    }

    #[test]
    fn repair_fills_a_missing_face() {
        let mut mesh = Mesh::unit_cube();
        mesh.tri.pop();
        assert_eq!(mesh.stats().edge_counts.boundary, 3);
        let report = mesh.repair(1e-6, 64);
        assert_eq!(report.holes_filled, 1, "report: {report:?}");
        assert!(mesh.stats().watertight, "report: {report:?}");
    }

    #[test]
    fn remesh_changes_topology() {
        let mut mesh = Mesh::unit_cube();
        let n = mesh.remesh(0.5, 48).unwrap();
        assert!(n > 0);
        let s = mesh.stats();
        assert!(s.vertices > 8, "voxel remesh adds vertices: {:?}", s);
    }

    #[test]
    fn obj_late_normals_are_dropped() {
        // Normals declared after the faces cannot be matched to vertices; the
        // decoder must drop them instead of writing a mismatched GLB.
        let obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\nvn 0 0 1\n";
        let m = decode_mesh_data(obj.as_bytes()).unwrap();
        assert_eq!(m.tri.len(), 1);
        assert!(m.nor.is_none());
        let glb = encode_mesh_format(&m, MeshFormat::Glb).unwrap();
        assert_eq!(decode_mesh_data(&glb).unwrap().tri.len(), 1);
    }

    #[test]
    fn uv_and_convert() {
        let mut mesh = Mesh::unit_cube();
        mesh.add_box_uvs();
        let uv = mesh.uv.as_ref().unwrap();
        assert_eq!(uv.len(), mesh.pos.len());
        assert!(uv.iter().all(|t| t[0].is_finite() && t[1].is_finite()));
        let obj = encode_mesh_format(&mesh, MeshFormat::Obj).unwrap();
        let back = decode_mesh_data(&obj).unwrap();
        assert!(back.uv.is_some());
    }

    fn b64(bytes: &[u8]) -> String {
        use base64::Engine;
        base64::engine::general_purpose::STANDARD.encode(bytes)
    }

    fn unb64(s: &str) -> Vec<u8> {
        use base64::Engine;
        base64::engine::general_purpose::STANDARD.decode(s).unwrap()
    }

    #[test]
    fn request_stats_and_convert_round_trip() {
        let cube = encode_mesh_format(&Mesh::unit_cube(), MeshFormat::Glb).unwrap();
        // Data-URL prefix and bare base64 are both accepted.
        let out = run_request(&json!({
            "mesh": format!("data:model/gltf-binary;base64,{}", b64(&cube)),
            "op": "stats",
        })).unwrap();
        assert_eq!(out["op"], "stats");
        assert_eq!(out["stats"]["vertices"], 8);
        assert_eq!(out["stats"]["triangles"], 12);
        assert_eq!(out["stats"]["watertight"], true);
        let glb = unb64(out["glb"].as_str().unwrap());
        assert_eq!(decode_mesh_data(&glb).unwrap().tri.len(), 12);

        // to3d-shaped `{ glb }` input and an OBJ conversion.
        let out = run_request(&json!({
            "glb": b64(&cube),
            "op": "convert",
            "params": { "format": "obj" },
        })).unwrap();
        assert_eq!(out["format"], "obj");
        assert_eq!(out["filename"], "mesh.obj");
        let obj = unb64(out["data"].as_str().unwrap());
        assert_eq!(obj.len() as u64, out["bytes"].as_u64().unwrap());
        assert_eq!(decode_mesh_data(&obj).unwrap().tri.len(), 12);
    }

    #[test]
    fn request_ops_produce_meshes() {
        let cube = encode_mesh_format(&Mesh::unit_cube(), MeshFormat::Glb).unwrap();
        for (op, params) in [
            ("normals", json!({})),
            ("center", json!({})),
            ("scale", json!({"scale": 2.0})),
            ("decimate", json!({"target_faces": 8})),
            ("remesh", json!({"target_edge": 0.5, "resolution": 32})),
            ("repair", json!({})),
            ("uv", json!({})),
        ] {
            let out = run_request(&json!({ "mesh": b64(&cube), "op": op, "params": params }))
                .unwrap_or_else(|e| panic!("op {op}: {e}"));
            let glb = unb64(out["glb"].as_str().unwrap());
            let back = decode_mesh_data(&glb).unwrap();
            assert_eq!(back.tri.len() as u64, out["triangles"].as_u64().unwrap(), "op {op}");
            assert!(back.tri.len() > 0, "op {op}");
        }
        let scaled = run_request(&json!({ "mesh": b64(&cube), "op": "scale", "params": {"scale": 2.0} })).unwrap();
        let (mn, mx) = decode_mesh_data(&unb64(scaled["glb"].as_str().unwrap())).unwrap().bbox();
        assert!(((mx[0] - mn[0]) - 2.0).abs() < 1e-4);
    }

    #[test]
    fn request_sdf_returns_grid_header_and_isosurface() {
        let cube = encode_mesh_format(&Mesh::unit_cube(), MeshFormat::Glb).unwrap();
        let out = run_request(&json!({
            "mesh": b64(&cube),
            "op": "sdf",
            "params": { "resolution": 16 },
        })).unwrap();
        let sdf = &out["sdf"];
        assert_eq!(sdf["dimensions"], json!([16, 16, 16]));
        assert_eq!(sdf["header"]["dtype"], "<f4");
        assert_eq!(sdf["sign_convention"], "negative inside");
        let grid = unb64(sdf["grid"].as_str().unwrap());
        assert_eq!(grid.len(), 16 * 16 * 16 * 4);
        assert_eq!(grid.len() as u64, sdf["grid_bytes"].as_u64().unwrap());
        let iso = unb64(sdf["isosurface"]["glb"].as_str().unwrap());
        assert!(decode_mesh_data(&iso).unwrap().tri.len() > 0);
    }

    #[test]
    fn request_rejects_bad_inputs() {
        let cube = encode_mesh_format(&Mesh::unit_cube(), MeshFormat::Glb).unwrap();
        assert!(run_request(&json!({ "op": "stats" })).is_err(), "missing mesh");
        assert!(run_request(&json!({ "mesh": "not base64!!", "op": "stats" })).is_err(), "bad base64");
        assert!(run_request(&json!({ "mesh": b64(b"hello"), "op": "stats" })).is_err(), "unknown format");
        assert!(run_request(&json!({ "mesh": b64(&cube), "op": "purple" })).is_err(), "unknown op");
        assert!(run_request(&json!({ "mesh": b64(&cube), "op": "convert", "params": {"format": "fbx"} })).is_err(), "unknown format");
    }

    #[test]
    fn smooth_damps_a_spike_and_drops_stale_normals() {
        let mut m = grid_plane(4);
        let w = 5;
        let center = 2 * w + 2;
        m.pos[center][2] = 1.0;
        m.nor = Some(vec![[0.0, 0.0, 1.0]; m.pos.len()]);
        m.smooth(4, 0.5, -0.53);
        assert!(m.pos[center][2] < 0.5, "spike stayed at {}", m.pos[center][2]);
        assert!(m.nor.is_none());
    }

    #[test]
    fn recolour_projects_colours_from_a_source_mesh() {
        let mut src = Mesh::unit_cube();
        src.col = Some(vec![[1.0, 0.0, 0.0, 1.0]; src.pos.len()]);
        let mut dst = Mesh::unit_cube();
        for p in &mut dst.pos {
            p[0] += 5.0;
        }
        let n = dst.recolour_from_mesh(&src).unwrap();
        assert_eq!(n, dst.pos.len());
        for c in dst.col.as_ref().unwrap() {
            assert!(c[0] > 0.99 && c[1] < 0.01 && c[2] < 0.01, "got {c:?}");
        }
    }

    #[test]
    fn recolour_samples_an_image_in_front_projection() {
        let mut img = image::RgbaImage::from_pixel(4, 4, image::Rgba([255, 0, 0, 255]));
        for y in 0..4 {
            for x in 2..4 {
                img.put_pixel(x, y, image::Rgba([0, 0, 255, 255]));
            }
        }
        let mut dst = Mesh::unit_cube();
        let n = dst.recolour_from_image(&img, None).unwrap();
        assert_eq!(n, dst.pos.len());
        for (p, c) in dst.pos.iter().zip(dst.col.as_ref().unwrap()) {
            if p[0] < 0.0 {
                assert!(c[0] > 0.99 && c[2] < 0.01, "left vertex {p:?} got {c:?}");
            } else {
                assert!(c[2] > 0.99 && c[0] < 0.01, "right vertex {p:?} got {c:?}");
            }
        }
    }

    #[test]
    fn request_smooth_and_recolour_from_mesh() {
        let grid = encode_mesh_format(&grid_plane(4), MeshFormat::Glb).unwrap();
        let smooth = run_request(&json!({
            "mesh": b64(&grid),
            "op": "smooth",
            "params": { "iterations": 3, "lambda": 0.5, "mu": -0.53 }
        })).unwrap();
        assert_eq!(smooth["smooth"]["iterations"], 3);

        let mut src = Mesh::unit_cube();
        src.col = Some(vec![[0.0, 1.0, 0.0, 1.0]; src.pos.len()]);
        let src_glb = encode_mesh_format(&src, MeshFormat::Glb).unwrap();
        let out = run_request(&json!({
            "mesh": b64(&grid),
            "op": "recolour",
            "params": { "source": "mesh", "mesh": b64(&src_glb) }
        })).unwrap();
        assert_eq!(out["recolour"]["source"], "mesh");
        let back = decode_mesh_data(&unb64(out["glb"].as_str().unwrap())).unwrap();
        let col = back.col.unwrap();
        assert!(col.iter().all(|c| c[1] > 0.99 && c[0] < 0.01), "some colours were not projected");
    }
}
