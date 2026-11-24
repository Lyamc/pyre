//! Geometric auto-rigging for the cabin: fit a small named skeleton to a mesh,
//! compute glTF-compatible skin weights, and emit a skinned GLB.
//!
//! This is deliberately not a learned rigger. The skeleton comes from the mesh
//! bounding box and anatomy-style fractions, joints are nudged toward the local
//! surface centroid, and weights use a documented inverse-distance falloff.
//! Assumptions: the subject is upright along +Y and faces +Z, which is the
//! frame [`crate::triposr::image_to_mesh`] writes. Imported meshes that are
//! lying down will get a skeleton along their Y extent, not their long axis.
//!
//! Pure Rust, no GPU, no weights.

use anyhow::{bail, Result};
use serde::Serialize;
use serde_json::{json, Value};

use crate::mesh::{decode_mesh_data, Mesh};

/// glTF allows four joint influences per vertex; more would be dropped by most
/// runtimes, so the fitter never asks for more.
pub const MAX_INFLUENCES: usize = 4;

#[derive(Clone, Debug, Serialize)]
pub struct Joint {
    pub name: String,
    pub parent: Option<usize>,
    pub position: [f32; 3],
}

#[derive(Clone, Debug)]
pub struct Rigged {
    pub template: String,
    pub joints: Vec<Joint>,
    /// Per-vertex joint indices, four slots, unused slots point at the root.
    pub joint_index: Vec<[u8; 4]>,
    /// Per-vertex weights, matching `joint_index`, normalised to sum to 1.
    pub weights: Vec<[f32; 4]>,
}

impl Rigged {
    pub fn bones(&self) -> Vec<[usize; 2]> {
        self.joints
            .iter()
            .enumerate()
            .filter_map(|(i, j)| j.parent.map(|p| [p, i]))
            .collect()
    }
}

/// Fit a skeleton and skin weights to `mesh`.
pub fn fit(mesh: &Mesh, template: &str) -> Result<Rigged> {
    if mesh.pos.is_empty() || mesh.tri.is_empty() {
        bail!("mesh has no triangles to rig");
    }
    let (mut joints, template_name) = match template.to_ascii_lowercase().as_str() {
        "humanoid" | "human" | "biped" => (skeleton_humanoid(mesh), "humanoid"),
        "character" | "simple" | "creature" => (skeleton_character(mesh), "character"),
        other => bail!("unknown rig template '{other}'; use humanoid or character"),
    };
    refine_joints(mesh, &mut joints);
    let (joint_index, weights) = skin_weights(mesh, &joints);
    Ok(Rigged { template: template_name.into(), joints, joint_index, weights })
}

/// `(name, parent, x fraction of width, y fraction of height, z fraction of depth)`
/// where x is centred, y runs from the bbox floor to the roof, and z is centred.
type Anchor = (&'static str, Option<usize>, f32, f32, f32);

fn skeleton_humanoid(m: &Mesh) -> Vec<Joint> {
    const A: &[Anchor] = &[
        ("hips", None, 0.0, 0.52, 0.0),
        ("spine", Some(0), 0.0, 0.61, 0.0),
        ("chest", Some(1), 0.0, 0.72, 0.0),
        ("neck", Some(2), 0.0, 0.81, 0.01),
        ("head", Some(3), 0.0, 0.90, 0.01),
        ("shoulder_l", Some(2), 0.16, 0.78, 0.0),
        ("upper_arm_l", Some(5), 0.33, 0.73, 0.0),
        ("forearm_l", Some(6), 0.40, 0.61, 0.01),
        ("hand_l", Some(7), 0.43, 0.49, 0.02),
        ("shoulder_r", Some(2), -0.16, 0.78, 0.0),
        ("upper_arm_r", Some(9), -0.33, 0.73, 0.0),
        ("forearm_r", Some(10), -0.40, 0.61, 0.01),
        ("hand_r", Some(11), -0.43, 0.49, 0.02),
        ("thigh_l", Some(0), 0.10, 0.49, 0.0),
        ("shin_l", Some(13), 0.11, 0.26, 0.0),
        ("foot_l", Some(14), 0.11, 0.03, 0.06),
        ("thigh_r", Some(0), -0.10, 0.49, 0.0),
        ("shin_r", Some(16), -0.11, 0.26, 0.0),
        ("foot_r", Some(17), -0.11, 0.03, 0.06),
    ];
    anchors_to_joints(m, A)
}

fn skeleton_character(m: &Mesh) -> Vec<Joint> {
    const A: &[Anchor] = &[
        ("hips", None, 0.0, 0.50, 0.0),
        ("spine", Some(0), 0.0, 0.64, 0.0),
        ("chest", Some(1), 0.0, 0.76, 0.0),
        ("head", Some(2), 0.0, 0.90, 0.01),
        ("arm_l", Some(2), 0.36, 0.67, 0.01),
        ("arm_r", Some(2), -0.36, 0.67, 0.01),
        ("leg_l", Some(0), 0.11, 0.27, 0.0),
        ("leg_r", Some(0), -0.11, 0.27, 0.0),
    ];
    anchors_to_joints(m, A)
}

fn anchors_to_joints(m: &Mesh, anchors: &[Anchor]) -> Vec<Joint> {
    let (mn, mx) = m.bbox();
    let c = [(mn[0] + mx[0]) * 0.5, (mn[1] + mx[1]) * 0.5, (mn[2] + mx[2]) * 0.5];
    let frame = fit_frame(m);
    let (w, h, d) = frame.extents;
    anchors
        .iter()
        .map(|&(name, parent, fx, fy, fz)| Joint {
            name: name.into(),
            parent,
            position: [
                c[0] + frame.right[0] * fx * w + frame.up[0] * (frame.min_up + fy * h) + frame.forward[0] * fz * d,
                c[1] + frame.right[1] * fx * w + frame.up[1] * (frame.min_up + fy * h) + frame.forward[1] * fz * d,
                c[2] + frame.right[2] * fx * w + frame.up[2] * (frame.min_up + fy * h) + frame.forward[2] * fz * d,
            ],
        })
        .collect()
}

/// Orthonormal frame used to lay the skeleton along the subject's long axis.
/// `up` is +Y for an upright character, but a mesh that is clearly lying down
/// (long principal axis along X or Z and a much shorter Y extent) is fitted
/// along its own long axis instead. `min_up` is the smallest vertex projection
/// on `up`, so `fy` fractions always start at the mesh floor.
struct Frame {
    right: [f32; 3],
    up: [f32; 3],
    forward: [f32; 3],
    extents: (f32, f32, f32),
    min_up: f32,
}

fn fit_frame(m: &Mesh) -> Frame {
    let axis = principal_axis(m);
    let (mn, mx) = m.bbox();
    let ext = [mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]];
    let dom = (0..3).max_by(|&a, &b| ext[a].total_cmp(&ext[b])).unwrap_or(1);
    // Only leave +Y when the mesh is clearly not standing: the principal axis
    // points along the dominant bbox axis and Y is much shorter. A T-pose with
    // an arm span wider than its height stays on +Y.
    let use_dom = dom != 1 && ext[1] < 0.6 * ext[dom] && axis[dom].abs() > 0.8;
    let up_idx = if use_dom { dom } else { 1 };
    let up = match up_idx {
        0 => [1.0, 0.0, 0.0],
        2 => [0.0, 0.0, 1.0],
        _ => [0.0, 1.0, 0.0],
    };
    let (right, forward) = match up_idx {
        0 => ([0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
        2 => ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0]),
        _ => ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
    };
    let axis_min = |a: [f32; 3]| m.pos.iter().map(|p| p[0] * a[0] + p[1] * a[1] + p[2] * a[2]).fold(f32::MAX, f32::min);
    let axis_max = |a: [f32; 3]| m.pos.iter().map(|p| p[0] * a[0] + p[1] * a[1] + p[2] * a[2]).fold(f32::MIN, f32::max);
    let (min_up, max_up) = if m.pos.is_empty() { (0.0, 0.0) } else { (axis_min(up), axis_max(up)) };
    Frame {
        right,
        up,
        forward,
        extents: (axis_max(right) - axis_min(right), max_up - min_up, axis_max(forward) - axis_min(forward)),
        min_up,
    }
}

/// Pull each joint a little toward the centroid of nearby surface vertices.
/// Crude or very low-poly meshes have no surface inside the search radius, so
/// they keep the bbox placement. The blend is small on purpose: the analytic
/// skeleton is more stable than a handful of nearest vertices.
fn refine_joints(m: &Mesh, joints: &mut [Joint]) {
    let (mn, mx) = m.bbox();
    let h = (mx[1] - mn[1]).max(1e-6);
    let radius = 0.06 * h;
    let radius2 = radius * radius;
    for joint in joints.iter_mut() {
        let p = joint.position;
        let mut sum = [0.0f64; 3];
        let mut wsum = 0.0f64;
        for v in &m.pos {
            let d2 = (v[0] - p[0]) * (v[0] - p[0]) + (v[1] - p[1]) * (v[1] - p[1]) + (v[2] - p[2]) * (v[2] - p[2]);
            if d2 > radius2 {
                continue;
            }
            // Inverse-square so the closest surface dominates without a hard cut.
            let w = 1.0 / (d2 as f64 + 1e-8);
            for q in 0..3 {
                sum[q] += w * v[q] as f64;
            }
            wsum += w;
        }
        if wsum > 0.0 {
            for q in 0..3 {
                let target = (sum[q] / wsum) as f32;
                joint.position[q] = p[q] * 0.7 + target * 0.3;
            }
        }
    }
}

/// Distance from `p` to segment `a`-`b`.
fn point_segment_distance(p: [f32; 3], a: [f32; 3], b: [f32; 3]) -> f32 {
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ap = [p[0] - a[0], p[1] - a[1], p[2] - a[2]];
    let len2 = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
    let t = if len2 > 1e-20 { ((ap[0] * ab[0] + ap[1] * ab[1] + ap[2] * ab[2]) / len2).clamp(0.0, 1.0) } else { 0.0 };
    let q = [a[0] + ab[0] * t, a[1] + ab[1] * t, a[2] + ab[2] * t];
    let d = [p[0] - q[0], p[1] - q[1], p[2] - q[2]];
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

/// Inverse-distance^4 over bone segments, top four, normalised.
///
/// A vertex near the middle of the torso is close to several bones, so the
/// result is a smooth blend. A vertex on a hand is far from everything except
/// that arm, so it stays nearly rigid. No heat solve, no linear system.
fn skin_weights(m: &Mesh, joints: &[Joint]) -> (Vec<[u8; 4]>, Vec<[f32; 4]>) {
    let bones: Vec<(usize, usize)> = joints
        .iter()
        .enumerate()
        .filter_map(|(i, j)| j.parent.map(|p| (p, i)))
        .collect();
    let mut indices = Vec::with_capacity(m.pos.len());
    let mut weights = Vec::with_capacity(m.pos.len());
    for &p in &m.pos {
        let mut cand: Vec<(f32, usize)> = bones
            .iter()
            .map(|&(a, b)| {
                let d = point_segment_distance(p, joints[a].position, joints[b].position);
                (1.0 / (d.powi(4) + 1e-8f32), b)
            })
            .collect();
        cand.sort_by(|a, b| b.0.total_cmp(&a.0));
        cand.truncate(MAX_INFLUENCES);
        let total: f32 = cand.iter().map(|x| x.0).sum();
        let mut ji = [0u8; 4];
        let mut w = [0.0f32; 4];
        if total > 0.0 {
            for (k, &(raw, joint)) in cand.iter().enumerate() {
                ji[k] = joint as u8;
                w[k] = raw / total;
            }
            // Guard against a zero/denormal total and renormalise after truncation.
            let s: f32 = w.iter().sum();
            if s > 0.0 && (s - 1.0).abs() > 1e-6 {
                for x in w.iter_mut() {
                    *x /= s;
                }
            }
        } else {
            ji[0] = joints.len().saturating_sub(1) as u8;
            w[0] = 1.0;
        }
        indices.push(ji);
        weights.push(w);
    }
    (indices, weights)
}

/// Largest eigenvector of the vertex covariance, via power iteration. Reported
/// in the response so callers can see what the mesh's long axis actually is.
fn principal_axis(m: &Mesh) -> [f32; 3] {
    let n = m.pos.len() as f64;
    if n < 2.0 {
        return [0.0, 1.0, 0.0];
    }
    let mut mean = [0.0f64; 3];
    for p in &m.pos {
        for q in 0..3 {
            mean[q] += p[q] as f64 / n;
        }
    }
    let mut cov = [[0.0f64; 3]; 3];
    for p in &m.pos {
        let d = [p[0] as f64 - mean[0], p[1] as f64 - mean[1], p[2] as f64 - mean[2]];
        for i in 0..3 {
            for j in 0..3 {
                cov[i][j] += d[i] * d[j];
            }
        }
    }
    let mut v = [1.0f64, 1.0f64, 1.0f64];
    for _ in 0..24 {
        let mut w = [0.0f64; 3];
        for i in 0..3 {
            for j in 0..3 {
                w[i] += cov[i][j] * v[j];
            }
        }
        let len = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt();
        if len < 1e-12 {
            break;
        }
        v = [w[0] / len, w[1] / len, w[2] / len];
    }
    // Stable sign: the dominant component is positive.
    let mut big = 0;
    for i in 1..3 {
        if v[i].abs() > v[big].abs() {
            big = i;
        }
    }
    if v[big] < 0.0 {
        for x in v.iter_mut() {
            *x = -*x;
        }
    }
    [v[0] as f32, v[1] as f32, v[2] as f32]
}

/// Decode one `POST /api/cabin/rig` body and produce a skinned GLB plus a joint
/// report. Kept free of axum types, like [`crate::mesh::run_request`].
pub fn run_request(b: &Value) -> Result<Value> {
    use base64::Engine;
    const MAX_INPUT: usize = 64 << 20;
    const MAX_VERTICES: usize = 500_000;
    let enc = base64::engine::general_purpose::STANDARD;
    let input = b
        .get("mesh")
        .and_then(|v| {
            v.as_str()
                .map(str::to_string)
                .or_else(|| v.get("glb").and_then(|g| g.as_str()).map(str::to_string))
        })
        .or_else(|| b.get("glb").and_then(|v| v.as_str()).map(str::to_string))
        .ok_or_else(|| anyhow::anyhow!("mesh missing: send base64 GLB/OBJ/PLY/STL in 'mesh' or {{'glb'}}"))?;
    let s = input.split_once("base64,").map(|x| x.1).unwrap_or(&input);
    let bytes = enc.decode(s.trim()).map_err(|e| anyhow::anyhow!("invalid base64: {e}"))?;
    if bytes.len() > MAX_INPUT {
        bail!("mesh input is {} bytes; the limit is {} bytes", bytes.len(), MAX_INPUT);
    }
    let template = b
        .get("template")
        .and_then(|v| v.as_str())
        .or_else(|| b.get("params").and_then(|p| p.get("template")).and_then(|v| v.as_str()))
        .unwrap_or("humanoid");
    let mesh = decode_mesh_data(&bytes)?;
    mesh.validate()?;
    if mesh.pos.len() > MAX_VERTICES {
        bail!("mesh has {} vertices; the rig limit is {MAX_VERTICES}", mesh.pos.len());
    }

    let rig = fit(&mesh, template)?;
    let mut nor = vec![[0.0f32; 3]; mesh.pos.len()];
    for t in &mesh.tri {
        let (a, b, c) = (mesh.pos[t[0] as usize], mesh.pos[t[1] as usize], mesh.pos[t[2] as usize]);
        let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let f = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
        for &i in t {
            for q in 0..3 {
                nor[i as usize][q] += f[q];
            }
        }
    }
    for n in nor.iter_mut() {
        let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-12);
        for q in 0..3 {
            n[q] /= l;
        }
    }

    let joints: Vec<crate::triposr::SkinJoint> = rig
        .joints
        .iter()
        .map(|j| crate::triposr::SkinJoint { name: j.name.clone(), parent: j.parent, translation: j.position })
        .collect();
    let glb = crate::triposr::write_glb_skinned(&mesh.pos, &nor, mesh.col.as_deref(), &mesh.tri, &rig.joint_index, &rig.weights, &joints);
    let bones: Vec<Value> = rig
        .bones()
        .iter()
        .map(|&[a, b]| json!({"from": rig.joints[a].name, "to": rig.joints[b].name}))
        .collect();
    let axis = principal_axis(&mesh);
    Ok(json!({
        "template": rig.template,
        "joints": rig.joints,
        "bones": bones,
        "joint_count": rig.joints.len(),
        "bone_count": rig.bones().len(),
        "skin": {
            "method": "inverse-distance^4 over bone segments, top 4, normalised",
            "max_influences": MAX_INFLUENCES,
            "vertices": mesh.pos.len(),
            "joints_attribute": "JOINTS_0 (u8 vec4)",
            "weights_attribute": "WEIGHTS_0 (f32 vec4)"
        },
        "assumptions": {
            "up_axis": "+Y",
            "forward_axis": "+Z (TripoSR glTF frame)",
            "principal_axis": axis,
            "refinement": "joints blended 30% toward the local surface centroid within 0.06 x height",
            "note": "geometric fit, not a learned rigger; animation quality is not claimed"
        },
        "vertices": mesh.pos.len(),
        "triangles": mesh.tri.len(),
        "glb": enc.encode(&glb),
        "glb_bytes": glb.len(),
        "bytes": glb.len()
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine;
    use crate::mesh::{encode_mesh_format, MeshFormat};

    fn cube() -> Mesh {
        Mesh::unit_cube()
    }

    #[test]
    fn humanoid_skeleton_has_named_chain_and_expected_bones() {
        let rig = fit(&cube(), "humanoid").unwrap();
        assert_eq!(rig.joints.len(), 19);
        assert_eq!(rig.bones().len(), 18);
        assert_eq!(rig.joints[0].name, "hips");
        assert_eq!(rig.joints[0].parent, None);
        assert_eq!(rig.joints[4].name, "head");
        assert!(rig.joints.iter().any(|j| j.name == "hand_l"));
        assert!(rig.joints.iter().any(|j| j.name == "foot_r"));
        // Every non-root joint names a parent earlier in the list.
        for (i, j) in rig.joints.iter().enumerate() {
            if let Some(p) = j.parent {
                assert!(p < i, "{} has parent {p} after itself", j.name);
            }
        }
        // The chain spans most of the bbox height.
        let (mn, mx) = cube().bbox();
        let ys: Vec<f32> = rig.joints.iter().map(|j| j.position[1]).collect();
        assert!(ys.iter().cloned().fold(f32::MAX, f32::min) < mn[1] + (mx[1] - mn[1]) * 0.1);
        assert!(ys.iter().cloned().fold(f32::MIN, f32::max) > mn[1] + (mx[1] - mn[1]) * 0.8);
    }

    #[test]
    fn character_template_is_shorter() {
        let humanoid = fit(&cube(), "humanoid").unwrap();
        let character = fit(&cube(), "character").unwrap();
        assert_eq!(character.joints.len(), 8);
        assert_eq!(character.bones().len(), 7);
        assert!(character.joints.len() < humanoid.joints.len());
        assert!(fit(&cube(), "no-such-template").is_err());
    }

    #[test]
    fn clearly_lying_mesh_is_fitted_along_its_long_axis() {
        let mut m = cube();
        for p in &mut m.pos {
            p[0] *= 3.0;
            p[1] *= 0.5;
            p[2] *= 0.5;
        }
        let rig = fit(&m, "humanoid").unwrap();
        let xs: Vec<f32> = rig.joints.iter().map(|j| j.position[0]).collect();
        let span = xs.iter().cloned().fold(f32::MIN, f32::max) - xs.iter().cloned().fold(f32::MAX, f32::min);
        assert!(span > 4.0, "skeleton should run along X, span was {span}");
    }

    #[test]
    fn weights_are_finite_normalised_and_at_most_four() {
        let rig = fit(&cube(), "humanoid").unwrap();
        assert_eq!(rig.weights.len(), cube().pos.len());
        for (ji, w) in rig.joint_index.iter().zip(&rig.weights) {
            let sum: f32 = w.iter().sum();
            assert!((sum - 1.0).abs() < 1e-4, "weights sum to {sum}");
            assert!(w.iter().all(|x| x.is_finite() && *x >= 0.0));
            assert!(w.iter().filter(|x| **x > 0.0).count() <= MAX_INFLUENCES);
            assert!(ji.iter().all(|&i| (i as usize) < rig.joints.len()));
        }
    }

    #[test]
    fn glb_request_has_skins_and_skin_attributes() {
        let glb = encode_mesh_format(&cube(), MeshFormat::Glb).unwrap();
        let b = json!({
            "mesh": base64::engine::general_purpose::STANDARD.encode(&glb),
            "template": "humanoid"
        });
        let out = run_request(&b).unwrap();
        assert_eq!(out["joint_count"], 19);
        assert_eq!(out["bone_count"], 18);
        assert_eq!(out["skin"]["max_influences"], 4);
        assert!(out["assumptions"]["principal_axis"].is_array());

        let bytes = base64::engine::general_purpose::STANDARD.decode(out["glb"].as_str().unwrap()).unwrap();
        assert_eq!(&bytes[0..4], b"glTF");
        let json_len = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) as usize;
        let js: Value = serde_json::from_slice(&bytes[20..20 + json_len]).unwrap();
        let skin = &js["skins"][0];
        assert_eq!(skin["joints"].as_array().unwrap().len(), 19);
        assert!(skin["inverseBindMatrices"].is_number());
        assert_eq!(skin["joints"].as_array().unwrap().len(), js["accessors"][skin["inverseBindMatrices"].as_u64().unwrap() as usize]["count"].as_u64().unwrap() as usize);
        let attrs = &js["meshes"][0]["primitives"][0]["attributes"];
        assert!(attrs["JOINTS_0"].is_number());
        assert!(attrs["WEIGHTS_0"].is_number());
        assert_eq!(js["nodes"].as_array().unwrap().len(), 20); // mesh + 19 joints
        assert_eq!(js["meshes"][0]["primitives"][0]["attributes"]["POSITION"].as_u64().unwrap(), 0);
        // The mesh tools must still be able to read a rigged GLB.
        let back = crate::mesh::decode_mesh_data(&bytes).unwrap();
        assert_eq!(back.pos.len(), 8);
        assert_eq!(back.tri.len(), 12);
        assert!(back.col.is_some(), "skinned writer always emits COLOR_0");
    }

    #[test]
    fn rejects_empty_and_unknown_input() {
        let b = json!({"mesh": "", "template": "humanoid"});
        assert!(run_request(&b).is_err());
        let glb = encode_mesh_format(&cube(), MeshFormat::Glb).unwrap();
        let b = json!({
            "mesh": base64::engine::general_purpose::STANDARD.encode(&glb),
            "template": "quadruped"
        });
        assert!(run_request(&b).is_err());
    }
}
