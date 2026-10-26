use std::collections::HashMap;

pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len().min(b.len()) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = (na.sqrt() * nb.sqrt()).max(1e-8);
    (1.0 - dot / denom).clamp(0.0, 2.0)
}

/// Centroid-linkage agglomerative clustering. `threshold` is cosine distance.
pub fn agglomerative(embeds: &[Vec<f32>], threshold: f32) -> Vec<usize> {
    let n = embeds.len();
    if n == 0 {
        return Vec::new();
    }
    let mut parent: Vec<usize> = (0..n).collect();
    let find = |parent: &mut [usize], mut x: usize| {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    };
    let mut clusters: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    let mut active: Vec<bool> = vec![true; n];
    loop {
        let mut best = f32::MAX;
        let mut pair = None;
        for i in 0..n {
            if !active[i] {
                continue;
            }
            for j in (i + 1)..n {
                if !active[j] {
                    continue;
                }
                let d = cluster_distance(&clusters[i], &clusters[j], embeds);
                if d < best {
                    best = d;
                    pair = Some((i, j));
                }
            }
        }
        let Some((i, j)) = pair else { break };
        if best > threshold {
            break;
        }
        let members = std::mem::take(&mut clusters[j]);
        clusters[i].extend(members);
        active[j] = false;
        let pi = find(&mut parent, i);
        let pj = find(&mut parent, j);
        parent[pj] = pi;
    }
    let mut labels = vec![0usize; n];
    let mut remap = HashMap::new();
    let mut next = 0usize;
    for i in 0..n {
        let mut x = i;
        while parent[x] != x {
            x = parent[x];
        }
        let id = *remap.entry(x).or_insert_with(|| {
            let v = next;
            next += 1;
            v
        });
        labels[i] = id;
    }
    labels
}

fn cluster_distance(a: &[usize], b: &[usize], embeds: &[Vec<f32>]) -> f32 {
    let ca = centroid(a, embeds);
    let cb = centroid(b, embeds);
    cosine_distance(&ca, &cb)
}

fn centroid(ids: &[usize], embeds: &[Vec<f32>]) -> Vec<f32> {
    let dim = embeds.first().map(|e| e.len()).unwrap_or(0);
    let mut c = vec![0f32; dim];
    if ids.is_empty() {
        return c;
    }
    for &i in ids {
        for (d, v) in c.iter_mut().zip(embeds[i].iter()) {
            *d += *v;
        }
    }
    let n = ids.len() as f32;
    for v in &mut c {
        *v /= n;
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_groups() {
        let a = vec![1.0, 0.0];
        let b = vec![0.9, 0.1];
        let c = vec![0.0, 1.0];
        let d = vec![0.1, 0.9];
        let labels = agglomerative(&[a, b, c, d], 0.4);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_ne!(labels[0], labels[2]);
    }
}
