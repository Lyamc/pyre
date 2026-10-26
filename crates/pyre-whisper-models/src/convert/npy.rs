use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::Context;

pub struct NpyDump {
    shapes: HashMap<String, Vec<usize>>,
    root: PathBuf,
}

impl NpyDump {
    pub fn new(root: impl AsRef<Path>) -> Self {
        Self {
            shapes: HashMap::new(),
            root: root.as_ref().to_path_buf(),
        }
    }

    pub fn write_f32(
        &mut self,
        rel_path: &str,
        data: &[f32],
        shape: &[usize],
    ) -> anyhow::Result<()> {
        let path = self.root.join(rel_path);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        // npy-rs only reads 1-D arrays; actual rank lives in shapes.json.
        write_npy_f32(&path, data, &[data.len()])?;
        self.shapes.insert(rel_path.replace('\\', "/"), shape.to_vec());
        Ok(())
    }

    pub fn write_scalar(&mut self, rel_path: &str, value: f32) -> anyhow::Result<()> {
        self.write_f32(rel_path, &[value], &[1])
    }

    pub fn flush_shapes(&self) -> anyhow::Result<()> {
        let path = self.root.join("shapes.json");
        let json = serde_json::to_string_pretty(&self.shapes)?;
        fs::write(path, json)?;
        Ok(())
    }
}

pub fn write_npy_f32(path: &Path, data: &[f32], shape: &[usize]) -> anyhow::Result<()> {
    let shape_str = if shape.is_empty() {
        String::new()
    } else {
        let inner = shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        format!("{inner},")
    };
    let header = format!(
        "{{ 'descr': '<f4', 'fortran_order': False, 'shape': ({shape_str}), }}"
    );
    let mut header_bytes = header.as_bytes().to_vec();
    while (header_bytes.len() + 10 + 1) % 16 != 0 {
        header_bytes.push(b' ');
    }
    header_bytes.push(b'\n');

    let mut out = Vec::new();
    out.push(0x93);
    out.extend_from_slice(b"NUMPY");
    out.extend_from_slice(&[0x01, 0x00]);
    let header_len = header_bytes.len() as u16;
    out.extend_from_slice(&header_len.to_le_bytes());
    out.extend_from_slice(&header_bytes);
    for &v in data {
        out.extend_from_slice(&v.to_le_bytes());
    }

    fs::write(path, out).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}