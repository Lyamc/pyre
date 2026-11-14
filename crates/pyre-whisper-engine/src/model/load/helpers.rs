use burn::tensor::{Tensor};
use npy::NpyData;
use std::error::Error;
use std::io::Read;
use std::fs::File;
use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

/// Replaced on every `init_shapes` call. A process converts v2 and then v3;
/// `OnceLock` kept v2's joint shape and reshaped v3's `[640, 8198]` head into `[640, 1030]`.
pub static SHAPES: LazyLock<Mutex<HashMap<String, Vec<usize>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

fn shapes_map() -> std::sync::MutexGuard<'static, HashMap<String, Vec<usize>>> {
    SHAPES.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

pub fn init_shapes(path: &str) {
    let shapes_path = format!("{}/shapes.json", path);
    let map = std::fs::read_to_string(&shapes_path)
        .ok()
        .and_then(|content| serde_json::from_str::<HashMap<String, Vec<usize>>>(&content).ok())
        .unwrap_or_default();
    *shapes_map() = map;
}

pub fn tensor_exists(name: &str, path: &str) -> bool {
    let tensor_path = format!("{}/{}.npy", path, name).replace("\\", "/");
    shapes_map().keys().any(|k| tensor_path.ends_with(k))
}

pub fn load_tensor<const D: usize>(
    name: &str,
    path: &str,
    device: &burn::tensor::Device,
) -> Result<Tensor<D>, Box<dyn Error>> {
    let tensor_path = format!("{}/{}.npy", path, name);
    
    // Check metadata first. If it's NOT in shapes.json, it's not just a missing file, 
    // it's a tensor that doesn't exist in the source model (like optional biases).
    if !tensor_exists(name, path) {
        return Err(format!("Tensor {} not found in metadata at {}", name, path).into());
    }

    let mut file = match File::open(&tensor_path) {
        Ok(f) => f,
        Err(e) => {
            // If it's in metadata but file is missing, this IS a real error.
            eprintln!("FATAL ERROR: Tensor '{}' exists in metadata but file is missing: {}", name, tensor_path);
            return Err(Box::new(e));
        }
    };

    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;

    let data = decode_npy_f32(&buf)?;

    let rel_path = tensor_path.replace("\\", "/");
    let mut shape_vec = Vec::new();
    let mut found = false;
    
    for (k, v) in shapes_map().iter() {
        if rel_path.ends_with(k) {
            shape_vec = v.clone();
            found = true;
            break;
        }
    }

    if !found {
        // Fallback for cases where metadata might be slightly off
        let mut fallback_shape = [0; D];
        for i in 0..D {
            fallback_shape[i] = if i < data.len() { data[i] as usize } else { 1 };
        }
        let tensor = Tensor::<1>::from_floats(&data[D..], device).reshape(fallback_shape);
        return Ok(tensor);
    }

    let target_shape = {
        let mut s = [0; D];
        if shape_vec.len() == D {
            for i in 0..D { s[i] = shape_vec[i]; }
        } else {
            if D == 3 && shape_vec.len() == 4 {
                s[0] = shape_vec[0];
                s[1] = shape_vec[1];
                s[2] = shape_vec[2] * shape_vec[3];
            } else if D == 2 && shape_vec.len() == 1 {
                s[0] = shape_vec[0];
                s[1] = 1;
            } else {
                for i in 0..D {
                    s[i] = if i < shape_vec.len() { shape_vec[i] } else { 1 };
                }
            }
        }
        s
    };

    let tensor = Tensor::<1>::from_floats(&data[..], device).reshape(target_shape);
    Ok(tensor)
}

/// Read an `.npy` vector as f32. `<f4` is native. `<f2` is IEEE f16 and a header that
/// names bfloat16 is bf16. Both half formats are widened here; the Conformer graph stays f32.
pub fn decode_npy_f32(buf: &[u8]) -> Result<Vec<f32>, Box<dyn Error>> {
    if buf.len() < 10 || buf[0] != 0x93 || &buf[1..6] != b"NUMPY" {
        return Err("not an npy file".into());
    }
    let major = buf[6];
    let (header, data) = if major == 1 {
        let n = u16::from_le_bytes([buf[8], buf[9]]) as usize;
        let start = 10;
        if buf.len() < start + n {
            return Err("truncated npy header".into());
        }
        (std::str::from_utf8(&buf[start..start + n])?, &buf[start + n..])
    } else {
        if buf.len() < 12 {
            return Err("truncated npy header".into());
        }
        let n = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize;
        let start = 12;
        if buf.len() < start + n {
            return Err("truncated npy header".into());
        }
        (std::str::from_utf8(&buf[start..start + n])?, &buf[start + n..])
    };
    let header_l = header.to_ascii_lowercase();
    if header_l.contains("bfloat16") || header_l.contains("bf16") {
        return Ok(data
            .chunks_exact(2)
            .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect());
    }
    if header_l.contains("<f2") || header_l.contains("|f2") || header_l.contains("float16") {
        return Ok(data
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect());
    }
    if header_l.contains("<f4") || header_l.contains("|f4") || header_l.contains("float32") {
        return Ok(data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect());
    }
    let npy = NpyData::<f32>::from_bytes(buf)?;
    Ok(npy.to_vec())
}

pub fn load_usize(name: &str, path: &str, device: &burn::tensor::Device) -> Result<usize, Box<dyn Error>> {
    let tensor = load_tensor::<1>(name, path, device)?;
    let data = tensor.into_data();
    let val = data.to_vec::<f32>().unwrap()[0] as usize;
    Ok(val)
}

#[cfg(test)]
mod tests {
    use super::init_shapes;

    #[test]
    fn second_conversion_replaces_the_shape_map() {
        let root = std::env::temp_dir().join(format!(
            "pyre-shapes-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let v2 = root.join("v2");
        let v3 = root.join("v3");
        std::fs::create_dir_all(&v2).unwrap();
        std::fs::create_dir_all(&v3).unwrap();
        // v2 joint is [640, 1030]. v3's same key is [640, 8198] (5246720 values).
        std::fs::write(
            v2.join("shapes.json"),
            r#"{"joint/weight.npy":[640,1030]}"#,
        )
        .unwrap();
        std::fs::write(
            v3.join("shapes.json"),
            r#"{"joint/weight.npy":[640,8198]}"#,
        )
        .unwrap();

        init_shapes(v2.to_str().unwrap());
        init_shapes(v3.to_str().unwrap());
        let path = format!("{}/joint/weight.npy", v3.display()).replace('\\', "/");
        let shape = super::shapes_map()
            .iter()
            .find(|(k, _)| path.ends_with(k.as_str()))
            .map(|(_, v)| v.clone());
        let _ = std::fs::remove_dir_all(&root);
        assert_eq!(shape, Some(vec![640, 8198]));
    }

    #[test]
    fn f16_and_bf16_npy_widen_to_f32() {
        let f16 = half::f16::from_f32(1.5).to_le_bytes();
        let mut buf = npy_header("<f2", 1);
        buf.extend_from_slice(&f16);
        let v = super::decode_npy_f32(&buf).unwrap();
        assert!((v[0] - 1.5).abs() < 1e-3);

        let bf = half::bf16::from_f32(-2.0).to_le_bytes();
        let mut buf = npy_header("bfloat16", 1);
        buf.extend_from_slice(&bf);
        let v = super::decode_npy_f32(&buf).unwrap();
        assert!((v[0] + 2.0).abs() < 1e-2);
    }

    fn npy_header(descr: &str, n: usize) -> Vec<u8> {
        let header = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': ({n},), }}");
        let mut header_bytes = header.into_bytes();
        while (header_bytes.len() + 10 + 1) % 16 != 0 {
            header_bytes.push(b' ');
        }
        header_bytes.push(b'\n');
        let mut out = vec![0x93];
        out.extend_from_slice(b"NUMPY");
        out.extend_from_slice(&[1, 0]);
        out.extend_from_slice(&(header_bytes.len() as u16).to_le_bytes());
        out.extend_from_slice(&header_bytes);
        out
    }
}
