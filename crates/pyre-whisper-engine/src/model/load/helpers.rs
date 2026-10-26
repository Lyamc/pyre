use burn::tensor::{Tensor};
use npy::NpyData;
use std::error::Error;
use std::io::Read;
use std::fs::File;
use std::collections::HashMap;
use std::sync::OnceLock;

pub static SHAPES: OnceLock<HashMap<String, Vec<usize>>> = OnceLock::new();

pub fn init_shapes(path: &str) {
    let shapes_path = format!("{}/shapes.json", path);
    if let Ok(content) = std::fs::read_to_string(&shapes_path) {
        if let Ok(map) = serde_json::from_str::<HashMap<String, Vec<usize>>>(&content) {
            let _ = SHAPES.set(map);
        }
    }
}

pub fn tensor_exists(name: &str, path: &str) -> bool {
    let tensor_path = format!("{}/{}.npy", path, name).replace("\\", "/");
    if let Some(m) = SHAPES.get() {
        for k in m.keys() {
            if tensor_path.ends_with(k) {
                return true;
            }
        }
    }
    false
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

    let npy = NpyData::<f32>::from_bytes(&buf)?;
    let data: Vec<f32> = npy.to_vec();

    let rel_path = tensor_path.replace("\\", "/");
    let mut shape_vec = Vec::new();
    let mut found = false;
    
    if let Some(m) = SHAPES.get() {
        for (k, v) in m {
            if rel_path.ends_with(k) {
                shape_vec = v.clone();
                found = true;
                break;
            }
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

pub fn load_usize(name: &str, path: &str, device: &burn::tensor::Device) -> Result<usize, Box<dyn Error>> {
    let tensor = load_tensor::<1>(name, path, device)?;
    let data = tensor.into_data();
    let val = data.to_vec::<f32>().unwrap()[0] as usize;
    Ok(val)
}
