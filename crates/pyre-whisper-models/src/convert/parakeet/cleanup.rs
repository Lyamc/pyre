use std::fs;
use std::path::Path;

use crate::convert::npy::NpyDump;

const NPY_SUBDIRS: &[&str] = &[
    "encoder",
    "decoder",
    "ctc_linear",
    "final_proj",
    "joint_pred",
];

const NPY_FILES: &[&str] = &["shapes.json", "actual_n_vocab.npy"];

pub fn cleanup_parakeet_npy_artifacts(model_dir: &Path, verbose: bool) -> anyhow::Result<()> {
    NPY_SUBDIRS
        .iter()
        .map(|subdir| model_dir.join(subdir))
        .filter(|path| path.exists())
        .try_for_each(|path| fs::remove_dir_all(path))?;

    NPY_FILES
        .iter()
        .map(|file| model_dir.join(file))
        .filter(|path| path.exists())
        .try_for_each(|path| fs::remove_file(path))?;

    if verbose {
        tracing::info!("removed parakeet npy intermediates from {}", model_dir.display());
    }
    Ok(())
}

pub fn write_attn_heads(
    dump: &mut NpyDump,
    layers: usize,
    n_head: usize,
) -> anyhow::Result<()> {
    let head_val = n_head as f32;
    (0..layers).try_for_each(|i| {
        dump.write_scalar(&format!("encoder/block_{i}/attn/n_head.npy"), head_val)
    })
}