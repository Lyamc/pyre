use std::path::Path;


use burn::config::Config;
use burn::module::Module;

use pyre_whisper_engine::model::load::{init_shapes, load_model_from_npy};

use crate::download::DownloadOptions;
use crate::paths::model_dir;

pub fn save_model_from_npy(name: &str, options: &DownloadOptions) -> anyhow::Result<()> {
    let dump_dir = model_dir(name);
    if !dump_dir.join("encoder").exists() && !dump_dir.join("decoder").exists() {
        anyhow::bail!(
            "no npy dump at {} for conversion",
            dump_dir.display()
        );
    }

    options.report(crate::download::PrepProgress::convert(
        format!("Building Burn weight file for '{name}'"),
        0.8,
    ));
    if options.verbose {
        tracing::info!("converting npy dump to burn mpk for '{name}'");
    }

    let device = burn::tensor::Device::default();
    let dump_str = dump_dir
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("invalid model path"))?;
    init_shapes(dump_str);
    let (model, model_config) =
        load_model_from_npy(dump_str, &device).map_err(|e| {
            anyhow::anyhow!("npy load failed for '{name}': {e}")
        })?;

    model
        .save_file(dump_dir.join("model.bpk"))
        .map_err(|e| anyhow::anyhow!("saving burnpack for '{name}' failed: {e:?}"))?;

    let cfg_path = dump_dir.join(format!("{name}.cfg"));
    model_config.save(&cfg_path)?;

    let alias_cfg = dump_dir.join("config.cfg");
    if !alias_cfg.exists() {
        model_config.save(&alias_cfg)?;
    }

    decompress_gz_to_aliases(&dump_dir, name)?;

    options.report(crate::download::PrepProgress::convert(
        format!("Burn bundle saved for '{name}'"),
        1.0,
    ));
    if options.verbose {
        tracing::info!("burn bundle saved to {}", dump_dir.display());
    }

    Ok(())
}

fn decompress_gz_to_aliases(dir: &Path, _name: &str) -> anyhow::Result<()> {
    use flate2::read::GzDecoder;
    use std::fs::File;
    use std::io::copy;

    let gz = dir.join("model.bpk");
    if !gz.exists() {
        return Ok(());
    }

    let mpk = dir.join("model.bpk");
    // Always refresh the uncompressed alias — a leftover model.mpk is
    // preferred by the loader and would keep serving pre-fix weights.
    let mut decoder = GzDecoder::new(File::open(&gz)?);
    let mut out = File::create(&mpk)?;
    copy(&mut decoder, &mut out)?;

    Ok(())
}