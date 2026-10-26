use burn::{
    module::{Module, Param, ParamId},
    tensor::{Tensor},
};
use crate::model::load::helpers::load_tensor;
use std::error::Error;

pub fn inject_weights<M: Module>(
    module: M,
    path: &str,
) -> Result<M, Box<dyn Error>> {
    // This is complex in Burn because modules are immutable and records are used for loading.
    // However, we can recursively build a record.
    // Given the current structure of the loader, it's easier to fix the component loaders.
    Ok(module)
}
