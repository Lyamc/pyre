//! Vocab embeddings kept on the host so the 7B table (~2.18 GB f32) never
//! sits on the GPU. Only the gathered rows for the current token ids are uploaded.

use std::fmt;
use std::sync::Arc;

use burn::module::{Module, Param};
use burn::tensor::{Int, Tensor};

#[derive(Module)]
pub struct HostEmbedding {
    #[module(skip)]
    weight: Arc<Vec<f32>>,
    #[module(skip)]
    vocab: usize,
    #[module(skip)]
    hidden: usize,
    marker: Param<Tensor<1>>,
}

impl fmt::Debug for HostEmbedding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HostEmbedding")
            .field("vocab", &self.vocab)
            .field("hidden", &self.hidden)
            .field("host_bytes", &(self.weight.len() * 4))
            .finish()
    }
}

impl HostEmbedding {
    pub fn empty(vocab: usize, hidden: usize, device: &burn::tensor::Device) -> Self {
        Self {
            weight: Arc::new(Vec::new()),
            vocab,
            hidden,
            marker: Param::from_tensor(Tensor::<1>::zeros([1], device)),
        }
    }

    pub fn from_weight(
        weight: Vec<f32>,
        vocab: usize,
        hidden: usize,
        device: &burn::tensor::Device,
    ) -> Self {
        debug_assert_eq!(weight.len(), vocab * hidden);
        Self {
            weight: Arc::new(weight),
            vocab,
            hidden,
            marker: Param::from_tensor(Tensor::<1>::zeros([1], device)),
        }
    }

    pub fn forward(&self, ids: Tensor<2, Int>) -> Tensor<3> {
        let [batch, seq] = ids.dims();
        let device = ids.device();
        let data = ids.into_data();
        let ids_i: Vec<i64> = data
            .clone()
            .to_vec::<i64>()
            .or_else(|_| {
                data.to_vec::<i32>()
                    .map(|v| v.into_iter().map(i64::from).collect())
            })
            .expect("embedding ids");
        let hidden = self.hidden;
        let vocab = self.vocab.max(1);
        let table = self.weight.as_slice();
        let mut rows = vec![0f32; ids_i.len() * hidden];
        if !table.is_empty() {
            for (i, &id) in ids_i.iter().enumerate() {
                let row = (id.max(0) as usize).min(vocab - 1);
                let src = row * hidden;
                let dst = i * hidden;
                if src + hidden <= table.len() {
                    rows[dst..dst + hidden].copy_from_slice(&table[src..src + hidden]);
                }
            }
        }
        Tensor::<1>::from_floats(rows.as_slice(), &device).reshape([batch, seq, hidden])
    }
}

#[cfg(test)]
mod tests {
    use super::HostEmbedding;
    use burn::backend::ndarray::{NdArray, NdArrayDevice};
    use burn::tensor::{Int, Tensor};

    #[test]
    fn gathers_requested_rows() {
        let device = NdArrayDevice::Cpu;
        // 3 tokens, hidden 2: rows [0,1], [2,3], [4,5]
        let table = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let emb = HostEmbedding::<NdArray>::from_weight(table, 3, 2, &device);
        let ids = Tensor::<NdArray, 1, Int>::from_ints([2i32, 0], &device).reshape([1, 2]);
        let out = emb.forward(ids).into_data().to_vec::<f32>().unwrap();
        assert_eq!(out, vec![4.0, 5.0, 0.0, 1.0]);
    }
}
