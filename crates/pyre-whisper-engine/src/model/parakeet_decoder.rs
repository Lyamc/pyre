use burn::config::Config;
use burn::module::{Module, Param};
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::tensor::activation::{sigmoid, tanh};
use burn::tensor::{Int, Tensor};

/// PyTorch-compatible LSTM layer using gate-ordered weights (i, f, g, o).
#[derive(Module, Debug)]
pub struct ParakeetLstmLayer {
    pub weight_ih: Param<Tensor<2>>,
    pub weight_hh: Param<Tensor<2>>,
    pub bias_ih: Param<Tensor<1>>,
    pub bias_hh: Param<Tensor<1>>,
}

impl ParakeetLstmLayer {
    pub fn hidden_size(&self) -> usize {
        self.bias_ih.dims()[0] / 4
    }

    pub fn step(
        &self,
        input: Tensor<2>,
        hidden: Tensor<2>,
        cell: Tensor<2>,
    ) -> (Tensor<2>, Tensor<2>) {
        // HF/PyTorch LSTM weights are [4*hidden, in]; matmul uses transposed layout.
        let gates = input.matmul(self.weight_ih.val().clone().transpose())
            + self.bias_ih.val().clone().unsqueeze()
            + hidden.matmul(self.weight_hh.val().clone().transpose())
            + self.bias_hh.val().clone().unsqueeze();

        let h = self.hidden_size();
        let i_gate = sigmoid(gates.clone().slice([0..1, 0..h]));
        let f_gate = sigmoid(gates.clone().slice([0..1, h..2 * h]));
        let g_gate = tanh(gates.clone().slice([0..1, 2 * h..3 * h]));
        let o_gate = sigmoid(gates.slice([0..1, 3 * h..4 * h]));

        let new_cell = f_gate * cell + i_gate * g_gate;
        let new_hidden = o_gate * tanh(new_cell.clone());
        (new_hidden, new_cell)
    }
}

#[derive(Module, Debug)]
pub struct ParakeetLstm {
    pub layers: Vec<ParakeetLstmLayer>,
}

impl ParakeetLstm {
    pub fn step(
        &self,
        input: Tensor<2>,
        hidden: &[Tensor<2>],
        cell: &[Tensor<2>],
    ) -> (Tensor<2>, Vec<Tensor<2>>, Vec<Tensor<2>>) {
        let mut x = input;
        let mut new_hidden = Vec::with_capacity(self.layers.len());
        let mut new_cell = Vec::with_capacity(self.layers.len());

        for (layer, (h, c)) in self.layers.iter().zip(hidden.iter().zip(cell.iter())) {
            let (nh, nc) = layer.step(x, h.clone(), c.clone());
            x = nh.clone();
            new_hidden.push(nh);
            new_cell.push(nc);
        }

        (x, new_hidden, new_cell)
    }
}

#[derive(Config, Debug)]
pub struct ParakeetDecoderConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
}

impl ParakeetDecoderConfig {
    pub fn init(&self, device: &burn::tensor::Device) -> ParakeetDecoder {
        let embedding = EmbeddingConfig::new(self.vocab_size, self.hidden_size).init(device);
        let projector = LinearConfig::new(self.hidden_size, self.hidden_size).init(device);
        let layers = (0..self.num_layers)
            .map(|_| ParakeetLstmLayer {
                weight_ih: Param::from_tensor(Tensor::zeros(
                    [4 * self.hidden_size, self.hidden_size],
                    device,
                )),
                weight_hh: Param::from_tensor(Tensor::zeros(
                    [4 * self.hidden_size, self.hidden_size],
                    device,
                )),
                bias_ih: Param::from_tensor(Tensor::zeros([4 * self.hidden_size], device)),
                bias_hh: Param::from_tensor(Tensor::zeros([4 * self.hidden_size], device)),
            })
            .collect();
        ParakeetDecoder {
            embedding,
            lstm: ParakeetLstm { layers },
            projector,
        }
    }
}

#[derive(Module, Debug)]
pub struct ParakeetDecoder {
    pub embedding: Embedding,
    pub lstm: ParakeetLstm,
    pub projector: Linear,
}

impl ParakeetDecoder {
    pub fn hidden_size(&self) -> usize {
        self.lstm
            .layers
            .first()
            .map(|l| l.hidden_size())
            .unwrap_or(self.projector.weight.dims()[1])
    }
}

pub struct ParakeetDecoderCache {
    pub output: Tensor<2>,
    pub hidden: Vec<Tensor<2>>,
    pub cell: Vec<Tensor<2>>,
    pub initialized: bool,
}

impl ParakeetDecoderCache {
    pub fn new(device: &burn::tensor::Device, num_layers: usize, hidden_size: usize) -> Self {
        let hidden = (0..num_layers)
            .map(|_| Tensor::zeros([1, hidden_size], device))
            .collect();
        let cell = (0..num_layers)
            .map(|_| Tensor::zeros([1, hidden_size], device))
            .collect();
        Self {
            output: Tensor::zeros([1, hidden_size], device),
            hidden,
            cell,
            initialized: false,
        }
    }

    pub fn reset(&mut self, device: &burn::tensor::Device, num_layers: usize, hidden_size: usize) {
        *self = Self::new(device, num_layers, hidden_size);
    }
}

impl ParakeetDecoder {
    pub fn forward_step(
        &self,
        token_id: usize,
        blank_token_id: usize,
        cache: &mut ParakeetDecoderCache,
        update_state: bool,
    ) -> Tensor<2> {
        if cache.initialized && token_id == blank_token_id && !update_state {
            return cache.output.clone();
        }

        let device = self.embedding.weight.device();
        // NeMo's predictor treats the blank / SOS as "no token": a zero vector, not
        // embedding[blank]. v3's blank id is 8192; feeding that row (often not the
        // pad) makes the first joint emit blank plus a long duration and the
        // utterance comes back empty or only as its last few words. v2's blank is
        // the pad row, which is why it was less affected.
        let hidden = self.embedding.weight.dims()[1];
        let input = if token_id == blank_token_id {
            Tensor::<2>::zeros([1, hidden], &device)
        } else {
            let token = Tensor::<2, Int>::from_ints([[token_id as i32]], &device);
            let embedded = self.embedding.forward(token);
            let [_, _, d_model] = embedded.dims();
            embedded.reshape([1, d_model])
        };

        let (lstm_out, new_hidden, new_cell) =
            self.lstm.step(input, &cache.hidden, &cache.cell);
        let projected = self.projector.forward(lstm_out);

        if update_state || !cache.initialized {
            cache.output = projected.clone();
            cache.hidden = new_hidden;
            cache.cell = new_cell;
            cache.initialized = true;
        }

        projected
    }
}