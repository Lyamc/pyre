use burn::tensor::{activation::relu, Tensor};
use crate::helper::*;

pub fn get_mel_filters(
    sample_rate: f64,
    n_fft: usize,
    n_mels: usize,
    htk: bool,
    device: &burn::tensor::Device,
    normalize: bool,
) -> Tensor<2> {
    let fmin = 0.0;
    let fmax = sample_rate * 0.5;

    // Center freqs of each FFT bin
    let fftfreqs = fft_frequencies(sample_rate, n_fft, device);
    let [n_ffefreqs] = fftfreqs.dims();

    // 'Center freqs' of mel bands - uniformly spaced between limits
    let mel_f_size = n_mels + 2;
    let mel_f = mel_frequencies(mel_f_size, fmin, fmax, htk, device);

    // fdiff = np.diff(mel_f)
    let fdiff = mel_f.clone().slice([1..mel_f_size]) - mel_f.clone().slice([0..(mel_f_size - 1)]);

    // ramps = np.subtract.outer(mel_f, fftfreqs)
    let ramps = mel_f
        .clone()
        .unsqueeze::<2>()
        .transpose()
        .repeat(&[1, n_ffefreqs])
        - fftfreqs.unsqueeze();

    // lower and upper slopes for all bins
    let lower = -ramps.clone().slice([0..n_mels])
        / fdiff
            .clone()
            .slice([0..n_mels])
            .unsqueeze::<2>()
            .transpose();
    let upper = ramps.slice([2..(2 + n_mels)])
        / fdiff.slice([1..(1 + n_mels)]).unsqueeze::<2>().transpose();

    // .. then intersect them with each other and zero
    let mut weights = relu(tensor_min(lower, upper));

    if normalize {
        // Slaney-style mel is scaled to be approx constant energy per channel
        let enorm = (mel_f.clone().slice([2..(n_mels + 2)]) - mel_f.clone().slice([0..n_mels]))
            .powf_scalar(-1)
            * 2.0;
        weights = weights * enorm.unsqueeze::<2>().transpose();
    }

    if !(all_zeros(mel_f.slice([0..(n_mels - 2)])) || all_zeros(relu(-weights.clone().max_dim(1))))
    {
        println!("Empty filters detected in mel frequency basis. 
Some channels will produce empty responses. 
Try increasing your sampling rate (and fmax) or reducing n_mels.");
    }

    return weights;
}

pub fn fft_frequencies(
    sample_rate: f64,
    n_fft: usize,
    device: &burn::tensor::Device,
) -> Tensor<1> {
    Tensor::arange(0..(n_fft / 2 + 1) as i64, device)
        .float()
        .mul_scalar(sample_rate / n_fft as f64)
}

pub fn mel_frequencies(
    n_mels: usize,
    fmin: f64,
    fmax: f64,
    htk: bool,
    device: &burn::tensor::Device,
) -> Tensor<1> {
    let min_mel = hz_to_mel(fmin, htk);
    let max_mel = hz_to_mel(fmax, htk);

    let mels = Tensor::arange(0..n_mels as i64, device)
        .float()
        .mul_scalar((max_mel - min_mel) / (n_mels - 1) as f64)
        .add_scalar(min_mel);

    mel_to_hz_tensor(mels, htk)
}

pub fn hz_to_mel(freq: f64, htk: bool) -> f64 {
    if htk {
        return 2595.0 * (1.0 + freq / 700.0).log10();
    }

    let f_min = 0.0;
    let f_sp = 200.0 / 3.0;

    let min_log_hz = 1000.0;
    let min_log_mel = (min_log_hz - f_min) / f_sp;
    let logstep = (6.4f64).ln() / 27.0;

    let mel = if freq >= min_log_hz {
        min_log_mel + (freq / min_log_hz).ln() / logstep
    } else {
        (freq - f_min) / f_sp
    };

    return mel;
}

pub fn mel_to_hz_tensor(mel: Tensor<1>, htk: bool) -> Tensor<1> {
    if htk {
        return (_10pow(mel / 2595.0) - 1.0) * 700.0;
    }

    let f_min = 0.0;
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = (min_log_hz - f_min) / f_sp;
    let logstep = (6.4f64).ln() / 27.0;

    let log_t = mel.clone().greater_equal_elem(min_log_mel).float();
    let freq = log_t.clone() * (((mel.clone() - min_log_mel) * logstep).exp() * min_log_hz)
        + (-log_t + 1.0) * (mel * f_sp + f_min);

    return freq;
}
