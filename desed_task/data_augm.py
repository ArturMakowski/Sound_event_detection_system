import random

import numpy as np
import torch
import torchaudio
from IPython.display import Audio
from typing import List, Callable


def frame_shift(mels, labels, net_pooling=4):
    bsz, n_bands, frames = mels.shape
    shifted = []
    new_labels = []
    for bindx in range(bsz):
        shift = int(random.gauss(0, 90))
        shifted.append(torch.roll(mels[bindx], shift, dims=-1))
        shift = -abs(shift) // net_pooling if shift < 0 else shift // net_pooling
        new_labels.append(torch.roll(labels[bindx], shift, dims=-1))
    return torch.stack(shifted), torch.stack(new_labels)


def mixup(data, target=None, alpha=0.2, beta=0.2, mixup_label_type="soft"):
    """Mixup data augmentation by permuting the data

    Args:
        data: input tensor, must be a batch so data can be permuted and mixed.
        target: tensor of the target to be mixed, if None, do not return targets.
        alpha: float, the parameter to the np.random.beta distribution
        beta: float, the parameter to the np.random.beta distribution
        mixup_label_type: str, the type of mixup to be used choice between {'soft', 'hard'}.
    Returns:
        torch.Tensor of mixed data and labels if given
    """
    with torch.no_grad():
        batch_size = data.size(0)
        c = np.random.beta(alpha, beta)

        perm = torch.randperm(batch_size)

        mixed_data = c * data + (1 - c) * data[perm, :]
        if target is not None:
            if mixup_label_type == "soft":
                mixed_target = torch.clamp(
                    c * target + (1 - c) * target[perm, :], min=0, max=1
                )
            elif mixup_label_type == "hard":
                mixed_target = torch.clamp(target + target[perm, :], min=0, max=1)
            else:
                raise NotImplementedError(
                    f"mixup_label_type: {mixup_label_type} not implemented. choice in "
                    f"{'soft', 'hard'}"
                )

            return mixed_data, mixed_target
        else:
            return mixed_data


def add_noise(mels, snrs=(6, 30), dims=(1, 2)):
    """Add white noise to mels spectrograms
    Args:
        mels: torch.tensor, mels spectrograms to apply the white noise to.
        snrs: int or tuple, the range of snrs to choose from if tuple (uniform)
        dims: tuple, the dimensions for which to compute the standard deviation (default to (1,2) because assume
            an input of a batch of mel spectrograms.
    Returns:
        torch.Tensor of mels with noise applied
    """
    if isinstance(snrs, (list, tuple)):
        snr = (snrs[0] - snrs[1]) * torch.rand(
            (mels.shape[0],), device=mels.device
        ).reshape(-1, 1, 1) + snrs[1]
    else:
        snr = snrs

    snr = 10 ** (snr / 20)  # linear domain
    sigma = torch.std(mels, dim=dims, keepdim=True) / snr
    mels = mels + torch.randn(mels.shape, device=mels.device) * sigma

    return mels


### FILTAUGMENT ###


def filt_aug_prototype(features, db_range=(-7.5, 0), n_bands=(2, 5)):
    # this is FilterAugment algorithm used for DCASE 2021 Challeng Task 4
    batch_size, n_freq_bin, _ = features.shape
    n_freq_band = torch.randint(
        low=n_bands[0], high=n_bands[1], size=(1,)
    ).item()  # [low, high)
    if n_freq_band > 1:
        band_bndry_freqs = torch.cat(
            (
                torch.tensor([0]),
                torch.sort(torch.randint(1, n_freq_bin - 1, (n_freq_band - 1,)))[0],
                torch.tensor([n_freq_bin]),
            )
        )
        band_factors = (
            torch.rand((batch_size, n_freq_band)).to(features)
            * (db_range[1] - db_range[0])
            + db_range[0]
        )
        band_factors = 10 ** (band_factors / 20)

        freq_filt = torch.ones((batch_size, n_freq_bin, 1)).to(features)
        for i in range(n_freq_band):
            freq_filt[:, band_bndry_freqs[i] : band_bndry_freqs[i + 1], :] = (
                band_factors[:, i].unsqueeze(-1).unsqueeze(-1)
            )
        return features * freq_filt
    else:
        return features


def freq_mask(features, mask_ratio=16):
    batch_size, n_freq_bin, _ = features.shape
    max_mask = int(n_freq_bin / mask_ratio)
    if max_mask == 1:
        f_widths = torch.ones(batch_size)
    else:
        f_widths = torch.randint(
            low=1, high=max_mask, size=(batch_size,)
        )  # [low, high)

    for i in range(batch_size):
        f_width = f_widths[i]
        f_low = torch.randint(low=0, high=n_freq_bin - f_width, size=(1,))

        features[i, f_low : f_low + f_width, :] = 0
    return features


def read_and_augment_audio(file_path: str, augmentations: List[Callable] = []) -> Audio:
    waveform, sample_rate = torchaudio.load(file_path)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    n_fft = 400
    win_length = n_fft
    hop_length = n_fft // 2
    n_mels = 128

    mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
    )

    mel_spectrogram = mel_spectrogram_transform(waveform)

    log_mel_spectrogram = torch.log(mel_spectrogram + 1e-6)

    augmented_log_mel_spectrogram = log_mel_spectrogram
    for augmentation in augmentations:
        augmented_log_mel_spectrogram = augmentation(augmented_log_mel_spectrogram)

    augmented_mel_spectrogram = torch.exp(augmented_log_mel_spectrogram) - 1e-6

    mel_to_linear_transform = torchaudio.transforms.InverseMelScale(
        n_stft=(n_fft // 2) + 1,
        n_mels=n_mels,
        sample_rate=sample_rate,
    )

    linear_spectrogram = mel_to_linear_transform(augmented_mel_spectrogram)

    griffin_lim_transform = torchaudio.transforms.GriffinLim(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        power=1,
    )

    augmented_waveform = griffin_lim_transform(linear_spectrogram)

    augmented_waveform = torch.clamp(augmented_waveform, min=-1.0, max=1.0)

    return Audio(augmented_waveform.numpy(), rate=sample_rate)


if __name__ == "__main__":
    audio_player = read_and_augment_audio(
        "/mnt/d/DESED_dataset/audio/train/strong_label_real_16k/Y_5wdyhlqBnM_0.000_10.000.wav",
        augmentations=[filt_aug_prototype, freq_mask],
    )

    audio_player
