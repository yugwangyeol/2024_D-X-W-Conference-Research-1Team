import torch
import torch.nn as nn
import torch.nn.functional as F

class FrequencyDomainNoiseEncoder(nn.Module):
    def __init__(self, n_fft=2048, freq_bins=1025):
        super(FrequencyDomainNoiseEncoder, self).__init__()
        self.n_fft = n_fft
        self.freq_bins = freq_bins

        # Frequency domain encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Frequency domain decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 2, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        batch_size, num_channels, _, freq_bins, time_steps = x.shape
        x = x.view(batch_size * num_channels, 2, freq_bins, time_steps)

        # Encode and decode
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        # Apply psychoacoustic masking
        psychoacoustic_mask = self.create_psychoacoustic_mask(freq_bins).to(decoded.device)  # Mask 텐서를 'decoded'와 동일한 장치로 이동
        noise = decoded * psychoacoustic_mask

        # Reshape back to original format
        noise = noise.view(batch_size, num_channels, 2, freq_bins, time_steps)
        return noise

    def create_psychoacoustic_mask(self, freq_bins):
        # Create a mask with weights for specific frequency bands
        mask = torch.ones((2, freq_bins, 1))  # Initialize mask
        high_freq_range_start = int(self.freq_bins * 8000 / (self.n_fft // 2))
        high_freq_range_end = int(self.freq_bins * 15000 / (self.n_fft // 2))
        mask[:, high_freq_range_start:high_freq_range_end, :] = 0.5  # Reduce noise in certain bands
        return mask


class SpectrogramConverter(nn.Module):
    def __init__(self, n_fft=2048, hop_length=512, sr=44100):
        super(SpectrogramConverter, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sr = sr

    def waveform_to_spectrogram(self, waveform):
        # Convert waveform to spectrogram
        stft = torch.stft(waveform, n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True)
        spect = torch.stack([stft.real, stft.imag], dim=1)
        return spect

    def spectrogram_to_waveform(self, spect):
        # Convert spectrogram back to waveform
        complex_spect = torch.complex(spect[:, 0], spect[:, 1])
        waveform = torch.istft(complex_spect, n_fft=self.n_fft, hop_length=self.hop_length)
        return waveform

    def forward(self, x):
        return x
