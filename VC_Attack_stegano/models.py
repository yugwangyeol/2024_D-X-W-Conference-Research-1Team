import torch
import torch.nn as nn
import torch.nn.functional as F

class FrequencyDomainNoiseEncoder(nn.Module):
    def __init__(self, n_fft=2048, freq_bins=1025):
        super(FrequencyDomainNoiseEncoder, self).__init__()
        self.n_fft = n_fft
        self.freq_bins = freq_bins

        # 인코더 (주파수 도메인)
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # 디코더 (주파수 도메인)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 2, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # x의 형태: (batch_size, num_channels, 2, freq_bins, time_steps)
        batch_size, num_channels, _, freq_bins, time_steps = x.shape
        x = x.view(batch_size * num_channels, 2, freq_bins, time_steps)

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        # 20kHz 이상 주파수에만 노이즈 적용
        high_freq_mask = torch.zeros_like(decoded)
        high_freq_mask[:, :, int(self.freq_bins * 15000 / (self.n_fft // 2)):, :] = 1
        noise = decoded * high_freq_mask

        # 원래 형태로 복원
        noise = noise.view(batch_size, num_channels, 2, freq_bins, time_steps)
        return noise

class SpectrogramConverter(nn.Module):
    def __init__(self, n_fft=2048, hop_length=512, sr=44100):
        super(SpectrogramConverter, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sr = sr

    def waveform_to_spectrogram(self, waveform):
        stft = torch.stft(waveform, n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True)
        spect = torch.stack([stft.real, stft.imag], dim=1)
        return spect

    def spectrogram_to_waveform(self, spect):
        complex_spect = torch.complex(spect[:, 0], spect[:, 1])
        waveform = torch.istft(complex_spect, n_fft=self.n_fft, hop_length=self.hop_length)
        return waveform

    def forward(self, x):
        # 이 메서드는 실제로 사용되지 않지만, nn.Module을 상속받기 위해 필요합니다.
        return x