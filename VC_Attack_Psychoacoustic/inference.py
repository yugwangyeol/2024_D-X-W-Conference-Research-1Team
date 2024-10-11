import torch
import torchaudio
import os
from models import FrequencyDomainNoiseEncoder, SpectrogramConverter

# 경로 설정
wav_name = "contents"
model_path = "checkpoints/frequency_domain_noise_encoder.pth"
input_wav_path = f"../Data/{wav_name}.wav"
output_wav_dir = "../output"
output_wav_path = os.path.join(output_wav_dir, f"noisy_{wav_name}_psy.wav")

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. FrequencyDomainNoiseEncoder 모델 불러오기
noise_encoder = FrequencyDomainNoiseEncoder().to(device)
noise_encoder.load_state_dict(torch.load(model_path, map_location=device))
noise_encoder.eval()

# SpectrogramConverter 초기화
spectrogram_converter = SpectrogramConverter().to(device)

# 2. 입력 wav 파일 로드
waveform, sample_rate = torchaudio.load(input_wav_path)

# 배치 차원 추가 (모델에 넣기 위해)
waveform = waveform.unsqueeze(0).to(device)  # [1, num_channels, num_samples]

# 3. 스펙트로그램으로 변환
spectrogram = spectrogram_converter.waveform_to_spectrogram(waveform.squeeze(1))
spectrogram = spectrogram.unsqueeze(1)  # [batch_size, 1, 2, freq_bins, time_steps]

# 4. 노이즈 적용
with torch.no_grad():
    noise_spectrogram = noise_encoder(spectrogram)
    noisy_spectrogram = spectrogram + noise_spectrogram

# 5. 스펙트로그램을 다시 파형으로 변환
noisy_waveform = spectrogram_converter.spectrogram_to_waveform(noisy_spectrogram.squeeze(1))
noisy_waveform = noisy_waveform.unsqueeze(1)  # [batch_size, 1, num_samples]

# 클램핑 적용 (선택사항)
noisy_waveform = torch.clamp(noisy_waveform, -1, 1)

# 6. 노이즈가 적용된 wav 파일 저장
if not os.path.exists(output_wav_dir):
    os.makedirs(output_wav_dir)

torchaudio.save(output_wav_path, noisy_waveform.squeeze(0).cpu(), sample_rate)

print(f"노이즈가 적용된 파일이 {output_wav_path}에 저장되었습니다.")