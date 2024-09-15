import torch
import torch.nn.functional as F
from dataloader import audio_to_melspectrogram
from noise_generator import UNET1D
from FreeVC.speaker_encoder.voice_encoder import SpeakerEncoder
import librosa
import soundfile as sf
import numpy as np

# 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
freevc_encoder = SpeakerEncoder('FreeVC/speaker_encoder/ckpt/pretrained_bak_5805000.pt').to(device)
noise_generator = UNET1D().to(device)

# 가중치 로드
noise_generator.load_state_dict(torch.load('pretrain/noise_generator.pth'))
noise_generator.eval()

# Mel spectrogram 생성
n_fft = 2048
hop_length = 512
n_mels = 120

def inference(input_audio_path, output_audio_path):
    y, sr = librosa.load(input_audio_path, sr=None)

    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # 텐서로 변환 및 노이즈 생성
    mel_spectrogram = torch.tensor(mel_spectrogram_db, dtype=torch.float32).unsqueeze(0).to(device)

    #with torch.no_grad():
        #noise = noise_generator(mel_spectrogram)

    # Mel-spectrogram에 노이즈 추가
    noisy_mel_spectrogram = mel_spectrogram# + noise  # 적절한 노이즈 스케일링

    # 데시벨에서 파워 스펙트로그램으로 변환
    noisy_mel_power = librosa.db_to_power(noisy_mel_spectrogram.squeeze(0).cpu().numpy())

    # Griffin-Lim을 사용한 음성 복원
    noisy_waveform = librosa.feature.inverse.mel_to_audio(
        noisy_mel_power, sr=sr, n_iter=1000, hop_length=hop_length, n_fft=n_fft
    )

    # 결과 음성을 정규화 및 스케일링
    noisy_waveform = librosa.util.normalize(noisy_waveform) * 1.0

    # 음성 파일 저장
    sf.write(output_audio_path, noisy_waveform, sr)
    print(f"Noisy audio saved at: {output_audio_path}")

if __name__ == "__main__":
    input_audio_path = '../Data/LJSpeech/wavs/LJ001-0001.wav'
    output_audio_path = 'result/output_noisy_audio.wav'
    inference(input_audio_path, output_audio_path)
