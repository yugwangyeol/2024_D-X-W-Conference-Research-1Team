import os
import torch
import librosa
import numpy as np

from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 오디오를 멜 스펙트로그램으로 변환하는 함수
def audio_to_melspectrogram(wav, sr=16000, n_mels=40, n_fft=1024, hop_length=256):
    mel_spectrogram = librosa.feature.melspectrogram(
        y=wav, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return mel_spectrogram_db

# KSS Dataset 클래스
class KSSDataset(Dataset):
    def __init__(self, root_dir, sr=16000, n_mels=40, transform=None, max_len=512):
        self.root_dir = root_dir  # KSS 데이터셋의 상위 폴더 경로
        self.sr = sr  # 샘플링 레이트
        self.n_mels = n_mels  # 멜 채널 수 (일반적으로 80)
        self.transform = transform  # 오디오를 멜 스펙트로그램으로 변환하는 함수
        self.max_len = max_len  # 고정된 시퀀스 길이
        
        # 각 폴더 내의 wav 파일 경로를 수집
        self.wav_files = []
        for subdir in os.listdir(root_dir):
            subdir_path = os.path.join(root_dir, subdir)
            if os.path.isdir(subdir_path):
                for file in os.listdir(subdir_path):
                    if file.endswith(".wav"):
                        self.wav_files.append(os.path.join(subdir_path, file))

    def __len__(self):
        return len(self.wav_files)

    def __getitem__(self, idx):
        wav_path = self.wav_files[idx]
        
        # 오디오 파일 로드
        wav, _ = librosa.load(wav_path, sr=self.sr)
        
        # 멜 스펙트로그램 변환
        if self.transform:
            mel_spectrogram = self.transform(wav)
        else:
            mel_spectrogram = audio_to_melspectrogram(wav, sr=self.sr, n_mels=self.n_mels)

        # 시퀀스 길이를 self.max_len으로 고정
        mel_spectrogram = torch.tensor(mel_spectrogram, dtype=torch.float32)
        if mel_spectrogram.shape[1] > self.max_len:
            mel_spectrogram = mel_spectrogram[:, :self.max_len]  # 고정된 길이로 자르기
        else:
            pad_size = self.max_len - mel_spectrogram.shape[1]
            mel_spectrogram = F.pad(mel_spectrogram, (0, pad_size))  # 패딩 추가

        return mel_spectrogram

# DataLoader 함수
def get_dataloader(root_dir, batch_size=32, num_workers=8, max_len=512):
    dataset = KSSDataset(root_dir=root_dir, transform=audio_to_melspectrogram, max_len=max_len)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    
    return dataloader
