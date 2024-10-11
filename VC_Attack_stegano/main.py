import torch
import torchaudio
from torch.utils.data import DataLoader

<<<<<<< HEAD:VC_Attack_stegano/main.py
from transformers import Wav2Vec2Model
from transformers import WavLMModel

from models import FrequencyDomainNoiseEncoder, SpectrogramConverter
=======

from transformers import WavLMModel

from models import NoiseEncoder
>>>>>>> fbce39285668345816cf80af92f653f02aca6a21:junhyeok/main.py
from train import train_noise_encoder
import wandb
import os

# 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 저장 경로 설정
save_dir = './checkpoints'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 하이퍼파라미터
batch_size = 2
num_epochs = 10
learning_rate = 0.001
lambda_spec = 0
lambda_emb = 1

# 데이터 로드
train_dataset = torchaudio.datasets.LIBRISPEECH(root='./data', url='train-clean-100', download=True)

def collate_fn(batch):
    waveforms = []
    sample_rate = 44100  # LibriSpeech의 샘플레이트
    max_length = max(waveform.shape[1] for waveform, _, _, _, _, _ in batch)
    for waveform, _, _, _, _, _ in batch:
        if waveform.shape[1] < max_length:
            padding = torch.zeros(1, max_length - waveform.shape[1])
            waveform = torch.cat([waveform, padding], dim=1)
        waveforms.append(waveform)
    return torch.stack(waveforms)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

<<<<<<< HEAD:VC_Attack_stegano/main.py
# Wav2Vec2 모델 초기화
#wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/hubert-large-ls960-ft").to(device)
wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large").to(device)
=======
# HuBERT 모델과 프로세서 초기화
# processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")


# WavLM 모델을 불러와서 장치에 올리는 코드
wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large").to(device)

>>>>>>> fbce39285668345816cf80af92f653f02aca6a21:junhyeok/main.py

# FrequencyDomainNoiseEncoder 모델 초기화
noise_encoder = FrequencyDomainNoiseEncoder().to(device)

# 옵티마이저 설정
optimizer = torch.optim.Adam(noise_encoder.parameters(), lr=learning_rate)

# wandb 초기화
wandb.init(project="VcAttack")

# 학습 실행
<<<<<<< HEAD:VC_Attack_stegano/main.py
spec_losses, emb_losses, total_losses = train_noise_encoder(noise_encoder,
                                                            wavlm,
=======
img_losses, emb_losses, total_losses = train_noise_encoder(noise_encoder,

                                                            wavlm,

>>>>>>> fbce39285668345816cf80af92f653f02aca6a21:junhyeok/main.py
                                                            train_loader, 
                                                            optimizer, 
                                                            num_epochs, 
                                                            device, 
                                                            lambda_spec=lambda_spec, 
                                                            lambda_emb=lambda_emb)

# 학습 완료 메시지 출력
print("학습 완료!")

model_save_path = os.path.join(save_dir, 'frequency_domain_noise_encoder.pth')
torch.save(noise_encoder.state_dict(), model_save_path)

print(f"모델이 {model_save_path} 경로에 저장되었습니다.")

# wandb 종료
wandb.finish()