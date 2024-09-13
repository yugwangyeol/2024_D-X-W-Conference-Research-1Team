from FreeVC.speaker_encoder.voice_encoder import SpeakerEncoder

from loss import loss_function
from dataloader import get_dataloader
from noise_generator import UNET1D

import wandb
import torch
import torch.optim as optim
from tqdm import tqdm 

# wadnb
run = wandb.init(project="VcAttack")

wandb.config = {
    "num_epochs": 50,
    "train_lr": 1e-4,
    "batch_size": 8,
    "mse_lambda_weight": 0.3,
    "embedding_lambda_weight" : 0.7,
    "speakers_per_batch":64,
    "utterances_per_speaker":10
}

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 경로 설정
root_dir = '../Data/kss/'  # 데이터셋 경로
train_loader = get_dataloader(root_dir=root_dir, batch_size=32)

# model load
freevc_encoder = SpeakerEncoder('FreeVC/speaker_encoder/ckpt/pretrained_bak_5805000.pt').to(device)
noise_generator = UNET1D().to(device)  # 1채널로 가정 (맬 스펙트로그램은 흑백 이미지)

# Optimizer 설정
optimizer = optim.Adam(noise_generator.parameters(), lr=wandb.config["train_lr"])

for epoch in range(wandb.config["num_epochs"]):
    noise_generator.train()
    running_loss = 0.0
    running_embedding_loss = 0.0
    running_mse_loss = 0.0

    for batch in tqdm(train_loader):
        inputs = batch.data.to(device)

        # 노이즈 생성
        noise = noise_generator(inputs)
        
        # 손실 함수 계산
        loss, embedding_loss, mse_loss = loss_function(freevc_encoder, inputs, noise, 
                            mse_lambda_weight=wandb.config["mse_lambda_weight"],
                            embedding_lambda_weight=wandb.config["embedding_lambda_weight"])

        # 역전파 및 최적화
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_embedding_loss += embedding_loss.item()
        running_mse_loss += mse_loss.item()

    # 로그 기록
    # Log the losses together in a single dictionary
    wandb.log({
        "loss": running_loss / len(train_loader),
        "embedding loss": running_embedding_loss / len(train_loader),
        "mse loss": running_mse_loss / len(train_loader)
    })


# 모델 저장
torch.save(noise_generator.state_dict(), 'noise_generator.pth')
print("Model saved!")


