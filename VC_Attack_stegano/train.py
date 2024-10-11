import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import Wav2Vec2Model
import wandb
from models import SpectrogramConverter

def cosine_similarity_loss(x, y):
    return 1 - F.cosine_similarity(x, y).mean()

def adjust_tensor_size(x, y):
    min_length = min(x.shape[1], y.shape[1])
    return x[:, :min_length, :], y[:, :min_length, :]

def train_noise_encoder(noise_encoder, wav2vec2, dataloader, optimizer, num_epochs, device, lambda_spec=0.95, lambda_emb=0.05):
    noise_encoder.train()
    spectrogram_converter = SpectrogramConverter().to(device)

    for epoch in range(num_epochs):
        epoch_spec_loss = 0.0
        epoch_emb_loss = 0.0
        epoch_total_loss = 0.0

        for waveforms in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            waveforms = waveforms.to(device)

            # 스펙트로그램으로 변환
            spectrograms = spectrogram_converter.waveform_to_spectrogram(waveforms.squeeze(1))
            spectrograms = spectrograms.unsqueeze(1)  # (batch_size, 1, 2, freq_bins, time_steps)

            # NoiseEncoder 통과
            noise_spectrograms = noise_encoder(spectrograms)
            noisy_spectrograms = spectrograms + noise_spectrograms

            # 스펙트로그램을 다시 파형으로 변환
            noisy_waveforms = spectrogram_converter.spectrogram_to_waveform(noisy_spectrograms.squeeze(1))
            noisy_waveforms = noisy_waveforms.unsqueeze(1)  # (batch_size, 1, time_steps)

            # Wav2Vec 2.0 모델을 사용하여 임베딩 계산
            with torch.no_grad():
                wav2vec2_original_output = wav2vec2(waveforms.squeeze(1)).last_hidden_state
                wav2vec2_noisy_output = wav2vec2(noisy_waveforms.squeeze(1)).last_hidden_state

            # Loss 계산
            wav2vec2_original_output, wav2vec2_noisy_output = adjust_tensor_size(wav2vec2_original_output, wav2vec2_noisy_output)

            spec_loss = F.mse_loss(noisy_spectrograms, spectrograms)
            emb_loss = cosine_similarity_loss(wav2vec2_noisy_output, wav2vec2_original_output)
            total_loss = lambda_spec * spec_loss + lambda_emb * emb_loss

            # 역전파 및 최적화
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # 손실 값 저장
            epoch_spec_loss += spec_loss.item()
            epoch_emb_loss += emb_loss.item()
            epoch_total_loss += total_loss.item()

        # wandb 로그 기록
        wandb.log({
            "epoch": epoch + 1,
            "Spectrogram Loss": epoch_spec_loss / len(dataloader),
            "Embedding Loss": epoch_emb_loss / len(dataloader),
            "Total Loss": epoch_total_loss / len(dataloader),
        })

    return epoch_spec_loss / len(dataloader), epoch_emb_loss / len(dataloader), epoch_total_loss / len(dataloader)