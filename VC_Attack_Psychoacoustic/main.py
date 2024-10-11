import torch
import torchaudio
from torch.utils.data import DataLoader
from transformers import WavLMModel
from models import FrequencyDomainNoiseEncoder
from train import train_noise_encoder
import wandb
import os

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model saving path
save_dir = './checkpoints'
os.makedirs(save_dir, exist_ok=True)

# Hyperparameters
batch_size = 2
num_epochs = 10
learning_rate = 0.001
lambda_spec = 0
lambda_emb = 1

# Data loading
train_dataset = torchaudio.datasets.LIBRISPEECH(root='./data', url='train-clean-100', download=True)

def collate_fn(batch):
    waveforms = []
    sample_rate = 44100  # Sample rate of LibriSpeech
    max_length = max(waveform.shape[1] for waveform, _, _, _, _, _ in batch)
    for waveform, _, _, _, _, _ in batch:
        if waveform.shape[1] < max_length:
            padding = torch.zeros(1, max_length - waveform.shape[1])
            waveform = torch.cat([waveform, padding], dim=1)
        waveforms.append(waveform)
    return torch.stack(waveforms)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Load pre-trained WavLM model
wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large").to(device)

# Initialize the FrequencyDomainNoiseEncoder model
noise_encoder = FrequencyDomainNoiseEncoder().to(device)

# Optimizer
optimizer = torch.optim.Adam(noise_encoder.parameters(), lr=learning_rate)

# Initialize wandb
wandb.init(project="VcAttack")

# Train the model
spec_losses, emb_losses, total_losses = train_noise_encoder(
    noise_encoder,
    wavlm,
    train_loader,
    optimizer,
    num_epochs,
    device,
    lambda_spec=lambda_spec,
    lambda_emb=lambda_emb
)

print("Training completed!")

# Save the trained model
model_save_path = os.path.join(save_dir, 'frequency_domain_noise_encoder.pth')
torch.save(noise_encoder.state_dict(), model_save_path)
print(f"Model saved at {model_save_path}")

# End wandb session
wandb.finish()
