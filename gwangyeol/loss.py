import torch
import torch.nn.functional as F

def cosine_loss(embedding_noisy, embedding_original):

    cosine_similarity = F.cosine_similarity(embedding_noisy, embedding_original, dim=-1)
    loss = 1 + cosine_similarity.mean()
    
    return loss

def loss_function(encoder, x, noise, mse_lambda_weight=0.5):
    x_noisy = x + noise
    x = x.transpose(1, 2) 
    x_noisy = x_noisy.transpose(1, 2) 

    embedding_original = encoder(x)
    embedding_noisy = encoder(x_noisy)

    loss_embedding_difference = cosine_loss(embedding_noisy, embedding_original)

    loss_signal_similarity = F.mse_loss(x_noisy, x)

    total_loss = loss_embedding_difference + mse_lambda_weight * loss_signal_similarity

    return total_loss, loss_embedding_difference, loss_signal_similarity
