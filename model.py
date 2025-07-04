import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from typing import Optional
from tqdm import tqdm


class DainDiffusionModel(nn.Module):
    def __init__(self, image_size=64, in_channels=3, out_channels=3, 
                 num_timesteps=1000, model_channels=128):
        super().__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_timesteps = num_timesteps
        self.model_channels = model_channels
        
        # Define the UNet architecture for diffusion
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, model_channels * 4),
            nn.SiLU(),
            nn.Linear(model_channels * 4, model_channels)
        )
        
        # Simplified UNet-like architecture
        self.conv1 = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)
        self.down1 = DownBlock(model_channels, model_channels * 2)
        self.down2 = DownBlock(model_channels * 2, model_channels * 4)
        
        self.mid = nn.Sequential(
            nn.Conv2d(model_channels * 4, model_channels * 4, kernel_size=3, padding=1),
            nn.GroupNorm(8, model_channels * 4),
            nn.SiLU(),
            nn.Conv2d(model_channels * 4, model_channels * 4, kernel_size=3, padding=1),
        )
        
        self.up1 = UpBlock(model_channels * 4, model_channels * 2)
        self.up2 = UpBlock(model_channels * 2, model_channels)
        self.out = nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1)
        
        # Timestep embedding
        self.timestep_embed = nn.Embedding(num_timesteps, model_channels)
        
    def forward(self, x, t):
        # Timestep embedding
        t_emb = self.timestep_embed(t)
        t_emb = self.time_embed(t_emb)
        
        # UNet forward pass
        h1 = self.conv1(x)
        h2 = self.down1(h1, t_emb)
        h3 = self.down2(h2, t_emb)
        
        h = self.mid(h3)
        
        h = self.up1(h, h2, t_emb)
        h = self.up2(h, h1, t_emb)
        h = self.out(h)
        
        return h


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
        self.pool = nn.AvgPool2d(2)
        
    def forward(self, x, t_emb):
        # Add time embedding as a bias
        x = self.conv(x)
        x = self.norm(x)
        x = x + t_emb[..., None, None]
        x = self.act(x)
        return self.pool(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
        
    def forward(self, x, skip, t_emb):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.norm(x)
        x = x + t_emb[..., None, None]
        x = self.act(x)
        return x


class DiffusionProcess:
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        
    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
            
        sqrt_alpha_bar = torch.sqrt(self.alpha_bars[t])[:, None, None, None]
        sqrt_one_minus_alpha_bar = torch.sqrt(1. - self.alpha_bars[t])[:, None, None, None]
        
        return sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
    
    def p_losses(self, model, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
            
        x_noisy = self.q_sample(x0, t, noise)
        predicted_noise = model(x_noisy, t)
        
        return F.mse_loss(noise, predicted_noise)
    
    @torch.no_grad()
    def p_sample(self, model, x, t, t_index):
        betas_t = self.betas[t]
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1. - self.alpha_bars[t])
        sqrt_alpha_bar_t = torch.sqrt(self.alpha_bars[t])
        
        # Predict noise using model
        pred_noise = model(x, t)
        
        # Calculate x0 and direction
        x0 = (x - sqrt_one_minus_alpha_bar_t * pred_noise) / sqrt_alpha_bar_t
        direction = torch.sqrt(betas_t) * pred_noise
        
        # Add noise if not last step
        if t_index > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)
            
        x_prev = x0 + direction + noise
        return x_prev
    
    @torch.no_grad()
    def sample(self, model, shape, device):
        # Start from pure noise
        img = torch.randn(shape, device=device)
        imgs = []
        
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling', total=self.num_timesteps):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            img = self.p_sample(model, img, t, i)
            imgs.append(img.cpu())
            
        return imgs
