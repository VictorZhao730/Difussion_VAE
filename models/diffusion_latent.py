import torch
import torch.nn as nn

class SimpleMLPUNet(nn.Module):
    def __init__(self, latent_dim=200, time_emb_dim=32):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        self.net = nn.Sequential(
            nn.Linear(latent_dim + time_emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
    def forward(self, z, t):
        # z: (B, latent_dim), t: (B,)
        t = t.float().unsqueeze(-1) / 1000  # 归一化
        t_emb = self.time_mlp(t)
        x = torch.cat([z, t_emb], dim=-1)
        return self.net(x)

class LatentDiffusion:
    @ staticmethod
    def linear_beta_schedule(timesteps):
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, timesteps)
    
    def __init__(self, latent_dim, timesteps=1000, device='cpu'):
        self.timesteps = timesteps
        self.device = device
        self.betas = self.linear_beta_schedule(timesteps).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.latent_dim = latent_dim

    def q_sample(self, z0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(z0)
        sqrt_alphas_cumprod = self.alphas_cumprod[t].sqrt().unsqueeze(-1)
        sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod[t]).sqrt().unsqueeze(-1)
        return sqrt_alphas_cumprod * z0 + sqrt_one_minus_alphas_cumprod * noise

    def sample_timesteps(self, batch_size):
        return torch.randint(0, self.timesteps, (batch_size,), device=self.device)
    
    def p_sample(self, model, z_noisy, t):
            pred_noise = model(z_noisy, t)
            
            betas_t = self.betas[t].unsqueeze(-1)                # (B, 1)
            sqrt_one_over_alpha_t = (1.0 / self.alphas[t]).sqrt().unsqueeze(-1)  # (B, 1)
            sqrt_inv_cumprod_t = (1.0 / self.alphas_cumprod[t].sqrt()).unsqueeze(-1)  # (B, 1)
            sqrt_one_minus_cumprod_t = (1 - self.alphas_cumprod[t]).sqrt().unsqueeze(-1)  # (B, 1)
            
            z0_pred = (z_noisy - sqrt_one_minus_cumprod_t * pred_noise) / sqrt_inv_cumprod_t
            
            coef1 = (
                self.betas[t].unsqueeze(-1) * self.alphas_cumprod[t-1].unsqueeze(-1).clamp(min=0)
                / (1 - self.alphas_cumprod[t]).unsqueeze(-1)
            )
            coef2 = (
                (self.alphas[t].unsqueeze(-1) ** 0.5) * (1 - self.alphas_cumprod[t-1].unsqueeze(-1).clamp(min=0))
                / (1 - self.alphas_cumprod[t]).unsqueeze(-1)
            )
            mean = coef1 * z0_pred + coef2 * z_noisy
            
            # No noise if t=0
            noise = torch.randn_like(z_noisy) if (t > 0).any() else torch.zeros_like(z_noisy)
            # variation
            var = self.betas[t].unsqueeze(-1)
            # sampling
            z_prev = mean + var.sqrt() * noise
            return z_prev