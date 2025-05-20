import torch
import torch.nn as nn

class DiffusionPriorNet(nn.Module):
    def __init__(self, cond_dim, latent_dim, time_emb_dim=32):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        self.net = nn.Sequential(
            nn.Linear(latent_dim + cond_dim + time_emb_dim, 128),  # 修正这里
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

    def forward(self, z_noisy, cond, t):
        # z_noisy: (B, latent_dim)
        # cond: (B, cond_dim)
        # t: (B,)
        t = t.float().unsqueeze(-1) / 1000
        t_emb = self.time_mlp(t)
        x = torch.cat([z_noisy, cond, t_emb], dim=-1)
        return self.net(x)
    
class DiffusionPrior:
    @staticmethod
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

    def p_sample(self, model, cond, z_noisy, t):

        pred_noise = model(cond, t)
        sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod[t]).sqrt().unsqueeze(-1)
        sqrt_alphas_cumprod = self.alphas_cumprod[t].sqrt().unsqueeze(-1)
        z0_pred = (z_noisy - sqrt_one_minus_alphas_cumprod * pred_noise) / sqrt_alphas_cumprod

        # sample next z
        noise = torch.randn_like(z_noisy) if (t > 0).any() else torch.zeros_like(z_noisy)
        beta_t = self.betas[t].unsqueeze(-1)
        mean = sqrt_alphas_cumprod * z0_pred + sqrt_one_minus_alphas_cumprod * noise
        return mean
