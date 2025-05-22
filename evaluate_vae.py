import torch
from models.vae import VAE
from utils.data import load_data, get_loader, load_cond, load_cond_data, get_cond_loader
from models.diffusion_latent import SimpleMLPUNet, LatentDiffusion
from models.diffusion_prior import DiffusionPriorNet
import os
from models.expression_decoder import ExpressionDecoder
import pandas as pd

DATA_DIR = 'dataset/iclr_final_truncated_fixed_powers.h5'
BATCH_SIZE = 64
SEED = 42
CSV_PATH = 'dataset/tsoulos_dataset_1.csv'
OUTPUT_DIR = './reconstructed_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_data = load_data(DATA_DIR, split="test", random_seed=SEED)
test_loader = get_loader(test_data, batch_size=BATCH_SIZE)

test_cond_data, test_indices = load_cond_data(DATA_DIR, split="test", random_seed=SEED)
test_cond = load_cond(CSV_PATH, test_indices)
test_cond_loader = get_cond_loader(test_cond_data, test_cond, batch_size=BATCH_SIZE)

df = pd.read_csv(CSV_PATH)
original_exprs = df.loc[test_indices, 'expr'].tolist()

vae = VAE(seq_len=72, feature_dim=53).to(device)
vae.load_state_dict(torch.load("trained_models/best_vae.pth", weights_only=True))
vae.eval()

diff_model = SimpleMLPUNet(latent_dim=vae.latent_dim).to(device)
diff_model.load_state_dict(torch.load("trained_models/best_diffusion_latent.pth", weights_only=True))
diff_model.eval()
diffusion = LatentDiffusion(latent_dim=vae.latent_dim, timesteps=1000, device=device)

prior_net = DiffusionPriorNet(cond_dim=test_cond.shape[-1], latent_dim=vae.latent_dim).to(device)
prior_net.load_state_dict(torch.load("trained_models/best_diffusion_prior.pth", weights_only=True))
prior_net.eval()

expression_decoder = ExpressionDecoder(
    checkpoint_path="trained_models/best_vae.pth",
    device=device,
    model=VAE(seq_len=72, feature_dim=53, hidden_dim=200, latent_dim=200)
)

all_mse = []
all_exprs = []
all_original_exprs = []
expr_idx = 0

with torch.no_grad():
    for batch in test_loader:
        x = batch[0].to(device)  # (batch, seq_len, rule_dim)
        batch_size = x.size(0)
        x_recon, mu, logvar = vae(x)  # x_recon: (batch, seq_len, rule_dim)
        recon_prob = torch.softmax(x_recon, dim=-1)
        mse = ((recon_prob - x) ** 2).mean().item()
        all_mse.append(mse)
        for i in range(batch_size):
            expr = expression_decoder.construct_expression(x_recon[i].unsqueeze(0))  # 或 recon_prob[i].unsqueeze(0)
            all_exprs.append(expr)
            orig_expr = original_exprs[expr_idx] if expr_idx < len(original_exprs) else "N/A"
            all_original_exprs.append(orig_expr)
            expr_idx += 1

vae_recon_mse = sum(all_mse) / len(all_mse)
output_path = os.path.join(OUTPUT_DIR, 'vae_reconstruction.txt')
with open(output_path, 'w', encoding='utf-8') as f:
    for orig, recon in zip(all_original_exprs, all_exprs):
        f.write(f"Original: {orig}\nReconstructed: {recon}\n\n")
print(f"MSE of VAE: {vae_recon_mse:.4f}")
print(f"Reconstructed expressions saved to: {output_path}")


all_mse = []
all_exprs = []
all_original_exprs = []
expr_idx = 0

with torch.no_grad():
    for batch in test_loader:
        x = batch[0].to(device)
        batch_size = x.size(0)
        mu, logvar = vae.encoder(x)
        z = mu
        t = diffusion.sample_timesteps(batch_size)
        z_noisy = diffusion.q_sample(z, t)
        z_pred = diffusion.p_sample(diff_model, z_noisy, t)
        x_recon = vae.decoder(z_pred)
        recon_prob = torch.softmax(x_recon, dim=-1)
        mse = ((recon_prob - x) ** 2).mean().item()
        all_mse.append(mse)
        for i in range(batch_size):
            expr = expression_decoder.construct_expression(x_recon[i].unsqueeze(0))
            all_exprs.append(expr)
            orig_expr = original_exprs[expr_idx] if expr_idx < len(original_exprs) else "N/A"
            all_original_exprs.append(orig_expr)
            expr_idx += 1

diff_recon_mse = sum(all_mse) / len(all_mse)
output_path = os.path.join(OUTPUT_DIR, 'vae_diffusion_latent_reconstruction.txt')
with open(output_path, 'w', encoding='utf-8') as f:
    for orig, recon in zip(all_original_exprs, all_exprs):
        f.write(f"Original: {orig}\nReconstructed: {recon}\n\n")
print(f"MSE of Diffusion on latent: {diff_recon_mse:.4f}")
print(f"Reconstructed expressions saved to: {output_path}")


all_mse = []
all_exprs = []
all_original_exprs = []
expr_idx = 0

with torch.no_grad():
    for batch_data, batch_cond in test_cond_loader:
        x = batch_data.to(device)
        cond = batch_cond.to(device)
        batch_size = x.shape[0]
        t_prior = torch.zeros(batch_size, device=device)
        z_noisy = torch.zeros(batch_size, vae.latent_dim, device=device)
        z_prior = prior_net(z_noisy, cond, t_prior)
        t = diffusion.sample_timesteps(batch_size)
        z_noisy = diffusion.q_sample(z_prior, t)
        z_pred = diffusion.p_sample(diff_model, z_noisy, t)
        x_recon = vae.decoder(z_pred)
        recon_prob = torch.softmax(x_recon, dim=-1)
        mse = ((recon_prob - x) ** 2).mean().item()
        all_mse.append(mse)
        for i in range(batch_size):
            expr = expression_decoder.construct_expression(x_recon[i].unsqueeze(0))
            all_exprs.append(expr)
            orig_expr = original_exprs[expr_idx] if expr_idx < len(original_exprs) else "N/A"
            all_original_exprs.append(orig_expr)
            expr_idx += 1

prior_recon_mse = sum(all_mse) / len(all_mse)
output_path = os.path.join(OUTPUT_DIR, 'vae_diffusion_prior_reconstruction.txt')
with open(output_path, 'w', encoding='utf-8') as f:
    for orig, recon in zip(all_original_exprs, all_exprs):
        f.write(f"Original: {orig}\nReconstructed: {recon}\n\n")
print(f"MSE of Diffusion Prior: {prior_recon_mse:.4f}")
print(f"Reconstructed expressions saved to: {output_path}")

