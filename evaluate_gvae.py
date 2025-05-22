import os
import torch
from torch.utils.data import DataLoader
import pandas as pd
from models.gvae import GVAE, MaskFn
from utils.data import load_data, get_loader, load_cond, load_cond_data, get_cond_loader
from models.diffusion_latent import SimpleMLPUNet, LatentDiffusion
from models.diffusion_prior import DiffusionPriorNet
from dataset.grammar_alltogether import GCFG
from models.expression_decoder import ExpressionDecoder

DATA_DIR = 'dataset/iclr_final_truncated_fixed_powers.h5'
BATCH_SIZE = 128
SEED = 42
CSV_PATH = 'dataset/tsoulos_dataset_1.csv'
OUTPUT_DIR = './reconstructed_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

productions = GCFG.productions()
start_symbol = GCFG.start()
mask_fn = MaskFn(productions, start_symbol)

test_data = load_data(DATA_DIR, split="test", random_seed=SEED)
test_loader = get_loader(test_data, batch_size=BATCH_SIZE)

test_cond_data, test_indices = load_cond_data(DATA_DIR, split="test", random_seed=SEED)
test_cond = load_cond(CSV_PATH, test_indices)
test_cond_loader = get_cond_loader(test_cond_data, test_cond, batch_size=BATCH_SIZE)

df = pd.read_csv(CSV_PATH)
original_exprs = df.loc[test_indices, 'expr'].tolist()

gvae = GVAE(seq_len=72, rule_dim=len(productions), mask_fn=mask_fn, device=device).to(device)
gvae.load_state_dict(torch.load("trained_models/best_gvae.pth", map_location=device, weights_only=True))
gvae.eval()

diff_model = SimpleMLPUNet(latent_dim=gvae.latent_dim).to(device)
diff_model.load_state_dict(torch.load("trained_models/best_diffusion_latent.pth", map_location=device, weights_only=True))
diff_model.eval()
diffusion = LatentDiffusion(latent_dim=gvae.latent_dim, timesteps=1000, device=device)

prior_net = DiffusionPriorNet(latent_dim=gvae.latent_dim, cond_dim=test_cond.shape[-1]).to(device)
prior_net.load_state_dict(torch.load("trained_models/best_diffusion_prior.pth", map_location=device, weights_only=True))
prior_net.eval()

start_token = torch.zeros(len(productions), device=device)
start_token[0] = 1

expression_decoder = ExpressionDecoder(
    checkpoint_path="trained_models/best_gvae.pth",
    device=device,
    model=GVAE(seq_len=72, rule_dim=len(productions), mask_fn=mask_fn, device=device,
               latent_dim=200, hidden_dim=200),
)

# 1. GVAE
all_mse = []
all_exprs = []
all_original_exprs = []
expr_idx = 0

with torch.no_grad():
    for batch in test_loader:
        x = batch[0].to(device)  # (batch, seq_len, rule_dim)
        batch_size = x.size(0)
        batch_stacks = mask_fn.init_stack(batch_size)
        recon_logits, mu, logvar = gvae(x, start_token, batch_stacks)
        recon_prob = torch.softmax(recon_logits, dim=-1)
        mse = ((recon_prob - x) ** 2).mean().item()
        all_mse.append(mse)
        for i in range(batch_size):
            expr = expression_decoder.construct_expression(recon_logits[i].unsqueeze(0))
            all_exprs.append(expr)
            all_original_exprs.append(original_exprs[expr_idx])
            expr_idx += 1

gvae_recon_mse = sum(all_mse) / len(all_mse)
output_path = os.path.join(OUTPUT_DIR, 'gvae_reconstruction.txt')
with open(output_path, 'w', encoding='utf-8') as f:
    for orig, recon in zip(all_original_exprs, all_exprs):
        f.write(f"Original: {orig}\nReconstructed: {recon}\n\n")
print(f"MSE of GVAE: {gvae_recon_mse:.4f}")
print(f"Reconstructed expressions saved to: {output_path}")

# 2. Diffusion on latent
all_mse = []
all_exprs = []
all_original_exprs = []
expr_idx = 0

with torch.no_grad():
    for batch in test_loader:
        x = batch[0].to(device)
        batch_size = x.size(0)
        mu, logvar = gvae.encoder(x)
        z = mu
        t = diffusion.sample_timesteps(batch_size)
        z_noisy = diffusion.q_sample(z, t)
        z_pred = diffusion.p_sample(diff_model, z_noisy, t)
        batch_stacks = mask_fn.init_stack(batch_size)
        recon_logits = gvae.decoder(z_pred, start_token, batch_stacks)
        recon_prob = torch.softmax(recon_logits, dim=-1)
        mse = ((recon_prob - x) ** 2).mean().item()
        all_mse.append(mse)
        for i in range(batch_size):
            expr = expression_decoder.construct_expression(recon_logits[i].unsqueeze(0))
            all_exprs.append(expr)
            all_original_exprs.append(original_exprs[expr_idx])
            expr_idx += 1

diff_recon_mse = sum(all_mse) / len(all_mse)
output_path = os.path.join(OUTPUT_DIR, 'diffusion_reconstruction.txt')
with open(output_path, 'w', encoding='utf-8') as f:
    for orig, recon in zip(all_original_exprs, all_exprs):
        f.write(f"Original: {orig}\nReconstructed: {recon}\n\n")
print(f"MSE of Diffusion on latent: {diff_recon_mse:.4f}")
print(f"Reconstructed expressions saved to: {output_path}")

# 3. Diffusion Prior
all_mse = []
all_exprs = []
all_original_exprs = []
expr_idx = 0

with torch.no_grad():
    for batch_data, batch_cond in test_cond_loader:
        x = batch_data.to(device)
        cond = batch_cond.to(device)
        batch_size = x.size(0)
        t_prior = torch.zeros(batch_size, device=device)  # t=0
        z_noisy = torch.zeros(batch_size, gvae.latent_dim, device=device)  # prior's input of z_noisy
        z_prior = prior_net(z_noisy, cond, t_prior)
        t = diffusion.sample_timesteps(batch_size)
        z_noisy = diffusion.q_sample(z_prior, t)
        z_pred = diffusion.p_sample(diff_model, z_noisy, t)
        batch_stacks = mask_fn.init_stack(batch_size)
        recon_logits = gvae.decoder(z_pred, start_token, batch_stacks)
        recon_prob = torch.softmax(recon_logits, dim=-1)
        mse = ((recon_prob - x) ** 2).mean().item()
        all_mse.append(mse)
        for i in range(batch_size):
            expr = expression_decoder.construct_expression(recon_logits[i].unsqueeze(0))
            all_exprs.append(expr)
            all_original_exprs.append(original_exprs[expr_idx])
            expr_idx += 1

prior_recon_mse = sum(all_mse) / len(all_mse)
output_path = os.path.join(OUTPUT_DIR, 'prior_reconstruction.txt')
with open(output_path, 'w', encoding='utf-8') as f:
    for orig, recon in zip(all_original_exprs, all_exprs):
        f.write(f"Original: {orig}\nReconstructed: {recon}\n\n")
print(f"MSE of Diffusion Prior: {prior_recon_mse:.4f}")
print(f"Reconstructed expressions saved to: {output_path}")
