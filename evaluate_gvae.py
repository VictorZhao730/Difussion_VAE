import os
import torch
import pandas as pd
from models.gvae import GVAE
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
seq_len = 72
input_feature_size = len(productions)   # 你的特征维度
output_feature_size = input_feature_size
hidden_n = 200

test_data = load_data(DATA_DIR, split="test", random_seed=SEED)
test_loader = get_loader(test_data, batch_size=BATCH_SIZE)

test_cond_data, test_indices = load_cond_data(DATA_DIR, split="test", random_seed=SEED)
test_cond = load_cond(CSV_PATH, test_indices)
test_cond_loader = get_cond_loader(test_cond_data, test_cond, batch_size=BATCH_SIZE)

df = pd.read_csv(CSV_PATH)
original_exprs = df.loc[test_indices, 'expr'].tolist()

# 加载GVAE
gvae = GVAE(input_feature_size=input_feature_size, seq_len=seq_len, hidden_n=hidden_n, output_feature_size=output_feature_size).to(device)
gvae.load_state_dict(torch.load("trained_models/best_gvae.pth", map_location=device))
gvae.eval()

# 加载Diffusion模型
diff_model = SimpleMLPUNet(latent_dim=hidden_n).to(device)
diff_model.load_state_dict(torch.load("trained_models/best_diffusion_latent.pth", map_location=device))
diff_model.eval()
diffusion = LatentDiffusion(latent_dim=hidden_n, timesteps=1000, device=device)

prior_net = DiffusionPriorNet(latent_dim=hidden_n, cond_dim=test_cond.shape[-1]).to(device)
prior_net.load_state_dict(torch.load("trained_models/best_diffusion_prior.pth", map_location=device))
prior_net.eval()

# 加载表达式解码器
expression_decoder = ExpressionDecoder(
    checkpoint_path="trained_models/best_gvae.pth",
    device=device,
    model=GVAE(input_feature_size=input_feature_size, seq_len=seq_len, hidden_n=hidden_n, output_feature_size=output_feature_size),
)

# 1. GVAE重建
all_mse = []
all_exprs = []
all_original_exprs = []
expr_idx = 0

with torch.no_grad():
    for batch in test_loader:
        x = batch[0].to(device)  # (batch, seq_len, input_feature_size)
        batch_size = x.size(0)
        recon_logits, mu, logvar = gvae(x)   # 只需x
        recon_prob = torch.sigmoid(recon_logits)  # 你的GVAE输出最后一层已是sigmoid+relu
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
        # decoder需要hidden
        h1, h2, h3 = gvae.decoder.init_hidden(batch_size, device)
        recon_logits, _, _, _ = gvae.decoder(z_pred, h1, h2, h3)
        recon_prob = torch.sigmoid(recon_logits)
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
        z_noisy = torch.zeros(batch_size, hidden_n, device=device)  # prior's input of z_noisy
        z_prior = prior_net(z_noisy, cond, t_prior)
        t = diffusion.sample_timesteps(batch_size)
        z_noisy = diffusion.q_sample(z_prior, t)
        z_pred = diffusion.p_sample(diff_model, z_noisy, t)
        h1, h2, h3 = gvae.decoder.init_hidden(batch_size, device)
        recon_logits, _, _, _ = gvae.decoder(z_pred, h1, h2, h3)
        recon_prob = torch.sigmoid(recon_logits)
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
