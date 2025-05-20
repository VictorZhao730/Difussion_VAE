import torch
from models.vae import VAE
from utils.data import load_data, get_loader, load_cond, load_cond_data, get_cond_loader
from models.diffusion_latent import SimpleMLPUNet, LatentDiffusion
from models.diffusion_prior import DiffusionPriorNet

DATA_DIR = 'dataset/iclr_final_truncated_fixed_powers.h5'
BATCH_SIZE = 64
SEED = 42
CSV_PATH = 'dataset/tsoulos_dataset_1.csv'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_data = load_data(DATA_DIR, split="test", random_seed=SEED)
test_loader = get_loader(test_data, batch_size=BATCH_SIZE)

test_cond_data, test_indices = load_cond_data(DATA_DIR, split="test", random_seed=SEED)
test_cond = load_cond(CSV_PATH, test_indices)
test_cond_loader = get_cond_loader(test_cond_data, test_cond, batch_size=BATCH_SIZE)

vae = VAE(seq_len=72, feature_dim=53).to(device)
vae.load_state_dict(torch.load("best_vae.pth"))
vae.eval()

diff_model = SimpleMLPUNet(latent_dim=vae.latent_dim).to(device)
diff_model.load_state_dict(torch.load("best_diffusion_latent.pth"))
diff_model.eval()
diffusion = LatentDiffusion(latent_dim=vae.latent_dim, timesteps=1000, device=device)

prior_net = DiffusionPriorNet(cond_dim=test_cond.shape[-1], latent_dim=vae.latent_dim).to(device)
prior_net.load_state_dict(torch.load("best_diffusion_prior.pth"))
prior_net.eval()

all_mse = []
with torch.no_grad():
    for batch in test_loader:
        x = batch[0].to(device)
        x_recon, mu, logvar = vae(x)
        mse = ((x_recon - x) ** 2).mean().item()
        all_mse.append(mse)
vae_recon_mse = sum(all_mse) / len(all_mse)
print(f"MSE of VAE: {vae_recon_mse:.4f}")

all_mse = []
with torch.no_grad():
    for batch in test_loader:
        x = batch[0].to(device)
        mu, logvar = vae.encoder(x)
        z = mu
        batch_size = z.shape[0]
        t = diffusion.sample_timesteps(batch_size)
        z_noisy = diffusion.q_sample(z, t)
        z_pred = diffusion.p_sample(diff_model, z_noisy, t)
        x_recon = vae.decoder(z_pred)
        mse = ((x_recon - x) ** 2).mean().item()
        all_mse.append(mse)
diff_recon_mse = sum(all_mse) / len(all_mse)
print(f"MSE of Diffusion on latent: {diff_recon_mse:.4f}")

all_mse = []
with torch.no_grad():
    for batch_data, batch_cond in test_cond_loader:
        x = batch_data.to(device)
        cond = batch_cond.to(device)
        batch_size = x.shape[0]
        t_prior = torch.zeros(batch_size, device=device)  # t=0
        z_noisy = torch.zeros(batch_size, vae.latent_dim, device=device)  # prior's input of z_noisy
        z_prior = prior_net(z_noisy, cond, t_prior)
        t = diffusion.sample_timesteps(batch_size)
        z_noisy = diffusion.q_sample(z_prior, t)
        z_pred = diffusion.p_sample(diff_model, z_noisy, t)
        x_recon = vae.decoder(z_pred)
        mse = ((x_recon - x) ** 2).mean().item()
        all_mse.append(mse)
prior_recon_mse = sum(all_mse) / len(all_mse)
print(f"MSE of Diffusion Prior: {prior_recon_mse:.4f}")
