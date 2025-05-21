import torch
from models.vae import VAE
from models.diffusion_latent import SimpleMLPUNet, LatentDiffusion
from utils.data import load_data, get_loader
from utils.train import train_diffusion
import os

LEARNING_RATE=1e-3
EPOCHS=50
BATCH_SIZE=64
DATA_DIR='dataset/iclr_final_truncated_fixed_powers.h5'
SEED=42

model = VAE(seq_len=72, feature_dim=53)
model.load_state_dict(torch.load("best_vae.pth"))
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

train_data = load_data(DATA_DIR, split="train", random_seed=SEED)
train_loader = get_loader(train_data, batch_size=64)
# get z
all_z = []
with torch.no_grad():
    for batch in train_loader:
        x = batch[0].to(device)
        mu, logvar = model.encoder(x)
        all_z.append(mu.cpu())
train_z = torch.cat(all_z, dim=0)

# Diffusion training
diff_model = SimpleMLPUNet(latent_dim=train_z.size(-1)).to(device)
diffusion = LatentDiffusion(latent_dim=train_z.size(-1), timesteps=1000, device=device)
train_diffusion(diff_model, diffusion, train_z, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE)

os.makedirs('./trained_models', exist_ok=True)
torch.save(diff_model.state_dict(), "./trained_models/best_diffusion_latent.pth")
