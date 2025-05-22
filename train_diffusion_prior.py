import torch
from torch.optim import Adam
from models.gvae import GVAE
from models.diffusion_prior import DiffusionPriorNet, DiffusionPrior
from utils.data import load_cond, load_cond_data, get_cond_loader
from utils.train import train_prior
import os
from dataset.grammar_alltogether import GCFG

LEARNING_RATE = 1e-3
EPOCHS = 50
BATCH_SIZE = 128
DATA_DIR = 'dataset/iclr_final_truncated_fixed_powers.h5'
CSV_PATH = 'dataset/tsoulos_dataset_1.csv'
SEED = 42

productions = GCFG.productions()


model = GVAE(seq_len=72, rule_dim=len(productions))
model.load_state_dict(torch.load("./trained_models/best_gvae.pth", weights_only=True))
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

train_data, train_indices = load_cond_data(DATA_DIR, split="train", random_seed=SEED)
train_cond = load_cond(CSV_PATH, train_indices)
train_loader = get_cond_loader(train_data, train_cond, batch_size=BATCH_SIZE, shuffle=True)

all_z = []
all_cond = []
with torch.no_grad():
    for batch_data, batch_cond in train_loader:
        x = batch_data.to(device)         # (B, 72, 53)
        cond = batch_cond.float()         # (B, cond_dim)
        mu, logvar = model.encoder(x)
        all_z.append(mu.cpu())
        all_cond.append(cond.cpu())
train_z = torch.cat(all_z, dim=0)       # (N, latent_dim)
train_cond = torch.cat(all_cond, dim=0) # (N, cond_dim)

# train Diffusion Prior
COND_DIM = train_cond.size(-1)
LATENT_DIM = train_z.size(-1)
TIMESTEPS = 1000

prior_net = DiffusionPriorNet(cond_dim=COND_DIM, latent_dim=LATENT_DIM).to(device)
prior = DiffusionPrior(latent_dim=LATENT_DIM, timesteps=TIMESTEPS, device=device)
optimizer = Adam(prior_net.parameters(), lr=LEARNING_RATE)

train_prior(prior_net, prior, train_z, train_cond, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE)
os.makedirs('./trained_models', exist_ok=True)
torch.save(prior_net.state_dict(), "./trained_models/best_diffusion_prior.pth")
