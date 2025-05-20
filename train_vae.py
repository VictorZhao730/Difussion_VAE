import torch
from models.vae import VAE, vae_loss
from utils.data import load_data, get_loader
from utils.train import train_vae

LEARNING_RATE=1e-3
EPOCHS=50
BATCH_SIZE=64
DATA_DIR='dataset/iclr_final_truncated_fixed_powers.h5'
SEED=42

data = load_data(DATA_DIR, split="train", random_seed=SEED)
train_loader = get_loader(data, batch_size=BATCH_SIZE, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VAE(seq_len=72, feature_dim=53).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_vae(model, train_loader, optimizer, device, epochs=EPOCHS)

torch.save(model.state_dict(), "best_vae.pth")
