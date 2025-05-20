import torch
from models.vae import VAE, vae_loss
from utils.data import load_data, get_loader
from utils.train import train_one_epoch

LEARNING_RATE=1e-3
EPOCHS=5
BATCH_SIZE=64
DATA_DIR='dataset/iclr_final_truncated_fixed_powers.h5'
SEED=42

data = load_data(DATA_DIR, split="train", random_seed=SEED)
train_loader = get_loader(data, batch_size=BATCH_SIZE, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VAE(seq_len=72, feature_dim=53).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    train_loss = train_one_epoch(model, train_loader, optimizer, device)
    print(f"[VAE] Epoch {epoch+1}: Train {train_loss:.4f}")

torch.save(model.state_dict(), "best_vae.pth")
