import torch
from models.gvae import GVAE,  MaskFn
from utils.data import load_data, get_loader
from utils.train import train_gvae
from dataset.grammar_alltogether import GCFG
import os
from nltk import CFG

LEARNING_RATE=5e-3
EPOCHS=5
BATCH_SIZE=256
DATA_DIR='dataset/iclr_final_truncated_fixed_powers.h5'   
SEED = 42

productions = GCFG.productions()
start_symbol = GCFG.start()
mask_fn = MaskFn(productions, start_symbol)

data = load_data(DATA_DIR, split="train", random_seed=SEED)
train_loader = get_loader(data, batch_size=BATCH_SIZE, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GVAE(
    seq_len=72,                      
    rule_dim=53,    
    hidden_dim=200,
    latent_dim=200,
    mask_fn=mask_fn,
    device=device
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

start_token = torch.zeros(len(productions))
start_token[0] = 1
train_gvae(model, train_loader, optimizer, device, start_token, epochs=EPOCHS)

os.makedirs('./trained_models', exist_ok=True)
torch.save(model.state_dict(), "best_gvae.pth")
