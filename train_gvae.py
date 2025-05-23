import torch
from models.gvae import GVAE         # 假设你把新GVAE代码存在 models/gvae_new.py
from utils.data import load_data, get_loader
from utils.train import train_gvae      # 训练函数建议新建
import os

LEARNING_RATE = 5e-3
EPOCHS = 150
BATCH_SIZE = 128
DATA_DIR = 'dataset/iclr_final_truncated_fixed_powers.h5'
SEED = 42

# 加载数据
data = load_data(DATA_DIR, split="train", random_seed=SEED)
train_loader = get_loader(data, batch_size=BATCH_SIZE, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 设定输入输出维度
INPUT_FEATURE_SIZE = 53      # 每个位置的特征数，与原VAE一致
SEQ_LEN = 72                 # 序列最大长度
HIDDEN_N = 200
OUTPUT_FEATURE_SIZE = 53     # 通常与输入一致

model = GVAE(
    input_feature_size=INPUT_FEATURE_SIZE,
    seq_len=SEQ_LEN,
    hidden_n=HIDDEN_N,
    output_feature_size=OUTPUT_FEATURE_SIZE
).to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 训练
train_gvae(model, train_loader, optimizer, device, epochs=EPOCHS)

os.makedirs('./trained_models', exist_ok=True)
torch.save(model.state_dict(), "./trained_models/best_gvae.pth")
