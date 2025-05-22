from models.vae import VAE, vae_loss
from models.gvae import gvae_loss
import matplotlib.pyplot as plt
import torch
import os
from tqdm import tqdm

def train_vae(model, loader, optimizer, device, epochs=10):
    loss_history = []
    for epoch in range(epochs):
        train_loss = None
        model.train()
        total_loss = 0
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x = batch[0].to(device)
            optimizer.zero_grad()
            recon_x, mu, logvar = model(x)
            loss = vae_loss(recon_x, x, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        train_loss = total_loss / len(loader.dataset)
        loss_history.append(train_loss)
        print(f"[VAE] Epoch {epoch+1}: Train {train_loss:.4f}")

    plt.figure()
    plt.plot(range(1, epochs + 1), loss_history, label="VAE Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve for Variational Autoencoder")
    plt.legend()
    plt.grid()
    os.makedirs('./plots', exist_ok=True)
    plt.savefig("./plots/vae_loss_curve.png") 
    plt.close()

def train_gvae(model, loader, optimizer, device, epochs=10, beta=0.3):
    """
    适用于新GVAE结构的训练函数
    model: NewGVAE模型
    loader: DataLoader, 每个batch返回(batch, seq_len, feature_dim)的tensor
    optimizer: 优化器
    device: 设备
    epochs: 训练轮数
    beta: 解码器teacher forcing权重
    """
    loss_history = []
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            # 如果你的loader返回(batch, seq_len, feature_dim)的tensor，直接用batch
            x = batch[0].to(device)  # (batch, seq_len, feature_dim)
            target = x  # 重建目标
            optimizer.zero_grad()
            output, mu, log_var = model(x, beta=beta, target_seq=target)
            loss = vae_loss(output, target, mu, log_var)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        train_loss = total_loss / len(loader.dataset)
        loss_history.append(train_loss)
        print(f"[GVAE] Epoch {epoch+1}: Train {train_loss:.4f}")

    plt.figure()
    plt.plot(range(1, epochs + 1), loss_history, label="GVAE Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve for GVAE")
    plt.legend()
    plt.grid()
    os.makedirs('./plots', exist_ok=True)
    plt.savefig("./plots/gvae_loss_curve.png")
    plt.close()


def train_diffusion(diff_model, diffusion, z_data, epochs=10, batch_size=64, lr=1e-3):
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(z_data), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(diff_model.parameters(), lr=lr)
    device = next(diff_model.parameters()).device

    loss_history = []

    for epoch in range(epochs):
        diff_model.train()
        total_loss = 0
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            z0 = batch[0].to(device)  # original z
            t = diffusion.sample_timesteps(z0.size(0)) 
            noise = torch.randn_like(z0)
            z_noisy = diffusion.q_sample(z0, t, noise)
            pred_noise = diff_model(z_noisy, t)
            loss = ((noise - pred_noise) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * z0.size(0)
        
        avg_loss = total_loss / len(z_data)
        loss_history.append(avg_loss)
        print(f"[Diffusion] Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    plt.figure()
    plt.plot(range(1, epochs + 1), loss_history, label="Diffusion on Latent Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve for Diffusion on Latent Model")
    plt.legend()
    plt.grid()
    os.makedirs('./plots', exist_ok=True)
    plt.savefig("./plots/diffusion_latent_loss_curve.png") 
    plt.close()


def train_prior(diff_model, diffusion, z_data, cond_data, epochs=10, batch_size=64, lr=1e-3):
    dataset = torch.utils.data.TensorDataset(z_data, cond_data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(diff_model.parameters(), lr=lr)
    device = next(diff_model.parameters()).device

    loss_history = []

    for epoch in range(epochs):
        diff_model.train()
        total_loss = 0
        for z0, cond in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            z0 = z0.to(device)
            cond = cond.to(device)
            t = diffusion.sample_timesteps(z0.size(0)) 
            noise = torch.randn_like(z0)
            z_noisy = diffusion.q_sample(z0, t, noise)
            pred_noise = diff_model(z_noisy, cond, t)
            loss = ((noise - pred_noise) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * z0.size(0)
        
        avg_loss = total_loss / len(z_data)
        loss_history.append(avg_loss)
        print(f"[DiffusionPrior] Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    plt.figure()
    plt.plot(range(1, epochs + 1), loss_history, label="Diffusion Prior Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve for Diffusion Prior Model")
    plt.legend()
    plt.grid()
    os.makedirs('./plots', exist_ok=True)
    plt.savefig("./plots/diffusion_prior_loss_curve.png")
    plt.close()
