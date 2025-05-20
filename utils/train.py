from models.vae import VAE, vae_loss
import torch

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in loader:
        x = batch[0].to(device)
        optimizer.zero_grad()
        recon_x, mu, logvar = model(x)
        loss = vae_loss(recon_x, x, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def train_diffusion(diff_model, diffusion, z_data, epochs=10, batch_size=128, lr=1e-3):

    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(z_data), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(diff_model.parameters(), lr=lr)
    device = next(diff_model.parameters()).device

    for epoch in range(epochs):
        diff_model.train()
        total_loss = 0
        for batch in loader:
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
        print(f"[Diffusion] Epoch {epoch+1}, Loss: {total_loss/len(z_data):.4f}")

        import torch

def train_prior(diff_model, diffusion, z_data, cond_data, epochs=10, batch_size=128, lr=1e-3):
    dataset = torch.utils.data.TensorDataset(z_data, cond_data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(diff_model.parameters(), lr=lr)
    device = next(diff_model.parameters()).device

    for epoch in range(epochs):
        diff_model.train()
        total_loss = 0
        for z0, cond in loader:
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
        print(f"[DiffusionPrior] Epoch {epoch+1}, Loss: {total_loss/len(z_data):.4f}")
