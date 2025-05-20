import torch
import torch.nn as nn
import torch.nn.functional as F

# Encoder
class Encoder(nn.Module):
    def __init__(self, seq_len=72, feature_dim=53, hidden_dim=200, latent_dim=200):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=feature_dim, out_channels=128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(256)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * seq_len, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        # x: (batch, seq_len, feature_dim)
        x = x.transpose(1, 2)  # (batch, feature_dim, seq_len)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

# Decoder
class Decoder(nn.Module):
    def __init__(self, seq_len=72, feature_dim=53, hidden_dim=200, latent_dim=200):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 256 * seq_len)
        self.deconv1 = nn.ConvTranspose1d(256, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.deconv2 = nn.ConvTranspose1d(128, feature_dim, kernel_size=3, padding=1)

        self.seq_len = seq_len
        self.feature_dim = feature_dim

    def forward(self, z):
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        x = x.view(-1, 256, self.seq_len)  # (batch, 256, seq_len)
        x = F.relu(self.bn1(self.deconv1(x)))
        x = torch.sigmoid(self.deconv2(x))  # (batch, feature_dim, seq_len)
        x = x.transpose(1, 2)  # (batch, seq_len, feature_dim)
        return x

# VAE
class VAE(nn.Module):
    def __init__(self, seq_len=72, feature_dim=53, hidden_dim=200, latent_dim=200):
        super(VAE, self).__init__()
        self.encoder = Encoder(seq_len, feature_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(seq_len, feature_dim, hidden_dim, latent_dim)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar


def vae_loss(recon_x, x, mu, logvar, weight = 0.1):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (BCE + weight*KLD) / x.size(0)

