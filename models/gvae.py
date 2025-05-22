import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvEncoder(nn.Module):
    def __init__(self, input_feature_size=53, seq_len=72, hidden_n=200, k1=2, k2=3, k3=4):
        super(ConvEncoder, self).__init__()
        self.conv_1 = nn.Conv1d(in_channels=input_feature_size, out_channels=12, kernel_size=k1)
        self.bn_1 = nn.BatchNorm1d(12)
        self.conv_2 = nn.Conv1d(in_channels=12, out_channels=12, kernel_size=k2)
        self.bn_2 = nn.BatchNorm1d(12)
        self.conv_3 = nn.Conv1d(in_channels=12, out_channels=12, kernel_size=k3)
        self.bn_3 = nn.BatchNorm1d(12)
        self.fc_0 = nn.Linear(12 * (seq_len - k1 - k2 - k3 + 3), hidden_n)  # 注意这里的长度要根据卷积核计算
        self.fc_mu = nn.Linear(hidden_n, hidden_n)
        self.fc_var = nn.Linear(hidden_n, hidden_n)

    def forward(self, x):
        # x: (batch, seq_len, feature_size)
        batch_size = x.size(0)
        x = x.transpose(1, 2).contiguous()  # (batch, feature_size, seq_len)
        x = F.relu(self.bn_1(self.conv_1(x)))
        x = F.relu(self.bn_2(self.conv_2(x)))
        x = F.relu(self.bn_3(self.conv_3(x)))
        x_ = x.view(batch_size, -1)
        h = self.fc_0(x_)
        return self.fc_mu(h), self.fc_var(h)
    
class MultiGRUDecoder(nn.Module):
    def __init__(self, input_size=200, hidden_n=200, output_feature_size=53, max_seq_length=72):
        super(MultiGRUDecoder, self).__init__()
        self.max_seq_length = max_seq_length
        self.hidden_n = hidden_n
        self.output_feature_size = output_feature_size
        self.batch_norm = nn.BatchNorm1d(input_size)
        self.fc_input = nn.Linear(input_size, hidden_n)
        self.gru_1 = nn.GRU(input_size=input_size, hidden_size=hidden_n, batch_first=True)
        self.gru_2 = nn.GRU(input_size=input_size, hidden_size=hidden_n, batch_first=True)
        self.gru_3 = nn.GRU(input_size=input_size, hidden_size=hidden_n, batch_first=True)
        self.fc_out = nn.Linear(hidden_n, output_feature_size)

    def forward(self, encoded, hidden_1, hidden_2, hidden_3, beta=0.3, target_seq=None):
        _batch_size = encoded.size()[0]
        embedded = F.relu(self.fc_input(self.batch_norm(encoded))) \
            .view(_batch_size, 1, -1) \
            .repeat(1, self.max_seq_length, 1)
        out_1, hidden_1 = self.gru_1(embedded, hidden_1)
        out_2, hidden_2 = self.gru_2(out_1, hidden_2)
        out_3, hidden_3 = self.gru_3(out_2, hidden_3)
        out = self.fc_out(out_3.contiguous().view(-1, self.hidden_n)).view(_batch_size, self.max_seq_length, self.output_feature_size)
        if self.training and target_seq is not None:
            # target_seq shape must be (batch, seq_len, output_feature_size)
            out = out * (1 - beta) + target_seq * beta
        return F.relu(torch.sigmoid(out)), hidden_1, hidden_2, hidden_3


    def init_hidden(self, batch_size, device):
        h1 = torch.zeros(1, batch_size, self.hidden_n, device=device)
        h2 = torch.zeros(1, batch_size, self.hidden_n, device=device)
        h3 = torch.zeros(1, batch_size, self.hidden_n, device=device)
        return h1, h2, h3
    
class GVAE(nn.Module):
    def __init__(self, input_feature_size=53, seq_len=15, hidden_n=200, output_feature_size=53):
        super(GVAE, self).__init__()
        self.encoder = ConvEncoder(input_feature_size=input_feature_size, seq_len=seq_len, hidden_n=hidden_n)
        self.decoder = MultiGRUDecoder(input_size=hidden_n, hidden_n=hidden_n, output_feature_size=output_feature_size, max_seq_length=seq_len)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, beta=0.3, target_seq=None):
        batch_size = x.size(0)
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        device = x.device
        h1, h2, h3 = self.decoder.init_hidden(batch_size, device)
        output, h1, h2, h3 = self.decoder(z, h1, h2, h3, beta=beta, target_seq=target_seq)
        return output, mu, log_var

def gvae_loss(recon_x, x, mu, log_var):
    batch_size = x.size(0)
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return (BCE + KLD) / batch_size

