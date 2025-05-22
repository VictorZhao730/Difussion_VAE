import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, input_size=200, hidden_n=200, output_size=53, seq_length=72):
        super(Decoder, self).__init__()
        self.seq_length = seq_length
        self.hidden_n = hidden_n
        self.output_size = output_size
        
        self.batch_norm = nn.BatchNorm1d(input_size)
        self.fc_input = nn.Linear(input_size, hidden_n)
        
        self.gru_1 = nn.GRU(input_size=hidden_n, hidden_size=hidden_n, batch_first=True)
        self.gru_2 = nn.GRU(input_size=hidden_n, hidden_size=hidden_n, batch_first=True)
        self.gru_3 = nn.GRU(input_size=hidden_n, hidden_size=hidden_n, batch_first=True)
        self.fc_out = nn.Linear(hidden_n, output_size)

    def forward(self, encoded, hidden_1, hidden_2, hidden_3, beta=0.3, target_seq=None):
        batch_size = encoded.size()[0]

        embedded = F.relu(self.fc_input(self.batch_norm(encoded))) \
            .view(batch_size, 1, -1) \
            .repeat(1, self.seq_length, 1)

        out_1, hidden_1 = self.gru_1(embedded, hidden_1)
        out_2, hidden_2 = self.gru_2(out_1, hidden_2)
        
        if self.training and target_seq is not None:
            out_2 = out_2 * (1 - beta) + target_seq * beta
            
        out_3, hidden_3 = self.gru_3(out_2, hidden_3)
        out = self.fc_out(out_3.contiguous().view(-1, self.hidden_n)) \
            .view(batch_size, self.seq_length, self.output_size)
            
        return F.sigmoid(out), hidden_1, hidden_2, hidden_3

    def init_hidden(self, batch_size):
        h1 = Variable(torch.zeros(1, batch_size, self.hidden_n), requires_grad=False)
        h2 = Variable(torch.zeros(1, batch_size, self.hidden_n), requires_grad=False)
        h3 = Variable(torch.zeros(1, batch_size, self.hidden_n), requires_grad=False)
        return h1, h2, h3


class Encoder(nn.Module):
    def __init__(self, input_channels=53, hidden_n=200):
        super(Encoder, self).__init__()
        # Adjust input channels to match the data shape
        self.conv_1 = nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=3)
        self.bn_1 = nn.BatchNorm1d(64)
        self.conv_2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
        self.bn_2 = nn.BatchNorm1d(128)
        self.conv_3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3)
        self.bn_3 = nn.BatchNorm1d(256)

        # Calculate the size after convolutions
        # Input shape: (batch_size, 53, 72)
        # After conv1: (batch_size, 64, 70)
        # After conv2: (batch_size, 128, 68)
        # After conv3: (batch_size, 256, 66)
        self.fc_0 = nn.Linear(256 * 66, hidden_n)
        self.fc_mu = nn.Linear(hidden_n, hidden_n)
        self.fc_var = nn.Linear(hidden_n, hidden_n)

    def forward(self, x):
        # Input shape: (batch_size, 72, 53)
        # Transpose to (batch_size, 53, 72) for conv1d
        x = x.transpose(1, 2).contiguous()
        
        x = F.relu(self.bn_1(self.conv_1(x)))
        x = F.relu(self.bn_2(self.conv_2(x)))
        x = F.relu(self.bn_3(self.conv_3(x)))
        
        x = x.view(x.size(0), -1)  # Flatten
        h = self.fc_0(x)
        return self.fc_mu(h), self.fc_var(h)


class VAELoss(nn.Module):
    def __init__(self):
        super(VAELoss, self).__init__()
        self.bce_loss = nn.BCELoss()
        self.bce_loss.size_average = False

    def forward(self, x, mu, log_var, recon_x):
        """gives the batch normalized Variational Error."""

        batch_size = x.size()[0]
        BCE = self.bce_loss(recon_x, x)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD_element = mu.pow(2).add_(log_var.exp()).mul_(-1).add_(1).add_(log_var)
        KLD = torch.sum(KLD_element).mul_(-0.5)

        return (BCE + KLD) / batch_size


class GrammarVariationalAutoEncoder(nn.Module):
    def __init__(self):
        super(GrammarVariationalAutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        batch_size = x.size()[0]
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        h1, h2, h3 = self.decoder.init_hidden(batch_size)
        output, h1, h2, h3 = self.decoder(z, h1, h2, h3)
        return output, mu, log_var

    def reparameterize(self, mu, log_var):
        """you generate a random distribution w.r.t. the mu and log_var from the embedding space."""
        vector_size = log_var.size()
        eps = Variable(torch.FloatTensor(vector_size).normal_())
        std = log_var.mul(0.5).exp_()
        return eps.mul(std).add_(mu)
