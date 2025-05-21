import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskFn:
    def __init__(self, productions, start_symbol):
        """
        productions: list of nltk.Production
        start_symbol: nltk.Nonterminal
        """
        self.productions = productions
        self.start_symbol = start_symbol
        self.idx2prod = {i: p for i, p in enumerate(self.productions)}

    def init_stack(self, batch_size=1):
        return [[self.start_symbol] for _ in range(batch_size)]

    def get_mask(self, stack):
        mask = torch.full((len(self.productions),), float('-inf'))
        if not stack:
            return mask
        top = stack[-1]
        for i, prod in enumerate(self.productions):
            if prod.lhs() == top:
                mask[i] = 0
        return mask

    def update_stack(self, stack, prod_idx):
        prod = self.idx2prod[prod_idx]
        stack = stack[:-1] 
        for symbol in reversed(prod.rhs()):
            if hasattr(symbol, 'symbol'):  # nltk.Nonterminal
                stack.append(symbol)
        return stack

# --------- Encoder ---------
class GEncoder(nn.Module):
    def __init__(self, seq_len=30, rule_dim=53, hidden_dim=200, latent_dim=200):
        super(GEncoder, self).__init__()
        self.gru = nn.GRU(input_size=rule_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        # x: (batch, seq_len, rule_dim)
        _, h = self.gru(x)  # h: (1, batch, hidden_dim)
        h = h.squeeze(0)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

# --------- Decoder ---------
class GDecoder(nn.Module):
    def __init__(self, seq_len=30, rule_dim=53, hidden_dim=200, latent_dim=200, mask_fn=None, device='cpu'):
        super(GDecoder, self).__init__()
        self.fc = nn.Linear(latent_dim, hidden_dim)
        self.gru = nn.GRU(input_size=rule_dim, hidden_size=hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, rule_dim)
        self.seq_len = seq_len
        self.rule_dim = rule_dim
        self.mask_fn = mask_fn
        self.device = device

    def forward(self, z, start_token, batch_stacks=None):
        """
        z: (batch, latent_dim)
        start_token: (rule_dim,) one-hot
        batch_stacks: list of stacks
        """
        batch = z.size(0)
        h = self.fc(z).unsqueeze(0)  # (1, batch, hidden_dim)
        inputs = start_token.unsqueeze(0).repeat(batch, 1).unsqueeze(1)  # (batch, 1, rule_dim)
        outputs = []
        stacks = batch_stacks if batch_stacks is not None else self.mask_fn.init_stack(batch)
        for t in range(self.seq_len):
            out, h = self.gru(inputs, h)  # out: (batch, 1, hidden_dim)
            logits = self.out(out.squeeze(1))  # (batch, rule_dim)
            masked_logits = []
            for i in range(batch):
                mask = self.mask_fn.get_mask(stacks[i]).to(self.device)
                masked_logit = logits[i] + mask
                masked_logit = torch.clamp(masked_logit, min=-10, max=10) # clip
                masked_logits.append(masked_logit)
            logits = torch.stack(masked_logits, dim=0)
            outputs.append(logits)
            next_tokens = torch.argmax(logits, dim=1)
            for i in range(batch):
                stacks[i] = self.mask_fn.update_stack(stacks[i], next_tokens[i].item())
            inputs = F.one_hot(next_tokens, num_classes=self.rule_dim).float().unsqueeze(1)
        # (batch, seq_len, rule_dim)
        return torch.stack(outputs, dim=1)

class GVAE(nn.Module):
    def __init__(self, seq_len=30, rule_dim=53, hidden_dim=200, latent_dim=200, mask_fn=None, device='cpu'):
        super(GVAE, self).__init__()
        self.encoder = GEncoder(seq_len, rule_dim, hidden_dim, latent_dim)
        self.decoder = GDecoder(seq_len, rule_dim, hidden_dim, latent_dim, mask_fn, device)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, start_token, batch_stacks=None):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_logits = self.decoder(z, start_token, batch_stacks)
        return recon_logits, mu, logvar

def gvae_loss(recon_logits, target, mu, logvar, weight=0.5):
    # recon_logits: (batch, seq_len, rule_dim)
    # target: (batch, seq_len, rule_dim) one-hot
    recon_logits = recon_logits.view(-1, recon_logits.size(-1))
    target = target.view(-1, target.size(-1))
    BCE = F.binary_cross_entropy_with_logits(recon_logits, target, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (BCE + weight * KLD) / target.size(0)

