import os
import random
import re
from typing import List, Optional, Union

from .gvae import GVAE
from .vae import VAE
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from dataset.grammar_alltogether import GCFG, S, get_mask

class Stack:
    def __init__(self, grammar, start_symbol):
        self.grammar = grammar
        self._stack = [start_symbol]

    @property
    def nonempty(self):
        return len(self._stack) > 0

    def push(self, symbol):
        self._stack.append(symbol)

    def pop(self):
        return self._stack.pop()

def default_device() -> torch.device:
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def load_checkpoint(checkpoint_path: str, device, model):
    print(f"→ Loading model from {checkpoint_path} …")
    model = model
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print("✔ Model loaded.")
    return model


class ExpressionDecoder:
    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[Union[str, torch.device]] = None,
        model: Union[GVAE, VAE, None] = None,
    ):
        self.device = torch.device(device) if device is not None else default_device()
        self.model = load_checkpoint(checkpoint_path, self.device, model)
        self.model = self.model.to(self.device)
        self.max_length = getattr(self.model, 'seq_len', 72)

    def construct_expression(
        self,
        logits: torch.Tensor,
        max_length: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> str:
        if seed is not None:
            random.seed(seed)
        logits = logits.to(self.device)
        stack = Stack(grammar=GCFG, start_symbol=S)
        rules = []
        t = 0
        limit = max_length or self.max_length

        while stack.nonempty and t < limit:
            nonterm = stack.pop()
            mask = get_mask(nonterm, stack.grammar, as_variable=True).to(self.device)
            probs = (logits[0, t].exp() * mask)
            probs = probs / probs.sum()
            idx = probs.argmax().item()
            prod = stack.grammar.productions()[idx]
            rules.append(prod)
            for sym in reversed(prod.rhs()):
                if hasattr(sym, 'symbol'): 
                    stack.push(sym)
            t += 1

        expr = "S"
        for prod in rules:
            rhs = " ".join(str(s) for s in prod.rhs())
            expr = expr.replace(prod.lhs().symbol(), rhs, 1)
        return re.sub(r"\s+", "", expr)

    def decode_latents(
        self,
        z: Union[np.ndarray, torch.Tensor],
        batch_size: int = 128,
        seed: Optional[int] = None,
    ) -> List[str]:
        if torch.is_tensor(z):
            z = z.detach().cpu().numpy()
        dataset = TensorDataset(torch.from_numpy(z).float())
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        results: List[str] = []
        self.model.eval()
        with torch.no_grad():
            for (batch_z,) in loader:
                logits = (
                    self.model.model
                         .decoder(batch_z.to(self.device).unsqueeze(1))
                         .mean(dim=1)
                )
                for single in logits:
                    expr = self.construct_expression(
                        single.unsqueeze(0),
                        seed=seed
                    )
                    results.append(expr)
        return results
