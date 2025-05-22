# Standard library imports
import os
import random
import re
import yaml
from typing import List, Optional, Union

# Scientific computing imports
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Custom model & grammar imports
from model_alltogether import GrammarVAE
from util_alltogether import GrammarVAEModel
from grammar_alltogether import GCFG, S, get_mask
from stack import Stack


def default_device() -> torch.device:
    """Pick GPU if available, else CPU."""
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def load_config(config_path: str) -> dict:
    """
    Load a YAML configuration file into a dict.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        dict: Configuration parameters
        
    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_checkpoint(checkpoint_path: str, config: dict) -> GrammarVAEModel:
    """
    Load a PyTorch-Lightning checkpoint for inference.

    Args:
        checkpoint_path: Path to .ckpt checkpoint file
        config: Configuration dictionary from load_config()

    Returns:
        GrammarVAEModel: Loaded model in eval mode
        
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    print(f"→ Loading model from {checkpoint_path} …")
    model = GrammarVAEModel.load_from_checkpoint(checkpoint_path, config=config)
    model.eval()
    print("✔ Model loaded.")
    return model


class ExpressionDecoder:
    """
    Load a trained GrammarVAE and decode latent vectors (z) into
    context-free-grammar expressions.

    This class handles:
    1. Loading the model and config
    2. Moving tensors to correct device
    3. Batched decoding of latent vectors
    4. Converting logits to expressions using grammar rules

    Attributes:
        model (GrammarVAEModel): Loaded model on specified device
        device (torch.device): CPU or GPU device
        max_length (int): Maximum number of grammar expansions
    """

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: Optional[Union[str, torch.device]] = None,
    ):
        # 1) Set computation device (GPU/CPU)
        self.device = torch.device(device) if device is not None else default_device()

        # 2) Load config and model
        print(f"→ Reading config from {config_path}")
        config = load_config(config_path)

        print(f"→ Loading checkpoint onto {self.device}")
        self.model = load_checkpoint(checkpoint_path, config)
        self.model = self.model.to(self.device)

        # 3) Get max sequence length from model (default 100)
        self.max_length = getattr(self.model.model, 'max_length', 100)

    def construct_expression(
        self,
        logits: torch.Tensor,
        max_length: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> str:
        """
        Convert a (1, T, V) logits tensor into a single expression string.
        Uses grammar rules to expand non-terminals into terminals.

        Args:
            logits: Model output logits tensor
            max_length: Maximum expansion steps (overrides self.max_length)
            seed: Random seed for reproducibility
            
        Returns:
            str: The decoded expression
        """
        if seed is not None:
            random.seed(seed)

        # Move logits to correct device
        logits = logits.to(self.device)

        # Initialize grammar stack and tracking variables
        stack = Stack(grammar=GCFG, start_symbol=S)
        rules = []
        t = 0
        limit = max_length or self.max_length

        # Expand grammar rules until stack empty or limit reached
        while stack.nonempty and t < limit:
            nonterm = stack.pop()

            # Get valid rules mask and normalize probabilities
            mask = get_mask(nonterm, stack.grammar, as_variable=True).to(self.device)
            probs = (logits[0, t].exp() * mask)
            probs = probs / probs.sum()

            # Select highest probability rule
            idx = probs.argmax().item()
            prod = stack.grammar.productions()[idx]
            rules.append(prod)

            # Push new non-terminals onto stack (right to left)
            for sym in reversed(prod.rhs()):
                if isinstance(sym, type(S)):
                    stack.push(sym)

            t += 1

        # Convert production rules to expression string
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
        """
        Decode N×latent_dim array into N expression strings.
        Processes latent vectors in batches for memory efficiency.

        Args:
            z: Latent vectors to decode
            batch_size: Number of vectors to process at once
            seed: Random seed for reproducibility
            
        Returns:
            list[str]: Decoded expressions
        """
        # Convert torch tensor to numpy if needed
        if torch.is_tensor(z):
            z = z.detach().cpu().numpy()

        # Create dataloader for batched processing
        dataset = TensorDataset(torch.from_numpy(z).float())
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        results: List[str] = []
        self.model.eval()
        with torch.no_grad():
            for (batch_z,) in loader:
                # Get decoder logits: shape [B, T, V]
                logits = (
                    self.model.model
                         .decoder(batch_z.to(self.device).unsqueeze(1))
                         .mean(dim=1)
                )
                # Decode each sequence in batch
                for single in logits:
                    expr = self.construct_expression(
                        single.unsqueeze(0),
                        seed=seed
                    )
                    results.append(expr)

        return results


if __name__ == "__main__":
    # Example usage
    # CONFIG_PATH = 'configs/config-alltogether_copy.yaml'
    # CKPT_PATH   = '/cluster/.../best.ckpt'

    decoder = ExpressionDecoder(CONFIG_PATH, CKPT_PATH)
    
    # Create test latent vector
    latent_dim = 20
    z_demo = np.random.randn(1, latent_dim)
    
    # Decode it to expression
    out = decoder.decode_latents(z_demo, batch_size=1, seed=42)
    print("Decoded:", out[0])
