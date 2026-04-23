from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    """
    Configuration container for the GPT-style decoder-only language model.
    """
    vocab_size: int
    context_length: int
    d_model: int
    n_layers: int
    n_heads: int
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    bias: bool = True
    tie_embeddings: bool = True

    @classmethod
    def from_dict(cls, cfg: dict, vocab_size: int):
        """
        Build a GPTConfig from a YAML/dict-style config.

        Notes:
        - The project config files use GPT-style names such as:
          n_embd, n_layer, n_head
        - vocab_size can either come from the config file or from the
          tokenizer metadata passed in at runtime
        """
        return cls(
            vocab_size=cfg.get("vocab_size", vocab_size),
            context_length=cfg["context_length"],
            d_model=cfg["n_embd"],
            n_layers=cfg["n_layer"],
            n_heads=cfg["n_head"],
            mlp_ratio=cfg.get("mlp_ratio", 4.0),
            dropout=cfg.get("dropout", 0.1),
            bias=cfg.get("bias", True),
            tie_embeddings=cfg.get("tie_embeddings", True),
        )


class CausalSelfAttention(nn.Module):
    """
    Standard multi-head causal self-attention.

    Each token can attend only to itself and previous tokens.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()

        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")

        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.dropout = config.dropout

        # One projection produces Q, K, and V together for efficiency.
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)

        # Output projection back to the model dimension.
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        """
        Args:
            x: input tensor of shape (B, T, C)
               B = batch size
               T = sequence length
               C = model dimension

        Returns:
            Tensor of shape (B, T, C)
        """
        B, T, C = x.shape

        # Project once, then split into queries, keys, and values.
        qkv = self.qkv_proj(x)  # (B, T, 3C)
        q, k, v = qkv.split(C, dim=2)

        # Reshape into multi-head format:
        # (B, T, C) -> (B, H, T, D)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # PyTorch fused scaled dot-product attention with causal masking.
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )  # (B, H, T, D)

        # Merge attention heads back into the model dimension.
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)

        # Final output projection + residual dropout.
        y = self.out_proj(y)
        y = self.resid_dropout(y)
        return y


class MLP(nn.Module):
    """
    Feed-forward network inside each Transformer block.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        hidden_dim = int(config.mlp_ratio * config.d_model)

        self.fc = nn.Linear(config.d_model, hidden_dim, bias=config.bias)
        self.proj = nn.Linear(hidden_dim, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # Standard Transformer MLP: linear -> GELU -> linear -> dropout
        x = self.fc(x)
        x = F.gelu(x)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """
    Pre-LN Transformer block:
    - LayerNorm
    - Causal self-attention
    - Residual connection
    - LayerNorm
    - MLP
    - Residual connection
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPTLanguageModel(nn.Module):
    """
    Minimal GPT-style decoder-only language model.

    Architecture:
    - token embeddings
    - positional embeddings
    - stack of Transformer blocks
    - final layer norm
    - language modeling head
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Token embedding table maps token ids -> vectors.
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Learned positional embeddings for sequence positions [0, context_length).
        self.position_embedding = nn.Embedding(config.context_length, config.d_model)

        self.dropout = nn.Dropout(config.dropout)

        # Main Transformer stack.
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)

        # Final projection from hidden states to vocabulary logits.
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Optionally tie output projection weights to input token embeddings.
        # This is common in language models and reduces parameter count.
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        # Initialize weights after all layers are created.
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        GPT-style weight initialization.
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        Forward pass.

        Args:
            idx: input token ids of shape (B, T)
            targets: optional target token ids of shape (B, T)

        Returns:
            logits: (B, T, vocab_size)
            loss: scalar cross-entropy loss if targets are provided, else None
        """
        B, T = idx.shape

        if T > self.config.context_length:
            raise ValueError(
                f"Sequence length {T} exceeds context_length {self.config.context_length}."
            )

        # Create position indices [0, 1, ..., T-1].
        positions = torch.arange(0, T, device=idx.device, dtype=torch.long).unsqueeze(0)

        # Look up token embeddings and positional embeddings.
        tok_emb = self.token_embedding(idx)            # (B, T, C)
        pos_emb = self.position_embedding(positions)   # (1, T, C)

        # Combine token + position information.
        x = self.dropout(tok_emb + pos_emb)

        # Pass through the Transformer blocks.
        for block in self.blocks:
            x = block(x)

        # Final normalization and projection to vocabulary logits.
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Flatten batch and time dimensions for token-level cross-entropy.
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

        return logits, loss

    def count_parameters(self, trainable_only: bool = True):
        """
        Count parameters in the model.

        Args:
            trainable_only: if True, count only parameters with requires_grad=True
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


def build_model(model_cfg: dict, vocab_size: int):
    """
    Convenience helper that builds the GPT model from a config dict.
    """
    config = GPTConfig.from_dict(model_cfg, vocab_size=vocab_size)
    model = GPTLanguageModel(config)
    return model
