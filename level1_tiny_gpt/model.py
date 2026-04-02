# model.py
# Defines our tiny GPT neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 1. Single Self-Attention Head ─────────────────────────────────────────────
class Head(nn.Module):
    """
    One head of self-attention.

    Think of it like this: every token asks a question (Query),
    every token advertises what it has (Key), and every token
    carries a value (Value). Attention scores decide how much
    each token 'listens' to every other token.
    """

    def __init__(self, head_size, embed_dim, block_size, dropout):
        super().__init__()
        # These three linear layers produce Q, K, V from the input
        self.key   = nn.Linear(embed_dim, head_size, bias=False)
        self.query = nn.Linear(embed_dim, head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_size, bias=False)

        # tril is a lower-triangular matrix of ones — used as a mask
        # so token at position i can only attend to positions 0..i
        # (it cannot "see the future")
        self.register_buffer(
            'tril',
            torch.tril(torch.ones(block_size, block_size))
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape          # Batch, Time (sequence length), Channels

        k = self.key(x)            # (B, T, head_size)
        q = self.query(x)          # (B, T, head_size)

        # Compute attention scores
        # Scaled by 1/sqrt(head_size) to keep values stable
        scale = k.shape[-1] ** -0.5
        scores = q @ k.transpose(-2, -1) * scale   # (B, T, T)

        # Mask future positions with -infinity so softmax zeros them out
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        # Softmax turns scores into probabilities (they sum to 1)
        weights = F.softmax(scores, dim=-1)         # (B, T, T)
        weights = self.dropout(weights)

        # Weighted sum of values
        v = self.value(x)                           # (B, T, head_size)
        return weights @ v                          # (B, T, head_size)


# ── 2. Multi-Head Attention ────────────────────────────────────────────────────
class MultiHeadAttention(nn.Module):
    """
    Several attention heads running in parallel, then concatenated.
    Multiple heads let the model attend to different things at once —
    one head might focus on grammar, another on meaning, etc.
    """

    def __init__(self, num_heads, head_size, embed_dim, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([
            Head(head_size, embed_dim, block_size, dropout)
            for _ in range(num_heads)
        ])
        # Project the concatenated heads back to embed_dim
        self.proj    = nn.Linear(num_heads * head_size, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Run all heads, concatenate along the last dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


# ── 3. FeedForward Network ─────────────────────────────────────────────────────
class FeedForward(nn.Module):
    """
    A simple two-layer MLP applied to each token independently.
    After attention mixes information across tokens, the feedforward
    network lets each token 'think' about what it just learned.
    """

    def __init__(self, embed_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),  # expand
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),  # contract back
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# ── 4. Transformer Block ───────────────────────────────────────────────────────
class Block(nn.Module):
    """
    One full transformer block = attention + feedforward.
    LayerNorm is applied before each sub-layer (this is the modern
    'pre-norm' style used by GPT-2 and later).
    The '+' signs are residual connections — they let gradients flow
    cleanly during backpropagation, making deep networks trainable.
    """

    def __init__(self, embed_dim, num_heads, block_size, dropout):
        super().__init__()
        head_size = embed_dim // num_heads
        self.attn = MultiHeadAttention(
            num_heads, head_size, embed_dim, block_size, dropout
        )
        self.ff   = FeedForward(embed_dim, dropout)
        self.ln1  = nn.LayerNorm(embed_dim)
        self.ln2  = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # attention with residual
        x = x + self.ff(self.ln2(x))     # feedforward with residual
        return x


# ── 5. The Full GPT Model ──────────────────────────────────────────────────────
class TinyGPT(nn.Module):
    """
    Puts it all together:
      token embedding → position embedding → N transformer blocks → output
    """

    def __init__(self, vocab_size, embed_dim, num_heads, num_layers,
                 block_size, dropout):
        super().__init__()

        # Lookup table: token index → embed_dim vector
        self.token_embedding    = nn.Embedding(vocab_size, embed_dim)

        # Lookup table: position index → embed_dim vector
        self.position_embedding = nn.Embedding(block_size, embed_dim)

        # Stack of transformer blocks
        self.blocks = nn.Sequential(*[
            Block(embed_dim, num_heads, block_size, dropout)
            for _ in range(num_layers)
        ])

        self.ln_final = nn.LayerNorm(embed_dim)

        # Final linear layer: maps from embed_dim → vocab_size (one score per token)
        self.head = nn.Linear(embed_dim, vocab_size)

        self.block_size = block_size

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # Token + positional embeddings added together
        tok_emb = self.token_embedding(idx)                        # (B, T, embed_dim)
        pos_emb = self.position_embedding(torch.arange(T))        # (T, embed_dim)
        x = tok_emb + pos_emb                                      # (B, T, embed_dim)

        # Pass through all transformer blocks
        x = self.blocks(x)
        x = self.ln_final(x)

        # Project to vocabulary scores (logits)
        logits = self.head(x)                                      # (B, T, vocab_size)

        # If we have targets, compute cross-entropy loss
        if targets is None:
            return logits, None

        B, T, C = logits.shape
        loss = F.cross_entropy(
            logits.view(B * T, C),     # flatten batch and time
            targets.view(B * T)        # flatten targets
        )
        return logits, loss