# generate.py
# Loads our trained TinyGPT and generates new Shakespeare-style text

import torch
import torch.nn.functional as F
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from level1_tiny_gpt.tokenizer import CharTokenizer
from level1_tiny_gpt.model import TinyGPT


def load_model(checkpoint_path):
    """
    Loads a saved model checkpoint.
    The checkpoint contains both the weights AND the settings
    used during training, so we can rebuild the exact same model.
    """
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    # Rebuild the tokenizer from the saved character list
    # (we need the same char<->int mappings as during training)
    tokenizer = CharTokenizer.__new__(CharTokenizer)
    tokenizer.chars       = checkpoint['chars']
    tokenizer.vocab_size  = checkpoint['vocab_size']
    tokenizer.char_to_int = { ch: i for i, ch in enumerate(tokenizer.chars) }
    tokenizer.int_to_char = { i: ch for i, ch in enumerate(tokenizer.chars) }

    # Rebuild the model with the same architecture as training
    model = TinyGPT(
        vocab_size  = checkpoint['vocab_size'],
        embed_dim   = checkpoint['embed_dim'],
        num_heads   = checkpoint['num_heads'],
        num_layers  = checkpoint['num_layers'],
        block_size  = checkpoint['block_size'],
        dropout     = checkpoint['dropout'],
    )

    # Load the trained weights into the model
    model.load_state_dict(checkpoint['model_state'])
    model.eval()  # switch off dropout for generation

    return model, tokenizer, checkpoint['block_size']


@torch.no_grad()  # no gradients needed — we're just running forward passes
def generate(model, tokenizer, block_size, prompt, max_new_chars, temperature):
    """
    Generates new text character by character.

    prompt        : the starting text to condition on
    max_new_chars : how many new characters to generate
    temperature   : controls randomness (lower = safer, higher = wilder)
    """

    # Encode the prompt into token integers
    idx = torch.tensor(
        tokenizer.encode(prompt),
        dtype=torch.long
    ).unsqueeze(0)  # add batch dimension: shape (1, T)

    generated = []

    for _ in range(max_new_chars):

        # Crop to the last block_size tokens if sequence gets too long
        idx_crop = idx[:, -block_size:]

        # Forward pass — get logits for the next token
        logits, _ = model(idx_crop)

        # Take only the logits at the very last position
        logits = logits[:, -1, :]   # shape: (1, vocab_size)

        # Apply temperature:
        # dividing by a small number makes the distribution sharper (more confident)
        # dividing by a large number makes it flatter (more random)
        logits = logits / temperature

        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)   # shape: (1, vocab_size)

        # Sample the next token from the probability distribution
        next_idx = torch.multinomial(probs, num_samples=1)  # shape: (1, 1)

        # Decode and store the new character
        generated.append(tokenizer.int_to_char[next_idx.item()])

        # Append the new token to the running sequence for next iteration
        idx = torch.cat([idx, next_idx], dim=1)

    return ''.join(generated)


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    checkpoint_path = "level1_tiny_gpt/checkpoints/model.pt"

    if not os.path.exists(checkpoint_path):
        print("No checkpoint found. Run train.py first!")
        sys.exit(1)

    print("Loading model...")
    model, tokenizer, block_size = load_model(checkpoint_path)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded — {total_params:,} parameters\n")

    # ── Experiment 1: cold start (empty prompt) ──
    print("=" * 50)
    print("GENERATION 1 — Cold start, temperature 1.0")
    print("=" * 50)
    output = generate(
        model, tokenizer, block_size,
        prompt       = "\n",
        max_new_chars= 300,
        temperature  = 1.0
    )
    print(output)

    # ── Experiment 2: prompted start ──
    print("\n" + "=" * 50)
    print("GENERATION 2 — Prompted: 'First Citizen:', temperature 1.0")
    print("=" * 50)
    output = generate(
        model, tokenizer, block_size,
        prompt       = "First Citizen:",
        max_new_chars= 300,
        temperature  = 1.0
    )
    print("First Citizen:" + output)

    # ── Experiment 3: low temperature (conservative) ──
    print("\n" + "=" * 50)
    print("GENERATION 3 — Low temperature 0.5 (more predictable)")
    print("=" * 50)
    output = generate(
        model, tokenizer, block_size,
        prompt       = "First Citizen:",
        max_new_chars= 300,
        temperature  = 0.5
    )
    print("First Citizen:" + output)

    # ── Experiment 4: high temperature (creative/chaotic) ──
    print("\n" + "=" * 50)
    print("GENERATION 4 — High temperature 1.5 (more chaotic)")
    print("=" * 50)
    output = generate(
        model, tokenizer, block_size,
        prompt       = "First Citizen:",
        max_new_chars= 300,
        temperature  = 1.5
    )
    print("First Citizen:" + output)