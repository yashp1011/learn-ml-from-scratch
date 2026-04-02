# train.py
# Trains our TinyGPT model on the Shakespeare text

import torch
import os
import sys

# This lets us import from sibling files cleanly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from level1_tiny_gpt.tokenizer import CharTokenizer
from level1_tiny_gpt.model import TinyGPT


# ── Hyperparameters ────────────────────────────────────────────────────────────
# These are the "settings" of training — you can tune them later

# Old parameters model size too large for the data
# BLOCK_SIZE  = 64    # how many characters the model sees at once
# BATCH_SIZE  = 16    # how many examples to process in parallel
# EMBED_DIM   = 64    # size of each token's vector representation
# NUM_HEADS   = 4     # number of attention heads
# NUM_LAYERS  = 4     # number of transformer blocks stacked
# DROPOUT     = 0.1   # randomly zero out 10% of connections (prevents overfitting)
# LEARNING_RATE = 3e-4  # how big each weight update step is
# MAX_STEPS   = 3000  # total number of training steps
# EVAL_EVERY  = 300   # print loss every N steps


# New parameters
BLOCK_SIZE    = 64
BATCH_SIZE    = 32    # was 16 — bigger batches = more stable gradients
EMBED_DIM     = 32    # was 64 — smaller model = less memorization
NUM_HEADS     = 4
NUM_LAYERS    = 3     # was 4 — one fewer layer
DROPOUT       = 0.3   # was 0.1 — more aggressive dropout
LEARNING_RATE = 3e-4
MAX_STEPS     = 3000
EVAL_EVERY    = 300



# ── 1. Load and encode the data ────────────────────────────────────────────────
print("Loading data...")
with open("data/input.txt", "r") as f:
    text = f.read()

tokenizer = CharTokenizer(text)
print(f"Vocab size: {tokenizer.vocab_size} characters")

# Encode the entire text into a tensor of integers
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
print(f"Total tokens: {len(data)}")


# ── 2. Split into train and validation sets ────────────────────────────────────
# We keep 10% of data aside to measure if the model generalises
# (if train loss drops but val loss doesn't, the model is memorising)
split = int(0.9 * len(data))
train_data = data[:split]
val_data   = data[split:]


# ── 3. Batch sampler ───────────────────────────────────────────────────────────
def get_batch(split):
    """
    Grabs a random batch of (input, target) pairs.

    For each example:
      input  = characters at positions [i : i+BLOCK_SIZE]
      target = characters at positions [i+1 : i+BLOCK_SIZE+1]

    So the model learns: given these chars, predict the next one.
    """
    source = train_data if split == 'train' else val_data

    # Pick BATCH_SIZE random starting positions
    starts = torch.randint(len(source) - BLOCK_SIZE, (BATCH_SIZE,))

    x = torch.stack([source[i     : i + BLOCK_SIZE    ] for i in starts])
    y = torch.stack([source[i + 1 : i + BLOCK_SIZE + 1] for i in starts])
    return x, y


# ── 4. Build the model ─────────────────────────────────────────────────────────
print("\nBuilding model...")
model = TinyGPT(
    vocab_size  = tokenizer.vocab_size,
    embed_dim   = EMBED_DIM,
    num_heads   = NUM_HEADS,
    num_layers  = NUM_LAYERS,
    block_size  = BLOCK_SIZE,
    dropout     = DROPOUT,
)

# Count how many parameters the model has
total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")


# ── 5. Optimizer ───────────────────────────────────────────────────────────────
# Adam is the most popular optimizer — it adapts the learning rate
# individually for each parameter, making training faster and stabler
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# ── 6. Loss estimator ──────────────────────────────────────────────────────────
@torch.no_grad()   # don't compute gradients here — saves memory
def estimate_loss():
    """
    Runs the model on several batches of train and val data
    and returns the average loss for each. Used just for reporting.
    """
    model.eval()   # switch to evaluation mode (disables dropout)
    losses = {}
    for split in ['train', 'val']:
        batch_losses = []
        for _ in range(50):   # average over 50 batches
            x, y = get_batch(split)
            _, loss = model(x, y)
            batch_losses.append(loss.item())
        losses[split] = sum(batch_losses) / len(batch_losses)
    model.train()  # switch back to training mode
    return losses


# ── 7. Training loop ───────────────────────────────────────────────────────────
print("\nStarting training...\n")

for step in range(MAX_STEPS):

    # Every EVAL_EVERY steps, print the current loss
    if step % EVAL_EVERY == 0:
        losses = estimate_loss()
        print(f"Step {step:4d} | train loss: {losses['train']:.4f} "
              f"| val loss: {losses['val']:.4f}")

    # Get a batch of training data
    x, y = get_batch('train')

    # Forward pass: compute predictions and loss
    logits, loss = model(x, y)

    # Backward pass: compute gradients
    optimizer.zero_grad()   # clear old gradients first
    loss.backward()         # backpropagate the loss

    # Update weights
    optimizer.step()


# ── 8. Save the trained model ──────────────────────────────────────────────────
os.makedirs("level1_tiny_gpt/checkpoints", exist_ok=True)
torch.save({
    'model_state': model.state_dict(),
    'vocab_size':  tokenizer.vocab_size,
    'chars':       tokenizer.chars,
    'embed_dim':   EMBED_DIM,
    'num_heads':   NUM_HEADS,
    'num_layers':  NUM_LAYERS,
    'block_size':  BLOCK_SIZE,
    'dropout':     DROPOUT,
}, "level1_tiny_gpt/checkpoints/model.pt")

print("\nTraining complete! Model saved to level1_tiny_gpt/checkpoints/model.pt")