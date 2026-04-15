# prepare_data.py
# Loads our text and prepares it for GPT-2 fine-tuning

from transformers import GPT2Tokenizer
from datasets import Dataset
import os

def prepare_dataset(data_path, tokenizer, block_size=128):
    """
    Reads the raw text and chops it into chunks of block_size tokens.

    Why chunks? GPT-2 can only process a fixed number of tokens at once
    (its context window). We slide through the text in overlapping chunks
    so every part of it gets used in training.

    block_size=128 means each training example is 128 tokens long.
    GPT-2 supports up to 1024 — we use 128 to keep training fast.
    """

    print(f"Loading text from {data_path}...")
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Total characters: {len(text):,}")

    # GPT-2's tokenizer works at the word/subword level — much smarter
    # than our character tokenizer. "unweeded" might become ["un", "weed", "ed"]
    print("Tokenizing with GPT-2 tokenizer...")
    tokens = tokenizer.encode(text)
    print(f"Total tokens: {len(tokens):,}")
    print(f"(Compare: {len(text):,} chars → {len(tokens):,} tokens, "
          f"ratio: {len(text)/len(tokens):.1f} chars/token)")

    # Chop the token list into chunks of block_size
    # We drop the last chunk if it's shorter than block_size
    chunks = []
    for i in range(0, len(tokens) - block_size, block_size):
        chunk = tokens[i : i + block_size]
        chunks.append(chunk)

    print(f"Created {len(chunks)} training chunks of {block_size} tokens each")

    # Each chunk serves as BOTH input and target (shifted by 1)
    # input:  tokens[0..126]
    # target: tokens[1..127]
    # This is the same next-token prediction we did in Level 1 — just at scale
    dataset = Dataset.from_dict({
        "input_ids": chunks,
        "labels":    chunks,   # same as input — loss computed internally
    })

    return dataset


if __name__ == "__main__":
    # Load GPT-2's tokenizer (downloads ~500KB vocab file first time)
    print("Loading GPT-2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad token by default

    dataset = prepare_dataset("data/input.txt", tokenizer, block_size=128)

    print("\nSample chunk decoded back to text:")
    print("-" * 40)
    print(tokenizer.decode(dataset[0]["input_ids"]))
    print("-" * 40)
    print("\nData preparation complete!")