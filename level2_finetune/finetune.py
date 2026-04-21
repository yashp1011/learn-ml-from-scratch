# finetune.py
# Downloads GPT-2 and fine-tunes it on our Shakespeare text

import os
import sys
import torch
from transformers import (
    GPT2LMHeadModel,   # GPT-2 with a language model head (predicts next token)
    GPT2Tokenizer,
    Trainer,           # Hugging Face's built-in training loop
    TrainingArguments, # All training settings in one object
    DataCollatorForLanguageModeling,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from level2_finetune.prepare_data import prepare_dataset


# ── Settings ───────────────────────────────────────────────────────────────────
MODEL_NAME  = "gpt2"          # smallest GPT-2 (117M parameters)
BLOCK_SIZE  = 128             # tokens per training chunk
EPOCHS      = 10              # how many times to pass through the full dataset
BATCH_SIZE  = 2               # small batch — fine for CPU training
OUTPUT_DIR  = "level2_finetune/checkpoints"


# ── 1. Load tokenizer and model ────────────────────────────────────────────────
print("Loading GPT-2 tokenizer and model...")
print("(This downloads ~500MB the first time — please wait)\n")

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# This downloads the full GPT-2 model with all 117M pre-trained weights
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

total_params = sum(p.numel() for p in model.parameters())
print(f"GPT-2 loaded — {total_params:,} parameters")
print(f"(That's {total_params / 1e6:.0f}M parameters vs your TinyGPT's 43K)\n")


# ── 2. Prepare dataset ─────────────────────────────────────────────────────────
print("Preparing dataset...")
dataset = prepare_dataset("data/input.txt", tokenizer, BLOCK_SIZE)

# Split 90% train, 10% validation
split      = dataset.train_test_split(test_size=0.1, seed=42)
train_data = split["train"]
val_data   = split["test"]
print(f"Train chunks: {len(train_data)} | Val chunks: {len(val_data)}\n")


# ── 3. Data collator ───────────────────────────────────────────────────────────
# This handles padding and automatically shifts inputs/labels by 1
# so the model learns to predict the next token
collator = DataCollatorForLanguageModeling(
    tokenizer = tokenizer,
    mlm       = False,   # mlm=False means causal LM (GPT-style), not masked LM (BERT-style)
)


# ── 4. Training arguments ──────────────────────────────────────────────────────
# TrainingArguments is Hugging Face's way of packaging all hyperparameters
training_args = TrainingArguments(
    output_dir              = OUTPUT_DIR,
    num_train_epochs        = EPOCHS,
    per_device_train_batch_size = BATCH_SIZE,
    per_device_eval_batch_size  = BATCH_SIZE,
    eval_strategy           = "epoch",   # evaluate after each epoch
    save_strategy           = "epoch",   # save checkpoint after each epoch
    logging_steps           = 5,
    learning_rate           = 5e-5,      # smaller than Level 1 — we're fine-tuning
                                         # not training from scratch, so smaller steps
    warmup_steps            = 10,        # gradually increase lr at the start
    weight_decay            = 0.01,      # mild regularization to prevent overfitting
    save_total_limit        = 2,         # only keep the 2 most recent checkpoints
    report_to               = "none",    # don't send logs to wandb etc.
)


# ── 5. Trainer ─────────────────────────────────────────────────────────────────
# Hugging Face's Trainer replaces the manual training loop we wrote in Level 1
# It handles: batching, gradient accumulation, evaluation, checkpointing
trainer = Trainer(
    model           = model,
    args            = training_args,
    train_dataset   = train_data,
    eval_dataset    = val_data,
    data_collator   = collator,
)


# ── 6. Fine-tune ───────────────────────────────────────────────────────────────
print("Starting fine-tuning...")
print("(Running on CPU — this will take 3-8 minutes)\n")
trainer.train()


# ── 7. Save the final model ────────────────────────────────────────────────────
final_path = os.path.join(OUTPUT_DIR, "final")
os.makedirs(final_path, exist_ok=True)
model.save_pretrained(final_path)
tokenizer.save_pretrained(final_path)
print(f"\nFine-tuning complete! Model saved to {final_path}")


# ── 8. Quick generation test ───────────────────────────────────────────────────
print("\n" + "="*50)
print("GENERATION TEST — after fine-tuning")
print("="*50)

model.eval()
prompts = [
    "Hamlet:",
    "Horatio:",
    "To be or not to be",
]

for prompt in prompts:
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    attention_mask = torch.ones(inputs.shape, dtype=torch.long)
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens  = 80,
            temperature     = 0.9,
            do_sample       = True,
            top_k           = 50,     # only sample from top 50 most likely tokens
            top_p           = 0.95,   # nucleus sampling — cuts off unlikely tail
            pad_token_id    = tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n{generated}")
    print("-" * 40)