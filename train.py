import math
import os
import time

import torch
from tqdm import tqdm

from batches_helpers import batch_to_device, load_batch, load_batch_count
from chess_helpers import LEN_POSSIBLE_MOVES
from constants import (
    DROPOUT,
    EPOCHS,
    GRAD_CLIP,
    INPUT_SIZE,
    LR,
    METADATA_FILE_PATH,
    MIN_LR_RATIO,
    MODEL_FILE_PATH,
    MODELS_FOLDER_PATH,
    N_EMBED,
    N_HEADS,
    N_LOOPS,
    SEED,
    WARMUP_FRAC,
    WEIGHT_DECAY,
)
from metadata import set_metadata
from model import LoopedGPT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.manual_seed(SEED)
model = LoopedGPT(
    vocab_size=LEN_POSSIBLE_MOVES,
    n_token_embd=N_EMBED,
    n_attn_embd=N_EMBED,
    n_head=N_HEADS,
    block_size=INPUT_SIZE,
    dropout=DROPOUT,
    loop_times=N_LOOPS,
).to(device=device)
print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")


def adamw_param_groups(module, weight_decay):
    decay, no_decay = [], []
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "embedding" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


optimizer = torch.optim.AdamW(
    adamw_param_groups(model, WEIGHT_DECAY),
    lr=LR,
    foreach=False,
    fused=False,
)

batch_count = load_batch_count("train")
print(f"Batch count: {batch_count}")
if batch_count <= 0:
    print("No train batches. Run generate_batches.py first.")
    raise SystemExit(1)

total_steps = EPOCHS * batch_count
warmup_steps = max(1, int(total_steps * WARMUP_FRAC))
print(
    f"Training for {EPOCHS} epoch(s), {batch_count} batches/epoch, "
    f"{total_steps} steps (warmup {warmup_steps})"
)


def lr_at(step):
    min_lr = LR * MIN_LR_RATIO
    if step < warmup_steps:
        return LR * float(step + 1) / float(warmup_steps)
    t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    t = min(1.0, max(0.0, t))
    return min_lr + 0.5 * (LR - min_lr) * (1.0 + math.cos(math.pi * t))


def set_lr(step):
    lr = lr_at(step)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


@torch.no_grad()
def evaluate(split):
    n = load_batch_count(split)
    if n <= 0:
        return None
    model.eval()
    total = 0.0
    for i in range(n):
        xs, ys, illegal_mask = load_batch(split, i)
        xs, ys, illegal_mask = batch_to_device(xs, ys, illegal_mask, device)
        _, split_loss = model(xs, ys, illegal_mask=illegal_mask)
        total += split_loss.item()
    model.train()
    return total / n


def save_checkpoint():
    os.makedirs(MODELS_FOLDER_PATH, exist_ok=True)
    torch.save(model.state_dict(), MODEL_FILE_PATH)


model.train()
start = time.perf_counter()
progress = tqdm(total=total_steps, dynamic_ncols=True)
last_loss = None
global_step = 0
val_loss = None

for epoch in range(EPOCHS):
    order = torch.randperm(batch_count)
    for batch_index in order.tolist():
        xs, ys, illegal_mask = load_batch("train", batch_index)
        xs, ys, illegal_mask = batch_to_device(xs, ys, illegal_mask, device)

        current_lr = set_lr(global_step)
        _, loss = model(xs, ys, illegal_mask=illegal_mask)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        last_loss = loss.item()

        progress.set_description(
            f"Epoch {epoch + 1}/{EPOCHS}, loss: "
            f"{(last_loss if last_loss is not None else float('nan')):.4f}, "
            f"lr: {current_lr:.2e}"
        )
        progress.update(1)
        global_step += 1

    save_checkpoint()
    val_loss = evaluate("val")
    if val_loss is not None:
        tqdm.write(f"Epoch {epoch + 1}/{EPOCHS} val loss: {val_loss:.4f}")

progress.close()
save_checkpoint()
print(f"\nSaved model to {MODEL_FILE_PATH}")

if last_loss is not None:
    set_metadata(METADATA_FILE_PATH, "train_loss", last_loss)
set_metadata(METADATA_FILE_PATH, "train_epochs", EPOCHS)
set_metadata(METADATA_FILE_PATH, "train_steps", global_step)

if val_loss is not None:
    print(f"Val loss: {val_loss:.4f}")
    set_metadata(METADATA_FILE_PATH, "val_loss", val_loss)

time_taken = time.perf_counter() - start
print(f"Training took {time_taken:.2f} seconds")
set_metadata(METADATA_FILE_PATH, "train_time", time_taken)
