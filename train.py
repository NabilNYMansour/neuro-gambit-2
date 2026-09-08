import os
import time

import torch
from tqdm import tqdm

from batches_helpers import batch_to_device, load_batch, load_batch_count
from chess_helpers import LEN_POSSIBLE_MOVES
from constants import (
    DROPOUT,
    INPUT_SIZE,
    ITERATIONS,
    LR,
    METADATA_FILE_PATH,
    MODEL_FILE_PATH,
    MODELS_FOLDER_PATH,
    N_EMBED,
    N_HEADS,
    N_LOOPS,
    SEED,
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

optimizer = torch.optim.Adam(model.parameters(), lr=LR, foreach=False, fused=False)

batch_count = load_batch_count("train")
print(f"Batch count: {batch_count}")
if batch_count <= 0:
    print("No train batches. Run generate_batches.py first.")
    raise SystemExit(1)


@torch.no_grad()
def evaluate(split):
    n = load_batch_count(split)
    if n <= 0:
        return None
    model.eval()
    total = 0.0
    for i in range(n):
        xs, ys = load_batch(split, i)
        xs, ys = batch_to_device(xs, ys, device)
        _, split_loss = model(xs, ys)
        total += split_loss.item()
    model.train()
    return total / n


model.train()
start = time.perf_counter()
progress = tqdm(total=ITERATIONS, dynamic_ncols=True)
last_loss = None
for step in range(ITERATIONS):
    rand_batch_index = torch.randint(0, batch_count, (1,)).item()

    xs, ys = load_batch("train", rand_batch_index)
    xs, ys = batch_to_device(xs, ys, device)

    logits, loss = model(xs, ys)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if not torch.isfinite(loss) or any(
        p.grad is not None and not torch.isfinite(p.grad).all()
        for p in model.parameters()
    ):
        print(f"Skipping non-finite step {step}")
        continue
    optimizer.step()
    last_loss = loss.item()

    progress.set_description(f"Iteration {step}, loss: {last_loss:.4f}")
    progress.update(1)

os.makedirs(MODELS_FOLDER_PATH, exist_ok=True)
torch.save(model.state_dict(), MODEL_FILE_PATH)
print(f"\nSaved model to {MODEL_FILE_PATH}")

if last_loss is not None:
    set_metadata(METADATA_FILE_PATH, "train_loss", last_loss)
set_metadata(METADATA_FILE_PATH, "train_iter", ITERATIONS)

val_loss = evaluate("val")
if val_loss is not None:
    print(f"Val loss: {val_loss:.4f}")
    set_metadata(METADATA_FILE_PATH, "val_loss", val_loss)

time_taken = time.perf_counter() - start
print(f"Training took {time_taken:.2f} seconds")
set_metadata(METADATA_FILE_PATH, "train_time", time_taken)
