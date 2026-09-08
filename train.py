import time

import torch
from tqdm import tqdm

from batches_helpers import batch_to_device, load_batch, load_batch_count
from constants import (
    EPOCHS,
    GRAD_CLIP,
    LR,
    METADATA_FILE_PATH,
    MODEL_FILE_PATH,
    SEED,
    WARMUP_FRAC,
    WEIGHT_DECAY,
)
from metadata import set_metadata
from model import (
    adamw_param_groups,
    create_model,
    evaluate,
    save_checkpoint,
    set_lr,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.manual_seed(SEED)
model = create_model(device)
print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

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

model.train()
start = time.perf_counter()
progress = tqdm(total=total_steps, dynamic_ncols=True)
last_loss = None
global_step = 0
val_metrics = None

for epoch in range(EPOCHS):
    order = torch.randperm(batch_count)
    for batch_index in order.tolist():
        xs, ys, illegal_mask = load_batch("train", batch_index)
        xs, ys, illegal_mask = batch_to_device(xs, ys, illegal_mask, device)

        current_lr = set_lr(optimizer, global_step, warmup_steps, total_steps)
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

    save_checkpoint(model)
    val_metrics = evaluate(model, "val", device)
    if val_metrics is not None:
        val_loss, val_acc = val_metrics
        tqdm.write(
            f"Epoch {epoch + 1}/{EPOCHS} val loss: {val_loss:.4f}, "
            f"val acc: {val_acc:.4f}"
        )

progress.close()
save_checkpoint(model)
print(f"\nSaved model to {MODEL_FILE_PATH}")

if last_loss is not None:
    set_metadata(METADATA_FILE_PATH, "train_loss", last_loss)
set_metadata(METADATA_FILE_PATH, "train_epochs", EPOCHS)
set_metadata(METADATA_FILE_PATH, "train_steps", global_step)

if val_metrics is not None:
    val_loss, val_acc = val_metrics
    print(f"Val loss: {val_loss:.4f}")
    print(f"Val accuracy: {val_acc:.4f}")
    set_metadata(METADATA_FILE_PATH, "val_loss", val_loss)
    set_metadata(METADATA_FILE_PATH, "val_accuracy", val_acc)

time_taken = time.perf_counter() - start
print(f"Training took {time_taken:.2f} seconds")
set_metadata(METADATA_FILE_PATH, "train_time", time_taken)
