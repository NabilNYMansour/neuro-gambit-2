import chess.pgn
from tqdm import tqdm

from batches_helpers import (
    clear_split_batches,
    ensure_batch_dirs,
    get_split_folder,
    iter_winner_examples,
    save_batch,
    split_name,
    write_batch_count,
)
from constants import BATCH_SIZE, DATA_WRITE_PATH, MAX_GAMES, SPLIT_NAMES

ensure_batch_dirs()
for split in SPLIT_NAMES:
    clear_split_batches(split)
buffers = {split: ([], []) for split in SPLIT_NAMES}
counts = {split: 0 for split in SPLIT_NAMES}

print("Generating data...")
game_idx = 0
progress = tqdm(total=MAX_GAMES, dynamic_ncols=True)
with open(DATA_WRITE_PATH, "r") as src:
    while True:
        game = chess.pgn.read_game(src)
        if game is None:
            break

        split = split_name(game_idx)
        xs, ys = buffers[split]
        for x, y in iter_winner_examples(game):
            xs.append(x)
            ys.append(y)
            if len(xs) == BATCH_SIZE:
                save_batch(split, counts[split], xs, ys)
                counts[split] += 1
                xs, ys = [], []

        buffers[split] = (xs, ys)
        game_idx += 1
        progress.update(1)
progress.close()

print("\nChecking for remaining batches...")
for split in SPLIT_NAMES:
    leftover = len(buffers[split][0])
    if leftover:
        print(f"{split}: dropping {leftover} leftover examples (incomplete batch)")
    write_batch_count(split, counts[split])
    print(f"{split}: {counts[split]} batches -> {get_split_folder(split)}")
print("Done.")
