import os

import torch

from chess_helpers import MTOI
from constants import (
    BATCHES_FOLDER_PATH,
    CONTEXT_SIZE,
    MAX_GAMES,
    SPLIT_NAMES,
    SPLIT_RATIOS,
)


def get_split_folder(split):
    return BATCHES_FOLDER_PATH + split + "/"


def get_batch_path(split, batch_num):
    return get_split_folder(split) + f"batch_{batch_num}.pth"


def get_batch_count_path(split):
    return get_split_folder(split) + "batch_count.txt"


def load_batch_count(split):
    batch_count_path = get_batch_count_path(split)
    if os.path.exists(batch_count_path):
        with open(batch_count_path, "r") as f:
            return int(f.read())
    print("No batch count file found. Run generate_batches.py first.")
    raise SystemExit(1)


def write_batch_count(split, count):
    with open(get_batch_count_path(split), "w") as f:
        f.write(str(count))


def load_batch(split, batch_num):
    return torch.load(get_batch_path(split, batch_num), weights_only=True)


def save_batch(split, batch_num, xs, ys):
    xs_t = torch.as_tensor(xs, dtype=torch.long)
    ys_t = torch.as_tensor(ys, dtype=torch.long)
    torch.save((xs_t, ys_t), get_batch_path(split, batch_num))


def batch_to_device(xs, ys, device):
    xs = torch.as_tensor(xs, dtype=torch.long).to(device)
    ys = torch.as_tensor(ys, dtype=torch.long).to(device)
    return xs, ys


def ensure_batch_dirs():
    os.makedirs(BATCHES_FOLDER_PATH, exist_ok=True)
    for split in SPLIT_NAMES:
        os.makedirs(get_split_folder(split), exist_ok=True)


def clear_split_batches(split):
    folder = get_split_folder(split)
    for name in os.listdir(folder):
        path = folder + name
        if os.path.isfile(path):
            os.remove(path)


def split_name(game_idx):
    n_train = int(SPLIT_RATIOS[0] * MAX_GAMES)
    n_val = int(SPLIT_RATIOS[1] * MAX_GAMES)
    if game_idx < n_train:
        return "train"
    if game_idx < n_train + n_val:
        return "val"
    return "test"


def is_winner_ply(winner, ply):
    return (winner == "white" and ply % 2 == 0) or (
        winner == "black" and ply % 2 == 1
    )


def game_winner(game):
    result = game.headers["Result"]
    if result == "1-0":
        return "white"
    if result == "0-1":
        return "black"
    raise ValueError(f"Expected a decisive game, got Result={result!r}")


def iter_winner_examples(game, context_size=CONTEXT_SIZE):
    moves = ["."] * (context_size - 1) + [
        move.uci() for move in game.mainline_moves()
    ]
    winner = game_winner(game)
    for i, sub_moves in enumerate(zip(*[moves[j:] for j in range(context_size)])):
        if is_winner_ply(winner, i):
            indices = [MTOI[move] for move in sub_moves]
            yield indices[:-1], indices[-1]
