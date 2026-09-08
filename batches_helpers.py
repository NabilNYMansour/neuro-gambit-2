import os

import torch

from chess_helpers import LEN_POSSIBLE_MOVES, MTOI, legal_move_indices
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
    data = torch.load(get_batch_path(split, batch_num), weights_only=True)
    if not isinstance(data, (tuple, list)) or len(data) != 3:
        print("Batches missing illegal-move masks. Re-run generate_batches.py.")
        raise SystemExit(1)
    return data


def save_batch(split, batch_num, xs, ys, legal_ids_batch):
    xs_t = torch.as_tensor(xs, dtype=torch.long)
    ys_t = torch.as_tensor(ys, dtype=torch.long)
    mask_t = torch.ones((len(legal_ids_batch), LEN_POSSIBLE_MOVES), dtype=torch.bool)
    for i, ids in enumerate(legal_ids_batch):
        if ids:
            mask_t[i, ids] = False
    torch.save((xs_t, ys_t, mask_t), get_batch_path(split, batch_num))


def batch_to_device(xs, ys, illegal_mask, device):
    xs = torch.as_tensor(xs, dtype=torch.long).to(device)
    ys = torch.as_tensor(ys, dtype=torch.long).to(device)
    illegal_mask = torch.as_tensor(illegal_mask, dtype=torch.bool).to(device)
    return xs, ys, illegal_mask


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
    mainline = list(game.mainline_moves())
    moves = ["."] * (context_size - 1) + [move.uci() for move in mainline]
    winner = game_winner(game)
    board = game.board()
    for i, move in enumerate(mainline):
        if is_winner_ply(winner, i):
            indices = [MTOI[m] for m in moves[i : i + context_size]]
            yield indices[:-1], indices[-1], legal_move_indices(board)
        board.push(move)
