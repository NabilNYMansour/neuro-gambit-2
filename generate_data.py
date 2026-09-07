import os
import sys

import chess.pgn
from tqdm import tqdm

from constants import BATCH_SIZE, DATA_PATH, DATA_WRITE_PATH, MAX_GAMES, MIN_ELO

if os.path.exists(DATA_WRITE_PATH):
    answer = input(
        f"{DATA_WRITE_PATH} already exists and will be cleared. Continue? [y/N] "
    )
    if answer.strip().lower() not in ("y", "yes"):
        print("Cancelled.")
        sys.exit()

kept = 0
with open(DATA_PATH, "r") as src, open(DATA_WRITE_PATH, "w") as out:
    progress = tqdm(total=MAX_GAMES, dynamic_ncols=True)
    while kept < MAX_GAMES:
        game = chess.pgn.read_game(src)
        if game is None:
            break

        result = game.headers["Result"]
        if (result == "1-0" and int(game.headers["WhiteElo"]) >= MIN_ELO) or (
            result == "0-1" and int(game.headers["BlackElo"]) >= MIN_ELO
        ):
            print(game, file=out, end="\n\n")
            kept += 1
            progress.update(1)
    progress.close()

print(f"Wrote {kept} games to {DATA_WRITE_PATH}")
