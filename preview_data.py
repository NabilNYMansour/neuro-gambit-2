import chess.pgn
from torch import sys

from constants import DATA_WRITE_PATH

green = "\033[92m"

game_index = int(input("Enter game index: ")) + 1
with open(DATA_WRITE_PATH, "r") as src:
    game = None
    for _ in range(game_index):
        game = chess.pgn.read_game(src)
    if game is None:
        print("Game not found. Exiting...")
        sys.exit(1)
    moves_list = game.mainline_moves()
    moves = [move.uci() for move in moves_list]
    winner = "white" if game.headers["Result"] == "1-0" else "black"
    for i, move in enumerate(moves):
        if (winner == "white" and i % 2 == 0) or (winner == "black" and i % 2 == 1):
            moves[i] = f"{green}{move}\033[0m"
    print(f"Winner: {winner}")
    print(" ".join(moves))
