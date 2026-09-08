import chess
import torch

POSSIBLE_MOVES = ['.']
for from_sq in chess.SQUARES:
    from_name = chess.SQUARE_NAMES[from_sq]
    from_rank = chess.square_rank(from_sq)
    for to_sq in chess.SQUARES:
        if from_sq == to_sq:
            continue
        to_name = chess.SQUARE_NAMES[to_sq]
        POSSIBLE_MOVES.append(from_name + to_name)
        to_rank = chess.square_rank(to_sq)
        if (from_rank == 6 and to_rank == 7) or (
            from_rank == 1 and to_rank == 0
        ):
            for promo in "qrbn":
                POSSIBLE_MOVES.append(from_name + to_name + promo)

LEN_POSSIBLE_MOVES = len(POSSIBLE_MOVES)

MTOI = {move: i for i, move in enumerate(POSSIBLE_MOVES)}
ITOM = {i: move for move, i in MTOI.items()}


def legal_move_indices(board):
    ids = []
    for move in board.legal_moves:
        idx = MTOI.get(move.uci())
        if idx is not None:
            ids.append(idx)
    return ids


def illegal_mask_from_board(board, device=None):
    mask = torch.ones(LEN_POSSIBLE_MOVES, dtype=torch.bool, device=device)
    ids = legal_move_indices(board)
    if ids:
        mask[torch.as_tensor(ids, dtype=torch.long, device=device)] = False
    return mask
