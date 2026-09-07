import chess

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
