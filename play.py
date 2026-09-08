import argparse
import io
import os

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import chess
import chess.svg
import pygame
import torch
from torch.nn import functional as F

from chess_helpers import (
    ITOM,
    MTOI,
    illegal_mask_from_board,
    legal_move_indices,
)
from constants import INPUT_SIZE
from model import load_model

SQ = 80
LIGHT = (240, 217, 181)
DARK = (181, 136, 99)
SELECT = (186, 202, 68)
ARROW = (80, 155, 70, 200)


def context_idx(board, device):
    history = ["."] * INPUT_SIZE + [move.uci() for move in board.move_stack]
    tokens = [MTOI[move] for move in history[-INPUT_SIZE:]]
    return torch.tensor([tokens], dtype=torch.long, device=device)


def square_at(pos, flipped):
    col, row = pos[0] // SQ, pos[1] // SQ
    if not (0 <= col < 8 and 0 <= row < 8):
        return None
    if flipped:
        return chess.square(7 - col, row)
    return chess.square(col, 7 - row)


def square_rect(square, flipped):
    file = chess.square_file(square)
    rank = chess.square_rank(square)
    col, row = (7 - file, rank) if flipped else (file, 7 - rank)
    return pygame.Rect(col * SQ, row * SQ, SQ, SQ)


def draw_arrow(screen, origin, dest, flipped):
    start = pygame.Vector2(square_rect(origin, flipped).center)
    end = pygame.Vector2(square_rect(dest, flipped).center)
    delta = end - start
    if delta.length() < 1:
        return
    direction = delta.normalize()
    start = start + direction * (SQ * 0.28)
    end = end - direction * (SQ * 0.18)
    shaft_end = end - direction * 20
    perp = pygame.Vector2(-direction.y, direction.x)
    overlay = pygame.Surface((SQ * 8, SQ * 8), pygame.SRCALPHA)
    pygame.draw.polygon(
        overlay,
        ARROW,
        [
            start + perp * 6,
            shaft_end + perp * 6,
            shaft_end - perp * 6,
            start - perp * 6,
        ],
    )
    pygame.draw.polygon(
        overlay,
        ARROW,
        [end, shaft_end + perp * 14, shaft_end - perp * 14],
    )
    screen.blit(overlay, (0, 0))


class Game:
    def __init__(self, model, device, human_color=chess.WHITE):
        self.model = model
        self.device = device
        self.human_color = human_color
        self.board = chess.Board()

    def your_turn(self):
        return (not self.board.is_game_over()) and self.board.turn == self.human_color

    def push(self, move):
        self.board.push(move)

    @torch.no_grad()
    def choose_move(self):
        idx = context_idx(self.board, self.device)
        mask = illegal_mask_from_board(self.board, device=self.device)
        logits, _ = self.model(idx, illegal_mask=mask)
        logits = logits[0, -1]
        ids = legal_move_indices(self.board)
        if not ids:
            return None
        move_id = int(torch.multinomial(F.softmax(logits, dim=-1), 1))
        try:
            move = chess.Move.from_uci(ITOM[move_id])
        except ValueError:
            move = None
        if move is None or move not in self.board.legal_moves:
            move = chess.Move.from_uci(ITOM[ids[int(logits[ids].argmax())]])
        return move


def make_pieces():
    pieces = {}
    for color in (chess.WHITE, chess.BLACK):
        for ptype in chess.PIECE_TYPES:
            svg = chess.svg.piece(chess.Piece(ptype, color), size=SQ).encode("utf-8")
            pieces[(color, ptype)] = pygame.image.load(io.BytesIO(svg)).convert_alpha()
    return pieces


def play(game):
    pygame.init()
    screen = pygame.display.set_mode((SQ * 8, SQ * 8))
    pygame.display.set_caption("Neuro Gambit")
    clock = pygame.time.Clock()
    pieces = make_pieces()
    flipped = game.human_color == chess.BLACK
    last_rank = 0 if flipped else 7
    selected = None
    pending_ai = not game.your_turn()
    bot_move = None

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type != pygame.MOUSEBUTTONDOWN or event.button != 1:
                continue
            if pending_ai or not game.your_turn():
                continue
            square = square_at(event.pos, flipped)
            if square is None:
                selected = None
                continue
            if selected is None:
                piece = game.board.piece_at(square)
                if piece is not None and piece.color == game.human_color:
                    selected = square
                continue
            move = chess.Move(selected, square)
            if game.board.piece_type_at(selected) == chess.PAWN and chess.square_rank(square) == last_rank:
                move = chess.Move(selected, square, promotion=chess.QUEEN)
            if move in game.board.legal_moves:
                game.push(move)
                selected = None
                pending_ai = not game.board.is_game_over()
            else:
                piece = game.board.piece_at(square)
                selected = (
                    square if piece is not None and piece.color == game.human_color else None
                )

        for square in chess.SQUARES:
            light = (chess.square_file(square) + chess.square_rank(square)) % 2 == 1
            color = SELECT if square == selected else (LIGHT if light else DARK)
            pygame.draw.rect(screen, color, square_rect(square, flipped))
        for square, piece in game.board.piece_map().items():
            sprite = pieces[(piece.color, piece.piece_type)]
            screen.blit(sprite, sprite.get_rect(center=square_rect(square, flipped).center))
        if bot_move is not None:
            draw_arrow(screen, bot_move.from_square, bot_move.to_square, flipped)
        if selected is not None:
            mark = pygame.Surface((SQ, SQ), pygame.SRCALPHA)
            for move in game.board.legal_moves:
                if move.from_square != selected:
                    continue
                mark.fill((0, 0, 0, 0))
                if game.board.piece_at(move.to_square):
                    pygame.draw.circle(mark, (0, 0, 0, 80), (SQ // 2, SQ // 2), SQ // 2 - 8, 6)
                else:
                    pygame.draw.circle(mark, (0, 0, 0, 55), (SQ // 2, SQ // 2), 12)
                screen.blit(mark, square_rect(move.to_square, flipped))
        pygame.display.flip()

        if pending_ai:
            pending_ai = False
            move = game.choose_move()
            if move is not None:
                game.push(move)
                bot_move = move

        clock.tick(60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "color",
        nargs="?",
        choices=("white", "black"),
        default="white",
        help="side to play",
    )
    args = parser.parse_args()
    human_color = chess.BLACK if args.color == "black" else chess.WHITE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    play(Game(load_model(device), device, human_color))


if __name__ == "__main__":
    main()
