import numpy as np
from agents.game_utils import *

DEPTH = 5


def generate_move_minmax(
    board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> tuple[PlayerAction, Optional[SavedState]]:
    return recursive_step_root(board, player, DEPTH), saved_state


def recursive_step_root(board: np.ndarray, player: BoardPiece, depth: int) -> int:
    other_player = PLAYER1
    if player == PLAYER1:
        other_player = PLAYER2

    if connected_four(board, other_player):
        return 1
    if not depth or (board[5, :] != np.zeros(7)).all():
        return 0

    best_action = (-2, -1)
    for action in get_valid_moves(board):
        apply_player_action(board, action, player)
        value = recursive_step(board, other_player, depth - 1)
        unapply_player_action(board, action, player)
        if value > best_action[0]:
            best_action = (value, action)
    return best_action[1]


def recursive_step(board: np.ndarray, player: BoardPiece, depth: int) -> int:
    other_player = PLAYER1
    if player == PLAYER1:
        other_player = PLAYER2

    if connected_four(board, other_player):
        return 1
    if not depth or (board[5, :] != np.zeros(7)).all():
        return 0

    best_value = -2
    for action in get_valid_moves(board):
        apply_player_action(board, action, player)
        value = recursive_step(board, other_player, depth - 1)
        unapply_player_action(board, action, player)
        if value > best_value:
            best_value = value
    return -best_value
