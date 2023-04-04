import numpy as np
from agents.game_utils import *


def generate_move_random(
    board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> tuple[PlayerAction, Optional[SavedState]]:
    # Choose a valid, non-full column randomly and return it as `action`
    valid_moves = []
    for row in range(6):
        if board[5, row] == 0:
            valid_moves.append(row)

    action = np.random.choice(get_valid_moves(board))
    return action, None