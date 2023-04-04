from enum import Enum
from typing import Callable, Optional
import numpy as np

BoardPiece = np.int8  # The data type (dtype) of the board
NO_PLAYER = BoardPiece(0)  # board[i, j] == NO_PLAYER where the position is empty
PLAYER1 = BoardPiece(1)  # board[i, j] == PLAYER1 where player 1 (player to move first) has a piece
PLAYER2 = BoardPiece(2)  # board[i, j] == PLAYER2 where player 2 (player to move second) has a piece

BoardPiecePrint = str  # dtype for string representation of BoardPiece
NO_PLAYER_PRINT = BoardPiecePrint(' ')
PLAYER1_PRINT = BoardPiecePrint('X')
PLAYER2_PRINT = BoardPiecePrint('O')

PlayerAction = np.int8  # The column to be played


class SavedState:
    pass


GenMove = Callable[
    [np.ndarray, BoardPiece, Optional[SavedState]],  # Arguments for the generate_move function
    tuple[PlayerAction, Optional[SavedState]]  # Return type of the generate_move function
]


class GameState(Enum):
    IS_WIN = 1
    IS_DRAW = -1
    STILL_PLAYING = 0


def initialize_game_state() -> np.ndarray:
    """
    Returns an ndarray, shape (6, 7) and data type (dtype) BoardPiece, initialized to 0 (NO_PLAYER).
    """
    return np.zeros((6, 7), BoardPiece)


def pretty_print_board(board: np.ndarray) -> str:
    """
    Should return `board` converted to a human readable string representation,
    to be used when playing or printing diagnostics to the console (stdout). The piece in
    board[0, 0] should appear in the lower-left. Here's an example output, note that we use
    PLAYER1_Print to represent PLAYER1 and PLAYER2_Print to represent PLAYER2):
    |==============|
    |              |
    |              |
    |    X X       |
    |    O X X     |
    |  O X O O     |
    |  O O X X     |
    |==============|
    |0 1 2 3 4 5 6 |
    """
    edge = "|===============|\n"
    bottom = "| 0 1 2 3 4 5 6 |"
    board_string = edge + bottom
    for i in range(6):
        current_row = "|"
        for token in board[i, :]:
            current_row += " "
            if int(token) == NO_PLAYER:
                current_row += NO_PLAYER_PRINT
            if int(token) == PLAYER1:
                current_row += PLAYER1_PRINT
            if int(token) == PLAYER2:
                current_row += PLAYER2_PRINT
        current_row += " |\n"
        board_string = current_row + board_string
    board_string = edge + board_string

    return board_string

def pretty_print_board_for_normalized_board(board: np.ndarray) -> str:
    """
    Should return `board` converted to a human readable string representation,
    to be used when playing or printing diagnostics to the console (stdout). The piece in
    board[0, 0] should appear in the lower-left. Here's an example output, note that we use
    PLAYER1_Print to represent PLAYER1 and PLAYER2_Print to represent PLAYER2):
    |==============|
    |              |
    |              |
    |    X X       |
    |    O X X     |
    |  O X O O     |
    |  O O X X     |
    |==============|
    |0 1 2 3 4 5 6 |
    """
    edge = "|===============|\n"
    bottom = "| 0 1 2 3 4 5 6 |"
    board_string = edge + bottom
    for i in range(6):
        current_row = "|"
        for token in board[i, :]:
            current_row += " "
            if float(token) == .5:
                current_row += NO_PLAYER_PRINT
            if float(token) == 1.:
                current_row += PLAYER1_PRINT
            if float(token) == 0.:
                current_row += PLAYER2_PRINT
        current_row += " |\n"
        board_string = current_row + board_string
    board_string = edge + board_string

    return board_string
def string_to_board(pp_board: str) -> np.ndarray:
    """
    Takes the output of pretty_print_board and turns it back into an ndarray.
    This is quite useful for debugging, when the agent crashed and you have the last
    board state as a string.
    """
    array_board = np.zeros((6, 7), BoardPiece)
    linewidth = 18
    for line in range(6):
        at = (6 - line) * linewidth
        for row in range(7):
            at += 2
            if pp_board[at] == PLAYER1_PRINT:
                array_board[line, row] = PLAYER1
            if pp_board[at] == PLAYER2_PRINT:
                array_board[line, row] = PLAYER2

    return array_board


def apply_player_action(board: np.ndarray, action: PlayerAction, player: BoardPiece) -> np.ndarray:
    """
    Sets board[i, action] = player, where i is the lowest open row. Raises a ValueError
    if action is not a legal move. If it is a legal move, the modified version of the
    board is returned and the original board should remain unchanged (i.e., either set
    back or copied beforehand).
    """
    # check if action is valid
    if action < 0 or action > 6:
        raise ValueError

    # if row is full
    if board[5, action] != 0:
        raise ValueError

    # apply action
    for i in range(6):
        if board[i, action] == 0:
            board[i, action] = player
            break

    return board


def unapply_player_action(board: np.ndarray, action: PlayerAction, player: BoardPiece) -> np.ndarray:
    """
    Sets board[i, action] = 0, where i is the lowest used row. Raises a ValueError
    if action is not a legal move. If it is a legal move, the modified version of the
    board is returned and the original board should remain unchanged (i.e., either set
    back or copied beforehand).
    """
    # check if action is valid
    if action < 0 or action > 6:
        raise ValueError

    # if row is empty
    if board[0, action] == 0:
        raise ValueError

    # apply action
    for i in range(6):
        if board[5 - i, action] == player:
            board[5 - i, action] = 0
            break

    return board


def get_valid_moves(board: np.ndarray) -> np.ndarray:
    """"
    Returns a List containing indexes of all rows, that are not full yet
    """
    valid_moves = []
    for i, row in enumerate(board[5, :]):
        if not row:
            valid_moves.append(i)
    return np.array(valid_moves)


def connected_four(board: np.ndarray, player: BoardPiece) -> bool:
    """
    Returns True if there are four adjacent pieces equal to `player` arranged
    in either a horizontal, vertical, or diagonal line. Returns False otherwise.
    """
    win_condition = np.ones(4) * player

    # check horizontal
    for line in range(6):
        for i in range(4):
            if (board[line, i:i + 4] == win_condition).all():
                return True

    # check vertical
    for row in range(7):
        for i in range(3):
            if (board[i:i + 4, row] == win_condition).all():
                return True

    # check diagonal \
    for line in range(3):
        for row in range(4):
            c = 0
            for i in range(4):
                if board[line + i, row + i] == player:
                    c += 1
            if c == 4:
                return True

    # check diagonal /
    for line in range(3):
        for row in range(4):
            c = 0
            for i in range(4):
                if board[line + i, 3 + row - i] == player:
                    c += 1
            if c == 4:
                return True

    return False


def check_end_state(board: np.ndarray, player: BoardPiece) -> GameState:
    """
    Returns the current game state for the current `player`, i.e. has their last
    action won (GameState.IS_WIN) or drawn (GameState.IS_DRAW) the game,
    or is play still on-going (GameState.STILL_PLAYING)?
    """
    # if player won
    if connected_four(board, player):
        return GameState.IS_WIN

    # if game is a draw
    if (board[5, :] != np.zeros(7)).all():
        return GameState.IS_DRAW

    # if game is still playing
    return GameState.STILL_PLAYING


def get_other_player(player: BoardPiece) -> BoardPiece:
    """Returns the other player"""
    if player == PLAYER1:
        return PLAYER2
    return PLAYER1
