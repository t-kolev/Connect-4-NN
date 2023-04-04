import numpy as np
import pytest
from agents.game_utils import *
import agents.minmax_agent as mma

def test_one_step_victory():
    board = initialize_game_state()
    board[0, 0:3] = np.ones(3) * PLAYER1
    saved_state = {PLAYER1: None, PLAYER2: None}
    assert mma.generate_move(board, PLAYER1, saved_state[PLAYER1])[0] == 3

def test_two_step_victory():
    board = initialize_game_state()
    board[0, 0] = PLAYER2
    board[1, 1] = PLAYER1
    board[2, 2] = PLAYER1
    board[3, 3] = PLAYER1
    board[0, 4] = PLAYER1
    board[1, 4] = PLAYER1
    board[0, 1:4] = np.ones(3) * 3
    board[0:3, 3] = np.ones(3) * 3
    board[1, 2] = 3
    saved_state = {PLAYER1: None, PLAYER2: None}
    assert mma.generate_move(board, PLAYER1, saved_state[PLAYER1])[0] == 4
    