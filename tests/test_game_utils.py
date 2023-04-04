import numpy as np
import pytest
from agents.game_utils import *


### pretty_print_board tests ###

def test_pretty_print_board_empty():
    game_state = initialize_game_state()
    edge = "|===============|\n"
    middle = "|               |\n"
    bottom = "| 0 1 2 3 4 5 6 |"
    expected = edge + middle * 6 + edge + bottom
    assert pretty_print_board(game_state) == expected


def test_pretty_print_board_tower():
    game_state = initialize_game_state()
    game_state[:, 3] = 2 * np.ones(6)
    edge = "|===============|\n"
    middle = "|       O       |\n"
    bottom = "| 0 1 2 3 4 5 6 |"
    expected = edge + middle * 6 + edge + bottom
    assert pretty_print_board(game_state) == expected


def test_pretty_print_board_reasonable_game_state():
    game_state = initialize_game_state()
    line3 = np.array([0, 0, 0, 2, 0, 0, 0], int)
    line2 = np.array([0, 0, 2, 1, 1, 0, 0], int)
    line1 = np.array([0, 2, 1, 1, 2, 0, 0], int)
    game_state[0, :] = line1
    game_state[1, :] = line2
    game_state[2, :] = line3
    edge = "|===============|\n"
    blank = "|               |\n"
    line3_s = "|       O       |\n"
    line2_s = "|     O X X     |\n"
    line1_s = "|   O X X O     |\n"
    bottom = "| 0 1 2 3 4 5 6 |"
    expected = edge + blank + blank + blank + line3_s + line2_s + line1_s + edge + bottom
    assert pretty_print_board(game_state) == expected


### string_to_board tests ###

def test_string_to_board_empty():
    expected = initialize_game_state()
    edge = "|===============|\n"
    middle = "|               |\n"
    bottom = "| 0 1 2 3 4 5 6 |"
    board_string = edge + middle + middle + middle + middle + middle + middle + edge + bottom
    assert (string_to_board(board_string) == expected).all()


def test_string_to_board_tower():
    expected = initialize_game_state()
    expected[:, 3] = 2 * np.ones(6)
    edge = "|===============|\n"
    middle = "|       O       |\n"
    bottom = "| 0 1 2 3 4 5 6 |"
    board_string = edge + middle + middle + middle + middle + middle + middle + edge + bottom
    assert (string_to_board(board_string) == expected).all()


def test_string_to_board_reasonable_game_state():
    expected = initialize_game_state()
    line3 = np.array([0, 0, 0, 2, 0, 0, 0], int)
    line2 = np.array([0, 0, 2, 1, 1, 0, 0], int)
    line1 = np.array([0, 2, 1, 1, 2, 0, 0], int)
    expected[0, :] = line1
    expected[1, :] = line2
    expected[2, :] = line3
    edge = "|===============|\n"
    blank = "|               |\n"
    line3_s = "|       O       |\n"
    line2_s = "|     O X X     |\n"
    line1_s = "|   O X X O     |\n"
    bottom = "| 0 1 2 3 4 5 6 |"
    board_string = edge + blank + blank + blank + line3_s + line2_s + line1_s + edge + bottom
    assert (string_to_board(board_string) == expected).all()


### apply_player_action tests ###

def test_apply_player_action_invalid_row_low():
    board = initialize_game_state()
    with pytest.raises(ValueError):
        apply_player_action(board, -1, PLAYER1)


def test_apply_player_action_invalid_row_high():
    board = initialize_game_state()
    with pytest.raises(ValueError):
        apply_player_action(board, 100, PLAYER1)


def test_apply_player_action_invalid_row_full():
    board = initialize_game_state()
    board[:, 3] = 2 * np.ones(6)
    with pytest.raises(ValueError):
        apply_player_action(board, 3, PLAYER1)


def test_apply_player_action_first_P1():
    for action in range(7):
        board = initialize_game_state()
        expected = board.copy()
        expected[0, action] = PLAYER1
        assert (apply_player_action(board, action, PLAYER1) == expected).all()


def test_apply_player_action_first_P2():
    for action in range(7):
        board = initialize_game_state()
        expected = board.copy()
        expected[0, action] = PLAYER2
        assert (apply_player_action(board, action, PLAYER2) == expected).all()


def test_apply_player_action_reasonable_game_P1():
    board = initialize_game_state()
    line3 = np.array([0, 0, 0, 2, 0, 0, 0], int)
    line2 = np.array([0, 0, 2, 1, 1, 0, 0], int)
    line1 = np.array([0, 2, 1, 1, 2, 0, 0], int)
    board[0, :] = line1
    board[1, :] = line2
    board[2, :] = line3
    expected = board.copy()
    expected[2, 4] = PLAYER1
    assert (apply_player_action(board, 4, PLAYER1) == expected).all()


def test_apply_player_action_reasonable_game_P2():
    board = initialize_game_state()
    line3 = np.array([0, 0, 0, 2, 0, 0, 0], int)
    line2 = np.array([0, 0, 2, 1, 1, 0, 0], int)
    line1 = np.array([0, 2, 1, 1, 2, 0, 0], int)
    board[0, :] = line1
    board[1, :] = line2
    board[2, :] = line3
    expected = board.copy()
    expected[2, 4] = PLAYER2
    assert (apply_player_action(board, 4, PLAYER2) == expected).all()


### connected_four tests ###

def test_connected_four_horizontal_P1():
    board = initialize_game_state()
    board[3, 3:7] = np.ones(4) * PLAYER1
    assert connected_four(board, PLAYER1) == True


def test_connected_four_horizontal_P2():
    board = initialize_game_state()
    board[3, 3:7] = np.ones(4) * PLAYER2
    assert connected_four(board, PLAYER2) == True


def test_connected_four_vertical_P1():
    board = initialize_game_state()
    board[2:6, 2] = np.ones(4) * PLAYER1
    assert connected_four(board, PLAYER1) == True


def test_connected_four_vertical_P2():
    board = initialize_game_state()
    board[2:6, 2] = np.ones(4) * PLAYER2
    assert connected_four(board, PLAYER2) == True


def test_connected_four_empty():
    board = initialize_game_state()
    assert connected_four(board, PLAYER2) == False


def test_connected_four_empty_reasonable_game_P2():
    board = initialize_game_state()
    line3 = np.array([0, 0, 0, 2, 0, 0, 0], int)
    line2 = np.array([0, 0, 2, 1, 1, 0, 0], int)
    line1 = np.array([0, 2, 1, 1, 2, 0, 0], int)
    board[0, :] = line1
    board[1, :] = line2
    board[2, :] = line3
    assert (connected_four(board, PLAYER1) == False and
            connected_four(board, PLAYER2) == False)


def test_connected_four_diagonal_P1_1():
    board = initialize_game_state()
    for i in range(4):
        board[i + 1, i] = PLAYER1
    assert connected_four(board, PLAYER1) == True


def test_connected_four_diagonal_P2_1():
    board = initialize_game_state()
    for i in range(4):
        board[i, i + 1] = PLAYER2
    assert connected_four(board, PLAYER2) == True


def test_connected_four_diagonal_P1():
    board = initialize_game_state()
    for i in range(4):
        board[4 - i, i] = PLAYER1
    assert connected_four(board, PLAYER1) == True


def test_connected_four_diagonal_P2():
    board = initialize_game_state()
    for i in range(4):
        board[5 - i, i + 1] = PLAYER2
    assert connected_four(board, PLAYER2) == True


### check_end_state tests ###


def test_check_end_state_full():
    board = np.array([[2, 2, 2, 1, 2, 1, 1],
                      [2, 1, 1, 2, 2, 1, 2],
                      [2, 2, 1, 2, 1, 1, 1],
                      [1, 2, 1, 2, 2, 2, 1],
                      [1, 2, 1, 1, 1, 1, 2],
                      [2, 1, 2, 2, 2, 1, 1]])
    assert (not check_end_state(board, PLAYER1) == GameState.STILL_PLAYING and
            not check_end_state(board, PLAYER2) == GameState.STILL_PLAYING)


def test_check_end_state_draw():
    board = np.ones((6, 7), BoardPiece) * 3
    assert (check_end_state(board, PLAYER1) == GameState.IS_DRAW and
            check_end_state(board, PLAYER2) == GameState.IS_DRAW)


def test_check_end_state_win():
    board = np.ones((6, 7), BoardPiece) * PLAYER2
    assert (check_end_state(board, PLAYER2) == GameState.IS_WIN)


def test_check_end_state_playing():
    board = initialize_game_state()
    assert (check_end_state(board, PLAYER2) == GameState.STILL_PLAYING)


### get_other_player tests ###


def test_get_other_player_P1():
    assert get_other_player(PLAYER1) == PLAYER2


def test_get_other_player_P2():
    assert get_other_player(PLAYER2) == PLAYER1
