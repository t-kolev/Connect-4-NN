import numpy as np
import pytest
from agents.game_utils import *
from agents.uct_agent.uct import *
import agents.uct_agent as ucta


def test_add_value():
    root = Node(None, np.zeros(1), False)
    for i in range(100):
        root.add_value(1)
    assert root.runs == 100


def test_mu():
    root = Node(None, np.zeros(1), False)
    root.add_value(1)
    root.add_value(2)
    assert root.mu() == 1.5


def test_ucb():
    board = initialize_game_state()
    root = Node(None, get_valid_moves(board), False)
    root.expand(board, PLAYER1)
    root.expand(board, PLAYER1)
    child1 = root.children[0]
    child2 = root.children[1]
    root.add_value(1)
    root.add_value(1)
    child1.add_value(1)
    child1.add_value(1)
    child2.add_value(1)
    assert child1.ucb() < child2.ucb()


def test_expand():
    board = initialize_game_state()
    root = Node(None, get_valid_moves(board), False)
    root.expand(board, PLAYER1)
    assert len(root.children) == 1


def test_expand_to_fully():
    board = initialize_game_state()
    root = Node(None, get_valid_moves(board), False)
    for i in range(get_valid_moves(board).size):
        root.expand(board, PLAYER1)
    assert root.fully_expanded


def test_expand_to_often():
    board = initialize_game_state()
    root = Node(None, get_valid_moves(board), False)
    for i in range(get_valid_moves(board).size+2):
        root.expand(board, PLAYER1)
    assert len(root.children) == get_valid_moves(board).size


def test_get_best_child_mu():
    board = initialize_game_state()
    root = Node(None, get_valid_moves(board), False)
    i = 0
    while not root.fully_expanded:
        root.expand(board, PLAYER1)
        root.children[-1].add_value(i)
        i += 1
    assert root.get_best_child_mu() == len(root.children)-1


def test_get_best_child_runs():
    board = initialize_game_state()
    root = Node(None, get_valid_moves(board), False)
    i = 0
    while not root.fully_expanded:
        root.expand(board, PLAYER1)
        root.children[-1].add_value(1)
        i += 1
    root.children[2].add_value(0)
    assert root.get_best_child_runs() == 2


def test_get_best_child_ucb():
    board = initialize_game_state()
    root = Node(None, get_valid_moves(board), False)
    i = 0
    while not root.fully_expanded:
        root.expand(board, PLAYER1)
        root.add_value(1)
        root.children[-1].add_value(1)
        if i != 2:
            root.children[-1].add_value(1)
        i += 1
    result = root.get_best_child_ucb()
    assert result[0] == root.children[2] and result[1] == 2


def test_back_propagate():
    board = np.ones((6, 7), BoardPiece) * 3
    board[:, 3] = np.zeros(6)
    root = generate_root(board, PLAYER1)
    root.expand(board, PLAYER1)
    at = root.children[0]
    at.expand(board, PLAYER2)
    at = at.children[0]
    at.expand(board, PLAYER1)
    leaf = at.children[0]
    back_propagation(leaf, 1.5)

    at = leaf
    p1 = True
    while at is not None:
        if p1:
            assert at.mu() == 1.5
        else:
            assert at.mu() == -1.5
        at = at.parent
        p1 = not p1


def test_select_next_node():
    board = np.ones((6, 7), BoardPiece) * 3
    board[:, 3] = np.zeros(6)
    original = board.copy()
    root = generate_root(board, PLAYER1)
    root.expand(board, PLAYER1)
    at = root.children[0]
    at.expand(board, PLAYER2)
    at = at.children[0]
    at.expand(board, PLAYER1)
    leaf = at.children[0]
    back_propagation(leaf, 1)
    assert leaf, PLAYER2 == select_next_node(root, original, PLAYER1)


def test_random_rollout():
    board = initialize_game_state()
    random_rollout(board, PLAYER1)
    assert (check_end_state(board, PLAYER1) != GameState.STILL_PLAYING or
            check_end_state(board, PLAYER1) != GameState.STILL_PLAYING)


def test_random_rollout_draw():
    board = np.ones((6, 7), BoardPiece) * 3
    assert random_rollout(board, PLAYER1) == 0


def test_random_rollout_win():
    board = np.ones((6, 7), BoardPiece)
    assert random_rollout(board, PLAYER1) == 1


def test_random_rollout_loose_classic():
    board = initialize_game_state()
    board[0, :] = np.array([2, 2, 2, 2, 0, 0, 0], int)
    assert random_rollout(board, PLAYER1, True) == 0


def test_random_rollout_loose_alternative():
    board = initialize_game_state()
    board[0, :] = np.array([2, 2, 2, 2, 0, 0, 0], int)
    assert random_rollout(board, PLAYER1, False) == -1


def test_generate_root_given_none():
    root = generate_root(initialize_game_state(), PLAYER1)
    assert root.parent is None and len(root.children) == 0


# the following tests could fail sometimes due to the agent following a random policy
# they can however be used to evaluate the quality of the agent (to some degree) by tracking their chance to pass
def test_one_step_victory():
    board = initialize_game_state()
    board[0, 0:3] = np.ones(3) * PLAYER1
    saved_state = {PLAYER1: None, PLAYER2: None}
    assert ucta.generate_move(board, PLAYER1, saved_state[PLAYER1])[0] == 3


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
    assert ucta.generate_move(board, PLAYER1, saved_state[PLAYER1])[0] == 4


def test_empty_P1():
    board = initialize_game_state()
    saved_state = {PLAYER1: None, PLAYER2: None}
    assert 1 < ucta.generate_move(board, PLAYER1, saved_state[PLAYER1])[0] < 5


def test_empty_P2():
    board = initialize_game_state()
    saved_state = {PLAYER1: None, PLAYER2: None}
    assert 1 < ucta.generate_move(board, PLAYER2, saved_state[PLAYER2])[0] < 5
