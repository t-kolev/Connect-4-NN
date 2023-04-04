import numpy as np
import pytest
from neuralNetwork.nn import *
import tensorflow as tf
import os

def test_initialize_neural_network():
    nn = NeuralNetwork()
    assert nn is not None


def test_nn():
    nn = NeuralNetwork()
    model = nn._generate_model()
    assert isinstance(model, tf.keras.Sequential)


def test_nn1():
    nn = NeuralNetwork()
    model = nn._generate_model()
    assert len(model.layers) == 6


def test_nn2():
    nn = NeuralNetwork()
    model = nn._generate_model()
    assert isinstance(model.layers[0], tf.keras.layers.Conv2D)


def test_nn3():
    nn = NeuralNetwork()
    model = nn._generate_model()
    assert model.layers[0].activation == tf.nn.relu


def test_nn4():
    nn = NeuralNetwork()
    model = nn._generate_model()
    assert model.layers[0].filters == 128


def test_nn5():
    nn = NeuralNetwork()
    model = nn._generate_model()
    assert model.layers[0].kernel_size == (4, 4)


def test_n6():
    nn = NeuralNetwork()
    model = nn._generate_model()
    assert isinstance(model.layers[1], tf.keras.layers.MaxPooling2D)


def test_nn7():
    nn = NeuralNetwork()
    model = nn._generate_model()
    assert model.layers[1].pool_size == (2, 2)


def test_nn8():
    nn = NeuralNetwork()
    model = nn._generate_model()
    assert isinstance(model.layers[2], tf.keras.layers.Flatten)


def test_nn9():
    nn = NeuralNetwork()
    model = nn._generate_model()
    assert isinstance(model.layers[3], tf.keras.layers.Dense)


def test_nn10():
    nn = NeuralNetwork()
    model = nn._generate_model()
    assert model.layers[3].activation == tf.nn.relu


def test_nn11():
    nn = NeuralNetwork()
    model = nn._generate_model()
    assert model.layers[3].units == 64


def test_nn12():
    nn = NeuralNetwork()
    model = nn._generate_model()
    assert model.layers[5].activation == tf.nn.softmax


def test_nn13():
    nn = NeuralNetwork()
    model = nn._generate_model()
    assert model.layers[5].units == 3

def test_save_model(tmp_path):
    nn = NeuralNetwork()
    model_path = os.path.join(tmp_path, 'my_model.h5')
    nn.save_model(model_path)
    assert os.path.isfile(model_path)


def test_load_model(tmp_path):
    model_path = os.path.join(tmp_path, 'my_model.h5')
    nn = NeuralNetwork()
    tf.keras.models.save_model(nn.model, model_path)
    nn.load_model(model_path)
    assert isinstance(nn.model, tf.keras.Sequential)


def test_normalize_board1():
    nn = NeuralNetwork()
    board = initialize_game_state()
    line3 = np.array([0, 0, 0, 2, 0, 0, 0], int)
    line2 = np.array([0, 0, 2, 1, 1, 0, 0], int)
    line1 = np.array([0, 2, 1, 1, 2, 0, 0], int)
    board[0, :] = line1
    board[1, :] = line2
    board[2, :] = line3
    board = nn.normalize_board(board, PLAYER1)
    assert ((board[0, :] == np.array([.5, 0., 1., 1., 0., .5, .5], float)).all() and
            (board[1, :] == np.array([.5, .5, 0., 1., 1., .5, .5], float)).all() and
            (board[2, :] == np.array([.5, .5, .5, 0., .5, .5, .5], float)).all())


def test_normalize_board2():
    nn = NeuralNetwork()
    board = initialize_game_state()
    line3 = np.array([0, 0, 0, 2, 0, 0, 0], int)
    line2 = np.array([0, 0, 2, 1, 1, 0, 0], int)
    line1 = np.array([0, 2, 1, 1, 2, 0, 0], int)
    board[0, :] = line1
    board[1, :] = line2
    board[2, :] = line3
    board = nn.normalize_board(board, PLAYER2)
    assert ((board[0, :] == np.array([.5, 1., 0., 0., 1., .5, .5], float)).all() and
            (board[1, :] == np.array([.5, .5, 1., 0., 0., .5, .5], float)).all() and
            (board[2, :] == np.array([.5, .5, .5, 1., .5, .5, .5], float)).all())


def test_save_data():
    nn = NeuralNetwork()
    board1 = initialize_game_state()
    board2 = board1.copy()
    board2[0, :] = np.array([0, 0, 1, 0, 0, 0, 0], int)
    board3 = board1.copy()
    board3[0, :] = np.array([0, 0, 1, 2, 0, 0, 0], int)
    board4 = board1.copy()
    board4[0, :] = np.array([0, 0, 0, 1, 0, 0, 0], int)
    board4[0, :] = np.array([0, 0, 1, 2, 0, 0, 0], int)
    nn.save_data(board1)
    nn.save_data(board2)
    nn.save_data(board3)
    nn.save_data(board4)
    assert ((nn.normalize_board(board1, PLAYER1) == nn.memory_player1[0]).all() and
            (nn.normalize_board(board2, PLAYER1) == nn.memory_player1[1]).all() and
            (nn.normalize_board(board3, PLAYER1) == nn.memory_player1[2]).all() and
            (nn.normalize_board(board4, PLAYER1) == nn.memory_player1[3]).all() and
            (nn.normalize_board(board1, PLAYER2) == nn.memory_player2[0]).all() and
            (nn.normalize_board(board2, PLAYER2) == nn.memory_player2[1]).all() and
            (nn.normalize_board(board3, PLAYER2) == nn.memory_player2[2]).all() and
            (nn.normalize_board(board4, PLAYER2) == nn.memory_player2[3]).all())


def test_label_data1():
    nn = NeuralNetwork()
    board1 = initialize_game_state()
    board2 = board1.copy()
    board2[0, :] = np.array([0, 0, 1, 0, 0, 0, 0], int)
    board3 = board1.copy()
    board3[0, :] = np.array([0, 0, 1, 2, 0, 0, 0], int)
    board4 = board1.copy()
    board4[0, :] = np.array([0, 0, 0, 1, 0, 0, 0], int)
    board4[0, :] = np.array([0, 0, 1, 2, 0, 0, 0], int)
    nn.save_data(board1)
    nn.save_data(board2)
    nn.save_data(board3)
    nn.save_data(board4)
    nn.label_data(PLAYER1)
    assert ((np.array([1, 0, 0]) == nn.memory_player1[0][1]).all() and
            (np.array([1, 0, 0]) == nn.memory_player1[1][1]).all() and
            (np.array([1, 0, 0]) == nn.memory_player1[2][1]).all() and
            (np.array([1, 0, 0]) == nn.memory_player1[3][1]).all() and
            (np.array([0, 0, 1]) == nn.memory_player2[0][1]).all() and
            (np.array([0, 0, 1]) == nn.memory_player2[1][1]).all() and
            (np.array([0, 0, 1]) == nn.memory_player2[2][1]).all() and
            (np.array([0, 0, 1]) == nn.memory_player2[3][1]).all())


def test_label_data2():
    nn = NeuralNetwork()
    board1 = initialize_game_state()
    board2 = board1.copy()
    board2[0, :] = np.array([0, 0, 1, 0, 0, 0, 0], int)
    board3 = board1.copy()
    board3[0, :] = np.array([0, 0, 1, 2, 0, 0, 0], int)
    board4 = board1.copy()
    board4[0, :] = np.array([0, 0, 0, 1, 0, 0, 0], int)
    board4[0, :] = np.array([0, 0, 1, 2, 0, 0, 0], int)
    nn.save_data(board1)
    nn.save_data(board2)
    nn.save_data(board3)
    nn.save_data(board4)
    nn.label_data(PLAYER2)
    assert ((np.array([0, 0, 1]) == nn.memory_player1[0][1]).all() and
            (np.array([0, 0, 1]) == nn.memory_player1[1][1]).all() and
            (np.array([0, 0, 1]) == nn.memory_player1[2][1]).all() and
            (np.array([0, 0, 1]) == nn.memory_player1[3][1]).all() and
            (np.array([1, 0, 0]) == nn.memory_player2[0][1]).all() and
            (np.array([1, 0, 0]) == nn.memory_player2[1][1]).all() and
            (np.array([1, 0, 0]) == nn.memory_player2[2][1]).all() and
            (np.array([1, 0, 0]) == nn.memory_player2[3][1]).all())


def test_label_data3():
    nn = NeuralNetwork()
    board1 = initialize_game_state()
    board2 = board1.copy()
    board2[0, :] = np.array([0, 0, 1, 0, 0, 0, 0], int)
    board3 = board1.copy()
    board3[0, :] = np.array([0, 0, 1, 2, 0, 0, 0], int)
    board4 = board1.copy()
    board4[0, :] = np.array([0, 0, 0, 1, 0, 0, 0], int)
    board4[0, :] = np.array([0, 0, 1, 2, 0, 0, 0], int)
    nn.save_data(board1)
    nn.save_data(board2)
    nn.save_data(board3)
    nn.save_data(board4)
    nn.label_data(0)
    assert ((np.array([0, 1, 0]) == nn.memory_player1[0][1]).all() and
            (np.array([0, 1, 0]) == nn.memory_player1[1][1]).all() and
            (np.array([0, 1, 0]) == nn.memory_player1[2][1]).all() and
            (np.array([0, 1, 0]) == nn.memory_player1[3][1]).all() and
            (np.array([0, 1, 0]) == nn.memory_player2[0][1]).all() and
            (np.array([0, 1, 0]) == nn.memory_player2[1][1]).all() and
            (np.array([0, 1, 0]) == nn.memory_player2[2][1]).all() and
            (np.array([0, 1, 0]) == nn.memory_player2[3][1]).all())


def test_learn():
    nn = NeuralNetwork()
    board1 = initialize_game_state()
    board2 = board1.copy()
    board2[0, :] = np.array([0, 0, 1, 0, 0, 0, 0], int)
    board3 = board1.copy()
    board3[0, :] = np.array([0, 0, 1, 2, 0, 0, 0], int)
    board4 = board1.copy()
    board4[0, :] = np.array([0, 0, 0, 1, 0, 0, 0], int)
    board4[0, :] = np.array([0, 0, 1, 2, 0, 0, 0], int)
    nn.save_data(board1)
    nn.save_data(board2)
    nn.save_data(board3)
    nn.save_data(board4)
    nn.label_data(PLAYER1)
    nn.learn()
    assert True


def test_evaluate():
    nn = NeuralNetwork()
    board = initialize_game_state()
    value = nn.evaluate_board(board, PLAYER1)
    assert value == nn.evaluate_board(board, PLAYER1)


def test_save_model2():
    board = initialize_game_state()
    nn = NeuralNetwork()
    evaluation = nn.evaluate_board(board, PLAYER1)
    nn.save_model()
    nn.load_model()
    assert evaluation == nn.evaluate_board(board, PLAYER1)


def test_reset_model():
    board = initialize_game_state()
    nn = NeuralNetwork()
    evaluation = nn.evaluate_board(board, PLAYER1)
    nn.reset()
    assert evaluation != nn.evaluate_board(board, PLAYER1)


def test_savedboard():
    nn = NeuralNetwork()
    board1 = initialize_game_state()
    board1[0, :] = np.array([0, 0, 1, 0, 0, 0, 0], int)
    nn.save_data(board1)
    nn.open_pickle("board.pickle")
    assert True
