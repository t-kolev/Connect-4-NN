import pickle
from agents.game_utils import *
import tensorflow as tf
import numpy as np
import random


class NeuralNetwork:

    def __init__(self):
        """Initializes a neural network"""
        self.memory_player1 = []
        self.memory_player2 = []

        try:
            self.load_model()
        except OSError:
            self.model = self._generate_model()

    @staticmethod
    def _generate_model():
        """Builds a neural network with random weights."""
        model = tf.keras.Sequential([
            # convolutional layer
            tf.keras.layers.Conv2D(128, (4, 4), activation=tf.nn.relu, input_shape=(6, 7, 1)),
            tf.keras.layers.MaxPooling2D(2, 2),

            # fully connected layers
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation=tf.nn.relu),
            tf.keras.layers.Dense(64, activation=tf.nn.relu),
            tf.keras.layers.Dense(3, activation=tf.nn.softmax),
        ])
        model.compile(optimizer='adam', loss=tf.keras.losses.MeanAbsoluteError(), metrics=['accuracy'])
        return model

    def reset(self):
        """Builds a neural network with random weights and saves it, overwriting the currently saved model"""
        self.model = self._generate_model()
        self.save_model()

    def save_model(self):
        """Saves the model in directory 'saved_model'."""
        self.model.save('saved_model\\my_model.h5')

    def load_model(self):
        """Load the model from directory 'saved_model'."""
        self.model = tf.keras.models.load_model('saved_model\\my_model.h5')

    @staticmethod
    def normalize_board(board: np.ndarray, player: BoardPiece):
        """
        Change the board so that the next player would be PLAYER2. Also change the pieces for 'empty' to 0.5,
        'player1' to 1 and 'player2' to 0.

            Args:
                board: the current board
                player: the current player

            Returns:
                the normalized board
        """
        board = board.astype(float)
        board[board == 0] = .5
        if player == PLAYER1:
            board[board == 2] = 0.
        else:
            board[board == 1] = 0.
            board[board == 2] = 1.
        return board

    def save_data(self, board: np.ndarray):
        """Saves the single board in memory after normalizing it.

            Args:
                board: the current board
        """
        self.memory_player1.append([self.normalize_board(board, PLAYER1)])
        self.memory_player2.append([self.normalize_board(board, PLAYER2)])

    def label_data(self, winner: BoardPiece):
        """Label the gathered data.

            Args:
                winner: The winning player or 0 in case of a draw
        """
        # choose labels
        label1 = np.array([0, 1, 0])
        label2 = np.array([0, 1, 0])
        if winner == PLAYER1:
            label1 = np.array([1, 0, 0])
            label2 = np.array([0, 0, 1])
        elif winner == PLAYER2:
            label1 = np.array([0, 0, 1])
            label2 = np.array([1, 0, 0])
        # set labels
        for board in self.memory_player1:
            board.append(label1)
        for board in self.memory_player2:
            board.append(label2)

    def history(self):
        """Saves the games for a player in the "board.pickle" file"""
        with open("board.pickle", "ab") as file:
            pickle.dump(pretty_print_board_for_normalized_board(self.memory_player1[-1][0]), file)

    @staticmethod
    def open_pickle(file_name):
        """
        Reading the file with the boards

            Args:
                file_name: path to the save file
        """
        with open(file_name, "rb") as file:
            data = pickle.load(file)


    @staticmethod
    def emptying_board_file(file_name):
        """Empties the file with the boards

            Args:
                file_name: path to the save file
        """
        with open(file_name, "wb") as file:
            pickle.dump({}, file)

    def learn(self):
        """The neural network learns from the generated data"""
        training_data = self.memory_player1 + self.memory_player2
        # shuffle the data to avoid bias for one label
        random.shuffle(training_data)

        X = []
        y = []

        for board, label in training_data:
            X.append(board)
            y.append(label)

        # reshape data to make it fit the input shape for the NN
        X = np.array(X).reshape((-1, 6, 7, 1))

        self.model.fit(X, np.array(y), epochs=1, verbose=0)

    def evaluate_board(self, board: np.ndarray, player: BoardPiece):
        """The neural network tries to predict the chance of current player winning/losing/draw for the given board.

            Args:
                board: the current board
                player: the current player

            Returns:
                a value calculated by some metric given win/loss/draw chance
        """
        # player = get_other_player(player)
        prediction = self.model.predict(np.array([self.normalize_board(board, player)]).reshape((-1, 6, 7, 1)),
                                        verbose=0)[0]
        win = prediction[0]
        draw = prediction[1]
        loss = prediction[2]
        return 1 + win - loss
