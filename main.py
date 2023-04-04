import time

import numpy as np

from agents.game_utils import PLAYER1, PLAYER2, PLAYER1_PRINT, PLAYER2_PRINT, GameState
from agents.game_utils import initialize_game_state, pretty_print_board, apply_player_action, check_end_state, string_to_board
from typing import Callable
from agents.game_utils import GenMove
from agents.agent_human_user import user_move
import agents.random_agent as ra
import agents.minmax_agent as mma
import agents.uct_agent as ucta
import neuralNetwork.nn as nn


def human_vs_agent(
    generate_move_1: GenMove,
    generate_move_2: GenMove = user_move,
    player_1: str = "Player 1",
    player_2: str = "Player 2",
    args_1: tuple = (),
    args_2: tuple = (),
    init_1: Callable = lambda board, player: None,
    init_2: Callable = lambda board, player: None,
):
    players = (PLAYER1, PLAYER2)
    for play_first in (1, -1):
        for init, player in zip((init_1, init_2)[::play_first], players):
            init(initialize_game_state(), player)

        saved_state = {PLAYER1: None, PLAYER2: None}
        board = initialize_game_state()
        gen_moves = (generate_move_1, generate_move_2)[::play_first]
        player_names = (player_1, player_2)[::play_first]
        gen_args = (args_1, args_2)[::play_first]

        playing = True
        while playing:
            for player, player_name, gen_move, args in zip(
                players, player_names, gen_moves, gen_args,
            ):
                t0 = time.time()
                print(pretty_print_board(board))
                print(
                    f'{player_name} you are playing with {PLAYER1_PRINT if player == PLAYER1 else PLAYER2_PRINT}'
                )
                action, saved_state[player] = gen_move(
                    board.copy(), player, saved_state[player], *args
                )
                print(f"Move time: {time.time() - t0:.3f}s")
                board = apply_player_action(board, action, player)
                end_state = check_end_state(board, player)
                if end_state != GameState.STILL_PLAYING:
                    print(pretty_print_board(board))
                    if end_state == GameState.IS_DRAW:
                        print("Game ended in draw")
                    else:
                        print(
                            f'{player_name} won playing {PLAYER1_PRINT if player == PLAYER1 else PLAYER2_PRINT}'
                        )
                    playing = False
                    break


def train_nn_vs(
    generate_move_1: GenMove,
    runs: int = 1,
    generate_move_2: GenMove = ucta.generate_move,
    args_1: tuple = (),
    args_2: tuple = (100, True),
):
    """Training the neural network.

        Args:
            generate_move_1: the move of the opponent
            runs: number of games
            generate_move_2: the move of the neural network
            args_1: arguments passed to the generate_move_1 function
            args_2: arguments passed to the generate_move_2 function
    """
    from agents.game_utils import get_other_player

    for r in range(runs):
        print(f"starting run {r+1}...")
        players = (PLAYER1, PLAYER2)
        for play_first in (1, -1):
            saved_state = {PLAYER1: None, PLAYER2: None}
            net = nn.NeuralNetwork()
            saved_state[players[::play_first][1]] = net
            board = initialize_game_state()
            gen_moves = (generate_move_1, generate_move_2)[::play_first]
            gen_args = (args_1, args_2)[::play_first]

            playing = True
            while playing:
                for player, gen_move, args in zip(
                    players, gen_moves, gen_args,
                ):
                    # print(pretty_print_board(board))
                    net.save_data(board.copy())
                    action, saved_state[player] = gen_move(
                        board.copy(), player, saved_state[player], *args
                    )
                    board = apply_player_action(board, action, player)
                    end_state = check_end_state(board, player)
                    if end_state != GameState.STILL_PLAYING:
                        net.save_data(board.copy())
                        # nn = saved_state[players[::play_first][1]]
                        if net is None:
                            net = saved_state[PLAYER2]
                        if end_state == GameState.IS_DRAW:
                            net.label_data(0)

                        else:
                            net.label_data(player)
                        net.learn()
                        net.save_model()
                        playing = False
                        break


def evaluate_nn(generate_move_1: GenMove,
    runs: int = 1,
    generate_move_2: GenMove = ucta.generate_move,
    args_1: tuple = (),
    args_2: tuple = (100, True),
):
    """Evaluating the neural network.

        Args:
             generate_move_1: the move of the opponent
             runs: number of games
             generate_move_2: the move of the neural network
             args_1: arguments passed to the generate_move_1 function
             args_2: arguments passed to the generate_move_2 function

        Returns:
            win_nn: the number of times when neural network won with the opponent
            draw_nn: number of times the game ended in a draw
            loss_nn: number of times when neural network lost against the opponent
    """
    from agents.game_utils import get_other_player
    win_nn = 0
    draw_nn = 0
    loss_nn = 0
    for r in range(runs):
        print(f"starting run {r+1}...")
        players = (PLAYER1, PLAYER2)
        for play_first in (1, -1):
            saved_state = {PLAYER1: None, PLAYER2: None}
            net = nn.NeuralNetwork()
            saved_state[players[::play_first][1]] = net
            board = initialize_game_state()
            gen_moves = (generate_move_1, generate_move_2)[::play_first]
            gen_args = (args_1, args_2)[::play_first]

            playing = True
            while playing:
                for player, gen_move, args in zip(
                    players, gen_moves, gen_args,
                ):
                    # print(pretty_print_board(board))
                    action, saved_state[player] = gen_move(
                        board.copy(), player, saved_state[player], *args
                    )
                    board = apply_player_action(board, action, player)
                    end_state = check_end_state(board, player)
                    if end_state != GameState.STILL_PLAYING:
                        playing = False
                        if end_state == GameState.IS_DRAW:
                            draw_nn += 1
                        elif end_state == GameState.IS_WIN and player == players[::play_first][1]:
                            win_nn += 1
                        else:
                            loss_nn += 1
                        break
    return win_nn, draw_nn, loss_nn


def train_evaluate(runs):
    """Trains and evaluates the neural network in the number of runs specified by the given paameter.

        Args:
            runs: number of training and evaluating rounds
    """
    nn.NeuralNetwork().model.save(f'saved_model\\my_model-0.h5')
    print(f"Evaluation for nn after 0 rounds of Training:")
    win, draw, loss = evaluate_nn(ra.generate_move, 5)
    print(f"win {win}, draw {draw}, loss {loss}")
    win, draw, loss = evaluate_nn(mma.generate_move, 1)
    print(f"win {win}, draw {draw}, loss {loss}")
    win, draw, loss = evaluate_nn(ucta.generate_move, 5)
    print(f"win {win}, draw {draw}, loss {loss}")
    for i in range(runs):
        train_nn_vs(ucta.generate_move, 5)
        nn.NeuralNetwork().model.save(f'saved_model\\my_model-{i+1}.h5')
        print(f"Evaluation for nn after {(i + 1) * 5} rounds of Training:")
        win, draw, loss = evaluate_nn(ra.generate_move, 5)
        print(f"win {win}, draw {draw}, loss {loss}")
        win, draw, loss = evaluate_nn(mma.generate_move, 1)
        print(f"win {win}, draw {draw}, loss {loss}")
        win, draw, loss = evaluate_nn(ucta.generate_move, 5)
        print(f"win {win}, draw {draw}, loss {loss}")


if __name__ == "__main__":
    # nn.NeuralNetwork().reset() # WARNING: all training will be lost
    # train_evaluate(20)
    # train_nn_vs(ucta.generate_move, 200)
    human_vs_agent(ucta.generate_move, args_1=(100, True))
