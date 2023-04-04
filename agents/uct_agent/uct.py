import numpy as np
from agents.game_utils import *
from neuralNetwork import nn


class Node:
    """class that can be used to construct and navigate a search tree

    This class is best used to bild a tree for MCTS using UCB as policy.

    Attributes:
        parent: The parent node (should be None for root node).
        actions: A np.ndarray actions that are possible on the board, the node represents.
        children: A list containing all children, that were created.
        self.fully_expanded: A boolean indicating if all possible children exist.
        value: Contains the cumulative reward over all runs.
        runs: Number of runs.
    """

    def __init__(self, parent, actions: np.ndarray, final: bool):
        """Initializes Node with no children created"""
        self.parent = parent
        self.actions = actions
        self.children = []
        self.fully_expanded = False
        self.final = final
        self.value = 0
        self.runs = 0

    def ucb(self, b=1):
        """Returns the UCB-b value of the node

        Standard is UCB1
        """
        return self.mu() + b * np.sqrt(2 * np.log(self.parent.runs) / self.runs)

    def mu(self):
        """Returns mu value of the node"""
        if self.runs == 0:
            raise ValueError
        return self.value / self.runs

    def add_value(self, value):
        """Adds reward to value and increments runs"""
        self.value += value
        self.runs += 1

    def expand(self, board: np.ndarray, player: BoardPiece):
        """Creates a new child node and changes bord"""
        if not self.fully_expanded:
            apply_player_action(board, self.actions[len(self.children)], player)
            new_child = Node(self, get_valid_moves(board), check_end_state(board, player) != GameState.STILL_PLAYING)
            self.children.append(new_child)
            if self.actions.size == len(self.children):
                self.fully_expanded = True

    def get_best_child_mu(self):
        """Returns the index of the child with the best mu value"""
        best_child = self.children[0]
        best_child_index = 0
        for i, child in enumerate(self.children):
            if best_child.mu() < child.mu():
                best_child = child
                best_child_index = i
        return best_child_index

    def get_best_child_runs(self):
        """Returns the index of the child with the most runs"""
        best_child = self.children[0]
        best_child_index = 0
        for i, child in enumerate(self.children):
            if best_child.runs < child.runs:
                best_child = child
                best_child_index = i
        return best_child_index

    def get_best_child_ucb(self, b=1):
        """Returns the child and corresponding index with the best ucb value"""
        best_child = self.children[0]
        best_child_index = 0
        for i, child in enumerate(self.children):
            if best_child.ucb(b) < child.ucb(b):
                best_child = child
                best_child_index = i
        return best_child, best_child_index


def random_rollout(board: np.ndarray, player: BoardPiece, classic: bool = False):
    """Plays with random policy until the game reaches a terminal state

    Args:
        board: starting board
        player: starting player
        classic: changes the evaluation of the terminal state.
                     If true, only wining will give a reward other than 0
                     If false, loosing wils also "punish" the agent by retuning the reward -1

    Returns:
         1 if the starting player won
        -1 if the other player won and classic is false
         0 otherwise
    """
    original_player = player
    other_player = get_other_player(player)

    while check_end_state(board, player) == GameState.STILL_PLAYING:
        player, other_player = other_player, player
        apply_player_action(board, np.random.choice(get_valid_moves(board)), player)

    if check_end_state(board, player) == GameState.IS_WIN:
        if player == original_player:
            return 1
        if not classic:
            return -1
    return 0


def back_propagation(at: Node, reward):
    """propagates the reward backwards until root node is reached

        A reward for one player will "punish" the other by adding the negated reward

        Args:
            at: the node where the back propagation starts
            reward: the number that will be added/subtracted from the nodes on the path from the root to "at"
    """
    while at is not None:
        at.add_value(reward)
        at = at.parent
        reward = -reward


def select_next_node(at: Node,
                     board: np.ndarray,
                     player: BoardPiece,
                     beta=1
                     ) -> tuple[Node, BoardPiece]:
    """traverses the tree until a non fully expanded node is reached

            Args:
                at: the node where the selection starts
                board: the current board
                player: the current player
                beta: parameter for UCB
                      1 means UCB1 will be used

            Returns:
                the next node to expand
                the player who´s turn it is
    """

    other_player = get_other_player(player)
    while not at.final and at.fully_expanded:
        # choose child with best UCB value
        best_child, best_child_index = at.get_best_child_ucb(beta)
        # apply action corresponding to chosen child
        action = at.actions[best_child_index]
        apply_player_action(board, action, player)
        # prepare next iteration
        player, other_player = other_player, player
        at = best_child
    return at, player


def generate_move_uct(board: np.ndarray,
                      player: BoardPiece,
                      saved_state: Optional[SavedState],
                      max_simulations: int = 1000,
                      use_nn: bool = False
                      ) -> tuple[PlayerAction, Optional[SavedState]]:
    """Constructs a tree step by step according to MCTS witch UCB as policy

        Args:
            board: the current board
            player: the current player
            saved_state: the neural network (if use_nn == true)
            max_simulations: maximal number of MCTS iterations (change this to adjust runtime/difficulty)
            use_nn: bool

        Returns:
            the best move that was found
    """
    # initialize
    if use_nn:
        if saved_state is None:
            net = nn.NeuralNetwork()
        else:
            net = saved_state
        evaluate = net.evaluate_board
    else:
        evaluate = random_rollout

    original_board = board.copy()
    original_player = player

    # create root node
    root = generate_root(board, player)
    c = 0

    while c < max_simulations:
        # initialize loop
        at = root
        board = original_board.copy()
        player = original_player

        # find a non fully expanded node
        at, player = select_next_node(at, board, player)
        other_player = get_other_player(player)

        # in case the node found is not a terminal state
        if not at.final:
            # create a new child for node
            at.expand(board, player)
            player, other_player = other_player, player
            at = at.children[-1]
        # evaluate current board state
        player, other_player = other_player, player
        reward = evaluate(board.copy(), player)

        # propagate reward backwards
        back_propagation(at, reward)
        c += 1

    # finally choose most explored child aka child with most runs
    move = PlayerAction(root.actions[root.get_best_child_runs()])
    if use_nn:
        return move, net
    return move, None


def generate_root(board: np.ndarray,
                  player: BoardPiece,
                  node: Node = None,
                  old_board: np.ndarray = None
                  ) -> Node:
    """Constructs a tree step by step according to MCTS witch UCB as policy

            Args:
                board: the current board
                player: the current player
                node: the node corresponding to "old_board"
                old_board: board one move before "board"

            Returns:
                a new root node
    """
    other_player = get_other_player(player)
    # if a new root node has to be generated
    if node is None:
        return Node(None, get_valid_moves(board), check_end_state(board, player) != GameState.STILL_PLAYING)
    # find root subtree corresponding to "board"
    # this part is not tested and therefore not used yet
    for action, child in zip(node.actions, node.children):
        unapply_player_action(board, action, other_player)
        if np.all(board == old_board):
            child.parent = None
            return child
        apply_player_action(board, action, other_player)
