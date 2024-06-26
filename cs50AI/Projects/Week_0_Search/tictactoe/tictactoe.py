"""
Tic Tac Toe Player
"""

import math
import copy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    count = 0
    for row in board:
        for item in row:
            if item != EMPTY:
                count += 1
    return 'X' if count%2 == 0 else 'O'


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    return {(i, j) for i in range(3) for j in range(3) if board[i][j] == EMPTY}


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    if action not in actions(board):
        print(action, actions(board))
        raise ValueError
    next = player(board)
    board_copy = copy.deepcopy(board)
    board_copy[action[0]][action[1]] = next
    return board_copy



def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    winning_combinations = [
    # Rows
    [(0, 0), (0, 1), (0, 2)],
    [(1, 0), (1, 1), (1, 2)],
    [(2, 0), (2, 1), (2, 2)],
    
    # Columns
    [(0, 0), (1, 0), (2, 0)],
    [(0, 1), (1, 1), (2, 1)],
    [(0, 2), (1, 2), (2, 2)],
    
    # Diagonals
    [(0, 0), (1, 1), (2, 2)],
    [(0, 2), (1, 1), (2, 0)]
    ]

    for wc in winning_combinations:
        a,b,c = wc
        if board[a[0]][a[1]] == board[b[0]][b[1]] == board[c[0]][c[1]] != EMPTY:
            return board[a[0]][a[1]]
    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    return len(actions(board)) == 0 or winner(board) is not None


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    champion = winner(board)
    if champion == 'X':
        return 1
    elif champion == 'O':
        return -1
    return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None
    current_player = player(board)
    if current_player == 'X':
        _, move = maxim(board)
    else:
        _, move = minim(board)
    
    return move

def minim(board):
    if terminal(board):
        return utility(board), None
    value = 100
    best_move = None
    for action in actions(board):
        max_value, temp = maxim(result(board, action))
        if max_value < value:
            value = max_value
            best_move = action
    return value, best_move
    # pass

def maxim(board):
    if terminal(board):
        return utility(board), None
    value = -100
    best_move = None
    for action in actions(board):
        min_value, temp = minim(result(board, action))
        if min_value > value:
            value = min_value
            best_move = action
    return value, best_move

