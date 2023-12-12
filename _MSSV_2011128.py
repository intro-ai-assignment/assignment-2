import numpy as np
from state import State_2
import logging

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

DEPTH = 5
ALPHA = -np.inf
BETA = np.inf

X = 1
O = -1


def pattern_to_score(board, blocking_multiplier, sequence_multiplier):
    ret = 0
    players = [(O, X), (X, O), (O, 0), (X, 0)]
    for player, third in players:
        if (
            (board[0] + board[1] == 2 * player and board[2] == third)
            or (board[1] + board[2] == 2 * player and board[0] == third)
            or (board[0] + board[2] == 2 * player and board[1] == third)
            or (board[3] + board[4] == 2 * player and board[5] == third)
            or (board[3] + board[5] == 2 * player and board[4] == third)
            or (board[5] + board[4] == 2 * player and board[3] == third)
            or (board[6] + board[7] == 2 * player and board[8] == third)
            or (board[6] + board[8] == 2 * player and board[7] == third)
            or (board[7] + board[8] == 2 * player and board[6] == third)
            or (board[0] + board[3] == 2 * player and board[6] == third)
            or (board[0] + board[6] == 2 * player and board[3] == third)
            or (board[3] + board[6] == 2 * player and board[0] == third)
            or (board[1] + board[4] == 2 * player and board[7] == third)
            or (board[1] + board[7] == 2 * player and board[4] == third)
            or (board[4] + board[7] == 2 * player and board[1] == third)
            or (board[2] + board[5] == 2 * player and board[8] == third)
            or (board[2] + board[8] == 2 * player and board[5] == third)
            or (board[5] + board[8] == 2 * player and board[2] == third)
            or (board[0] + board[4] == 2 * player and board[8] == third)
            or (board[0] + board[8] == 2 * player and board[4] == third)
            or (board[4] + board[8] == 2 * player and board[0] == third)
            or (board[2] + board[4] == 2 * player and board[6] == third)
            or (board[2] + board[6] == 2 * player and board[4] == third)
            or (board[4] + board[6] == 2 * player and board[2] == third)
        ):
            if third != 0:
                ret += blocking_multiplier * third
            else:
                ret += sequence_multiplier * player
    return ret


def evaluation(state: State_2):
    value = 0
    # Winning position is worth -10000
    # Losing position is worth 10000
    # Tie is -5000
    game_result = state.game_result(state.global_cells.reshape(3, 3))
    if game_result != None:
        if game_result == O:
            value -= 10000
        elif game_result == X:
            value += 10000
        else:
            value -= 5000

        return value

    # Smaller board wins worth 20 points
    # Winning the center board adds 20 => total 40
    # Winning a corner board adds 10 => total 30
    # Getting a center square in any small board is worth 2
    # Getting a center square in the center board adds 2 => total 4
    # Two board wins which can be continued for a winning sequence are worth 40
    # Blocking opponent's sequence is worth 50
    # And similar patterns inside a small board are worth 4 and 5 points
    global_cells = state.global_cells

    # global
    value += (state.count_O + state.count_X) * 20
    value += state.global_cells[4] * 20
    value += (
        state.global_cells[0]
        + state.global_cells[2]
        + state.global_cells[6]
        + state.global_cells[8]
    ) * 10
    value += pattern_to_score(state.global_cells, 50, 40)

    # similar for local
    for i, board in enumerate(state.blocks):
        if global_cells[i] != 0:
            continue
        value += board[1, 1] * 2
        ret = pattern_to_score(board.reshape(-1), 5, 4)
        value += ret
    value += state.blocks[4][1, 1] * 2

    return value


def minimax(position: State_2, depth, alpha, beta, maximizingPlayer):
    if depth == 0 or position.game_over:
        return (evaluation(position), None)

    valid_moves = position.get_valid_moves
    picked_move = None

    if maximizingPlayer:
        maxEval = -np.inf
        for child in valid_moves:
            temp_state = State_2(position)
            temp_state.free_move = position.free_move
            temp_state.act_move(child)
            eval, _ = minimax(temp_state, depth - 1, alpha, beta, False)
            # maxEval = max(maxEval, eval)
            if eval > maxEval:
                maxEval = eval
                picked_move = child
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return maxEval, picked_move
    else:
        minEval = np.inf
        for child in valid_moves:
            temp_state = State_2(position)
            temp_state.free_move = position.free_move
            temp_state.act_move(child)
            eval, _ = minimax(temp_state, depth - 1, alpha, beta, True)
            # minEval = min(minEval, eval)
            if eval < minEval:
                minEval = eval
                picked_move = child
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return minEval, picked_move


def select_move(cur_state: State_2, remain_time):
    # empty local cells
    valid_moves = cur_state.get_valid_moves
    search_state = State_2(cur_state)
    search_state.free_move = cur_state.free_move
    # logger.info(f"Search state: {search_state}")
    if cur_state.player_to_move == 1:
        maximize = True
    else:
        maximize = False
    if len(valid_moves) != 0:
        min_value, picked_move = minimax(search_state, DEPTH, ALPHA, BETA, maximize)
        return picked_move
    return None
