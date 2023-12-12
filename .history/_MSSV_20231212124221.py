import numpy as np
from state import *


def calc_twos(local_board, value: int, player, opponent):
    change = 0
    for y in range(len(local_board)):
        to_change = 0

        if local_board[y][0] == local_board[y][1] and local_board[y][2] == 0:
            to_change = value

        if local_board[y][0] == local_board[y][2] and local_board[y][1] == 0:
            to_change = value

        if local_board[y][1] == local_board[y][2] and local_board[y][0] == 0:
            to_change = value

        if player in (local_board[y][1], local_board[y][2]):
            change += to_change

        if opponent in (local_board[y][1], local_board[y][2]):
            change -= to_change

    for x in range(len(local_board[0])):
        to_change = 0

        if local_board[0][x] == local_board[1][x] and local_board[2][x] == 0:
            to_change = value

        if local_board[0][x] == local_board[2][x] and local_board[1][x] == 0:
            to_change = value

        if local_board[1][x] == local_board[2][x] and local_board[0][x] == 0:
            to_change = value

        if player in (local_board[1][x], local_board[2][x]):
            change += to_change

        if opponent in (local_board[1][x], local_board[2][x]):
            change -= to_change

    to_change = 0

    if local_board[0][0] == local_board[1][1] and local_board[2][2] == 0:
        to_change = value

    if local_board[0][0] == local_board[2][2] and local_board[1][1] == 0:
        to_change = value

    if player in (local_board[0][0], local_board[2][2]):
        change += to_change

    if opponent in (local_board[0][0], local_board[2][2]):
        change -= to_change

    if local_board[2][0] == local_board[1][1] and local_board[0][2] == 0:
        to_change = value

    if local_board[2][0] == local_board[0][2] and local_board[1][1] == 0:
        to_change = value

    if player in (local_board[2][0], local_board[0][2]):
        change += to_change

    if opponent in (local_board[2][0], local_board[0][2]):
        change -= to_change

    return change


def calc_block(local_board, value, player, opponent):
    change = 0

    for y in range(len(local_board)):
        to_change = 0

        if local_board[y][0] == local_board[y][1] == opponent and local_board[y][2] == player:
            to_change = value

        if local_board[y][0] == local_board[y][2] == opponent and local_board[y][1] == player:
            to_change = value

        if local_board[y][1] == local_board[y][2] == opponent and local_board[y][0] == player:
            to_change = value

        change += to_change

    for x in range(len(local_board[0])):
        to_change = 0

        if local_board[0][x] == local_board[1][x] == opponent and local_board[2][x] == player:
            to_change = value

        if local_board[0][x] == local_board[2][x] == opponent and local_board[1][x] == player:
            to_change = value

        if local_board[1][x] == local_board[2][x] == opponent and local_board[0][x] == player:
            to_change = value

        change += to_change

    to_change = 0

    if local_board[0][0] == local_board[1][1] == opponent and local_board[2][2] == player:
        to_change = value

    if local_board[0][0] == local_board[2][2] == opponent and local_board[1][1] == player:
        to_change = value

    change += to_change

    if local_board[2][0] == local_board[1][1] == opponent and local_board[0][2] == player:
        to_change = value

    if local_board[2][0] == local_board[0][2] == opponent and local_board[1][1] == player:
        to_change = value

    change += to_change

    return change

def calc_score_global(cur_state, player, value_board=None, turn_amount=0):
        if value_board is None:
            value_board = {'won 1': 100, 'won 2 in a row': 200, 'won game': 9999999999, '2 in a row': 5,
                           'blocked 2': 12, 'won block 2': 120}

        opponent = 'o' if player == 'x' else 'x'

        score = 0

        # Calculate score for individual boards based on the amount of "useful twos" they have

        global_board = cur_state.global_cells.reshape((3,3))

        boards = np.array([np.array([block for block in cur_state.blocks] * 3) for _ in range(3)])
        for large_y in range(3):
            for large_x in range(3):
                local_board = boards[large_y][large_x]
                if global_board[large_y][large_x] != 0:
                    continue


                score += calc_twos(local_board, value_board['2 in a row'], player, opponent)

        # Checking for single won boards

        for y in range(len(global_board)):
            for x in range(len(global_board[y])):
                if global_board[y][x] == player:
                    score += value_board['won 1']

                if global_board[y][x] == opponent:
                    score -= value_board['won 1']

        # Same algorithm as with the boards but now for whole game

        score += calc_twos(global_board, value_board['won 2 in a row'], player, opponent)

        # If they win they get infinity points
        result = cur_state.game_result(global_board)
        if result is not None and result != 0:
            if result == player:
                score += value_board['won game'] * (81 - turn_amount)
            if result == opponent:
                score -= value_board['won game'] * (81 - turn_amount)

        # Smaller board blocking score
        for y in range(len(boards)):
            for x in range(len(boards[y])):
                score += calc_block(boards[y][x], value_board['blocked 2'], player, opponent)
                score -= calc_block(boards[y][x], value_board['blocked 2'], opponent, player)

        # Won boards blocking score
        score += calc_block(global_board, value_board['won block 2'], player, opponent)
        score -= calc_block(global_board, value_board['won block 2'], opponent, player)

        return score

def new_minimax_ab(cur_state, depth, player, opponent, alpha=-np.inf, beta=np.inf, maximizing=True, starting=False,
                   initial_depth=81):

    if cur_state.game_over or depth <= 0:
        score =  calc_score_global(cur_state,player,None,turn_amount=initial_depth - depth)
        return score
    if maximizing:
        max_eval = -np.inf
        best_move = None

        for move in cur_state.get_valid_moves:

            cur_state_copy = State_2(cur_state)
            cur_state_copy.free_move = cur_state.free_move
            cur_state_copy.act_move(move)

            evaluation = new_minimax_ab(cur_state_copy, depth - 1, player, opponent, alpha, beta, False, False,
                                        initial_depth=initial_depth)
            if evaluation > max_eval:
                max_eval = evaluation
                best_move = move

            alpha = max(evaluation, alpha)
            del cur_state_copy
            if beta <= alpha:
                break
        if starting:
            return best_move, max_eval

        return max_eval

    else:

        min_eval = np.inf
        best_move = None

        for move in cur_state.get_valid_moves:
            cur_state_copy = State_2(cur_state)
            # Thực hiện bước đi mới đã chọn cho cur_state_copy
            cur_state_copy.free_move = cur_state.free_move
            cur_state_copy.act_move(move)
            evaluation = new_minimax_ab(cur_state_copy, depth - 1, player, opponent, alpha, beta, True, False,
                                        initial_depth=initial_depth)
            if evaluation < min_eval:
                min_eval = evaluation
                best_move = move

            beta = min(evaluation, beta)

            if beta <= alpha:
                break

        if starting:
            return best_move, min_eval

        return min_eval
#def new_minimax_ab(cur_state, depth, player, opponent, alpha=-np.inf, beta=np.inf, maximizing=True, starting=False,
                  # initial_depth=81):


def select_move(cur_state, remain_time):
    valid_moves = cur_state.get_valid_moves
    if valid_moves:
        best_move = new_minimax_ab(cur_state, 2, cur_state.player_to_move, -cur_state.player_to_move,-np.inf,  np.inf, True, True, 81)[0]
        return best_move
    return None

