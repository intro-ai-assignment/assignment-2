import numpy as np
from copy import *
from math import *
import time
import random


from state import State


class Node():
    def __init__(self, parent, state, reachMove):
        self.totalSimulations = 0
        self.score = 0
        self.children = []
        self.parent = parent
        self.state = state
        self.reachMove = reachMove

    def expand(self):
        nextMoves = self.state.get_valid_moves
        for move in nextMoves:
            new_state = deepcopy(self.state)
            new_state.act_move(move)

            new_node = Node(self, new_state, move)
            self.children.append(new_node)

    def backPropogate(self, result):
        self.totalSimulations += 1
        self.score += result

        if self.parent is not None:  # Non-Root node
            self.parent.backPropogate(result)

    def getExplorationTerm(self):
        return sqrt(log(self.parent.totalSimulations)/(self.totalSimulations or 1))

    def getExploitationTerm(self):
        return self.score/(self.totalSimulations or 1)


class MCTS():
    def __init__(self, symbol, compTime=2, C=sqrt(2)):
        self.symbol = symbol
        self.C = C
        self.compTime = compTime  # In seconds
        self.opponentMap = {
            1: -1,
            -1: 1
        }

    def simulate(self, state, prevMove):
        is_terminal = False
        if state.game_over:
            is_terminal = True
        elif len(state.get_valid_moves) == 0:
            is_terminal = True

        if not is_terminal:
            nextMoves = state.get_valid_moves

            # Randmoly choose the next move
            randomMove = random.choice(nextMoves)
            state.act_move(randomMove)

            return self.simulate(state, randomMove)
        else:
            value = state.game_result(state.global_cells.reshape(3, 3))
            if value != 0:
                if value == self.symbol:
                    return 1
                else:
                    return -1  # Loss
            else:
                return 0  # Draw

    def selection(self, currNode, symbol):
        is_terminal = False
        if currNode.state.game_over:
            is_terminal = True
        elif len(currNode.state.get_valid_moves) == 0:
            is_terminal = True

        if is_terminal:  # Terminal node
            return currNode

        if len(currNode.children) == 0:  # Not expanded
            return currNode

        # Selecting best child based on exploration Term and exploitation term
        if symbol == self.symbol:
            sortedChildren = sorted(currNode.children, key=lambda child: child.getExploitationTerm(
            ) + self.C*child.getExplorationTerm(), reverse=True)
        else:
            sortedChildren = sorted(currNode.children, key=lambda child: -
                                    child.getExploitationTerm() + self.C*child.getExplorationTerm(), reverse=True)

        return self.selection(sortedChildren[0], self.opponentMap[symbol])

    def getMove(self, state, prevMove):
        # Creting a root node
        rootNode = Node(None, deepcopy(state), prevMove)

        # Monte Carlo Iterations
        startTime = time.time()
        while time.time() - startTime < 2 and time.time() - startTime < self.compTime:

            selectedNode = self.selection(
                rootNode, self.symbol)  # Selection step

            if selectedNode.totalSimulations == 0:  # First simulation
                result = self.simulate(
                    deepcopy(selectedNode.state), selectedNode.reachMove)
                selectedNode.backPropogate(result)
            else:  # Expansion
                selectedNode.expand()

        # Final move selection
        sortedChildren = sorted(
            rootNode.children, key=lambda child: child.getExploitationTerm(), reverse=True)

        return sortedChildren[0].reachMove


def select_move(cur_state, remain_time):
    # print("*Current state: ", cur_state)
    # print("valid_moves:", cur_state.get_valid_moves)
    # valid_moves = cur_state.get_valid_moves
    # if len(valid_moves) != 0:
    #     result = np.random.choice(valid_moves)
    #     print(type(result))
    #     print(result)
    #     return result

    # return None

    mcts = MCTS(cur_state.player_to_move, compTime=remain_time)
    move = mcts.getMove(deepcopy(cur_state), cur_state.previous_move)
    if move:
        return move

    return None
