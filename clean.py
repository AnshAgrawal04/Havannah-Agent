import time
import math
import random
import numpy as np
from helper import *

class MCTSNode:
    def __init__(self, state, player, parent=None, action=None):

        self.state = state
        self.parent = parent
        self.action = action
        self.visits = 0
        self.wins = 0
        self.rave_visits = 0
        self.rave_wins = 0
        self.children = []
        self.player = player 
        # actions is where state is 0
        self.actions = get_valid_actions(state)
        self.expandable = get_valid_actions(state)

class AIPlayer:

    def __init__(self, player_number: int, timer):

        self.player_number = player_number
        self.type = 'ai'
        self.player_string = 'Player {}: ai'.format(player_number)
        self.timer = timer
        self.root = None
        self.total_time = fetch_remaining_time(timer, self.player_number)
        self.board_size = None
        self.rollouts = 0

    def get_move(self, state: np.array) -> Tuple[int, int]:
        
        if not self.board_size:
            self.board_size = (state.shape[0] + 1 )//2
        
        move = self.forwardCheck(state.copy(), self.player_number)
        if move: return move
        move = self.forwardCheck(state.copy(), 3 - self.player_number)
        if move: return move

        best_child = self.MCTS(state)
        self.root = best_child
        return best_child.action

    def MCTS(self, state):
        if self.root is None:  
            self.root = MCTSNode(state, self.player_number)
        else:
            for child in self.root.children:
                if np.array_equal(child.state, state):
                    self.root = child
                    break
            else:
                self.root = MCTSNode(state, self.player_number)

        timeout = self.set_timeout()
        start = time.time()
        while time.time() - start < timeout:
            self.search(self.root)

        print("Number of rollouts: ", self.rollouts)
        self.rollouts = 0

        best_child = self.best_child(self.root)
        return best_child
    
    def search(self, node):
        if len(node.children) == 0 and node.visits == 0:
            outcome = self.rollout(node)
            self.backpropagate(node, outcome)
            # print("Rollout")
        else:
            if self.add_child(node):
                new_move = node.expandable.pop()
                new_state = node.state.copy()
                new_state[new_move] = node.player
                node.children.append(MCTSNode(new_state, 3 - node.player, 
                                              parent=node, action=new_move))
            if len(node.children) == 0:
                return
            best_idx = 0
            for i in range(1, len(node.children)):
                if self.UCT(node.children[i]) > self.UCT(node.children[best_idx]):
                    best_idx = i
            self.search(node.children[best_idx])

    def set_timeout(self):
        if self.total_time <= 180:
            return 9
        elif self.total_time <= 360:
            time_list = [18,17,16,15,14,13,12,11,10,9,8,7]
            ret_time = time_list[0]
            if time_list==[7]:
                return 7
            time_list.pop(0)
            return ret_time
        else:
            time_list = [20,19,18,17,16,15,14,13,12,11,10,9,8]
            ret_time = time_list[0]
            if time_list==[8]:
                return 8
            time_list.pop(0)
            return ret_time

    def UCT(self, node):
        if node.visits == 0 or node.parent.visits == 0: return math.inf
        exploit = node.wins / node.visits
        explore = 0.9 * math.sqrt(math.log(node.parent.visits) / node.visits)
        if node.rave_visits == 0: 
            return exploit + explore
        rave = node.rave_wins / node.rave_visits
        k = 100
        beta = k / (k + node.visits)
        return beta * exploit + (1 - beta) * rave + explore
    
    def best_child(self, node): 
        best_child = node.children[0]
        for child in node.children:
            # print(f'Wins: {child.wins}, Visits: {child.visits}')
            if child.visits > best_child.visits:
                best_child = child
        return best_child

    def add_child(self, node):
        if node.expandable == []: 
            return False
        return True

    def rollout(self, node):
        state = node.state.copy()
        player = node.player
        available_moves = get_valid_actions(state)
        available_moves = np.random.permutation(available_moves)
        available_moves = available_moves.tolist()
        moves_played = []
        self.rollouts += 1
        outcome = 0
        while True:
            if len(available_moves) == 0:
                outcome = 0.5
                break
            move = available_moves.pop()
            move = tuple(move)
            moves_played.append((move, player))
            state[move] = player
            if check_win(state, move, player)[0]:
                outcome = (player == self.player_number)
                break
            player = 3 - player

        for move, player in moves_played:
            self.rave_backpropagate(node, move, player, outcome)
        return outcome
    
    def rave_backpropagate(self, node, move, player, outcome):
        while node is not None:
            state = node.state.copy()
            state[move] = player
            self.update_rave(node, state, outcome)
            node = node.parent
    
    def update_rave(self, node, state, outcome):
        for child in node.children:
            if np.array_equal(child.state, state):
                child.rave_visits += 1
                child.rave_wins += outcome
                return

    def backpropagate(self, node, outcome):
        while node is not None:
            node.visits += 1
            node.wins += outcome
            node = node.parent

    def forwardCheck(self, state, player):
        copy = state.copy()
        available_moves = get_valid_actions(copy)
        for move in available_moves:
            copy[move] = player
            if check_win(copy, move, player)[0]:   return move
            copy[move] = 0
        return None