import math
import random
import numpy as np
from helper import *


class AIPlayer:

    def __init__(self, player_number: int, timer):
        """
        Intitialize the AIPlayer Agent

        # Parameters
        `player_number (int)`: Current player number, num==1 starts the game
        
        `timer: Timer`
            - a Timer object that can be used to fetch the remaining time for any player
            - Run `fetch_remaining_time(timer, player_number)` to fetch remaining time of a player
        """
        self.player_number = player_number
        self.type = 'ai'
        self.player_string = 'Player {}: ai'.format(player_number)
        self.timer = timer
        self.state_tree = {}

    
    def RandomRollout(self,state,player):
        while True:
            available_moves = np.argwhere(state == 0)
            if len(available_moves) == 0:
                break
            move = random.choice(available_moves)
            if check_win(state,tuple(move),player):
                return 1
            state[move[0], move[1]] = player
            available_moves = np.argwhere(state == 0)
            if len(available_moves) == 0:
                break
            move = random.choice(available_moves)
            if check_win(state,tuple(move),3-player):
                return -1
            state[move[0], move[1]] = 3 - player
        return 0
    
    def UCB(self, node):
        if node['parent']['visits'] == 0:
            return float('inf')
        return node['wins'] / (node['visits']+1)+ 1.41 * math.sqrt(math.log(node['parent']['visits']) / (node['visits']+1))
    
    def softmax(self,x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def SelectNode(self, node):
        if node['children'] == []:
            return node
        ucb_values = [self.UCB(child) for child in node['children']]
        probabilities = self.softmax(ucb_values)
        selected_index = np.random.choice(len(node['children']), p=probabilities)
        return node['children'][selected_index]
    
    def ExpandNode(self, node):
        available_moves = np.argwhere(node['state'] == 0)   
        for move in available_moves:
            new_state = np.copy(node['state'])
            new_state[move[0], move[1]] = node['player']
            new_node = {'state': new_state, 'parent': node, 'player': 3 - node['player'], 'wins': 0, 'visits': 0, 'children': []}
            node['children'].append(new_node)

    def Backpropagate(self, node, result):
        while node is not None:
            node['visits'] += 1
            node['wins'] += result
            node = node['parent']
    
    def MCTS(self, state):
        if state.tobytes() not in self.state_tree:
            self.state_tree[state.tobytes()] = {'state': state, 'parent': None, 'player': self.player_number, 'wins': 0, 'visits': 0, 'children': []}
            # 
        root = self.state_tree[state.tobytes()]
        for _ in range(1000):
            node = root
            while node['children'] != []:
                node = self.SelectNode(node)
            
            # Expand the node
            self.ExpandNode(node)
            print(len(node['children']))
            if node['visits'] == 0:
                result = self.RandomRollout(np.copy(node['state']), node['player'])
            else:
                result = self.RandomRollout(np.copy(node['state']), node['player'])
            self.Backpropagate(node, result)
        return root['children'][np.argmax([child['visits'] for child in root['children']])]['state']
    
    def get_move(self, state: np.array) -> Tuple[int, int]:
        """
        Return the next move for the AIPlayer

        # Parameters
        `state (np.array)`: Current state of the board
        
        # Returns
        `Tuple[int, int]`: The next move for the AIPlayer
        """
        state = np.copy(state)
        new_state = self.MCTS(state)
        return np.argwhere(new_state != state)[0]
