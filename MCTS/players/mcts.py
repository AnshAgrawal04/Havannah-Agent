import time
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

    def random_rollout(self, state):
        """
        Randomly simulate a game from the current state to the end of the game
        """
        while True:
            available_moves = np.argwhere(state == 0)
            if len(available_moves) == 0:
                break 
            move = random.choice(available_moves)
            if check_win(state,move,self.player_number):
                return 1
            state[move[0], move[1]] = self.player_number
            available_moves.remove(move)
            if len(available_moves) == 0:
                break
            move = random.choice(available_moves)
            if check_win(state,move,3-self.player_number):
                return -1
            state[move[0], move[1]] = 3 - self.player_number
        return 0


    def UCB(self, node):
        """
        Calculate the UCB value of a node
        """
        return node['wins'] / node['visits'] + 1.41 * math.sqrt(math.log(node['parent']['visits']) / node['visits'])
    
    def select_node(self, node):
        """
        Select the best child node based on UCB values
        """
        if node['children'] == []:
            return node
        return self.select_node(max(node['children'], key=self.UCB))
    
    def expand_node(self, node):
        """
        Expand the node by adding all possible children
        """
        available_moves = np.argwhere(node['state'] == 0)
        for move in available_moves:
            new_state = node['state'].copy()
            new_state[move[0], move[1]] = node['player']
            node['children'].append({'state': new_state, 'player': 3 - node['player'], 'parent': node, 'children': [], 'visits': 0, 'wins': 0})

    def backpropagate(self, node, result):
        """
        Backpropagate the result of the simulation to all parent nodes
        """
        node['visits'] += 1
        node['wins'] += result
        if node['parent'] != None:
            self.backpropagate(node['parent'], result)
    
    def check_win1(self, node, player):  
        """
        Check if the player has won the game
        """
        # find the move using the node state and the parent node state  
        if (node['parent']==None):
            return False
        move = np.argwhere(np.logical_xor(node['state'] == 0, node['parent']['state'] == 0))[0]
        # check if the player has won the game
        if check_win(node['state'], move,player):
            return True
        return False


    def monte_carlo_tree_search(self, state):

        if tuple(map(tuple, state)) not in self.state_tree:
            self.state_tree[tuple(map(tuple, state))] = {'state': state, 'player': self.player_number, 'parent': None, 'children': [], 'visits': 0, 'wins': 0}
        root = self.state_tree[tuple(map(tuple, state))]

        start_time = time.time()
        while time.time() - start_time < 1.5:
            node = self.select_node(root)
            if self.check_win1(node,self.player_number):
                result = 1
            elif self.check_win1(node, 3 - self.player_number):
                result = -1
            else:
                self.expand_node(node)
                result = self.random_rollout(node['state'])
            self.backpropagate(node, result)
        
        best_child = max(root['children'], key=lambda x: x['visits'])
        best_move = np.argwhere(np.logical_xor(root['state'] == 0, best_child['state'] == 0))[0]
        return (best_move[0], best_move[1])


    def get_move(self, state: np.array) -> Tuple[int, int]:
        """
        Given the current state of the board, return the next move

        # Parameters
        `state: Tuple[np.array]`
            - a numpy array containing the state of the board using the following encoding:
            - the board maintains its same two dimensions
            - spaces that are unoccupied are marked as 0
            - spaces that are blocked are marked as 3
            - spaces that are occupied by player 1 have a 1 in them
            - spaces that are occupied by player 2 have a 2 in them

        # Returns
        Tuple[int, int]: action (coordinates of a board cell)
        """
        # Applying Monte Carlo Tree Search
        return self.monte_carlo_tree_search(state)
        



        # Do the rest of your implementation here
        raise NotImplementedError('Whoops I don\'t know what to do')

