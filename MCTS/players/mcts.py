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
        # the state tree is of the form, we store the state, visits, moves, wins, parent , children and move that led to this state
        self.state_tree = {}

        self.loose_bridge_pattern = [{'pattern': [(0, 0), (0, 1), (1, 1), (0, 2)], 'stones': [1, 0, 0, 1]},
                                     {'pattern': [(0, 0), (1, 0), (1, 1), (2, 1)], 'stones': [1, 0, 0, 1]},
                                     {'pattern': [(0, 0), (0, -1), (1, 0), (1, -1)], 'stones': [1, 0, 0, 1]}]


    def forwardCheck(self, state, player):
        for i in range(0, state.shape[0]):
            for j in range(0, state.shape[1]):
                if state[i][j] == 0:
                    state[i][j] = player
                    if check_win(state, (i, j), player)[0]:
                        state[i][j] = 0
                        return (i, j)
                    state[i][j] = 0
        return None

# i n t s e a r c h ( Node node , S t a t e s t a t e ) {
# // r o l l o u t
# i f ( node . n u m c h i l d r e n == 0 && node . s i m s == 0 ) {
# w h i l e ( ! s t a t e . t e r m i n a l ( ) )
# s t a t e . randmove ( ) ;
# r e t u r n s t a t e . outcome ( ) ; // win = 1 , draw = 0 . 5 o r l o s s = 0
# }
# // expand
# i f ( node . n u m c h i l d r e n == 0 )
# f o r e a c h ( s t a t e . s u c c e s s o r s a s s u c c )
# node . a d d c h i l d ( Node ( s u c c ) ) ;
# // d e s c e n t
# Node b e s t = node . c h i l d r e n . f i r s t ( ) ;
# f o r e a c h ( node . c h i l d r e n a s c h i l d )
# i f ( b e s t . v a l u e ( ) < c h i l d . v a l u e ( ) )
# b e s t = c h i l d ;
# i n t outcome = 1 − s e a r c h ( b e s t , s t a t e . move ( b e s t . move ) ) ;
# // back−p r o p a g a t e
# b e s t . s i m s += 1 ;
# b e s t . w i n s += outcome ;
# r e t u r n outcome ;
# }





    def search(self, node, state):
        if (len(node['children']) == 0 and node['visits'] == 0):
            while not check_win(state, None, 0)[0]:
                available_moves = np.argwhere(state == 0)
                move = random.choice(available_moves)
                state[move[0], move[1]] = self.player_number
                if check_win(state, tuple(move), self.player_number)[0]:
                    return 1
                available_moves = np.argwhere(state == 0)
                move = random.choice(available_moves)
                state[move[0], move[1]] = 3 - self.player_number
                if check_win(state, tuple(move), 3 - self.player_number)[0]:
                    return -1
            return 0
        
        if (len(node['children']) == 0):
            for move in node['moves']:
                new_state = state.copy()
                new_state[move[0], move[1]] = self.player_number
                self.state_tree[state.tobytes()]['children'].append(self.state_tree[new_state.tobytes()])
                self.state_tree[new_state.tobytes()]['parent'] = node
                self.state_tree[new_state.tobytes()]['move'] = move
                node['children'].append(self.state_tree[new_state.tobytes()])
        
        best = node['children'][0]
        for child in node['children']:
            if best['wins'] < child['wins']:
                best = child

        best_changed_state = state.copy()
        best_changed_state[best['move'][0], best['move'][1]] = self.player_number
        outcome = 1 - self.search(best, best_changed_state)
        best['visits'] += 1
        best['wins'] += outcome
        return outcome




    def MCTS(self, state):
        if state.tobytes() not in self.state_tree:
            self.state_tree[state.tobytes()] = {'state': state, 'visits': 0, 'moves': [], 'wins': 0, 'parent': None, 'move': None , 'children': []}

        root = self.state_tree[state.tobytes()]

        timeout = 5
        start = time.time()
        while time.time() - start < timeout:
            search_state = state.copy()
            node = root
            self.search(root,search_state)

        # fidn the best child
        best = root['children'][0]
        for child in root['children']:
            if best['wins'] < child['wins']:
                best = child
        return best['move']

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
        # forward check for winning move
        state_copy = state.copy()   
        move = self.forwardCheck(state_copy, self.player_number)
        if move is not None:
            return move
        # forward check for blocking move
        move = self.forwardCheck(state_copy, 3 - self.player_number)
        if move is not None:
            return move
        
        return self.MCTS(state)

        
        
        


        # Do the rest of your implementation here
        raise NotImplementedError('Whoops I don\'t know what to do')

