# import math
# import random
# import numpy as np
# from helper import *


# class AIPlayer:

#     def __init__(self, player_number: int, timer):
#         """
#         Intitialize the AIPlayer Agent

#         # Parameters
#         `player_number (int)`: Current player number, num==1 starts the game
        
#         `timer: Timer`
#             - a Timer object that can be used to fetch the remaining time for any player
#             - Run `fetch_remaining_time(timer, player_number)` to fetch remaining time of a player
#         """
#         self.player_number = player_number
#         self.type = 'ai'
#         self.player_string = 'Player {}: ai'.format(player_number)
#         self.timer = timer
#         self.state_tree = {}

    
#     def RandomRollout(self, state, player):
#         while True:
#             available_moves = np.argwhere(state == 0)
#             if len(available_moves) == 0:
#                 break
#             move = random.choice(available_moves)
#             if check_win(state, tuple(move), player)[0]:
#                 # print(f"Player {player} won rollout")
#                 return 1
#             state[move[0], move[1]] = player

#             # print("Change", np.argwhere(state != old_state))

#             available_moves = np.argwhere(state == 0)
#             if len(available_moves) == 0:
#                 break
#             move = random.choice(available_moves)
#             if check_win(state,tuple(move), 3 - player)[0]:
#                 # print(f"Player {player} lost rollout")
#                 return -1
#             state[move[0], move[1]] = 3 - player

#         return 0
    
#     def UCB(self, node):
#         if node['visits'] == 0 or node['parent']['visits'] == 0:
#             return float('inf')
#         return node['wins'] / (node['visits'] + 1) + 1.41 * math.sqrt(math.log(node['parent']['visits']) / (node['visits'] + 1))
    
#     def softmax(self,x):
#         e_x = np.exp(x - np.max(x))
#         return e_x / np.sum(e_x)

#     def SelectNode(self, node):
#         if node['children'] == []:
#             return node
#         ucb_values = [self.UCB(child) for child in node['children']]
#         # probabilities = self.softmax(ucb_values)
#         # selected_index = np.random.choice(len(node['children']), p=probabilities)
#         selected_index = np.argmax([self.UCB(child) for child in node['children']])
#         return node['children'][selected_index]
    
#     def ExpandNode(self, node):
#         available_moves = np.argwhere(node['state'] == 0)
#         for move in available_moves:
#             new_state = np.copy(node['state'])
#             new_state[move[0], move[1]] = node['player']
#             if new_state.tobytes() in self.state_tree:
#                 node['children'].append(self.state_tree[new_state.tobytes()])
#             else:
#                 new_node = {'state': new_state, 'parent': node, 'player': 3 - node['player'], 'wins': 0, 'visits': 0, 'children': []}
#                 self.state_tree[new_state.tobytes()] = new_node
#                 node['children'].append(new_node)
#         # return self.state[new_state.tobytes()]
            

#     def Backpropagate(self, node, result):
#         while node is not None:
#             node['visits'] += 1
#             node['wins'] += result
#             node = node['parent']
    
#     def MCTS(self, state):
#         if state.tobytes() not in self.state_tree:
#             # print("Adding state to tree")
#             self.state_tree[state.tobytes()] = {'state': state, 'parent': None, 'player': self.player_number, 'wins': 0, 'visits': 0, 'children': []}
            
#         root = self.state_tree[state.tobytes()]
#         for _ in range(1000):
#             node = root
#             while len(node['children']) > 0:
#                 node = self.SelectNode(node)

#             if node['visits'] == 0:
#                 result = self.RandomRollout(np.copy(node['state']), node['player'])
#             else:
#                 # Expand the node
#                 self.ExpandNode(node)
#                 child_node = self.SelectNode(node)
#                 result = self.RandomRollout(np.copy(child_node['state']), node['player'])

#             self.Backpropagate(node, result)

#         selected_child = root['children'][np.argmax([child['wins'] / child['visits'] for child in root['children']])]
#         with open(f'child_stats/state.txt', 'a') as f:
#             f.write(f'Wins: {selected_child["wins"]}\n')
#             f.write(f'Visits: {selected_child["visits"]}\n')
#             f.write(f'UCB: {self.UCB(selected_child)}\n')
#             f.write(selected_child['state'].__str__() + '\n')
#             f.write('------------------------------------------------------------\n')
#         return selected_child['state']
    
#     def forwardCheck(self, state, player):
#         for i in range(0, state.shape[0]):
#             for j in range(0, state.shape[1]):
#                 if state[i][j] == 0:
#                     state[i][j] = player
#                     if check_win(state, (i, j), player)[0]:
#                         state[i][j] = 0
#                         return (i, j)
#                     state[i][j] = 0
#         return None
    
#     def get_move(self, state: np.array) -> Tuple[int, int]:
#         """
#         Return the next move for the AIPlayer

#         # Parameters
#         `state (np.array)`: Current state of the board
        
#         # Returns
#         `Tuple[int, int]`: The next move for the AIPlayer
#         """
#         state_copy = state.copy()   
#         move = self.forwardCheck(state_copy, self.player_number)
#         if move is not None:
#             return move
#         # forward check for blocking move
#         move = self.forwardCheck(state_copy, 3 - self.player_number)
#         if move is not None:
#             return move
#         state = np.copy(state)
#         new_state = self.MCTS(state)
#         return np.argwhere(new_state != state)[0]





import time
import math
import random
import numpy as np
from helper import *


# class AIPlayer:

#     def __init__(self, player_number: int, timer):
#         """
#         Intitialize the AIPlayer Agent

#         # Parameters
#         `player_number (int)`: Current player number, num==1 starts the game
        
#         `timer: Timer`
#             - a Timer object that can be used to fetch the remaining time for any player
#             - Run `fetch_remaining_time(timer, player_number)` to fetch remaining time of a player
#         """
#         self.player_number = player_number
#         self.type = 'ai'
#         self.player_string = 'Player {}: ai'.format(player_number)
#         self.timer = timer
#         self.board_size = None
#         # the state tree is of the form, we store the state, visits, moves, wins, parent , children and move that led to this state
#         self.state_tree = {}

#         self.loose_bridge_pattern = [{'pattern':[(0,0), (0,1), (1,1), (1,2)], 'stones': [1, 0, 0, 1]},
#                                      {'pattern':[(0,0), (1,0), (1,1), (2,1)], 'stones': [1, 0, 0, 1]},
#                                      {'pattern':[(0,0), (-1,0), (0,1), (-1,1)], 'stones': [1, 0, 0, 1]},
#                                      {'pattern':[(0,0), (-1,-1), (-1,0), (-2,-1)], 'stones': [1, 0, 0, 1]},
#                                      {'pattern':[(0,0), (-1,-1), (0,-1), (-1,-2)], 'stones': [1, 0, 0, 1]},
#                                      {'pattern':[(0,0), (0,-1), (1,0), (1,-1)], 'stones': [1, 0, 0, 1]}]

#     def get_move(self, state: np.array) -> Tuple[int, int]:
#         """
#         Given the current state of the board, return the next move

#         # Parameters
#         `state: Tuple[np.array]`
#             - a numpy array containing the state of the board using the following encoding:
#             - the board maintains its same two dimensions
#             - spaces that are unoccupied are marked as 0
#             - spaces that are blocked are marked as 3
#             - spaces that are occupied by player 1 have a 1 in them
#             - spaces that are occupied by player 2 have a 2 in them

#         # Returns
#         Tuple[int, int]: action (coordinates of a board cell)
#         """
#         # Applying Monte Carlo Tree Search
#         # forward check for winning move\
        
#         move = self.forwardCheck(state, self.player_number)
#         if move is not None:
#             return move
#         # forward check for blocking move
#         move = self.forwardCheck(state, 3 - self.player_number)
#         if move is not None:
#             return move
        
#         if not self.board_size:
#             self.board_size = (state.shape[0] + 1) // 2
        
#         state_copy = self.convert_to_us(state)
#         move = self.MCTS(state_copy)
#         return self.convert_move_to_ta(move)

#         # Do the rest of your implementation here
#         raise NotImplementedError('Whoops I don\'t know what to do')
    
#     def MCTS(self, state):
#         if state.tobytes() not in self.state_tree:
#             self.state_tree[state.tobytes()] = {'state': state, 'visits': 0, 'wins': 0, 'parent': None, 'move': None , 'children': [], 'player': self.player_number}

#         root = self.state_tree[state.tobytes()]

#         timeout = 10
#         start = time.time()
#         while time.time() - start < timeout:
#             search_state = state.copy()
#             self.search(root, search_state)

#         best = root['children'][0]
#         for child in root['children']:
#             # print(self.win_rate(child))
#             if self.win_rate(child) > self.win_rate(best):
#                 best = child
#         # for child in root['children']:
#         #     print(child['wins'], child['visits'])
#         # print("AI playing ", best['move'])
#         return best['move']

#     def search(self, node, state):
#         outcome = 0
#         iters = 1

#         if (len(node['children']) == 0 and node['visits'] == 0):
#             for _ in range(iters):
#                 rollout_state = state.copy()
#                 outcome += self.rollout(rollout_state)
#             node['visits'] += iters
#         else:
#             if (len(node['children']) == 0):
#                 available_moves = np.argwhere(state == 0)
#                 if len(available_moves) == 0:
#                     return 0
#                 move = random.choice(available_moves)
#                 new_state = state.copy()
#                 new_state[move] = self.player_number

#                 if new_state.to_bytes() not in self.state_tree:
#                     new_player = 3 - node['player']
#                 # for move in available_moves:
#                 #     new_state = state.copy()
#                 #     new_state[move[0], move[1]] = self.player_number

#                 #     # loose_bridge_bonus = self.matchPattern(tuple(move), new_state, self.player_number)
#                 #     if new_state.tobytes() not in self.state_tree:
#                 #         new_player = 3 - node['player']
#                 #         new_node = {'state': new_state, 'visits': 0, 'wins': 0, 'parent': node, 'move': move, 
#                 #                     'children': [], 'player': new_player}
#                 #         self.state_tree[new_state.tobytes()] = new_node
#                 #     else:
#                 #         new_node = self.state_tree[new_state.tobytes()]
#                 #     node['children'].append(new_node)

#             best = node['children'][0]
#             for child in node['children']:
#                 if self.UCB(child) > self.UCB(best):
#                     best = child
#             outcome = iters - self.search(best, best['state'])
        
#         node['visits'] += iters
#         node['wins'] += outcome
#         self.state_tree[state.tobytes()] = node

#         return outcome

#     def check_win(self, state, move, player):
#         converted_state = self.convert_to_ta(state)
#         converted_move = self.convert_move_to_ta(move)
#         return check_win(converted_state, converted_move, player)

#     def convert_to_us(self, state):
#         sz = self.board_size
#         new_state = np.zeros((2*sz-1, 2*sz - 1))
#         for i in range(2*sz - 1):
#             for j in range(2*sz - 1):
#                 if abs(i - j) >= sz:
#                     new_state[(i, j)] = 3

#         for i in range(2*sz - 1):
#             for j in range(sz):
#                 new_state[(i, j)] = state[(i, j)]

#         for i in range(2*sz - 1):
#             for j in range(sz, 2*sz-1):
#                 if abs(i - j) >= sz:
#                     continue
#                 diff = j - sz + 1
#                 new_state[(i, j)] = state[(i - diff, j)]
        
#         return new_state
                  
#     def convert_to_ta(self, state):
#         sz = self.board_size
#         new_state = np.zeros((2*sz - 1, 2*sz - 1))
#         for i in range(2*sz - 1):
#             for j in range(2*sz - 1):
#                 if i - j >= sz or i + j > 3*sz - 3:
#                     new_state[(i, j)] = 3

#         for i in range(2*sz - 1):
#             for j in range(sz):
#                 new_state[(i, j)] = state[(i, j)]

#         for i in range(2*sz - 1):
#             for j in range(sz, 2*sz-1):
#                 if abs(i - j) >= sz:
#                     continue
#                 diff = j - sz + 1
#                 new_state[(i - diff, j)] = state[(i, j)]
                
#         return new_state

#     def convert_move_to_ta(self, move):
#         i = move[0]
#         j = move[1]
#         if j <= self.board_size-1:
#             return move
#         else:
#             diff = j - self.board_size + 1
#             return tuple((i - diff, j))

#     def matchPattern(self, move, state, player):
#         connections = 0
#         connection_dict = []
#         state[move] = player
#         scaling_factor = 1.0  # Adjust based on distance to edge/corner

#         for pattern in self.loose_bridge_pattern:
#             valid = True
#             for dir in range(4):
#                 x = pattern['pattern'][dir][0] + move[0]
#                 y = pattern['pattern'][dir][1] + move[1]
#                 if (x < 0 or x > 2*self.board_size-2) or (y < 0 or y > 2*self.board_size-2):
#                     valid = False
#                     break
#                 if pattern['stones'][dir] == 0 and state[(x,y)] != 0 and state[(x,y)] != player:
#                     valid = False
#                     break
#                 if state[(x,y)] != pattern['stones'][dir] * player:
#                     valid = False
#                     break
#             if valid:
#                 # Increase bonus if closer to edge or strategic area
#                 connections += scaling_factor
#                 connection_dict.append([move, (x, y)])
#         return connections

#     def rollout(self, rollout_state):
#         while np.any(rollout_state == 0):
#             available_moves = np.argwhere(rollout_state == 0)
#             move = random.choice(available_moves)
#             # Bias towards moves forming loose bridges
#             # move_scores = [self.matchPattern(tuple(move), rollout_state, self.player_number) for move in available_moves]
#             # if sum(move_scores) == 0:
#             #     move = random.choice(available_moves)
#             # else:
#             #     move_probs = np.array(move_scores) / sum(move_scores)
#             #     move = random.choices(available_moves, weights=move_probs, k=1)[0]
            
#             rollout_state[move[0], move[1]] = self.player_number
#             if self.check_win(rollout_state, tuple(move), self.player_number)[0]:
#                 return 1

#             # Opponent's move
#             if np.any(rollout_state == 0):
#                 available_moves = np.argwhere(rollout_state == 0)
#                 move = random.choice(available_moves)
#                 rollout_state[move[0], move[1]] = 3 - self.player_number
#                 if self.check_win(rollout_state, tuple(move), 3 - self.player_number)[0]:
#                     return 0
#         return 0.5

#     def position_bonus(self, pos):
#         corners = [(0, 0), (0, self.board_size-1), (self.board_size-1, 0), (self.board_size-1, 2*self.board_size-2),
#                    (2*self.board_size-2, 0), (2*self.board_size - 2, self.board_size-1)]
#         for corner in corners:
#             if pos[0] == corner[0] and pos[1] == corner[1]:
#                 return 50
#         return 0

#     def UCB(self, node):
#         if node['visits'] == 0 or node['parent']['visits'] == 0:
#             return float('inf')
#         pattern_const = 20  # Reduced to prevent overpowering exploration
#         pattern_bonus = pattern_const * self.matchPattern(node['move'], node['state'], node['player'])
        
#         exploit = node['wins'] / (node['visits'])
#         explore = 1.41 * math.sqrt(math.log(node['parent']['visits']) / (node['visits']))
#         position_bonus = self.position_bonus(node['move'])
        
#         # Add a dynamic adjustment based on game progress
#         total_visits = sum(child['visits'] for child in node['parent']['children'])
#         dynamic_bonus = pattern_bonus * (1 - math.exp(-total_visits / 100))
        
#         return exploit + explore




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

        self.board_size = None
        self.exploit_threshold = 5

        self.available_moves = {}

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
        # forward check for winning move\
        
        move = self.forwardCheck(state.copy(), self.player_number)
        if move is not None:
            return move
        # forward check for blocking move
        move = self.forwardCheck(state.copy(), 3 - self.player_number)
        if move is not None:
            return move
        
        if not self.board_size:
            self.board_size = (state.shape[0] + 1) // 2

        state_copy = state.copy()
        return self.MCTS(state_copy)

        # Do the rest of your implementation here
        raise NotImplementedError('Whoops I don\'t know what to do')
    
    def MCTS(self, state):
        root = {'state': state, 'visits': 0, 'wins': 0, 
                'move': None, 'parent': None, 'children': [], 'player': self.player_number}

        timeout = 10
        start = time.time()
        while time.time() - start < timeout:
            # print("Calling from root")
            self.search(root, state.copy())

        best = root['children'][0]
        for child in root['children']:
            if self.win_rate(child) > self.win_rate(best):
                best = child
        return best['move']
    
    def search(self, node, state):
        if len(node['children']) == 0 and node['visits'] == 0:
            # print(f"Rollout on node {state}")
            outcome = self.rollout(state.copy())
            self.backprop(node, outcome)
        else:
            if self.add_new_child(node):
                if state.tobytes() not in self.available_moves:
                    self.available_moves[state.tobytes()] = np.argwhere(state == 0)
                available_moves = self.available_moves[state.tobytes()]
                if len(available_moves) == 0: 
                    return
                move = random.choice(available_moves)
                self.remove_move(state, move)
                new_state = state.copy()
                new_state[move[0], move[1]] = self.player_number
                new_node = {'state': new_state, 'visits': 0, 'wins': 0, 'parent': node, 
                            'move': tuple(move), 'children': [], 'player': 3 - node['player']}
                node['children'].append(new_node)
            
            best = node['children'][0]
            for child in node['children']:
                if self.UCB(child) > self.UCB(best):
                    best = child
            self.search(best, best['state'])
    
    def remove_move(self, state, move):
        new_moves = []
        for old_move in self.available_moves[state.tobytes()]:
            if old_move[0] != move[0] or old_move[1] != move[1]:
                new_moves.append(old_move)
        self.available_moves[state.tobytes()] = new_moves

    def add_new_child(self, node):
        if len(node['children']) == 0:
            return True
        if len(self.available_moves[node['state'].tobytes()]) == 0:
            return False
        mean_child_visits = sum(child['visits'] for child in node['children']) / len(node['children'])
        return mean_child_visits < self.exploit_threshold

    def rollout(self, rollout_state):
        while True:
            available_moves = np.argwhere(rollout_state == 0)
            if len(available_moves) == 0: 
                break
            move = random.choice(available_moves)
            rollout_state[move[0], move[1]] = self.player_number
            if check_win(rollout_state, tuple(move), self.player_number)[0]:
                return 1
            
            available_moves = np.argwhere(rollout_state == 0)
            if len(available_moves) == 0: 
                break
            move = random.choice(available_moves)
            rollout_state[move[0], move[1]] = 3 - self.player_number
            if check_win(rollout_state, tuple(move), 3 - self.player_number)[0]:
                return 0
            
        return 0.5

    def backprop(self, node, outcome):
        while node is not None:
            node['visits'] += 1
            node['wins'] += outcome
            node = node['parent']

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
    
    def UCB(self, node):
        if node['parent']['visits'] == 0 or node['visits'] == 0:
            return float('inf')
        exploit = node['wins'] / node['visits']
        explore = 1 * math.sqrt(math.log(node['parent']['visits']) / node['visits'])
        return exploit + explore

    def win_rate(self, node):
        if node['visits'] == 0:
            return 0
        return node['wins'] / node['visits']


