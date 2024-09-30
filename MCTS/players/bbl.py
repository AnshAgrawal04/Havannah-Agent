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

        self.board_size = None
        self.exploit_threshold = 5

        self.available_moves = {}
        self.node_connections = {}
        self.rollout_timer = 0

        self.state_tree = {}
        self.iters = 5

        self.init_time = 20
        self.total_time = fetch_remaining_time(timer, self.player_number)


    def get_move(self, state: np.array) -> Tuple[int, int]:
        
        # Applying Monte Carlo Tree Search        
        if not self.board_size:
            self.board_size = (state.shape[0] + 1) // 2

        # state_copy = self.convert_to_us(state)
        move = self.MCTS(state.copy())
        return move
        # return self.convert_move_to_ta(move)

        # Do the rest of your implementation here
        raise NotImplementedError('Whoops I don\'t know what to do')
    
    def MCTS(self, state):
        if state.tobytes() not in self.state_tree:
            node = {'state': state, 'visits': 0, 'wins': 0, 
                    'move': None, 'parent': None, 'children': [], 
                    'player': self.player_number}
            self.state_tree[state.tobytes()] = node

        move = self.forwardCheck(state.copy(), self.player_number)
        if move is not None:
            return move

        move = self.forwardCheck(state.copy(), 3 - self.player_number)
        if move is not None:
            return move

        timeout = self.set_timeout()
        start = time.time()
        while time.time() - start < timeout:
            self.search(state.copy())
        
        print("Number of rollouts for this move are:", self.rollout_timer)
        self.rollout_timer = 0

        children = self.state_tree[state.tobytes()]['children']
        best = children[0]
        for child in children:
            if child['wins'] > best['wins']:
                best = child
            # print(f"Wins: {child['wins']}, Visits: {child['visits']}")
            
        return best['move']      
    
    def search(self, state):
        node = self.state_tree[state.tobytes()]
        if len(node['children']) == 0 and node['visits'] == 0:
            outcome = self.rollout(state.copy(), node['player'])
            self.backprop(state.copy(), outcome)
        else:
            if self.add_new_child(node):

                if state.tobytes() not in self.available_moves:
                    self.available_moves[state.tobytes()] = np.argwhere(state == 0).tolist()
                    random.shuffle(self.available_moves[state.tobytes()])

                if len(self.available_moves[state.tobytes()]) == 0: 
                    return
                move = self.available_moves[state.tobytes()].pop()

                new_state = state.copy()
                new_state[move[0], move[1]] = node['player']
                new_player = 3 - node['player']
                new_node = {'state': new_state, 'visits': 0, 'wins': 0, 'parent': node, 
                            'move': tuple(move), 'children': [], 'player': new_player}
                
                self.state_tree[new_state.tobytes()] = new_node
                self.state_tree[state.tobytes()]['children'].append(new_node)
            
            children = self.state_tree[state.tobytes()]['children']
            if len(children) == 0:
                return
            best = children[0]
            for child in children:
                if self.UCB(child) > self.UCB(best):
                    best = child
            self.search(best['state'])
    
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
        return True

    def rollout(self, rollout_state, player):
        outcome = 0
        itr = 0
        self.rollout_timer += 1
        while itr < self.iters:
            result = None
            curr_state = rollout_state.copy()
            draw = True
            while True:
                available_moves = np.argwhere(curr_state == 0)
                if len(available_moves) == 0:
                    break
                move = random.choice(available_moves)
                curr_state[move[0], move[1]] = player
                if check_win(curr_state, tuple(move), player)[0]:
                    result = 1
                    draw = False
                    break
                
                available_moves = np.argwhere(curr_state == 0)
                if len(available_moves) == 0: 
                    break
                move = random.choice(available_moves)
                curr_state[move[0], move[1]] = 3 - player
                if check_win(curr_state, tuple(move), 3 - player)[0]:
                    result = 0
                    draw = False
                    break  
            if draw:          
                result = 0.5
            itr += 1
            outcome += result
        return outcome

    def choose_rollout_move(self, prev_move, available_moves):
        if prev_move and prev_move in self.LGR:
            return self.LGR[prev_move]
        if len(available_moves) == 0:
            return None
        return random.choice(available_moves)

    def backprop(self, state, outcome):
        node = self.state_tree[state.tobytes()]
        while node is not None:
            state = node['state']
            self.state_tree[state.tobytes()]['visits'] += self.iters
            self.state_tree[state.tobytes()]['wins'] += outcome
            node = node['parent']

    def forwardCheck(self, state, player):
        old_state = state
        for i in range(0, state.shape[0]):
            for j in range(0, state.shape[1]):
                if state[i][j] == 0:
                    state[i][j] = player
                    if check_win(state, (i, j), player)[0]:
                        if state.tobytes() not in self.state_tree:
                            node = {'state': state, 'visits': 1, 'wins': (self.player_number == player), 
                                'move': (i, j), 'parent': None, 'children': [], 'player': 3 - player}
                            self.state_tree[state.tobytes()] = node
                            self.state_tree[old_state.tobytes()]['children'].append(node)
                        state[i][j] = 0
                        return (i, j)
                    state[i][j] = 0
        return None
    
    def UCB(self, node):
        if node['parent']['visits'] == 0 or node['visits'] == 0:
            return float('inf')
        exploit = node['wins'] / node['visits']
        explore = 1.414 * math.sqrt(math.log(node['parent']['visits']) / node['visits'])
        return exploit + explore

    def win_rate(self, node):
        if node['visits'] == 0:
            return 0
        return node['wins'] / node['visits']

    def convert_to_us(self, state):
        sz = self.board_size
        new_state = np.zeros((2*sz-1, 2*sz - 1))
        for i in range(2*sz - 1):
            for j in range(2*sz - 1):
                if abs(i - j) >= sz:
                    new_state[(i, j)] = 3

        for i in range(2*sz - 1):
            for j in range(sz):
                new_state[(i, j)] = state[(i, j)]

        for i in range(2*sz - 1):
            for j in range(sz, 2*sz-1):
                if abs(i - j) >= sz:
                    continue
                diff = j - sz + 1
                new_state[(i, j)] = state[(i - diff, j)]
        
        return new_state
                  
    def convert_to_ta(self, state):
        sz = self.board_size
        new_state = np.zeros((2*sz - 1, 2*sz - 1))
        for i in range(2*sz - 1):
            for j in range(2*sz - 1):
                if i - j >= sz or i + j > 3*sz - 3:
                    new_state[(i, j)] = 3

        for i in range(2*sz - 1):
            for j in range(sz):
                new_state[(i, j)] = state[(i, j)]

        for i in range(2*sz - 1):
            for j in range(sz, 2*sz-1):
                if abs(i - j) >= sz:
                    continue
                diff = j - sz + 1
                new_state[(i - diff, j)] = state[(i, j)]
                
        return new_state

    def convert_move_to_ta(self, move):
        i = move[0]
        j = move[1]
        if j <= self.board_size-1:
            return move
        else:
            diff = j - self.board_size + 1
            return tuple((i - diff, j))

    def matchPattern(self, move, state, player):
        connections = 0
        connection_dict = []
        state[move[0], move[1]] = player
        scaling_factor = 1.0  # Adjust based on distance to edge/corner

        for pattern in self.loose_bridge_pattern:
            valid = True
            for dir in range(4):
                x = pattern['pattern'][dir][0] + move[0]
                y = pattern['pattern'][dir][1] + move[1]
                if (x < 0 or x > 2*self.board_size-2) or (y < 0 or y > 2*self.board_size-2):
                    valid = False
                    break
                if pattern['stones'][dir] == 0 and state[(x,y)] != 0 and state[(x,y)] != player:
                    valid = False
                    break
                if state[(x,y)] != pattern['stones'][dir] * player:
                    valid = False
                    break
            if valid:
                # Increase bonus if closer to edge or strategic area
                connections += scaling_factor
                connection_dict.append([move, (x, y)])
        return connections

    def checkVC(self, node):
        if node['parent'] is None:
            return 0
        state_ = self.convert_to_us(node['state'])
        parent_ = self.convert_to_us(node['parent']['state'])
        move = np.argwhere(state_ != parent_)

        if len(move) == 0:
            return 0
        move= move[0]
        # access the first element of the tuple
        move = tuple(move)
        move_x = move[0]
        move_y = move[1]
        # print(move_x, move_y)

        patterns =  [{'pattern':[(0,0), (0,1), (1,1), (1,2)], 'stones': [1, 0, 0, 1]},
                                     {'pattern':[(0,0), (1,0), (1,1), (2,1)], 'stones': [1, 0, 0, 1]},
                                     {'pattern':[(0,0), (-1,0), (0,1), (-1,1)], 'stones': [1, 0, 0, 1]},
                                     {'pattern':[(0,0), (-1,-1), (-1,0), (-2,-1)], 'stones': [1, 0, 0, 1]},
                                     {'pattern':[(0,0), (-1,-1), (0,-1), (-1,-2)], 'stones': [1, 0, 0, 1]},
                                     {'pattern':[(0,0), (0,-1), (1,0), (1,-1)], 'stones': [1, 0, 0, 1]}]
        connections = 0

        for pattern in patterns:
            valid = True
            for dir in [0,1,2,3]:
                x = pattern['pattern'][dir][0] + move_x
                y = pattern['pattern'][dir][1] + move_y
                if (x < 0 or x > 2*self.board_size-2) or (y < 0 or y > 2*self.board_size-2) or (state_[(x,y)] == 3):
                    valid = False
                    break

                if pattern['stones'][dir] == 0 :
                    if state_[(x,y)] == node['player']:
                        valid = False
                        break
                else:
                    if state_[(x,y)] != 3-node['player']:
                        valid = False
                        break
            if valid:
                connections += 1
                # print("Connection found with ", move , " and the second move is ", (x, y))
        return connections




# import time
# import math
# import random
# import numpy as np
# from helper import *

# class AIPlayer:

#     def __init__(self, player_number: int, timer):
        
#         self.player_number = player_number
#         self.type = 'ai'
#         self.player_string = 'Player {}: ai'.format(player_number)

#         self.board_size = None
#         self.exploit_threshold = 5

#         self.available_moves = {}
#         self.node_connections = {}
#         self.rollout_timer = 0
#         self.current_state = None
#         self.tree = {}

#     def get_move(self, state: np.array) -> Tuple[int, int]:
       
#         move = self.forwardCheck(state.copy(), self.player_number)
#         if move is not None:
#             return move
#         # forward check for blocking move
#         move = self.forwardCheck(state.copy(), 3 - self.player_number)
#         if move is not None:
#             return move
        
#         if not self.board_size:
#             self.board_size = (state.shape[0] + 1) // 2

#         # state_copy = self.convert_to_us(state)
#         move = self.MCTS(state.copy())
#         return move
    
#     def forwardCheck(self, state, player):
#         for i in range(state.shape[0]):
#             for j in range(state.shape[1]):
#                 if state[i][j] == 0:
#                     state[i][j] = player
#                     if check_win(state, (i,j), player)[0]:
#                         return (i, j)
#                     state[i][j] = 0
#         return None
    
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
    
#     def MCTS(self, state: np.array) -> Tuple[int, int]:
#         #-----------------Initialisation-----------------
#         if state.tobytes() not in self.tree:
#             self.tree[state.tobytes()] = {'state': state, 'visits': 0, 'wins': 0, 
#                 'move': None, 'parent': None, 'children': [], 'player': self.player_number, 'our_state':self.convert_to_us(state),'childrenDistribution': None}
#         self.tree[state.tobytes()]['parent'] = None
#         self.current_state = state
#         root = self.tree[state.tobytes()]
#         #-----------------Growing Tree--------------------
#         timeout = 10
#         start = time.time()
#         while time.time() - start < timeout:
#             self.search(root, state.copy())
        
#         print("Number of rollouts for this move are:", self.rollout_timer)
#         self.rollout_timer = 0
#         #-----------------Selecting Child-------------------
#         best = root['children'][0]
#         for child in root['children']:
#             if self.win_rate(child) > self.win_rate(best):
#                 best = child
#         return best['move']
    
#     def search(self, node, state):
#         if len(node['children']) == 0 and node['visits'] == 0:
#             outcome = self.rollout(state.copy())
#             self.backpropagate(node, outcome)
        
#         else:
#             if self.add_new_child(node):
#                 self.add_children(node)
#             if len(node['children']) == 0:
#                 return
#             best = node['children'][0]  
#             for child in node['children']:
#                 if self.win_rate(child) > self.win_rate(best):
#                     best = child
#             self.search(best, best['state'])

#     def add_children(self, node):
#         if node['state'].tobytes() not in self.available_moves:
#             self.available_moves[node['state'].tobytes()] = self.getMoves(node['state'])
#         if len(self.available_moves[node['state'].tobytes()]) == 0:
#             return
#         if node['childrenDistribution'] is None:
#             node['childrenDistribution']=self.generateDist(node)
#         if (node['childrenDistribution'] == []): return
#         _, move = random.choices(node['childrenDistribution'], weights=[x[0] for x in node['childrenDistribution']], k=1)[0]
#         new_state = node['state'].copy()
#         new_state[move] = node['player']
#         self.tree[new_state.tobytes()] = {'state': new_state, 'visits': 0, 'wins': 0, 'move': move, 'parent': node, 'children': [], 'player': 3 - node['player'], 'our_state':self.convert_to_us(new_state), 'childrenDistribution': None}
#         node['children'].append(self.tree[new_state.tobytes()])
#         return  
        
#     def generateDist(self, node):
#         node['childrenDistribution'] = []
#         for move in self.available_moves[node['state'].tobytes()]:
#             node['childrenDistribution'].append((0.1 + self.checkVC(node, move) + 0.01 * self.checkLoc(node, move), move))
#         node['childrenDistribution'].sort(reverse = True)
#         return node['childrenDistribution']

#     def add_new_child(self, node):
#         if len(node['children']) == 0:
#             return True
#         if len(self.available_moves[node['state'].tobytes()]) == 0:
#             return False
#         return True

#     def getMoves(self, state):
#         moves = []
#         for i in range(state.shape[0]):
#             for j in range(state.shape[1]):
#                 if state[i][j] == 0:
#                     moves.append((i, j))
#         return moves
    
#     def checkVC(self, node, move):
#         if node['parent'] is None:
#             return 0
#         parent_= node['our_state']
#         move = self.convert_move_to_us(move)
#         state_ = parent_.copy()
#         state_[move] = node['player']
#         move_x = move[0]
#         move_y = move[1]

#         patterns =  [{'pattern':[(0,0), (0,1), (1,1), (1,2)], 'stones': [1, 0, 0, 1]},
#                     {'pattern':[(0,0), (1,0), (1,1), (2,1)], 'stones': [1, 0, 0, 1]},
#                     {'pattern':[(0,0), (-1,0), (0,1), (-1,1)], 'stones': [1, 0, 0, 1]},
#                     {'pattern':[(0,0), (-1,-1), (-1,0), (-2,-1)], 'stones': [1, 0, 0, 1]},
#                     {'pattern':[(0,0), (-1,-1), (0,-1), (-1,-2)], 'stones': [1, 0, 0, 1]},
#                     {'pattern':[(0,0), (0,-1), (1,0), (1,-1)], 'stones': [1, 0, 0, 1]}]
#         connections = 0

#         for pattern in patterns:
#             valid = True
#             for dir in [0,1,2,3]:
#                 x = pattern['pattern'][dir][0] + move_x
#                 y = pattern['pattern'][dir][1] + move_y
#                 if (x < 0 or x > 2*self.board_size-2) or (y < 0 or y > 2*self.board_size-2) or (state_[(x,y)] == 3):
#                     valid = False
#                     break

#                 if pattern['stones'][dir] == 0 :
#                     if state_[(x,y)] == 3 - node['player']:
#                         valid = False
#                         break
#                 else:
#                     if state_[(x,y)] != node['player']:
#                         valid = False
#                         break
#             if valid:
#                 connections += 1
#                 # print("Connection found with ", move , " and the second move is ", (x, y))
#         return connections
    
#     def checkLoc(self,node,move):
#         if node['parent'] is None:
#             return 0
#         parent_= node['our_state']
#         move = self.convert_move_to_us(move)
#         state_ = parent_.copy()
#         state_[move] = node['player']
#         move_x = move[0]
#         move_y = move[1]

#         localities = 0

#         for tup in [(-1,-1),(-1,0),(0,-1),(0,1),(1,0),(1,1)]:
#             x = move_x + tup[0]
#             y = move_y + tup[1]
#             if (x < 0 or x > 2*self.board_size-2) or (y < 0 or y > 2*self.board_size-2) or (state_[(x,y)] == 3):
#                 continue
#             if state_[(x,y)] == node['player']:
#                 localities += 1
#         return localities

#     def convert_move_to_us(self, move):
#         x, y = move
#         if y < self.board_size -1:
#             return (x, y)
#         else:
#             diff = y - self.board_size + 1
#             return (x + diff, y)
        
#     def UCB(self, node):
#         if node['parent']['visits'] == 0 or node['visits'] == 0:
#             return float('inf')
#         exploit = node['wins'] / node['visits']
#         explore = 1.5 * math.sqrt(math.log(node['parent']['visits']) / node['visits'])
#         # pattern_bonus = 2 * self.checkVC(node) / node['visits']
#         return exploit + explore
        
#     def rollout(self, rollout_state):
#         outcome = 0
#         iters = 1
#         itr = 0
#         self.rollout_timer += 1
#         while itr < iters:
#             curr_state = rollout_state.copy()
#             while True:
#                 available_moves = np.argwhere(curr_state == 0)
#                 if len(available_moves) == 0: 
#                     break
#                 move = random.choice(available_moves)
#                 curr_state[move[0], move[1]] = self.player_number
#                 if check_win(curr_state, tuple(move), self.player_number)[0]:
#                     outcome += 1
                
#                 available_moves = np.argwhere(curr_state == 0)
#                 if len(available_moves) == 0: 
#                     break
#                 move = random.choice(available_moves)
#                 curr_state[move[0], move[1]] = 3 - self.player_number
#                 if check_win(curr_state, tuple(move), 3 - self.player_number)[0]:
#                     outcome += 0
#             outcome += 0.5
#             itr += 1
#         return outcome / iters 
    
#     def backpropagate(self, node, outcome):
#         while node is not None:
#             node['visits'] += 1
#             node['wins'] += outcome
#             node = node['parent']

#     def win_rate(self, node):
#         return node['wins'] / node['visits'] if node['visits'] != 0 else 0