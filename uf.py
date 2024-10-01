import time
import math
import random
import numpy as np
from helper import *

# Store C_STRING as 12 bits of 0s
C_STRING = 0b000000000000

def list_deep_copy(l):
    nl = []
    for i in l:
        if type(i) == list:
            nl.append(list_deep_copy(i))
        else:
            nl.append(i)
    return nl

class UnionFind:
    def __init__(self, rows, flag=1, state=None):
        if flag == 0:
            return
        self.rows = rows
        self.parent = [[(i, j) for j in range(rows)] 
                       for i in range(rows)]
        self.sz = [[1 for _ in range(rows)] 
                   for __ in range(rows)]
        self.bits = [[C_STRING for _ in range(rows)] 
                       for __ in range(rows)]
        if state is not None:
            self.initialise(state)
    
    def copy(self):
        uf = UnionFind(1, flag=0)
        uf.rows = self.rows
        uf.parent = list_deep_copy(self.parent)
        uf.sz = list_deep_copy(self.sz)
        uf.bits = list_deep_copy(self.bits)
        return uf

    def find(self, x, y):
        if self.parent[x][y] == (x, y):
            return x, y
        self.parent[x][y] = self.find(self.parent[x][y][0], self.parent[x][y][1])
        return self.parent[x][y]
    
    def union(self, x1, y1, x2, y2):
        rx1, ry1 = self.find(x1, y1)
        rx2, ry2 = self.find(x2, y2)
        if (rx1, ry1) == (rx2, ry2):
            return
        if self.sz[rx1][ry1] < self.sz[rx2][ry2]:
            self.parent[rx1][ry1] = (rx2, ry2)
            self.sz[rx2][ry2] += self.sz[rx1][ry1]
            self.bits[rx2][ry2] |= self.bits[rx1][ry1]
        else:
            self.parent[rx2][ry2] = (rx1, ry1)
            self.sz[rx1][ry1] += self.sz[rx2][ry2]
            self.bits[rx1][ry1] |= self.bits[rx2][ry2]

    def initialise(self, state):
        corners = get_all_corners(self.rows)
        edges = get_all_edges(self.rows)
        for i in range(len(corners)):
            x, y = corners[i]
            self.bits[x][y] |= (1 << i)
        for i in range(len(edges)):
            for e in edges[i]:
                x, y = e
                self.bits[x][y] |= (1 << (i + 6))
        for i in range(self.rows):
            for j in range(self.rows):
                if state[i][j] == 0 or state[i][j] == 3:
                    continue
                neighbors = get_neighbours(self.rows, (i, j))
                for x, y in neighbors:
                    if state[x][y] == state[i][j]:
                        self.union(i, j, x, y)
    
    def update_bits(self, state, move):
        x, y = move
        neighbors = get_neighbours(self.rows, move)
        if state[x][y] == 0 or state[x][y] == 3:
            return
        for x, y in neighbors:
            if state[x][y] == 0 or state[x][y] == 3:
                continue
            if state[x][y] == state[move]:
                self.union(x, y, move[0], move[1])
    
    def check_win(self, state, move):
        x, y = move
        rx, ry = self.find(x, y)

        num_corners = 0
        for i in range(6):
            if self.bits[rx][ry] & (1 << i):
                num_corners += 1
        if num_corners >= 2:
            # print("Corner win")
            return True
        
        num_edges = 0
        for i in range(6, 12):
            if self.bits[rx][ry] & (1 << i):
                num_edges += 1
        if num_edges >= 3:
            # print("Edge win")
            return True

        player = state[move]
        new_state = (state == player)
        if check_ring(new_state, move):
            # print("Ring win")
            return True
        return False

class MCTSNode:
    def __init__(self, state, player, uf, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.uf = uf
        self.visits = 0
        self.wins = 0
        self.children = []
        self.player = player 
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
        self.rollout_iters = 10

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
            uf = UnionFind(state.shape[0], state=state)
            self.root = MCTSNode(state, self.player_number, uf)
        else:
            for child in self.root.children:
                if np.array_equal(child.state, state):
                    self.root = child
                    break
            else:
                uf = UnionFind(state.shape[0], state=state)
                self.root = MCTSNode(state, self.player_number, uf)

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
                new_uf = node.uf.copy()
                new_uf.update_bits(new_state, new_move)
                node.children.append(MCTSNode(new_state, 3 - node.player, new_uf, 
                                              parent=node, action=new_move))
            if len(node.children) == 0:
                return
            best_idx = 0
            for i in range(1, len(node.children)):
                if self.UCB(node.children[i]) > self.UCB(node.children[best_idx]):
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

    def UCB(self, node):
        if node.visits == 0 or node.parent.visits == 0: return math.inf
        return node.wins / node.visits + 0.9 * math.sqrt(math.log(node.parent.visits) / node.visits)
    
    def best_child(self, node): 
        best_child = node.children[0]
        for child in node.children:
            # print(f'Wins: {child.wins}, Visits: {child.visits}')
            if child.wins > best_child.wins:
                best_child = child
        return best_child

    def add_child(self, node):
        if node.expandable == []: 
            return False
        return True

    def rollout(self, node):
        outcome = 0
        for i in range(self.rollout_iters):
            state = node.state.copy()
            player = node.player
            available_moves = list_deep_copy(node.actions)
            available_moves = np.random.permutation(available_moves)
            available_moves = available_moves.tolist()
            uf_copy = node.uf.copy()
            # print(i, uf_copy.bits)

            while True:
                if len(available_moves) == 0:
                    outcome += 0.5
                    break
                move = available_moves.pop()
                move = tuple(move)
                state[move] = player
                uf_copy.update_bits(state, move)
                if uf_copy.check_win(state, move):
                    # print(f"Win at state: \n{state} \n f{move}")
                    outcome += 1 if player == self.player_number else 0
                    break
                player = 3 - player

        self.rollouts += 1
        return outcome

    def backpropagate(self, node, outcome):
        while node is not None:
            node.visits += self.rollout_iters
            node.wins += outcome
            node = node.parent

    def forwardCheck(self, state, player):
        available_moves = get_valid_actions(state)
        for move in available_moves:
            state[move] = player
            if check_win(state, move, player)[0]:   return move
            state[move] = 0
        return None