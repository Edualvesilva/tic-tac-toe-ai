import numpy as np
import random
import pickle

class TicTacToeQLearning():
    def __init__(self,alpha = 0.1,gamma = 0.9,epsilon = 0.2):
        self.q_table = {}  # Q-Table stored as a dictionary
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        
    def get_state(self, board):
        return str(board)

    def get_valid_moves(self, board):
        return [(i, j) for i in range(3) for j in range(3) if board[i][j] not in ['X', 'O']]
    
    def choose_action(self,board):
        state = self.get_state(board)
        valid_moves = self.get_valid_moves(board)
        
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(valid_moves)  # Explore
        
        if state not in self.q_table:
            self.q_table[state] = {move: 0 for move in valid_moves}
            return random.choice(valid_moves)
        
        return max(self.q_table[state], key=self.q_table[state].get)  # Exploit
    
    def update_q_table(self, board, action, reward, next_board):
        state = self.get_state(board)
        next_state = self.get_state(next_board)
        
        if state not in self.q_table:
            self.q_table[state] = {move: 0 for move in self.get_valid_moves(board)}
        
        if next_state not in self.q_table:
            self.q_table[next_state] = {move: 0 for move in self.get_valid_moves(next_board)}
        
        max_future_q = max(self.q_table[next_state].values(), default=0)
        self.q_table[state][action] += self.alpha * (reward + self.gamma * max_future_q - self.q_table[state][action])
    
        if reward == 0:  # If it's a neutral move (neither win nor loss)
            self.q_table[state][action] -= 0.1  # Small penalty
    
    def save_q_table(self, filename="q_table.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self, filename="q_table.pkl"):
        with open(filename, "rb") as f:
            self.q_table = pickle.load(f)

def check_winner(board):
    for row in board:
        if row[0] == row[1] == row[2]:
            return row[0]
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col]:
            return board[0][col]
    if board[0][0] == board[1][1] == board[2][2]:
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0]:
        return board[0][2]
    if all(board[i][j] in ['X', 'O'] for i in range(3) for j in range(3)):
        return "DRAW"
    return None

def train_q_learning(episodes=10000):
    agent = TicTacToeQLearning()
    for _ in range(episodes):
        board = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        turn = 'X'
        game_over = False
        move_history = []

        while not game_over:
            if turn == 'O':
                action = agent.choose_action(board)
                board[action[0]][action[1]] = 'O'
                move_history.append((board, action))

            winner = check_winner(board)
            if winner is not None:
                reward = 1 if winner == 'O' else -1 if winner == 'X' else 0
                for b, a in move_history:
                    agent.update_q_table(b, a, reward, board)
                game_over = True
            
            turn = 'X' if turn == 'O' else 'O'
    
    agent.save_q_table()
    print("Training completed and Q-table saved!")

if __name__ == "__main__":
    train_q_learning()
