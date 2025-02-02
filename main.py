import pygame, sys
from q_learning import TicTacToeQLearning

pygame.init()

WIDTH, HEIGHT = 900, 820
FONT = pygame.font.Font('assets/Roboto-Regular.ttf', 100)
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Tic Tac Toe With AI")

BOARD = pygame.image.load('assets/Board.png')
X_IMG = pygame.image.load('assets/X.png')
O_IMG = pygame.image.load('assets/O.png')

BG_COLOR = (214, 201, 227)

board = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
graphical_board = [[[None, None], [None, None], [None, None]], 
                   [[None, None], [None, None], [None, None]], 
                   [[None, None], [None, None], [None, None]]]

to_move = 'X'

# Initialize the Q-learning agent
ai_agent = TicTacToeQLearning()
ai_agent.save_q_table("q_table.pkl")
# Load the pre-trained Q-table (if available)
try:
    ai_agent.load_q_table("q_table.pkl")
    print("Q-table loaded successfully!")
except FileNotFoundError:
    print("No pre-trained Q-table found. Starting with an empty Q-table.")

def render_board(board, ximg, oimg):
    global graphical_board
    for i in range(3):
        for j in range(3):
            if board[i][j] == 'X':
                graphical_board[i][j][0] = ximg
                graphical_board[i][j][1] = ximg.get_rect(center=(j * 300 + 150, i * 300 + 150))
            elif board[i][j] == 'O':
                graphical_board[i][j][0] = oimg
                graphical_board[i][j][1] = oimg.get_rect(center=(j * 300 + 150, i * 300 + 150))

SCREEN.fill(BG_COLOR)
SCREEN.blit(BOARD, (64, 64))
pygame.display.update()

def add_XO(board, graphical_board, to_move):
    current_pos = pygame.mouse.get_pos()
    converted_x = (current_pos[0] - 64) // 300
    converted_y = (current_pos[1] - 64) // 300
    
    if 0 <= converted_x < 3 and 0 <= converted_y < 3:
        if board[converted_y][converted_x] not in ['X', 'O']:
            board[converted_y][converted_x] = to_move
            to_move = 'O' if to_move == 'X' else 'X'
    
    render_board(board, X_IMG, O_IMG)
    
    for i in range(3):
        for j in range(3):
            if graphical_board[i][j][0] is not None:
                SCREEN.blit(graphical_board[i][j][0], graphical_board[i][j][1])
    
    return board, to_move

def check_win(board):
    for row in range(3):
        if board[row][0] == board[row][1] == board[row][2]:
            return board[row][0]
    
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

def get_ai_move(board):
    return ai_agent.choose_action(board)

def update_q_table(board, action, reward, next_board):
    ai_agent.update_q_table(board, action, reward, next_board)

def show_winner(winner):
    if winner == "DRAW":
        text = FONT.render("It's a draw!", True, (0, 0, 0))
    else:
        text = FONT.render(f"Player {winner} wins!", True, (0, 0, 0))
    SCREEN.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT // 2 - text.get_height() // 2))
    pygame.display.update()
    pygame.time.wait(2000)

def reset_game():
    global board, graphical_board, to_move, game_finished
    board = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    graphical_board = [[[None, None], [None, None], [None, None]], 
                       [[None, None], [None, None], [None, None]], 
                       [[None, None], [None, None], [None, None]]]
    to_move = 'X'
    game_finished = False
    SCREEN.fill(BG_COLOR)
    SCREEN.blit(BOARD, (64, 64))
    pygame.display.update()

game_finished = False

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        
        if event.type == pygame.MOUSEBUTTONDOWN and not game_finished:
            # Player's move
            prev_board = [row[:] for row in board]  # Save the previous board state
            board, to_move = add_XO(board, graphical_board, to_move)
            winner = check_win(board)
            if winner is not None:
                game_finished = True
                show_winner(winner)
                reset_game()
            else:
                # AI's move
                if to_move == 'O':
                    ai_move = get_ai_move(board)
                    if ai_move:
                        prev_ai_board = [row[:] for row in board]  # Save the board state before AI's move
                        board[ai_move[0]][ai_move[1]] = 'O'
                        to_move = 'X'
                        render_board(board, X_IMG, O_IMG)
                        for i in range(3):
                            for j in range(3):
                                if graphical_board[i][j][0] is not None:
                                    SCREEN.blit(graphical_board[i][j][0], graphical_board[i][j][1])
                        winner = check_win(board)
                        if winner is not None:
                            game_finished = True
                            show_winner(winner)
                            reset_game()
                        else:
                            # Update Q-table for the AI's move
                            reward = 0  # No reward unless the game ends
                            update_q_table(prev_ai_board, ai_move, reward, board)
            
            pygame.display.update()