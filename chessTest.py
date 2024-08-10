import torch
import torch.nn as nn
import torch.optim as optim
import chess
import random
import numpy as np

class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.fc1 = nn.Linear(64, 128)  # 64 squares on the chessboard
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)  # Output a value for a given board state

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

def board_to_tensor(board):
    piece_map = {
        'p': 1, 'P': -1,  # pawns
        'n': 2, 'N': -2,  # knights
        'b': 3, 'B': -3,  # bishops
        'r': 4, 'R': -4,  # rooks
        'q': 5, 'Q': -5,  # queens
        'k': 6, 'K': -6   # kings
    }
    board_tensor = torch.zeros(64)
    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            board_tensor[i] = piece_map[piece.symbol()]
    return board_tensor

def train_chess_net(model, optimizer, criterion, board, user_move):
    model.train()

    optimizer.zero_grad()

    board_tensor = board_to_tensor(board)
    
    output = model(board_tensor)
    
    target = torch.tensor([1.0])  # Assume the user's move is good
    loss = criterion(output, target)
    
    loss.backward()
    optimizer.step()
    
    return loss.item()

def get_user_move(board):
    while True:
        user_move = input("Your move (UCI format): ")
        try:
            move = chess.Move.from_uci(user_move)
            if move in board.legal_moves:
                return move
            else:
                print("Illegal move. Try again.")
        except ValueError:
            print("Invalid UCI format. Try again.")

def bot_move(board, model):
    legal_moves = list(board.legal_moves)
    best_move = None
    best_value = -float('inf')
    
    for move in legal_moves:
        board.push(move)
        board_tensor = board_to_tensor(board)
        value = model(board_tensor).item()
        board.pop()
        
        if value > best_value:
            best_value = value
            best_move = move
    
    return best_move

# Initialize the neural network, optimizer, and loss function
model = ChessNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

board = chess.Board()

while not board.is_game_over():
    if board.turn == chess.WHITE:
        move = bot_move(board, model)
        print("Bot's move:", move)
        board.push(move)
    else:
        move = get_user_move(board)
        board.push(move)
        
        loss = train_chess_net(model, optimizer, criterion, board, move)
        print(f"Training loss: {loss}")
    
    print(board)
