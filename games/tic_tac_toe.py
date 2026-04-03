import numpy as np
import copy
from games.base_game import BaseGame

class TicTacToe(BaseGame):
    def __init__(self):
        self.board = np.zeros(9, dtype=int)
        self.current_player = 1
        self.done = False
        self.winner = None

    def reset(self) -> np.ndarray:
        self.board = np.zeros(9, dtype=int)
        self.current_player = 1
        self.done = False
        self.winner = None
        return self.board.copy()

    def get_valid_moves(self) -> list:
        if self.done:
            return []
        return [i for i in range(9) if self.board[i] == 0]

    def make_move(self, move: int, player: int) -> tuple:
        if self.done:
            return self.board.copy(), 0, True, {'winner': self.winner}
        if self.board[move] != 0:
            return self.board.copy(), -1, True, {'winner': -player, 'invalid': True}
        self.board[move] = player
        winner = self.check_winner()
        if winner is not None:
            self.done = True
            self.winner = winner
            if winner == player:
                reward = 1.0
            elif winner == 0:
                reward = 0.3
            else:
                reward = -1.0
            return self.board.copy(), reward, True, {'winner': winner}
        self.current_player = -player
        return self.board.copy(), 0.0, False, {'winner': None}

    def check_winner(self) -> int:
        b = self.board.reshape(3, 3)
              
        for row in b:
            if abs(row.sum()) == 3:
                return row[0]
              
        for col in b.T:
            if abs(col.sum()) == 3:
                return col[0]
                   
        d1 = b[0, 0] + b[1, 1] + b[2, 2]
        d2 = b[0, 2] + b[1, 1] + b[2, 0]
        if abs(d1) == 3:
            return b[1, 1]
        if abs(d2) == 3:
            return b[1, 1]
              
        if 0 not in self.board:
            return 0
        return None

    def clone(self):
        g = TicTacToe()
        g.board = self.board.copy()
        g.current_player = self.current_player
        g.done = self.done
        g.winner = self.winner
        return g

    def render(self) -> str:
        symbols = {0: '.', 1: 'X', -1: 'O'}
        b = self.board.reshape(3, 3)
        rows = []
        for row in b:
            rows.append(' | '.join(symbols[v] for v in row))
        sep = '-' * 9
        return f'\n{sep}\n'.join(rows)

    def get_state_key(self) -> str:
        return str(tuple(self.board))
