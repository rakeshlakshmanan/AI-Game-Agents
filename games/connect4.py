import numpy as np
from games.base_game import BaseGame

class Connect4(BaseGame):
    ROWS = 6
    COLS = 7

    def __init__(self):
        self.board = np.zeros((self.ROWS, self.COLS), dtype=int)
        self.current_player = 1
        self.done = False
        self.winner = None

    def reset(self) -> np.ndarray:
        self.board = np.zeros((self.ROWS, self.COLS), dtype=int)
        self.current_player = 1
        self.done = False
        self.winner = None
        return self.board.copy()

    def get_valid_moves(self) -> list:
        if self.done:
            return []
        return [c for c in range(self.COLS) if self.board[0, c] == 0]

    def make_move(self, move: int, player: int) -> tuple:
        if self.done:
            return self.board.copy(), 0, True, {'winner': self.winner}
        if move not in self.get_valid_moves():
            return self.board.copy(), -1, True, {'winner': -player, 'invalid': True}
                               
        row = None
        for r in range(self.ROWS - 1, -1, -1):
            if self.board[r, move] == 0:
                row = r
                break
        self.board[row, move] = player
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
        b = self.board
                    
        for r in range(self.ROWS):
            for c in range(self.COLS - 3):
                window = b[r, c:c+4]
                if abs(window.sum()) == 4 and 0 not in window:
                    return b[r, c]
                  
        for r in range(self.ROWS - 3):
            for c in range(self.COLS):
                window = b[r:r+4, c]
                if abs(window.sum()) == 4 and 0 not in window:
                    return b[r, c]
                                           
        for r in range(self.ROWS - 3):
            for c in range(self.COLS - 3):
                window = [b[r+i, c+i] for i in range(4)]
                if abs(sum(window)) == 4 and 0 not in window:
                    return window[0]
                                           
        for r in range(self.ROWS - 3):
            for c in range(3, self.COLS):
                window = [b[r+i, c-i] for i in range(4)]
                if abs(sum(window)) == 4 and 0 not in window:
                    return window[0]
              
        if len(self.get_valid_moves()) == 0:
            return 0
        return None

    def clone(self):
        g = Connect4()
        g.board = self.board.copy()
        g.current_player = self.current_player
        g.done = self.done
        g.winner = self.winner
        return g

    def render(self) -> str:
        symbols = {0: '.', 1: 'X', -1: 'O'}
        rows = []
        for r in range(self.ROWS):
            rows.append(' '.join(symbols[v] for v in self.board[r]))
        header = ' '.join(str(c) for c in range(self.COLS))
        return header + '\n' + '\n'.join(rows)

    def get_state_key(self) -> str:
        return str(tuple(self.board.flatten()))
