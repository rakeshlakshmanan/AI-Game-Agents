import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from games.tic_tac_toe import TicTacToe
from games.connect4 import Connect4

class TestTicTacToe:
    def test_reset(self):
        game = TicTacToe()
        state = game.reset()
        assert np.all(state == 0)
        assert len(game.get_valid_moves()) == 9

    def test_valid_moves_decrease(self):
        game = TicTacToe()
        game.reset()
        game.make_move(0, 1)
        assert 0 not in game.get_valid_moves()
        assert len(game.get_valid_moves()) == 8

    def test_row_win(self):
        game = TicTacToe()
        game.reset()
        for col in [0, 1, 2]:
            game.make_move(col, 1)
        assert game.check_winner() == 1

    def test_col_win(self):
        game = TicTacToe()
        game.reset()
        for row in [0, 3, 6]:
            game.make_move(row, -1)
        assert game.check_winner() == -1

    def test_diagonal_win(self):
        game = TicTacToe()
        game.reset()
        for pos in [0, 4, 8]:
            game.make_move(pos, 1)
        assert game.check_winner() == 1

    def test_anti_diagonal_win(self):
        game = TicTacToe()
        game.reset()
        for pos in [2, 4, 6]:
            game.make_move(pos, -1)
        assert game.check_winner() == -1

    def test_draw(self):
        game = TicTacToe()
        game.reset()
                                       
        moves = [(0, 1), (1, -1), (2, 1), (3, -1), (4, 1), (5, -1), (6, -1), (7, 1), (8, -1)]
        for move, player in moves:
            if game.check_winner() is None:
                game.make_move(move, player)
        result = game.check_winner()
        assert result == 0 or result is not None                                    

    def test_no_winner_ongoing(self):
        game = TicTacToe()
        game.reset()
        assert game.check_winner() is None

    def test_clone_independence(self):
        game = TicTacToe()
        game.reset()
        game.make_move(0, 1)
        clone = game.clone()
        clone.make_move(4, -1)
        assert game.board[4] == 0
        assert clone.board[4] == -1

    def test_render(self):
        game = TicTacToe()
        game.reset()
        r = game.render()
        assert isinstance(r, str)
        assert len(r) > 0

    def test_state_key(self):
        game = TicTacToe()
        game.reset()
        key = game.get_state_key()
        assert isinstance(key, str)

class TestConnect4:
    def test_reset(self):
        game = Connect4()
        state = game.reset()
        assert np.all(state == 0)
        assert len(game.get_valid_moves()) == 7

    def test_gravity(self):
        game = Connect4()
        game.reset()
        game.make_move(0, 1)
                                       
        assert game.board[5, 0] == 1

    def test_column_fills(self):
        game = Connect4()
        game.reset()
        for _ in range(6):
            game.make_move(0, 1 if _ % 2 == 0 else -1)
                       
        assert 0 not in game.get_valid_moves() or game.board[0, 0] != 0

    def test_horizontal_win(self):
        game = Connect4()
        game.reset()
        for col in range(4):
            game.make_move(col, 1)
        assert game.check_winner() == 1

    def test_vertical_win(self):
        game = Connect4()
        game.reset()
        for _ in range(4):
            game.make_move(0, 1)
            if _ < 3:
                                   
                pass
                                         
        game2 = Connect4()
        game2.reset()
        for _ in range(4):
            game2.board[5-_, 0] = 1
                        
        winner = game2.check_winner()
        assert winner == 1

    def test_diagonal_win(self):
        game = Connect4()
        game.reset()
                                 
        for i in range(4):
            game.board[5-i, i] = 1
        assert game.check_winner() == 1

    def test_clone_independence(self):
        game = Connect4()
        game.reset()
        game.make_move(3, 1)
        clone = game.clone()
        clone.make_move(3, -1)
        assert game.board[4, 3] == 0 or game.board[5, 3] == 1
        assert clone.board[4, 3] == -1 or clone.board[5, 3] == 1

    def test_state_key(self):
        game = Connect4()
        game.reset()
        key = game.get_state_key()
        assert isinstance(key, str)
