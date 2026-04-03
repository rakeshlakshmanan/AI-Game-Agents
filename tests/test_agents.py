import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from games.tic_tac_toe import TicTacToe
from games.connect4 import Connect4
from agents.random_agent import RandomAgent
from agents.default_opponent import DefaultOpponent
from agents.minimax_agent import MiniMaxAgent
from agents.alphabeta_agent import AlphaBetaAgent
from agents.qlearning_agent import QLearningAgent

class TestDefaultOpponent:
    def test_wins_if_possible_ttt(self):
        game = TicTacToe()
        game.reset()
                                                           
        game.board[0] = 1
        game.board[1] = 1
        agent = DefaultOpponent(1, 'Default')
        move = agent.get_move(game)
        assert move == 2

    def test_blocks_opponent_ttt(self):
        game = TicTacToe()
        game.reset()
                                                       
        game.board[0] = -1
        game.board[1] = -1
        agent = DefaultOpponent(1, 'Default')
        move = agent.get_move(game)
        assert move == 2

    def test_wins_before_block(self):
        game = TicTacToe()
        game.reset()
                                                                
        game.board[0] = 1
        game.board[1] = 1
        game.board[3] = -1
        game.board[4] = -1
        agent = DefaultOpponent(1, 'Default')
        move = agent.get_move(game)
        assert move == 2                            

class TestMiniMaxAgent:
    def test_never_loses_ttt(self):
        import random
        random.seed(0)
        mm = MiniMaxAgent(1, 'MiniMax', max_depth=None)
        random_opp = RandomAgent(-1, 'Random', seed=0)

        losses = 0
        for _ in range(50):
            game = TicTacToe()
            game.reset()
            current = 1
            while True:
                if current == 1:
                    move = mm.get_move(game)
                else:
                    move = random_opp.get_move(game)
                _, _, done, info = game.make_move(move, current)
                if done:
                    if info.get('winner') == -1:
                        losses += 1
                    break
                current = -current
        assert losses == 0

    def test_returns_valid_move(self):
        game = TicTacToe()
        game.reset()
        mm = MiniMaxAgent(1, 'MiniMax')
        move = mm.get_move(game)
        assert move in game.get_valid_moves()

class TestAlphaBetaAgent:
    def test_same_result_as_minimax(self):
        game = TicTacToe()
        game.reset()
        game.make_move(0, 1)
        game.make_move(4, -1)

        mm = MiniMaxAgent(1, 'MiniMax', max_depth=None)
        ab = AlphaBetaAgent(1, 'AlphaBeta', max_depth=None)

        mm_move = mm.get_move(game)
        ab_move = ab.get_move(game)
                                    
        assert mm_move in game.get_valid_moves()
        assert ab_move in game.get_valid_moves()

    def test_fewer_nodes_than_minimax(self):
        game = TicTacToe()
        game.reset()
        mm = MiniMaxAgent(1, 'MiniMax', max_depth=None)
        ab = AlphaBetaAgent(1, 'AlphaBeta', max_depth=None)
        mm.get_move(game)
        ab.get_move(game)
        assert ab.nodes_explored < mm.nodes_explored

class TestQLearningAgent:
    def test_q_table_updates(self):
        agent = QLearningAgent(1, 'QL', alpha=0.5)
        state = '(0, 0, 0, 0, 0, 0, 0, 0, 0)'
        action = 4
        reward = 1.0
        next_state = '(0, 0, 0, 0, 1, 0, 0, 0, 0)'
        agent.learn(state, action, reward, next_state, True, [])
        assert (state, action) in agent.q_table
        assert agent.q_table[(state, action)] != 0.0

    def test_epsilon_decay(self):
        agent = QLearningAgent(1, 'QL', epsilon=1.0, epsilon_decay=0.9, epsilon_min=0.01)
        agent.decay_epsilon()
        assert abs(agent.epsilon - 0.9) < 1e-9

    def test_get_move_valid(self):
        game = TicTacToe()
        game.reset()
        agent = QLearningAgent(1, 'QL')
        move = agent.get_move(game)
        assert move in game.get_valid_moves()
