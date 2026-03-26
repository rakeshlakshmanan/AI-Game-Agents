import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from games.tic_tac_toe import TicTacToe
from agents.random_agent import RandomAgent
from experiments.runner import run_tournament


class TestTournamentRunner:
    def test_game_count(self):
        a1 = RandomAgent(1, 'R1', seed=0)
        a2 = RandomAgent(-1, 'R2', seed=1)
        results = run_tournament(a1, a2, TicTacToe, num_games=100)
        total = results['agent1_wins'] + results['agent2_wins'] + results['draws']
        assert total == 100

    def test_results_format(self):
        a1 = RandomAgent(1, 'R1', seed=0)
        a2 = RandomAgent(-1, 'R2', seed=1)
        results = run_tournament(a1, a2, TicTacToe, num_games=10)
        assert 'agent1_wins' in results
        assert 'agent2_wins' in results
        assert 'draws' in results
        assert 'agent1_win_rate' in results
        assert 'agent2_win_rate' in results
        assert 'draw_rate' in results
        assert 'avg_game_length' in results
        assert len(results['results_per_game']) == 10

    def test_rates_sum_to_one(self):
        a1 = RandomAgent(1, 'R1', seed=0)
        a2 = RandomAgent(-1, 'R2', seed=1)
        results = run_tournament(a1, a2, TicTacToe, num_games=50)
        total_rate = results['agent1_win_rate'] + results['agent2_win_rate'] + results['draw_rate']
        assert abs(total_rate - 1.0) < 1e-9
