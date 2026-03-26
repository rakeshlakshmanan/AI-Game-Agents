import random
from agents.base_agent import BaseAgent


class DefaultOpponent(BaseAgent):
    """Rule-based smart opponent: win > block > strategic > random."""

    def __init__(self, player: int, name: str = "DefaultOpponent", seed: int = None):
        super().__init__(player, name)
        if seed is not None:
            random.seed(seed)

    def get_move(self, game) -> int:
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            return None

        # 1. Win if possible
        for move in valid_moves:
            g = game.clone()
            g.make_move(move, self.player)
            if g.check_winner() == self.player:
                return move

        # 2. Block opponent win
        opponent = -self.player
        for move in valid_moves:
            g = game.clone()
            g.make_move(move, opponent)
            if g.check_winner() == opponent:
                return move

        # 3. Strategic placement
        from games.tic_tac_toe import TicTacToe
        from games.connect4 import Connect4
        if isinstance(game, TicTacToe):
            preferred = [4, 0, 2, 6, 8, 1, 3, 5, 7]
            for move in preferred:
                if move in valid_moves:
                    return move
        elif isinstance(game, Connect4):
            preferred = [3, 2, 4, 1, 5, 0, 6]
            for move in preferred:
                if move in valid_moves:
                    return move

        # 4. Random fallback
        return random.choice(valid_moves)
