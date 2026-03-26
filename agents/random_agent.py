import random
from agents.base_agent import BaseAgent


class RandomAgent(BaseAgent):
    def __init__(self, player: int, name: str = "RandomAgent", seed: int = None):
        super().__init__(player, name)
        if seed is not None:
            random.seed(seed)

    def get_move(self, game) -> int:
        valid_moves = game.get_valid_moves()
        return random.choice(valid_moves)
