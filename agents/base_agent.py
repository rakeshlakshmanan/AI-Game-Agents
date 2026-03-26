from abc import ABC, abstractmethod


class BaseAgent(ABC):
    def __init__(self, player: int, name: str = "Agent"):
        self.player = player  # 1 or -1
        self.name = name

    @abstractmethod
    def get_move(self, game) -> int:
        """Given current game state, return chosen move."""
        pass

    def learn(self, *args, **kwargs):
        """Optional: update agent after a game (for RL agents)."""
        pass
