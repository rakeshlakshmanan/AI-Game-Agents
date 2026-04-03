from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self, player: int, name: str = "Agent"):
        self.player = player           
        self.name = name

    @abstractmethod
    def get_move(self, game) -> int:
        pass

    def learn(self, *args, **kwargs):
        pass
