from abc import ABC, abstractmethod
import numpy as np

class BaseGame(ABC):
    @abstractmethod
    def reset(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_valid_moves(self) -> list:
        pass

    @abstractmethod
    def make_move(self, move: int, player: int) -> tuple:
        pass

    @abstractmethod
    def check_winner(self) -> int:
        pass

    @abstractmethod
    def clone(self):
        pass

    @abstractmethod
    def render(self) -> str:
        pass

    @abstractmethod
    def get_state_key(self) -> str:
        pass
