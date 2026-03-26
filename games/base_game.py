from abc import ABC, abstractmethod
import numpy as np

class BaseGame(ABC):
    @abstractmethod
    def reset(self) -> np.ndarray:
        """Reset the game and return initial state."""
        pass

    @abstractmethod
    def get_valid_moves(self) -> list:
        """Return list of valid moves (integers)."""
        pass

    @abstractmethod
    def make_move(self, move: int, player: int) -> tuple:
        """Make a move. Return (new_state, reward, done, info)."""
        pass

    @abstractmethod
    def check_winner(self) -> int:
        """Return 1 if player 1 wins, -1 if player 2 wins, 0 if draw, None if ongoing."""
        pass

    @abstractmethod
    def clone(self):
        """Return a deep copy of the game (needed for Minimax tree search)."""
        pass

    @abstractmethod
    def render(self) -> str:
        """Return string representation of current board."""
        pass

    @abstractmethod
    def get_state_key(self) -> str:
        """Return hashable string representation of state (for Q-table)."""
        pass
