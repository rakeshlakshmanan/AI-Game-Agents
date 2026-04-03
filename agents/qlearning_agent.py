import random
import pickle
import os
from agents.base_agent import BaseAgent


class QLearningAgent(BaseAgent):
    def __init__(self, player: int, name: str = "QLearning",
                 alpha: float = 0.1, gamma: float = 0.95,
                 epsilon: float = 1.0, epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.9995, seed: int = None):
        super().__init__(player, name)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = {}
        if seed is not None:
            random.seed(seed)

    def _get_q(self, state_key, action):
        return self.q_table.get((state_key, action), 0.0)

    def state_key(self, game) -> str:
        """Normalize board to agent's perspective so pieces are always +1 for self."""
        if self.player == 1:
            return game.get_state_key()
        # flip signs so agent's pieces appear as +1 regardless of assigned player
        import numpy as np
        return str(tuple((-game.board).flatten()))

    def get_move(self, game) -> int:
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            return None
        if random.random() < self.epsilon:
            return random.choice(valid_moves)
        sk = self.state_key(game)
        q_values = {a: self._get_q(sk, a) for a in valid_moves}
        return max(q_values, key=q_values.get)

    def learn(self, state, action, reward, next_state, done, valid_next_moves):
        current_q = self._get_q(state, action)
        if done or not valid_next_moves:
            target = reward
        else:
            max_next_q = max(self._get_q(next_state, a) for a in valid_next_moves)
            target = reward + self.gamma * max_next_q
        self.q_table[(state, action)] = current_q + self.alpha * (target - current_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({'q_table': self.q_table, 'epsilon': self.epsilon}, f)

    def load(self, filepath: str):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.q_table = data['q_table']
        self.epsilon = data.get('epsilon', self.epsilon_min)
