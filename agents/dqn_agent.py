import random
import os
import numpy as np
from collections import deque
from agents.base_agent import BaseAgent

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class DQNNetwork(nn.Module if TORCH_AVAILABLE else object):
    def __init__(self, input_size: int, output_size: int):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for DQNAgent")
        super().__init__()
        if input_size == 9:
            # TTT architecture
            self.net = nn.Sequential(
                nn.Linear(input_size, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, output_size),
            )
        else:
            # Connect4 architecture
            self.net = nn.Sequential(
                nn.Linear(input_size, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, output_size),
            )

    def forward(self, x):
        return self.net(x)


class DQNAgent(BaseAgent):
    def __init__(self, player: int, name: str = "DQN",
                 input_size: int = 9, output_size: int = 9,
                 alpha: float = 0.001, gamma: float = 0.95,
                 epsilon: float = 1.0, epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.9995, batch_size: int = 64,
                 replay_buffer_size: int = 10000, target_update: int = 1000,
                 seed: int = None):
        super().__init__(player, name)
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for DQNAgent")
        self.input_size = input_size
        self.output_size = output_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.steps = 0

        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

        self.device = torch.device('cpu')
        self.policy_net = DQNNetwork(input_size, output_size).to(self.device)
        self.target_net = DQNNetwork(input_size, output_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=alpha)
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        self.loss_history = []

    def _state_to_tensor(self, state):
        return torch.FloatTensor(state.flatten()).unsqueeze(0).to(self.device)

    def get_move(self, game) -> int:
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            return None
        if random.random() < self.epsilon:
            return random.choice(valid_moves)
        state = self._state_to_tensor(game.board)
        with torch.no_grad():
            q_values = self.policy_net(state).squeeze()
        # mask invalid actions
        mask = torch.full((self.output_size,), float('-inf'))
        for m in valid_moves:
            mask[m] = 0
        q_values = q_values + mask
        return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((
            state.flatten().copy(),
            action,
            reward,
            next_state.flatten().copy(),
            done
        ))

    def train_step(self) -> float:
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
        target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        loss_val = loss.item()
        self.loss_history.append(loss_val)
        return loss_val

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'epsilon': self.epsilon,
        }, filepath)

    def load(self, filepath: str):
        data = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(data['policy_net'])
        self.target_net.load_state_dict(data['target_net'])
        self.epsilon = data.get('epsilon', self.epsilon_min)
