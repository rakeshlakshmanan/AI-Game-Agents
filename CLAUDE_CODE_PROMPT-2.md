# Claude Code Project Prompt: AI Game Agents

## PROJECT OVERVIEW
Build a complete Python project called `ai-game-agents` that implements Minimax and Reinforcement Learning algorithms for playing Tic Tac Toe and Connect 4. This is a university AI assignment (CS7IS2) that requires implementing, comparing, and analyzing multiple game-playing agents.

---

## REPOSITORY STRUCTURE
Create this exact directory structure:

```
ai-game-agents/
├── README.md
├── requirements.txt
├── setup.py
├── main.py                      # Entry point: run games, experiments, demos
├── games/
│   ├── __init__.py
│   ├── base_game.py             # Abstract base class for games
│   ├── tic_tac_toe.py           # Tic Tac Toe implementation
│   └── connect4.py              # Connect 4 implementation
├── agents/
│   ├── __init__.py
│   ├── base_agent.py            # Abstract base class for agents
│   ├── default_opponent.py      # Rule-based smart opponent
│   ├── random_agent.py          # Fully random agent (for baseline)
│   ├── minimax_agent.py         # Minimax without pruning
│   ├── alphabeta_agent.py       # Minimax with alpha-beta pruning
│   ├── qlearning_agent.py       # Tabular Q-learning
│   └── dqn_agent.py             # Deep Q-Network agent
├── experiments/
│   ├── __init__.py
│   ├── runner.py                # Run tournaments and collect results
│   ├── analysis.py              # Generate graphs, tables, statistics
│   └── results/                 # Auto-generated results (CSV, JSON)
├── plots/                       # Auto-generated plots (PNG)
├── models/                      # Saved trained models (Q-tables, DQN weights)
├── report/
│   └── report_template.md       # Markdown report template with placeholders
└── tests/
    ├── __init__.py
    ├── test_games.py
    ├── test_agents.py
    └── test_experiments.py
```

---

## DETAILED SPECIFICATIONS

### 1. GAMES (`games/`)

#### `base_game.py` — Abstract Base Class
```python
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
```

#### `tic_tac_toe.py`
- 3x3 board stored as numpy array
- Players are 1 and -1, empty cells are 0
- Moves are integers 0-8 (index into flattened board)
- Win check: rows, columns, diagonals
- `clone()` must return a full deep copy for minimax search
- `get_state_key()`: convert board to tuple then string

#### `connect4.py`
- 6 rows x 7 columns board stored as numpy array
- Players are 1 and -1, empty cells are 0
- Moves are column indices 0-6 (piece drops to lowest empty row in that column)
- Win check: horizontal, vertical, diagonal (both directions) — check for 4 in a row
- `clone()` must return a full deep copy for minimax search
- `get_state_key()`: convert board to tuple then string

---

### 2. AGENTS (`agents/`)

#### `base_agent.py` — Abstract Base Class
```python
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
```

#### `random_agent.py`
- Picks a uniformly random valid move
- Used as a baseline only

#### `default_opponent.py` — Rule-Based Smart Opponent
This is critical. It must be smarter than random. Implement this priority:
1. **Win**: If there's a move that wins immediately, take it
2. **Block**: If the opponent has a move that would win on their next turn, block it
3. **Center/Strategic**: For TTT prefer center (4), then corners (0,2,6,8), then edges. For Connect4 prefer center column (3), then adjacent columns
4. **Random fallback**: Pick a random valid move if none of the above apply

#### `minimax_agent.py` — Basic Minimax
- Implement standard minimax algorithm
- For TTT: search full game tree (depth is manageable, max ~9 levels)
- For Connect4: USE DEPTH LIMITING (default max_depth=5) with a heuristic evaluation function
- Heuristic for Connect4 evaluation (when depth limit reached):
  - Score based on number of 2-in-a-row, 3-in-a-row, 4-in-a-row for each player
  - Center column preference
  - Weight: 4-in-a-row = 10000, 3-in-a-row with open space = 100, 2-in-a-row with open spaces = 10, center column bonus = 5
- Track and print number of nodes explored (for comparison with alpha-beta)
- Constructor: `MiniMaxAgent(player, max_depth=None)` — None means unlimited depth (for TTT)

#### `alphabeta_agent.py` — Minimax with Alpha-Beta Pruning
- Same as minimax but with alpha-beta pruning
- Same heuristic evaluation for Connect4
- Same depth limiting
- Track and print number of nodes explored (should be significantly fewer than basic minimax)
- Constructor: `AlphaBetaAgent(player, max_depth=None)`

#### `qlearning_agent.py` — Tabular Q-Learning
- Q-table stored as a dictionary: `{(state_key, action): q_value}`
- Hyperparameters (with these defaults, all configurable):
  - `alpha` (learning rate): 0.1
  - `gamma` (discount factor): 0.95
  - `epsilon` (exploration rate): 1.0
  - `epsilon_min`: 0.01
  - `epsilon_decay`: 0.9995
- Epsilon-greedy action selection
- Update rule: `Q(s,a) = Q(s,a) + alpha * (reward + gamma * max(Q(s',a')) - Q(s,a))`
- Methods:
  - `get_move(game)`: epsilon-greedy selection from valid moves
  - `learn(state, action, reward, next_state, done, valid_moves)`: Q-update
  - `save(filepath)`: save Q-table to pickle file
  - `load(filepath)`: load Q-table from pickle file
  - `decay_epsilon()`: multiply epsilon by epsilon_decay, floor at epsilon_min
- Reward scheme:
  - Win: +1.0
  - Loss: -1.0
  - Draw: 0.3 (slightly positive to encourage not losing)
  - Each non-terminal move: 0.0 (or a tiny negative like -0.01 to encourage faster wins)
- Training function (standalone or method):
  - Train by self-play OR against default opponent for N episodes
  - Track win rates over training for plotting learning curves
  - Decay epsilon after each episode

#### `dqn_agent.py` — Deep Q-Network
- Use PyTorch
- Neural network architecture:
  - For TTT: Input = 9 (flattened board) → Dense(128) → ReLU → Dense(128) → ReLU → Dense(9) (one Q-value per action)
  - For Connect4: Input = 42 (flattened 6x7 board) → Dense(256) → ReLU → Dense(256) → ReLU → Dense(128) → ReLU → Dense(7) (one Q-value per column)
- Experience replay buffer (capacity 10000)
- Target network (updated every `target_update` steps, default 1000)
- Hyperparameters (all configurable):
  - `alpha` (learning rate): 0.001
  - `gamma` (discount factor): 0.95
  - `epsilon`: 1.0
  - `epsilon_min`: 0.01
  - `epsilon_decay`: 0.9995
  - `batch_size`: 64
  - `replay_buffer_size`: 10000
  - `target_update`: 1000
- Methods:
  - `get_move(game)`: epsilon-greedy, mask invalid actions (set their Q-values to -inf before argmax)
  - `store_transition(state, action, reward, next_state, done)`
  - `train_step()`: sample batch from replay buffer, compute loss, backprop
  - `save(filepath)`: save model weights
  - `load(filepath)`: load model weights
- Same reward scheme as Q-learning
- Training function:
  - Train by self-play OR against default opponent for N episodes
  - Track win rates over training
  - Log loss values for plotting

---

### 3. EXPERIMENTS (`experiments/`)

#### `runner.py` — Tournament Runner
```python
def run_tournament(agent1, agent2, game_class, num_games=1000, verbose=False):
    """
    Play num_games between agent1 and agent2.
    Alternate who goes first each game.
    Return dict: {
        'agent1_wins': int,
        'agent2_wins': int,
        'draws': int,
        'agent1_win_rate': float,
        'agent2_win_rate': float,
        'draw_rate': float,
        'avg_game_length': float,
        'results_per_game': list  # for plotting over time
    }
    """

def train_rl_agent(agent, opponent, game_class, num_episodes=50000, eval_every=1000, eval_games=100):
    """
    Train an RL agent (Q-learning or DQN) against an opponent.
    Every eval_every episodes, run eval_games to measure current win rate.
    Return training_history: list of {
        'episode': int,
        'win_rate': float,
        'draw_rate': float,
        'loss_rate': float,
        'epsilon': float
    }
    """
```

#### `analysis.py` — Visualization and Analysis
Generate these specific plots (save to `plots/` directory):

1. **`ttt_vs_default.png`**: Bar chart — Win/Draw/Loss rates of each algorithm vs default opponent in TTT
2. **`c4_vs_default.png`**: Bar chart — Win/Draw/Loss rates of each algorithm vs default opponent in Connect4
3. **`ttt_head_to_head.png`**: Heatmap — Head-to-head results matrix of all algorithms in TTT
4. **`c4_head_to_head.png`**: Heatmap — Head-to-head results matrix of all algorithms in Connect4
5. **`ttt_qlearning_curve.png`**: Line plot — Q-learning win rate over training episodes in TTT
6. **`ttt_dqn_curve.png`**: Line plot — DQN win rate over training episodes in TTT
7. **`c4_qlearning_curve.png`**: Line plot — Q-learning win rate over training episodes in Connect4
8. **`c4_dqn_curve.png`**: Line plot — DQN win rate over training episodes in Connect4
9. **`nodes_explored_comparison.png`**: Bar chart — Minimax vs Alpha-Beta nodes explored (TTT and Connect4)
10. **`overall_comparison.png`**: Summary bar chart — Overall win rates of all algorithms across both games

Use matplotlib and seaborn. Make plots publication-quality:
- Clear labels, titles, legends
- Consistent color scheme across plots
- Error bars where applicable (run experiments multiple times)
- Grid lines for readability
- Save at 300 DPI

---

### 4. MAIN ENTRY POINT (`main.py`)

Create a comprehensive CLI using argparse:

```
python main.py --mode play --game ttt --agent1 minimax --agent2 default
python main.py --mode play --game c4 --agent1 alphabeta --agent2 default --depth 5
python main.py --mode train --game ttt --agent qlearning --episodes 50000
python main.py --mode train --game c4 --agent dqn --episodes 20000
python main.py --mode tournament --game ttt --num-games 1000
python main.py --mode tournament --game c4 --num-games 500
python main.py --mode full-experiment  # Run ALL experiments and generate ALL plots
python main.py --mode interactive --game ttt --opponent minimax  # Human plays against AI
```

Modes:
- `play`: Watch two agents play a single game with board visualization
- `train`: Train an RL agent and save the model
- `tournament`: Run a tournament between two specific agents
- `full-experiment`: Run the complete experiment suite (all matchups, training, plots)
- `interactive`: Human player vs AI agent (for the video demo)

---

### 5. REQUIREMENTS (`requirements.txt`)

```
numpy>=1.21.0
torch>=1.9.0
matplotlib>=3.4.0
seaborn>=0.11.0
pandas>=1.3.0
tqdm>=4.62.0
```

---

### 6. REPORT TEMPLATE (`report/report_template.md`)

Generate a markdown template with these sections:
1. Introduction
2. Game Implementations
3. Agent Implementations
   - 3.1 Default Opponent
   - 3.2 Minimax
   - 3.3 Alpha-Beta Pruning
   - 3.4 Q-Learning
   - 3.5 Deep Q-Network
4. Experimental Setup
5. Results
   - 5.1 Algorithms vs Default Opponent (TTT)
   - 5.2 Algorithms vs Default Opponent (Connect4)
   - 5.3 Head-to-Head (TTT)
   - 5.4 Head-to-Head (Connect4)
   - 5.5 Overall Comparison
6. Analysis and Discussion
7. Conclusions
8. References

Include placeholders like `![TTT vs Default](../plots/ttt_vs_default.png)` for each plot.

---

### 7. TESTS (`tests/`)

Write pytest-compatible tests:

#### `test_games.py`
- Test TTT: valid moves, win detection (row/col/diagonal), draw detection, invalid move handling
- Test Connect4: gravity (pieces fall), valid moves (full column), win detection (horizontal/vertical/diagonal), draw detection
- Test clone() produces independent copies

#### `test_agents.py`
- Test default opponent: always wins if winning move exists, always blocks if opponent about to win
- Test minimax on TTT: should never lose (TTT is a solved game — minimax should always draw or win)
- Test alpha-beta produces same moves as minimax (just faster)
- Test Q-learning agent: Q-table updates correctly after learn()
- Test DQN agent: can forward pass through network, invalid moves are masked

#### `test_experiments.py`
- Test tournament runner: correct game counts, alternating first player
- Test results format

---

## IMPLEMENTATION PRIORITIES AND NOTES

### Critical Implementation Details:
1. **State representation consistency**: Both games must use the same convention: player 1 = 1, player 2 = -1, empty = 0
2. **Move representation**: Integers only. TTT: 0-8. Connect4: 0-6
3. **Minimax for Connect4 MUST be depth-limited**: Full search tree is impossibly large (~4.5 trillion nodes). Use max_depth=5 as default
4. **Q-learning for Connect4**: State space is huge. Consider using board symmetry to reduce states, or accept that it won't converge perfectly and note this in analysis
5. **DQN must mask invalid actions**: Set Q-values of invalid columns/positions to -infinity before taking argmax
6. **Always alternate who goes first**: In tournaments, swap first player each game to ensure fairness
7. **Reproducibility**: Set random seeds (numpy, torch, random) with a configurable seed parameter
8. **Progress bars**: Use tqdm for all training and tournament loops
9. **Console output**: Print clear, formatted results after each experiment

### Training Recommendations:
- TTT Q-learning: 50,000 episodes should be sufficient
- TTT DQN: 20,000-30,000 episodes
- Connect4 Q-learning: 100,000+ episodes (larger state space)
- Connect4 DQN: 50,000+ episodes
- Evaluate every 1000 episodes during training

### Performance Notes:
- Minimax on Connect4 at depth 5 should complete a move in <2 seconds
- Alpha-beta on Connect4 at depth 5 should be noticeably faster (track node counts)
- Q-learning training should complete in minutes
- DQN training may take 10-30 minutes for Connect4

---

## WHAT SUCCESS LOOKS LIKE

After running `python main.py --mode full-experiment`, the project should:
1. Train all RL agents (Q-learning and DQN for both games)
2. Run all tournament matchups (each algorithm vs default, and all head-to-head pairs)
3. Generate all 10 plots in the `plots/` directory
4. Print a summary table of all results to console
5. Save detailed results as CSV files in `experiments/results/`

Expected behavior:
- Minimax and Alpha-Beta should NEVER lose at TTT (it's a solved game — they should always draw or win)
- Alpha-Beta should explore significantly fewer nodes than basic Minimax
- Q-learning should show improving win rates over training episodes
- DQN should show improving win rates but may be more unstable than tabular Q-learning
- Against the default opponent, Minimax/Alpha-Beta should perform best, followed by trained RL agents
- Connect4 is harder — RL agents may not reach perfect play

---

## BUILD ORDER

Please implement in this exact order:
1. Project structure and `requirements.txt`
2. `base_game.py` and `base_agent.py`
3. `tic_tac_toe.py` with full tests
4. `connect4.py` with full tests
5. `random_agent.py`
6. `default_opponent.py` with tests
7. `minimax_agent.py` with tests
8. `alphabeta_agent.py` with tests
9. `qlearning_agent.py` with tests
10. `dqn_agent.py` with tests
11. `runner.py` (tournament infrastructure)
12. `analysis.py` (plotting)
13. `main.py` (CLI)
14. `report_template.md`
15. `README.md`
16. Run full experiment suite and verify everything works end-to-end

After each file, run the relevant tests to make sure everything works before moving on.

---

## FINAL CHECKLIST
- [ ] Both games work correctly with all win/draw/invalid-move edge cases
- [ ] Default opponent wins when it can and blocks when it must
- [ ] Minimax never loses at TTT
- [ ] Alpha-Beta produces same results as Minimax with fewer nodes
- [ ] Q-learning trains and improves over time
- [ ] DQN trains and improves over time
- [ ] All 10 plots generate correctly
- [ ] Interactive mode works (human can play)
- [ ] Full experiment completes without errors
- [ ] All tests pass
- [ ] Results are reproducible with seed setting
