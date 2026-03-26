# AI Game Agents

A Python implementation of Minimax and Reinforcement Learning algorithms for Tic Tac Toe and Connect 4.

## Algorithms
- **Default Opponent** — Rule-based (win > block > strategic > random)
- **Minimax** — Full tree search (TTT) / depth-limited with heuristic (Connect 4)
- **Alpha-Beta** — Minimax with alpha-beta pruning
- **Q-Learning** — Tabular reinforcement learning
- **DQN** — Deep Q-Network with experience replay

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Watch two agents play
```bash
python main.py --mode play --game ttt --agent1 minimax --agent2 default
python main.py --mode play --game c4 --agent1 alphabeta --agent2 default --depth 5
```

### Train an RL agent
```bash
python main.py --mode train --game ttt --agent qlearning --episodes 50000
python main.py --mode train --game c4 --agent dqn --episodes 25000
```

### Run a tournament
```bash
python main.py --mode tournament --game ttt --agent1 minimax --agent2 default --num-games 1000
python main.py --mode tournament --game c4 --agent1 alphabeta --agent2 default --num-games 500
```

### Play interactively (Human vs AI)
```bash
python main.py --mode interactive --game ttt --opponent minimax
python main.py --mode interactive --game c4 --opponent alphabeta --depth 5
```

### Run full experiment suite
```bash
python main.py --mode full-experiment
```

## Run Tests
```bash
pytest tests/ -v
```

## Project Structure
```
ai-game-agents/
├── main.py                  # CLI entry point
├── games/                   # Game implementations
│   ├── tic_tac_toe.py
│   └── connect4.py
├── agents/                  # Agent implementations
│   ├── default_opponent.py
│   ├── minimax_agent.py
│   ├── alphabeta_agent.py
│   ├── qlearning_agent.py
│   └── dqn_agent.py
├── experiments/             # Tournaments and analysis
│   ├── runner.py
│   └── analysis.py
├── plots/                   # Generated plots (PNG)
├── models/                  # Saved trained models
├── report/                  # Report template
└── tests/                   # Pytest test suite
```

## Agent Names for CLI
| `--agent` / `--agent1` / `--agent2` | Description |
|---|---|
| `random` | Random moves |
| `default` | Rule-based opponent |
| `minimax` | Minimax (use `--depth 5` for Connect 4) |
| `alphabeta` | Alpha-Beta pruning (use `--depth 5` for Connect 4) |
| `qlearning` | Tabular Q-Learning |
| `dqn` | Deep Q-Network |
