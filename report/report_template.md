# AI Game Agents: Minimax and Reinforcement Learning
## CS7IS2 Assignment Report

**Student Name:** [Your Name]
**Student ID:** [Your ID]
**Date:** [Date]

---

## 1. Introduction

This report presents the implementation and evaluation of multiple AI game-playing agents for Tic Tac Toe (TTT) and Connect 4. The agents implemented include:
- A rule-based Default Opponent (baseline)
- Minimax algorithm (with depth-limiting for Connect 4)
- Minimax with Alpha-Beta Pruning
- Tabular Q-Learning
- Deep Q-Network (DQN)

The goal is to compare algorithm performance, analyze learning curves, and evaluate trade-offs between classical tree-search and reinforcement learning approaches.

---

## 2. Game Implementations

### 2.1 Tic Tac Toe
[Describe the board representation, move encoding, win conditions, and state space size.]

### 2.2 Connect 4
[Describe the board representation, gravity mechanic, move encoding, win conditions (4-in-a-row), and state space complexity.]

---

## 3. Agent Implementations

### 3.1 Default Opponent
[Describe the rule-based priority: win > block > strategic > random.]

### 3.2 Minimax
For Tic Tac Toe, Minimax performs a full game tree search (max 9 moves, manageable). For Connect 4, full search is infeasible — we first confirmed this experimentally (see Section 4.1) and then adopted depth-limited search with a heuristic evaluation function.

**Heuristic weights (Connect 4 evaluation function):**
| Pattern | Weight |
|---------|--------|
| 4-in-a-row | 10,000 |
| 3-in-a-row (open end) | 100 |
| 2-in-a-row (open ends) | 10 |
| Centre column occupancy | 5 |
| Opponent 3-in-a-row threat | −80 |

### 3.3 Alpha-Beta Pruning
Alpha-Beta pruning extends Minimax by maintaining upper (β) and lower (α) bounds, pruning branches that cannot influence the final decision. In theory, optimal move ordering reduces the effective branching factor from b to √b. In practice (see Section 4.1), pruning reduces the number of nodes explored compared to plain Minimax, particularly in mid-game positions where more pieces constrain the search space.

### 3.4 Q-Learning
[Describe the Q-table representation, epsilon-greedy exploration, update rule, and training procedure.]

**Hyperparameters:**
| Parameter | Value |
|-----------|-------|
| Learning rate (α) | 0.1 |
| Discount factor (γ) | 0.95 |
| Initial epsilon | 1.0 |
| Epsilon min | 0.01 |
| Epsilon decay | 0.9995 |

### 3.5 Deep Q-Network
[Describe the neural network architecture, experience replay, target network, and training procedure.]

**Architecture (TTT):** Input(9) → Dense(128) → ReLU → Dense(128) → ReLU → Output(9)
**Architecture (Connect 4):** Input(42) → Dense(256) → ReLU → Dense(256) → ReLU → Dense(128) → ReLU → Output(7)

---

## 4. Experimental Setup

- **Number of tournament games:** 200 per matchup
- **Training episodes:** TTT Q-Learning: 30,000 | TTT DQN: 15,000 | C4 Q-Learning: 50,000 | C4 DQN: 25,000
- **Training opponent:** TTT — DefaultOpponent | Connect 4 — RandomAgent (see Section 4.2)
- **Evaluation frequency:** Every 1,000 episodes
- **Random seed:** 42

### 4.1 Connect 4 Scalability Test

To confirm that full-depth Minimax is infeasible for Connect 4, both agents were run on an empty board for 30 minutes with no depth limit. Results:

| Metric | Minimax (no pruning) | Alpha-Beta (with pruning) |
|--------|---------------------|--------------------------|
| Nodes explored in 30 min | 28,262,928 | 3,872,513 |
| % of game tree explored | 0.00021% | 0.00003% |
| Estimated time for full tree | ~27 years | ~243 years |

Both agents timed out having explored a negligible fraction of the ~4.5 trillion node game tree. This conclusively confirms that full-depth search is computationally infeasible.

**Chosen approach:** Depth-limited search (`max_depth=5`) with a heuristic evaluation function, as recommended in the assignment. This allows each move to be computed in under 2 seconds while still playing strategically meaningful moves.

**Note on Alpha-Beta at the start position:** Alpha-Beta explored fewer nodes (3.9M vs 28.3M), but its per-node throughput was lower due to the overhead of maintaining bounds on an empty, symmetric board where pruning opportunities are scarce. Alpha-Beta's advantage becomes more pronounced mid-game when prior pieces constrain the search.

### 4.2 RL Training Opponent for Connect 4

Connect 4's state space (~4.5 × 10¹² positions) is too large for tabular Q-Learning to converge against a smart opponent in a feasible number of episodes. Training against `DefaultOpponent` yields sparse rewards early in training (the agent loses almost every game), slowing learning severely. Following the assignment recommendation, Connect 4 RL agents are trained against `RandomAgent` instead, which provides positive reward signals early and allows the agent to learn basic strategy before being evaluated against stronger opponents.

---

## 5. Results

### 5.1 Algorithms vs Default Opponent (TTT)

![TTT vs Default](../plots/ttt_vs_default.png)

[Table of win/draw/loss rates]

### 5.2 Algorithms vs Default Opponent (Connect 4)

![Connect4 vs Default](../plots/c4_vs_default.png)

[Table of win/draw/loss rates]

### 5.3 Head-to-Head Results (TTT)

![TTT Head-to-Head](../plots/ttt_head_to_head.png)

[Analysis of head-to-head matrix]

### 5.4 Head-to-Head Results (Connect 4)

![Connect4 Head-to-Head](../plots/c4_head_to_head.png)

[Analysis of head-to-head matrix]

### 5.5 Learning Curves

![TTT Q-Learning Curve](../plots/ttt_qlearning_curve.png)

![TTT DQN Curve](../plots/ttt_dqn_curve.png)

![Connect4 Q-Learning Curve](../plots/c4_qlearning_curve.png)

![Connect4 DQN Curve](../plots/c4_dqn_curve.png)

### 5.6 Nodes Explored: Minimax vs Alpha-Beta

![Nodes Explored](../plots/nodes_explored_comparison.png)

### 5.7 Overall Comparison

![Overall Comparison](../plots/overall_comparison.png)

---

## 6. Analysis and Discussion

### 6.1 Minimax vs Alpha-Beta
At depth 5 (depth-limited Connect 4), Alpha-Beta explores significantly fewer nodes than basic Minimax (see `plots/nodes_explored_comparison.png`). For the full unlimited search (30-minute run), Minimax explored 28.3M nodes vs Alpha-Beta's 3.9M — a 7.3× reduction — yet both covered less than 0.001% of the tree, confirming that pruning alone cannot make full Connect 4 search tractable.

### 6.2 Q-Learning Convergence
[Fill in: convergence speed from learning curve, final win rate vs DefaultOpponent, and note that tabular Q-Learning is fundamentally limited for Connect 4 due to state space size — Q-table cannot enumerate all ~10¹² states.]

### 6.3 DQN Performance
[Fill in: DQN training stability (loss curve), comparison with Q-Learning win rates, and whether the neural network generalises better than the Q-table for Connect 4.]

### 6.4 Comparison Across Games
TTT is a solved game (perfect Minimax always draws or wins), making it a controlled environment to verify algorithm correctness. Connect 4 is dramatically harder: the state space is ~10¹² vs ~5,478 for TTT, making RL convergence difficult and requiring both depth-limiting for Minimax and a weaker training opponent for RL agents.

---

## 7. Conclusions

[Summarize key findings: which algorithm performed best, trade-offs, and limitations.]

---

## 8. References

1. Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed.)
2. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.)
3. Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518, 529–533.
4. [Any additional references]
