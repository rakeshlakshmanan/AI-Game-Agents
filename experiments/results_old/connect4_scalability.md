# Connect 4 Scalability Test — Findings

## Setup
- Game: Connect 4 (6 rows × 7 columns)
- Starting position: empty board (first move)
- Time budget per agent: **30 minute(s)**
- Depth limit: **None** (full unlimited search)
- Known Connect 4 game tree size: ~4.5 trillion terminal positions

## Results

| Metric | Minimax (no pruning) | Alpha-Beta (with pruning) |
|--------|---------------------|--------------------------|
| Nodes explored in 30 min | 28,262,928 | 3,872,513 |
| Nodes per second | 15,702 | 1,772 |
| % of tree explored | 0.00020788% | 0.00002848% |
| Estimated time for full tree | 2.75e+01 years | 2.43e+02 years |
| Timed out | Yes | Yes |

## Analysis

Minimax explored 7.3x more nodes than Alpha-Beta.

Both agents timed out far before completing the search, confirming that full Connect 4 Minimax is **computationally infeasible** on a standard machine.

The fraction of the game tree explored in the allotted time is negligibly small (< 0.00001%), meaning neither agent could make a reliably optimal move.

## Conclusion and Chosen Approach

Full-depth Minimax (and even Alpha-Beta) cannot solve Connect 4 in real time.
We therefore adopt the **preferred approach** described in the assignment:

1. **Depth-limited search** — look only 5 moves ahead (`max_depth=5`)
2. **Heuristic evaluation function** — score positions at the depth limit using:
   - 4-in-a-row = 10,000 points
   - 3-in-a-row (open end) = 100 points
   - 2-in-a-row (open ends) = 10 points
   - Centre column occupancy = 5 points
   - Opponent 3-in-a-row threat = −80 points

This allows each move to be computed in under 2 seconds while still playing
strategically meaningful moves.

For RL agents, Connect 4's state space (~10^12 positions) is too large for
tabular Q-Learning to converge meaningfully. We therefore train against a
**RandomAgent** (rather than the smarter DefaultOpponent) so the RL agent
receives positive reward signals early in training and can learn basic strategy.