#!/usr/bin/env python3
"""
AI Game Agents - Main Entry Point
CS7IS2 Assignment
"""

import argparse
import os
import sys
import random
import numpy as np

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def get_game(game_name: str):
    from games.tic_tac_toe import TicTacToe
    from games.connect4 import Connect4
    if game_name == 'ttt':
        return TicTacToe
    elif game_name == 'c4':
        return Connect4
    else:
        raise ValueError(f"Unknown game: {game_name}")


def get_agent(agent_name: str, player: int, depth: int = None, game_name: str = 'ttt'):
    from agents.random_agent import RandomAgent
    from agents.default_opponent import DefaultOpponent
    from agents.minimax_agent import MiniMaxAgent
    from agents.alphabeta_agent import AlphaBetaAgent
    from agents.qlearning_agent import QLearningAgent
    from agents.dqn_agent import DQNAgent

    input_size = 9 if game_name == 'ttt' else 42
    output_size = 9 if game_name == 'ttt' else 7

    agents = {
        'random': lambda: RandomAgent(player, 'Random'),
        'default': lambda: DefaultOpponent(player, 'Default'),
        'minimax': lambda: MiniMaxAgent(player, 'MiniMax', max_depth=depth),
        'alphabeta': lambda: AlphaBetaAgent(player, 'AlphaBeta', max_depth=depth),
        'qlearning': lambda: QLearningAgent(player, 'QLearning'),
        'dqn': lambda: DQNAgent(player, 'DQN', input_size=input_size, output_size=output_size),
    }
    if agent_name not in agents:
        raise ValueError(f"Unknown agent: {agent_name}")
    return agents[agent_name]()


def mode_play(args):
    """Watch two agents play a single game."""
    game_class = get_game(args.game)
    game = game_class()
    game.reset()

    depth = args.depth if hasattr(args, 'depth') else None
    agent1 = get_agent(args.agent1, 1, depth=depth, game_name=args.game)
    agent2 = get_agent(args.agent2, -1, depth=depth, game_name=args.game)

    print(f"\n=== {agent1.name} (X) vs {agent2.name} (O) ===")
    print(f"Game: {args.game.upper()}\n")
    print(game.render())
    print()

    current = agent1
    move_num = 0
    while True:
        move = current.get_move(game)
        _, _, done, info = game.make_move(move, current.player)
        move_num += 1
        print(f"Move {move_num}: {current.name} plays {move}")
        print(game.render())
        print()
        if done:
            winner = info.get('winner')
            if winner == 1:
                print(f"Winner: {agent1.name} (X)")
            elif winner == -1:
                print(f"Winner: {agent2.name} (O)")
            else:
                print("Result: Draw")
            break
        current = agent2 if current is agent1 else agent1


def mode_train(args):
    """Train an RL agent."""
    from experiments.runner import train_rl_agent
    from agents.default_opponent import DefaultOpponent

    game_class = get_game(args.game)
    episodes = args.episodes

    input_size = 9 if args.game == 'ttt' else 42
    output_size = 9 if args.game == 'ttt' else 7

    if args.agent == 'qlearning':
        from agents.qlearning_agent import QLearningAgent
        agent = QLearningAgent(1, 'QLearning')
    elif args.agent == 'dqn':
        from agents.dqn_agent import DQNAgent
        agent = DQNAgent(1, 'DQN', input_size=input_size, output_size=output_size)
    else:
        print(f"Agent '{args.agent}' does not require training.")
        return

    opponent = DefaultOpponent(-1, 'Default')
    print(f"\nTraining {agent.name} on {args.game.upper()} for {episodes} episodes...")
    history = train_rl_agent(agent, opponent, game_class, num_episodes=episodes)

    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', f'{args.agent}_{args.game}.pkl' if args.agent == 'qlearning' else f'{args.agent}_{args.game}.pt')
    agent.save(model_path)
    print(f"Model saved to {model_path}")

    if history:
        final = history[-1]
        print(f"\nFinal evaluation (episode {final['episode']}):")
        print(f"  Win rate:  {final['win_rate']:.2%}")
        print(f"  Draw rate: {final['draw_rate']:.2%}")
        print(f"  Loss rate: {final['loss_rate']:.2%}")


def mode_tournament(args):
    """Run a tournament between two agents."""
    from experiments.runner import run_tournament

    game_class = get_game(args.game)
    depth = args.depth if hasattr(args, 'depth') else None
    agent1 = get_agent(args.agent1 if hasattr(args, 'agent1') else 'alphabeta', 1, depth=depth, game_name=args.game)
    agent2 = get_agent(args.agent2 if hasattr(args, 'agent2') else 'default', -1, depth=depth, game_name=args.game)
    num_games = args.num_games

    print(f"\n=== Tournament: {agent1.name} vs {agent2.name} ({args.game.upper()}) ===")
    results = run_tournament(agent1, agent2, game_class, num_games=num_games)

    print(f"\nResults over {num_games} games:")
    print(f"  {agent1.name} wins: {results['agent1_wins']} ({results['agent1_win_rate']:.2%})")
    print(f"  {agent2.name} wins: {results['agent2_wins']} ({results['agent2_win_rate']:.2%})")
    print(f"  Draws:           {results['draws']} ({results['draw_rate']:.2%})")
    print(f"  Avg game length: {results['avg_game_length']:.1f} moves")


def mode_interactive(args):
    """Human plays against an AI agent."""
    game_class = get_game(args.game)
    game = game_class()
    game.reset()

    depth = args.depth if hasattr(args, 'depth') else 5
    ai = get_agent(args.opponent if hasattr(args, 'opponent') else 'minimax', -1, depth=depth, game_name=args.game)

    print(f"\n=== Interactive Mode: You (X) vs {ai.name} (O) ===")
    print(f"Game: {args.game.upper()}")
    if args.game == 'ttt':
        print("Moves: 0-8 (0=top-left, 8=bottom-right)")
    else:
        print("Moves: 0-6 (column index)")
    print()
    print(game.render())
    print()

    current_player = 1
    while True:
        if current_player == 1:
            # Human
            valid = game.get_valid_moves()
            while True:
                try:
                    move = int(input(f"Your move {valid}: "))
                    if move in valid:
                        break
                    print("Invalid move, try again.")
                except (ValueError, KeyboardInterrupt):
                    print("\nGame aborted.")
                    return
        else:
            move = ai.get_move(game)
            print(f"{ai.name} plays: {move}")

        _, _, done, info = game.make_move(move, current_player)
        print(game.render())
        print()

        if done:
            winner = info.get('winner')
            if winner == 1:
                print("You win!")
            elif winner == -1:
                print(f"{ai.name} wins!")
            else:
                print("Draw!")
            break
        current_player = -current_player


def mode_full_experiment(args):
    """Run the complete experiment suite."""
    import json
    import pandas as pd
    from experiments.runner import run_tournament, train_rl_agent
    from experiments.analysis import (plot_vs_default, plot_head_to_head,
                                       plot_learning_curve, plot_nodes_explored,
                                       plot_overall_comparison, save_results_csv,
                                       RESULTS_DIR)
    from games.tic_tac_toe import TicTacToe
    from games.connect4 import Connect4
    from agents.random_agent import RandomAgent
    from agents.default_opponent import DefaultOpponent
    from agents.minimax_agent import MiniMaxAgent
    from agents.alphabeta_agent import AlphaBetaAgent
    from agents.qlearning_agent import QLearningAgent
    from agents.dqn_agent import DQNAgent

    os.makedirs('models', exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    N_GAMES = 200  # quick run; increase for final paper

    print("\n" + "="*60)
    print("FULL EXPERIMENT SUITE")
    print("="*60)

    overall_data = {}

    for game_name, game_class, mm_depth in [('TTT', TicTacToe, None), ('Connect4', Connect4, 5)]:
        print(f"\n{'='*40}\n{game_name}\n{'='*40}")
        input_size = 9 if game_name == 'TTT' else 42
        output_size = 9 if game_name == 'TTT' else 7
        ql_episodes = 30000 if game_name == 'TTT' else 50000
        dqn_episodes = 15000 if game_name == 'TTT' else 25000

        # Agents
        default = DefaultOpponent(-1, 'Default')

        # --- Train RL Agents ---
        print(f"\n[1/4] Training Q-Learning on {game_name}...")
        ql_agent = QLearningAgent(1, 'QLearning')
        ql_history = train_rl_agent(ql_agent, DefaultOpponent(-1, 'Default'), game_class, num_episodes=ql_episodes)
        ql_model_path = f'models/qlearning_{game_name.lower()}.pkl'
        ql_agent.save(ql_model_path)

        print(f"\n[2/4] Training DQN on {game_name}...")
        dqn_agent = DQNAgent(1, 'DQN', input_size=input_size, output_size=output_size)
        dqn_history = train_rl_agent(dqn_agent, DefaultOpponent(-1, 'Default'), game_class, num_episodes=dqn_episodes)
        dqn_model_path = f'models/dqn_{game_name.lower()}.pt'
        dqn_agent.save(dqn_model_path)

        # Save learning curves
        short = game_name.lower().replace('connect4', 'c4')
        plot_learning_curve(ql_history, 'Q-Learning', game_name, f'{short}_qlearning_curve.png')
        plot_learning_curve(dqn_history, 'DQN', game_name, f'{short}_dqn_curve.png')

        # --- vs Default ---
        print(f"\n[3/4] Running vs-Default tournaments on {game_name}...")
        ql_agent.epsilon = 0.0
        dqn_agent.epsilon = 0.0

        algo_agents = [
            MiniMaxAgent(1, 'MiniMax', max_depth=mm_depth),
            AlphaBetaAgent(1, 'AlphaBeta', max_depth=mm_depth),
            ql_agent,
            dqn_agent,
        ]
        vs_default_results = {}
        for agent in algo_agents:
            res = run_tournament(agent, DefaultOpponent(-1, 'Default'), game_class, num_games=N_GAMES)
            vs_default_results[agent.name] = res

        plot_vs_default(vs_default_results, game_name, f'{short}_vs_default.png')

        # Save CSV
        rows = []
        for name, r in vs_default_results.items():
            rows.append({'agent': name, 'game': game_name, **{k: v for k, v in r.items() if k != 'results_per_game'}})
        pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, f'{short}_vs_default.csv'), index=False)

        # overall data
        for agent in algo_agents:
            if agent.name not in overall_data:
                overall_data[agent.name] = {}
            overall_data[agent.name][game_name] = {
                'win_rate': vs_default_results[agent.name]['agent1_win_rate']
            }

        # --- Head-to-head ---
        print(f"\n[4/4] Head-to-head tournaments on {game_name}...")
        all_agents = [
            MiniMaxAgent(1, 'MiniMax', max_depth=mm_depth),
            AlphaBetaAgent(1, 'AlphaBeta', max_depth=mm_depth),
            QLearningAgent(1, 'QLearning'),
            DQNAgent(1, 'DQN', input_size=input_size, output_size=output_size),
        ]
        # load trained RL agents
        all_agents[2].load(ql_model_path)
        all_agents[2].epsilon = 0.0
        all_agents[3].load(dqn_model_path)
        all_agents[3].epsilon = 0.0

        n = len(all_agents)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i, j] = 0.5
                    continue
                res = run_tournament(all_agents[i], all_agents[j], game_class, num_games=N_GAMES // 2)
                matrix[i, j] = res['agent1_win_rate']

        labels = [a.name for a in all_agents]
        plot_head_to_head(matrix, labels, game_name, f'{short}_head_to_head.png')

        # --- Node comparison ---
        if game_name == 'TTT':
            mm = MiniMaxAgent(1, 'MiniMax', max_depth=None)
            ab = AlphaBetaAgent(1, 'AlphaBeta', max_depth=None)
            test_game = TicTacToe()
            test_game.reset()
            mm.get_move(test_game)
            ab.get_move(test_game)
            nodes_data = {'MiniMax (TTT)': mm.nodes_explored, 'AlphaBeta (TTT)': ab.nodes_explored}
        else:
            mm = MiniMaxAgent(1, 'MiniMax', max_depth=5)
            ab = AlphaBetaAgent(1, 'AlphaBeta', max_depth=5)
            test_game = Connect4()
            test_game.reset()
            mm.get_move(test_game)
            ab.get_move(test_game)
            nodes_data = {'MiniMax (C4)': mm.nodes_explored, 'AlphaBeta (C4)': ab.nodes_explored}

        if game_name == 'Connect4':
            plot_nodes_explored(nodes_data, 'nodes_explored_comparison.png')

    # Overall comparison
    plot_overall_comparison(overall_data, 'overall_comparison.png')

    # Print summary table
    print("\n" + "="*60)
    print("SUMMARY: Win rates vs Default Opponent")
    print("="*60)
    print(f"{'Algorithm':<15} {'TTT':>10} {'Connect4':>10}")
    print("-"*40)
    for algo, games in overall_data.items():
        ttt_wr = games.get('TTT', {}).get('win_rate', float('nan'))
        c4_wr = games.get('Connect4', {}).get('win_rate', float('nan'))
        print(f"{algo:<15} {ttt_wr:>9.2%} {c4_wr:>9.2%}")

    print("\nAll plots saved to plots/")
    print("All results saved to experiments/results/")


def main():
    parser = argparse.ArgumentParser(description='AI Game Agents')
    parser.add_argument('--mode', choices=['play', 'train', 'tournament', 'full-experiment', 'interactive'],
                        default='play', help='Mode to run')
    parser.add_argument('--game', choices=['ttt', 'c4'], default='ttt', help='Game to play')
    parser.add_argument('--agent1', default='minimax', help='First agent')
    parser.add_argument('--agent2', default='default', help='Second agent')
    parser.add_argument('--agent', default='qlearning', help='Agent to train (for train mode)')
    parser.add_argument('--opponent', default='minimax', help='Opponent for interactive mode')
    parser.add_argument('--depth', type=int, default=None, help='Search depth for minimax agents')
    parser.add_argument('--episodes', type=int, default=50000, help='Training episodes')
    parser.add_argument('--num-games', type=int, default=1000, dest='num_games', help='Number of tournament games')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()
    set_seed(args.seed)

    if args.mode == 'play':
        mode_play(args)
    elif args.mode == 'train':
        mode_train(args)
    elif args.mode == 'tournament':
        mode_tournament(args)
    elif args.mode == 'interactive':
        mode_interactive(args)
    elif args.mode == 'full-experiment':
        mode_full_experiment(args)


if __name__ == '__main__':
    main()
