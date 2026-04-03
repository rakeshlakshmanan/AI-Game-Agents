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

def _launch_gui(game_name: str, agent1, agent2, move_delay: int = 700):
    from gui.game_window import TicTacToeWindow, Connect4Window
    if game_name == 'ttt':
        TicTacToeWindow(agent1, agent2, move_delay=move_delay).run()
    else:
        Connect4Window(agent1, agent2, move_delay=move_delay).run()

def mode_play(args):
    depth = args.depth if hasattr(args, 'depth') else None
    agent1 = get_agent(args.agent1, 1, depth=depth, game_name=args.game)
    agent2 = get_agent(args.agent2, -1, depth=depth, game_name=args.game)

    if getattr(args, 'ui', False):
        _launch_gui(args.game, agent1, agent2)
        return

    game_class = get_game(args.game)
    game = game_class()
    game.reset()

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
    from experiments.runner import train_rl_agent
    from agents.default_opponent import DefaultOpponent
    from agents.random_agent import RandomAgent

    game_class = get_game(args.game)
    episodes = args.episodes
    game_label = args.game.upper()

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

                                                                 
    if args.game == 'c4':
        opponent = RandomAgent(-1, 'Random')
    else:
        opponent = DefaultOpponent(-1, 'Default')

    print(f"\nTraining {agent.name} on {game_label} for {episodes} episodes vs {opponent.name}...")
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

                                                              
        plot_dir = getattr(args, 'plot_dir', None)
        if plot_dir:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            os.makedirs(plot_dir, exist_ok=True)

            episodes_x = [h['episode'] for h in history]
            win_rates  = [h['win_rate']  for h in history]
            draw_rates = [h['draw_rate'] for h in history]
            loss_rates = [h['loss_rate'] for h in history]

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(episodes_x, win_rates,  label='Win Rate',  color='#2ecc71', linewidth=2)
            ax.plot(episodes_x, draw_rates, label='Draw Rate', color='#f39c12', linewidth=2, linestyle='--')
            ax.plot(episodes_x, loss_rates, label='Loss Rate', color='#e74c3c', linewidth=2, linestyle=':')
            ax.set_xlabel('Training Episodes', fontsize=12)
            ax.set_ylabel('Rate', fontsize=12)
            ax.set_title(f'{game_label}: {agent.name} Learning Curve (vs {opponent.name})', fontsize=14)
            ax.legend(fontsize=11)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_ylim(0, 1.05)

            plot_filename = f'{args.agent}_{args.game}_learning_curve.png'
            plot_path = os.path.join(plot_dir, plot_filename)
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300)
            plt.close()
            print(f"Learning curve saved to {plot_path}")

def mode_tournament(args):
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
    depth = args.depth if hasattr(args, 'depth') else 5
    ai = get_agent(args.opponent if hasattr(args, 'opponent') else 'minimax', -1, depth=depth, game_name=args.game)

    if getattr(args, 'ui', False):
        from gui.game_window import HumanGUIAgent
        human = HumanGUIAgent(1, "You")
        _launch_gui(args.game, human, ai)
        return

    game_class = get_game(args.game)
    game = game_class()
    game.reset()

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

def mode_vs_default(args):
    import pandas as pd
    from experiments.runner import run_tournament
    from agents.default_opponent import DefaultOpponent
    from agents.minimax_agent import MiniMaxAgent
    from agents.alphabeta_agent import AlphaBetaAgent
    from agents.qlearning_agent import QLearningAgent
    from agents.dqn_agent import DQNAgent

    game_class = get_game(args.game)
    num_games = args.num_games
    depth = args.depth
    game_label = args.game.upper()
    input_size = 9 if args.game == 'ttt' else 42
    output_size = 9 if args.game == 'ttt' else 7

                            
    ql = QLearningAgent(1, 'QLearning')
    ql_path = f'models/qlearning_{args.game}.pkl'
    if os.path.exists(ql_path):
        ql.load(ql_path)
        ql.epsilon = 0.0
    else:
        print(f"  [warn] No saved model at {ql_path}, using untrained QLearning")

    dqn = DQNAgent(1, 'DQN', input_size=input_size, output_size=output_size)
    dqn_path = f'models/dqn_{args.game}.pt'
    if os.path.exists(dqn_path):
        dqn.load(dqn_path)
        dqn.epsilon = 0.0
    else:
        print(f"  [warn] No saved model at {dqn_path}, using untrained DQN")

    algorithms = [
        MiniMaxAgent(1, 'MiniMax', max_depth=depth),
        AlphaBetaAgent(1, 'AlphaBeta', max_depth=depth),
        ql,
        dqn,
    ]

    print(f"\n=== {game_label}: All Algorithms vs Default ({num_games} games, fixed first mover) ===\n")
    print(f"{'Matchup':<35} {'First':>10} {'W':>6} {'D':>6} {'L':>6}")
    print("-" * 65)

    rows = []
    for algo in algorithms:
        for combo in ['algo_first', 'default_first']:
            if combo == 'algo_first':
                a1 = algo
                a1.player = 1
                a2 = DefaultOpponent(-1, 'Default')
                first_mover = algo.name
            else:
                a1 = DefaultOpponent(1, 'Default')
                a2 = algo
                a2.player = -1
                first_mover = 'Default'

            res = run_tournament(a1, a2, game_class, num_games=num_games, alternate=False)

                                                                    
            if combo == 'algo_first':
                algo_wins  = res['agent1_wins']
                algo_losses = res['agent2_wins']
            else:
                algo_wins  = res['agent2_wins']
                algo_losses = res['agent1_wins']
            algo_draws = res['draws']

            matchup = f"{a1.name} vs {a2.name}"
            print(f"{matchup:<35} {first_mover:>10} "
                  f"{algo_wins/num_games:>5.1%} {algo_draws/num_games:>5.1%} {algo_losses/num_games:>5.1%}")

            rows.append({
                'game':            game_label,
                'algorithm':       algo.name,
                'first_mover':     first_mover,
                'matchup':         matchup,
                'num_games':       num_games,
                'algo_wins':       algo_wins,
                'algo_draws':      algo_draws,
                'algo_losses':     algo_losses,
                'algo_win_rate':   round(algo_wins  / num_games, 4),
                'algo_draw_rate':  round(algo_draws / num_games, 4),
                'algo_loss_rate':  round(algo_losses / num_games, 4),
                'avg_game_length': round(res['avg_game_length'], 2),
            })

    os.makedirs(os.path.join('experiments', 'results'), exist_ok=True)
    csv_path = os.path.join('experiments', 'results', f'{args.game}_vs_default.csv')
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

def mode_head_to_head(args):
    import pandas as pd
    from experiments.runner import run_tournament
    from agents.minimax_agent import MiniMaxAgent
    from agents.alphabeta_agent import AlphaBetaAgent
    from agents.qlearning_agent import QLearningAgent
    from agents.dqn_agent import DQNAgent

    game_class = get_game(args.game)
    num_games = args.num_games
    depth = args.depth
    game_label = args.game.upper()
    input_size = 9 if args.game == 'ttt' else 42
    output_size = 9 if args.game == 'ttt' else 7

                            
    ql = QLearningAgent(1, 'QLearning')
    ql_path = f'models/qlearning_{args.game}.pkl'
    if os.path.exists(ql_path):
        ql.load(ql_path)
        ql.epsilon = 0.0
    else:
        print(f"  [warn] No saved model at {ql_path}, using untrained QLearning")

    dqn = DQNAgent(1, 'DQN', input_size=input_size, output_size=output_size)
    dqn_path = f'models/dqn_{args.game}.pt'
    if os.path.exists(dqn_path):
        dqn.load(dqn_path)
        dqn.epsilon = 0.0
    else:
        print(f"  [warn] No saved model at {dqn_path}, using untrained DQN")

    def make_agents():
        return [
            MiniMaxAgent(1, 'MiniMax', max_depth=depth),
            AlphaBetaAgent(1, 'AlphaBeta', max_depth=depth),
            ql,
            dqn,
        ]

    agents = make_agents()
    names = [a.name for a in agents]

    print(f"\n=== {game_label}: Head-to-Head ({num_games} games, fixed first mover) ===\n")
    print(f"{'Matchup':<35} {'First':>12} {'W':>6} {'D':>6} {'L':>6}")
    print("-" * 65)

    rows = []
                                                                          
    for i, a1 in enumerate(agents):
        for j, a2 in enumerate(agents):
            if i == j:
                continue

            a1.player = 1
            a2.player = -1

            res = run_tournament(a1, a2, game_class, num_games=num_games, alternate=False)

            matchup = f"{a1.name} vs {a2.name}"
            w = res['agent1_win_rate']
            d = res['draw_rate']
            l = res['agent2_win_rate']
            print(f"{matchup:<35} {a1.name:>12} {w:>5.1%} {d:>5.1%} {l:>5.1%}")

            rows.append({
                'game':              game_label,
                'first_mover':       a1.name,
                'second_mover':      a2.name,
                'matchup':           matchup,
                'num_games':         num_games,
                'first_wins':        res['agent1_wins'],
                'draws':             res['draws'],
                'second_wins':       res['agent2_wins'],
                'first_win_rate':    round(w, 4),
                'draw_rate':         round(d, 4),
                'second_win_rate':   round(l, 4),
                'avg_game_length':   round(res['avg_game_length'], 2),
            })

    os.makedirs(os.path.join('experiments', 'results'), exist_ok=True)
    csv_path = os.path.join('experiments', 'results', f'{args.game}_head_to_head.csv')
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

def mode_generate_plots(args):
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import seaborn as sns

    plot_dir = args.plot_dir or 'plots'
    os.makedirs(plot_dir, exist_ok=True)
    results_dir = os.path.join('experiments', 'results')

    ALGOS  = ['MiniMax', 'AlphaBeta', 'QLearning', 'DQN']
    WIN_C  = '#27ae60'
    DRAW_C = '#f39c12'
    LOSS_C = '#e74c3c'
    FIRST_EDGE  = '#2c3e50'
    SECOND_EDGE = '#7f8c8d'

    def _bar_label(ax, x, val, base=0, fontsize=9):
        if val >= 0.06:
            ax.text(x, base + val / 2, f'{val:.0%}',
                    ha='center', va='center', fontsize=fontsize,
                    color='white', fontweight='bold')

    def _annotate_top(ax, x, total, fontsize=8):
        ax.text(x, total + 0.02, f'{total:.0%}',
                ha='center', va='bottom', fontsize=fontsize,
                color='#2c3e50', fontweight='bold')

                                                                          
                                                                            
                                                                          
    def plot_vs_default(df, game_label, filename):
        fig, ax = plt.subplots(figsize=(14, 7))
        fig.patch.set_facecolor('#ffffff')

        algos = ALGOS
        n = len(algos)
        bar_w = 0.32
        gap   = 1.1
        positions = np.arange(n) * gap

        for idx, algo in enumerate(algos):
            for k, (fm_key, offset, edge, lbl) in enumerate([
                (algo,      -bar_w / 2 - 0.02, FIRST_EDGE,  'Goes 1st'),
                ('Default', +bar_w / 2 + 0.02, SECOND_EDGE, 'Goes 2nd'),
            ]):
                row = df[(df['algorithm'] == algo) & (df['first_mover'] == fm_key)]
                if row.empty:
                    continue
                w = row['algo_win_rate'].values[0]
                d = row['algo_draw_rate'].values[0]
                l = row['algo_loss_rate'].values[0]
                x = positions[idx] + offset

                b1 = ax.bar(x, w, bar_w, color=WIN_C,  edgecolor=edge, linewidth=1.2, zorder=3)
                b2 = ax.bar(x, d, bar_w, bottom=w,     color=DRAW_C, edgecolor=edge, linewidth=1.2, zorder=3)
                b3 = ax.bar(x, l, bar_w, bottom=w + d, color=LOSS_C, edgecolor=edge, linewidth=1.2, zorder=3)

                _bar_label(ax, x, w, 0)
                _bar_label(ax, x, d, w)
                _bar_label(ax, x, l, w + d)

                                                           
                ax.text(x, -0.06, lbl, ha='center', va='top', fontsize=8,
                        color=edge, fontweight='bold',
                        transform=ax.get_xaxis_transform())

        ax.set_xticks(positions)
        ax.set_xticklabels(algos, fontsize=13, fontweight='bold')
        ax.set_ylabel('Rate (Win / Draw / Loss)', fontsize=12)
        ax.set_ylim(0, 1.12)
        ax.set_title(f'{game_label}:  Algorithm  vs  Default Opponent\n'
                     f'Dark border = algorithm goes first   |   Grey border = Default goes first',
                     fontsize=13, fontweight='bold', pad=14)
        ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
        ax.spines[['top', 'right']].set_visible(False)

        legend_handles = [
            mpatches.Patch(facecolor=WIN_C,  edgecolor='grey', label='Algorithm Wins'),
            mpatches.Patch(facecolor=DRAW_C, edgecolor='grey', label='Draw'),
            mpatches.Patch(facecolor=LOSS_C, edgecolor='grey', label='Algorithm Loses'),
            mpatches.Patch(facecolor='white', edgecolor=FIRST_EDGE,  linewidth=2, label='Algorithm goes 1st'),
            mpatches.Patch(facecolor='white', edgecolor=SECOND_EDGE, linewidth=2, label='Default goes 1st'),
        ]
        ax.legend(handles=legend_handles, loc='upper right', fontsize=9,
                  framealpha=0.9, edgecolor='lightgrey')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved {filename}")

                                                                          
                                                                            
                                                                          
    def plot_head_to_head(df, game_label, filename):
        algos   = ALGOS
        n       = len(algos)
        idx_map = {a: i for i, a in enumerate(algos)}

        first_wins  = np.full((n, n), np.nan)
        first_draws = np.full((n, n), np.nan)
        first_loss  = np.full((n, n), np.nan)

        for _, row in df.iterrows():
            fm = row['first_mover']
            sm = row['second_mover']
            if fm not in idx_map or sm not in idx_map:
                continue
            i, j = idx_map[fm], idx_map[sm]
            first_wins[i][j]  = row['first_win_rate']
            first_draws[i][j] = row['draw_rate']
            first_loss[i][j]  = row['second_win_rate']

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.patch.set_facecolor('#ffffff')
        fig.suptitle(
            f'{game_label}: Head-to-Head Results\n'
            f'Row agent = first mover (left)  |  Row agent = second mover (right)',
            fontsize=13, fontweight='bold')

                                                               
        def build_annot(win_m, draw_m, loss_m):
            annot = []
            for i in range(n):
                row_ann = []
                for j in range(n):
                    if np.isnan(win_m[i][j]):
                        row_ann.append('—')
                    else:
                        w = win_m[i][j]
                        d = draw_m[i][j]
                        l = loss_m[i][j]
                        winner = ('ROW WINS' if w > l else
                                  'COL WINS' if l > w else 'DRAW')
                        row_ann.append(f'W {w:.0%}\nD {d:.0%}\nL {l:.0%}\n{winner}')
                annot.append(row_ann)
            return annot

                              
        annot_first = build_annot(first_wins, first_draws, first_loss)
        mask_first  = np.isnan(first_wins)

                                            
        second_wins  = first_loss.T.copy()
        second_draws = first_draws.T.copy()
        second_loss  = first_wins.T.copy()
        annot_second = build_annot(second_wins, second_draws, second_loss)
        mask_second  = np.isnan(second_wins)

        for ax, win_m, annot, mask, title in [
            (axes[0], first_wins,  annot_first,  mask_first,
             'Row goes FIRST\n(value = row agent win rate)'),
            (axes[1], second_wins, annot_second, mask_second,
             'Row goes SECOND\n(value = row agent win rate)'),
        ]:
            sns.heatmap(
                win_m, annot=annot, fmt='', cmap='RdYlGn',
                xticklabels=algos, yticklabels=algos,
                vmin=0, vmax=1, ax=ax, linewidths=1, linecolor='white',
                annot_kws={'size': 8, 'va': 'center'},
                cbar_kws={'label': 'Row Agent Win Rate', 'shrink': 0.8},
                mask=mask,
            )
            ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
            ax.set_xlabel('Opponent (column agent)', fontsize=10)
            ax.set_ylabel('Agent (row agent)', fontsize=10)
            ax.tick_params(axis='both', labelsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved {filename}")

                                                                          
                                                                            
                                                                          
    def plot_overall(ttt_df, c4_df, filename):
        algos = ALGOS
        n     = len(algos)
        bar_w = 0.18
        x     = np.arange(n)

        configs = [
            (ttt_df, 'TTT  Goes 1st',  True,  -1.5*bar_w, '#2980b9', ''),
            (ttt_df, 'TTT  Goes 2nd',  False, -0.5*bar_w, '#2980b9', '//'),
            (c4_df,  'C4   Goes 1st',  True,  +0.5*bar_w, '#c0392b', ''),
            (c4_df,  'C4   Goes 2nd',  False, +1.5*bar_w, '#c0392b', '//'),
        ]

        fig, ax = plt.subplots(figsize=(14, 7))
        fig.patch.set_facecolor('#ffffff')

        for df, label, is_first, offset, color, hatch in configs:
            vals = []
            for algo in algos:
                fm_val = algo if is_first else 'Default'
                row = df[(df['algorithm'] == algo) & (df['first_mover'] == fm_val)]
                vals.append(row['algo_win_rate'].values[0] if not row.empty else 0)
            bars = ax.bar(x + offset, vals, bar_w, color=color, hatch=hatch,
                          edgecolor='white', linewidth=0.8, label=label, zorder=3)
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.015,
                            f'{val:.0%}', ha='center', va='bottom',
                            fontsize=8, fontweight='bold', color='#2c3e50')

        ax.set_xticks(x)
        ax.set_xticklabels(algos, fontsize=13, fontweight='bold')
        ax.set_ylabel('Win Rate vs Default Opponent', fontsize=12)
        ax.set_ylim(0, 1.18)
        ax.set_title('Overall Win Rate vs Default Opponent — All Algorithms & Games\n'
                     '(solid = algorithm goes first  |  hatched = Default goes first)',
                     fontsize=13, fontweight='bold', pad=14)
        ax.legend(fontsize=9, loc='upper left', framealpha=0.9,
                  ncol=2, edgecolor='lightgrey')
        ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
        ax.spines[['top', 'right']].set_visible(False)
        ax.axhline(0.5, color='grey', linestyle=':', linewidth=1, label='50% line')
        ax.text(n - 0.1, 0.51, '50%', fontsize=8, color='grey', va='bottom')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved {filename}")

                                                                          
                                                                            
                                                                          
    def plot_learning_curves_summary(filename):
        curve_files = {
            'TTT — Q-Learning': os.path.join(plot_dir, 'qlearning_ttt_learning_curve.png'),
            'TTT — DQN':        os.path.join(plot_dir, 'dqn_ttt_learning_curve.png'),
            'C4 — Q-Learning':  os.path.join(plot_dir, 'qlearning_c4_learning_curve.png'),
            'C4 — DQN':         os.path.join(plot_dir, 'dqn_c4_learning_curve.png'),
        }
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('RL Agent Learning Curves (Win / Draw / Loss rate over training episodes)',
                     fontsize=14, fontweight='bold')
        for ax, (title, path) in zip(axes.flat, curve_files.items()):
            if os.path.exists(path):
                img = plt.imread(path)
                ax.imshow(img)
                ax.set_title(title, fontsize=12, fontweight='bold', pad=6)
            else:
                ax.text(0.5, 0.5, f'Missing:\n{os.path.basename(path)}\nRun --mode train first',
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=10, color='grey')
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, filename), dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Saved {filename}")

                                                                          
                                                                            
                                                                          
    print(f"\nGenerating plots → {plot_dir}/\n")

    ttt_vs  = pd.read_csv(os.path.join(results_dir, 'ttt_vs_default.csv'))
    c4_vs   = pd.read_csv(os.path.join(results_dir, 'c4_vs_default.csv'))
    ttt_h2h = pd.read_csv(os.path.join(results_dir, 'ttt_head_to_head.csv'))
    c4_h2h  = pd.read_csv(os.path.join(results_dir, 'c4_head_to_head.csv'))

    plot_vs_default(ttt_vs,  'Tic Tac Toe', 'ttt_vs_default.png')
    plot_vs_default(c4_vs,   'Connect 4',   'c4_vs_default.png')
    plot_head_to_head(ttt_h2h, 'Tic Tac Toe', 'ttt_head_to_head.png')
    plot_head_to_head(c4_h2h,  'Connect 4',   'c4_head_to_head.png')
    plot_overall(ttt_vs, c4_vs, 'overall_comparison.png')
    plot_learning_curves_summary('learning_curves_summary.png')

    print(f"\nDone. All plots saved to {plot_dir}/")

def mode_full_experiment(args):
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

    N_GAMES = 200                                       

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

                
        default = DefaultOpponent(-1, 'Default')

                                                                             
                                                                           
                                                                           
                                                          
        training_opponent_ql = (
            RandomAgent(-1, 'Random') if game_name == 'Connect4'
            else DefaultOpponent(-1, 'Default')
        )
        training_opponent_dqn = (
            RandomAgent(-1, 'Random') if game_name == 'Connect4'
            else DefaultOpponent(-1, 'Default')
        )
        print(f"  Training opponent: {training_opponent_ql.name}")

                                 
        print(f"\n[1/4] Training Q-Learning on {game_name}...")
        ql_agent = QLearningAgent(1, 'QLearning')
        ql_history = train_rl_agent(ql_agent, training_opponent_ql, game_class, num_episodes=ql_episodes)
        ql_model_path = f'models/qlearning_{game_name.lower()}.pkl'
        ql_agent.save(ql_model_path)

        print(f"\n[2/4] Training DQN on {game_name}...")
        dqn_agent = DQNAgent(1, 'DQN', input_size=input_size, output_size=output_size)
        dqn_history = train_rl_agent(dqn_agent, training_opponent_dqn, game_class, num_episodes=dqn_episodes)
        dqn_model_path = f'models/dqn_{game_name.lower()}.pt'
        dqn_agent.save(dqn_model_path)

                              
        short = game_name.lower().replace('connect4', 'c4')
        plot_learning_curve(ql_history, 'Q-Learning', game_name, f'{short}_qlearning_curve.png')
        plot_learning_curve(dqn_history, 'DQN', game_name, f'{short}_dqn_curve.png')

                            
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

                  
        rows = []
        for name, r in vs_default_results.items():
            rows.append({'agent': name, 'game': game_name, **{k: v for k, v in r.items() if k != 'results_per_game'}})
        pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, f'{short}_vs_default.csv'), index=False)

                      
        for agent in algo_agents:
            if agent.name not in overall_data:
                overall_data[agent.name] = {}
            overall_data[agent.name][game_name] = {
                'win_rate': vs_default_results[agent.name]['agent1_win_rate']
            }

                              
        print(f"\n[4/4] Head-to-head tournaments on {game_name}...")
        all_agents = [
            MiniMaxAgent(1, 'MiniMax', max_depth=mm_depth),
            AlphaBetaAgent(1, 'AlphaBeta', max_depth=mm_depth),
            QLearningAgent(1, 'QLearning'),
            DQNAgent(1, 'DQN', input_size=input_size, output_size=output_size),
        ]
                                
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
                res = run_tournament(all_agents[i], all_agents[j], game_class, num_games=N_GAMES)
                matrix[i, j] = res['agent1_win_rate']

        labels = [a.name for a in all_agents]
        plot_head_to_head(matrix, labels, game_name, f'{short}_head_to_head.png')

                                 
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

                        
    plot_overall_comparison(overall_data, 'overall_comparison.png')

                         
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
    parser.add_argument('--mode', choices=['play', 'train', 'tournament', 'vs-default', 'head-to-head', 'generate-plots', 'full-experiment', 'interactive'],
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
    parser.add_argument('--ui', action='store_true', help='Show game in a GUI window (play / interactive modes)')
    parser.add_argument('--plot-dir', type=str, default=None, dest='plot_dir', help='Directory to save plots after training')

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
    elif args.mode == 'vs-default':
        mode_vs_default(args)
    elif args.mode == 'head-to-head':
        mode_head_to_head(args)
    elif args.mode == 'generate-plots':
        mode_generate_plots(args)
    elif args.mode == 'full-experiment':
        mode_full_experiment(args)

if __name__ == '__main__':
    main()
