import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

PLOTS_DIR = 'plots'
RESULTS_DIR = os.path.join('experiments', 'results')
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

COLORS = {
    'win': '#2ecc71',
    'draw': '#f39c12',
    'loss': '#e74c3c',
}
ALGO_COLORS = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']


def save_results_csv(data: dict, filename: str):
    filepath = os.path.join(RESULTS_DIR, filename)
    df = pd.DataFrame([data])
    df.to_csv(filepath, index=False)
    return filepath


def plot_vs_default(results: dict, game_name: str, filename: str):
    """Bar chart of win/draw/loss rates vs default opponent."""
    algos = list(results.keys())
    win_rates = [results[a]['agent1_win_rate'] for a in algos]
    draw_rates = [results[a]['draw_rate'] for a in algos]
    loss_rates = [results[a]['agent2_win_rate'] for a in algos]

    x = np.arange(len(algos))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, win_rates, width, label='Win', color=COLORS['win'])
    ax.bar(x, draw_rates, width, label='Draw', color=COLORS['draw'])
    ax.bar(x + width, loss_rates, width, label='Loss', color=COLORS['loss'])

    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Rate')
    ax.set_title(f'{game_name}: Algorithm Performance vs Default Opponent')
    ax.set_xticks(x)
    ax.set_xticklabels(algos, rotation=15)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=300)
    plt.close()


def plot_head_to_head(matrix: np.ndarray, labels: list, game_name: str, filename: str):
    """Heatmap of head-to-head win rates."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt='.2f', cmap='RdYlGn',
                xticklabels=labels, yticklabels=labels,
                vmin=0, vmax=1, ax=ax, linewidths=0.5)
    ax.set_title(f'{game_name}: Head-to-Head Win Rate Matrix\n(row agent win rate vs col agent)')
    ax.set_xlabel('Opponent')
    ax.set_ylabel('Agent')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=300)
    plt.close()


def plot_learning_curve(history: list, agent_name: str, game_name: str, filename: str):
    """Line plot of win rate over training episodes."""
    episodes = [h['episode'] for h in history]
    win_rates = [h['win_rate'] for h in history]
    draw_rates = [h['draw_rate'] for h in history]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(episodes, win_rates, label='Win Rate', color=COLORS['win'], linewidth=2)
    ax.plot(episodes, draw_rates, label='Draw Rate', color=COLORS['draw'], linewidth=2, linestyle='--')
    ax.set_xlabel('Training Episodes')
    ax.set_ylabel('Rate')
    ax.set_title(f'{game_name}: {agent_name} Learning Curve')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=300)
    plt.close()


def plot_nodes_explored(nodes_data: dict, filename: str):
    """Bar chart comparing minimax vs alpha-beta node counts."""
    labels = list(nodes_data.keys())
    values = list(nodes_data.values())

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=ALGO_COLORS[:len(labels)])
    ax.set_ylabel('Nodes Explored')
    ax.set_title('Minimax vs Alpha-Beta: Nodes Explored')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                f'{val:,}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=300)
    plt.close()


def plot_overall_comparison(data: dict, filename: str):
    """Summary bar chart of win rates across all algorithms and games."""
    algorithms = list(data.keys())
    games = list(next(iter(data.values())).keys())

    x = np.arange(len(algorithms))
    width = 0.35 / len(games)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, game in enumerate(games):
        win_rates = [data[a].get(game, {}).get('win_rate', 0) for a in algorithms]
        offset = (i - len(games) / 2 + 0.5) * width
        ax.bar(x + offset, win_rates, width, label=game, color=ALGO_COLORS[i % len(ALGO_COLORS)])

    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Win Rate vs Default Opponent')
    ax.set_title('Overall Algorithm Comparison Across Games')
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, rotation=15)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=300)
    plt.close()
