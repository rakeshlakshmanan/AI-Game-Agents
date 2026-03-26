import random
import numpy as np
from tqdm import tqdm


def run_tournament(agent1, agent2, game_class, num_games: int = 1000, verbose: bool = False) -> dict:
    """
    Play num_games between agent1 and agent2.
    Alternate who goes first each game.
    """
    agent1_wins = 0
    agent2_wins = 0
    draws = 0
    total_moves = 0
    results_per_game = []

    for i in tqdm(range(num_games), desc=f"{agent1.name} vs {agent2.name}", leave=False):
        game = game_class()
        game.reset()

        # alternate first player
        if i % 2 == 0:
            first, second = agent1, agent2
        else:
            first, second = agent2, agent1

        agents = {first.player: first, second.player: second}
        # assign players for this game
        first.player = 1
        second.player = -1

        current = first
        moves = 0
        while True:
            move = current.get_move(game)
            _, _, done, info = game.make_move(move, current.player)
            moves += 1
            if done:
                winner = info.get('winner')
                if winner == agent1.player:
                    agent1_wins += 1
                    results_per_game.append(1)
                elif winner == agent2.player:
                    agent2_wins += 1
                    results_per_game.append(-1)
                else:
                    draws += 1
                    results_per_game.append(0)
                break
            current = second if current is first else first

        total_moves += moves

        if verbose:
            print(game.render())

    # restore original players
    agent1.player = 1
    agent2.player = -1

    return {
        'agent1_wins': agent1_wins,
        'agent2_wins': agent2_wins,
        'draws': draws,
        'agent1_win_rate': agent1_wins / num_games,
        'agent2_win_rate': agent2_wins / num_games,
        'draw_rate': draws / num_games,
        'avg_game_length': total_moves / num_games,
        'results_per_game': results_per_game,
    }


def train_rl_agent(agent, opponent, game_class, num_episodes: int = 50000,
                   eval_every: int = 1000, eval_games: int = 100,
                   seed: int = 42) -> list:
    """
    Train an RL agent against an opponent.
    Returns training_history list.
    """
    random.seed(seed)
    np.random.seed(seed)
    training_history = []
    is_dqn = hasattr(agent, 'store_transition')

    for episode in tqdm(range(1, num_episodes + 1), desc=f"Training {agent.name}"):
        game = game_class()
        game.reset()

        # alternate first player
        if episode % 2 == 0:
            agent.player = 1
            opponent.player = -1
        else:
            agent.player = -1
            opponent.player = 1

        current_player_id = 1  # player 1 always goes first
        state = game.board.copy()

        while True:
            if current_player_id == agent.player:
                action = agent.get_move(game)
                next_state, reward, done, info = game.make_move(action, agent.player)

                if is_dqn:
                    agent.store_transition(state, action, reward, next_state, done)
                    agent.train_step()
                else:
                    valid_next = game.get_valid_moves()
                    agent.learn(game.get_state_key() if not done else str(tuple(next_state.flatten())),
                                action,
                                reward if done else 0.0,
                                str(tuple(next_state.flatten())),
                                done,
                                valid_next)

                state = next_state
                if done:
                    break
            else:
                action = opponent.get_move(game)
                next_state, reward, done, info = game.make_move(action, opponent.player)
                if done:
                    # agent lost or draw
                    winner = info.get('winner')
                    if winner == agent.player:
                        agent_reward = 1.0
                    elif winner == -agent.player:
                        agent_reward = -1.0
                    else:
                        agent_reward = 0.3
                    if is_dqn:
                        agent.store_transition(state, action if action is not None else 0,
                                               agent_reward, next_state, True)
                    state = next_state
                    break
                state = next_state

            current_player_id = -current_player_id

        agent.decay_epsilon()

        if episode % eval_every == 0:
            wins = draws = losses = 0
            saved_eps = agent.epsilon
            agent.epsilon = 0.0  # greedy eval

            for g_idx in range(eval_games):
                eval_game = game_class()
                eval_game.reset()
                if g_idx % 2 == 0:
                    agent.player = 1
                    opponent.player = -1
                else:
                    agent.player = -1
                    opponent.player = 1

                cur = 1
                while True:
                    if cur == agent.player:
                        m = agent.get_move(eval_game)
                    else:
                        m = opponent.get_move(eval_game)
                    _, _, done, info = eval_game.make_move(m, cur)
                    if done:
                        w = info.get('winner')
                        if w == agent.player:
                            wins += 1
                        elif w == 0:
                            draws += 1
                        else:
                            losses += 1
                        break
                    cur = -cur

            agent.epsilon = saved_eps
            training_history.append({
                'episode': episode,
                'win_rate': wins / eval_games,
                'draw_rate': draws / eval_games,
                'loss_rate': losses / eval_games,
                'epsilon': agent.epsilon,
            })

    # restore players
    agent.player = 1
    opponent.player = -1
    return training_history
