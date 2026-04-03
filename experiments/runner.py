import random
import numpy as np
from tqdm import tqdm

def run_tournament(agent1, agent2, game_class, num_games: int = 1000,
                   verbose: bool = False, alternate: bool = True) -> dict:
    agent1_wins = 0
    agent2_wins = 0
    draws = 0
    total_moves = 0
    results_per_game = []

    for i in tqdm(range(num_games), desc=f"{agent1.name} vs {agent2.name}", leave=False):
        game = game_class()
        game.reset()

        if alternate and i % 2 == 1:
            first, second = agent2, agent1
        else:
            first, second = agent1, agent2

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
                if winner == first.player and first is agent1:
                    agent1_wins += 1
                    results_per_game.append(1)
                elif winner == second.player and second is agent2:
                    agent2_wins += 1
                    results_per_game.append(-1)
                elif winner == first.player and first is agent2:
                    agent2_wins += 1
                    results_per_game.append(-1)
                elif winner == second.player and second is agent1:
                    agent1_wins += 1
                    results_per_game.append(1)
                else:
                    draws += 1
                    results_per_game.append(0)
                break
            current = second if current is first else first

        total_moves += moves

        if verbose:
            print(game.render())

                              
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
    random.seed(seed)
    np.random.seed(seed)
    training_history = []
    is_dqn = hasattr(agent, 'store_transition')

    for episode in tqdm(range(1, num_episodes + 1), desc=f"Training {agent.name}"):
        game = game_class()
        game.reset()

                                
        if episode % 2 == 0:
            agent.player = 1
            opponent.player = -1
        else:
            agent.player = -1
            opponent.player = 1

        current_player_id = 1                              

                                                                         
                                                                              
        agent_board_before = None                                            
        prev_agent_sk     = None                                                          
        prev_agent_action = None

        while True:
            if current_player_id == agent.player:
                                                                
                agent_board_before = game.board.copy()
                if not is_dqn:
                    prev_agent_sk = agent.state_key(game)

                action = agent.get_move(game)
                next_state, reward, done, info = game.make_move(action, agent.player)
                prev_agent_action = action

                if done:
                                                                                   
                    if is_dqn:
                        agent.store_transition(agent_board_before, action, reward, next_state, done)
                        agent.train_step()
                    else:
                        agent.learn(prev_agent_sk, action, reward, prev_agent_sk, True, [])
                    break
                                                                              

            else:
                opp_action = opponent.get_move(game)
                next_state, _, done, info = game.make_move(opp_action, opponent.player)

                if done:
                    winner = info.get('winner')
                    agent_reward = (1.0  if winner == agent.player  else
                                    0.3  if winner == 0             else
                                    -1.0)
                else:
                    agent_reward = 0.0

                                                                                
                if agent_board_before is not None:
                    if is_dqn:
                        agent.store_transition(agent_board_before, prev_agent_action,
                                               agent_reward, next_state, done)
                        agent.train_step()
                    else:
                        if done:
                            agent.learn(prev_agent_sk, prev_agent_action,
                                        agent_reward, prev_agent_sk, True, [])
                        else:
                            next_sk = agent.state_key(game)
                            agent.learn(prev_agent_sk, prev_agent_action,
                                        0.0, next_sk, False, game.get_valid_moves())

                                         
                    agent_board_before = None
                    prev_agent_sk      = None
                    prev_agent_action  = None

                if done:
                    break

            current_player_id = -current_player_id

        agent.decay_epsilon()

        if episode % eval_every == 0:
            wins = draws = losses = 0
            saved_eps = agent.epsilon
            agent.epsilon = 0.0               

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

                     
    agent.player = 1
    opponent.player = -1
    return training_history
