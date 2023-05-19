import gym
from Q_Learner import QLearner
from Hyperparameters import hyperparameters
import pandas as pd
from tqdm import tqdm

NUMBER_OF_LOG_MESSAGES = 100

# create the environment
env = gym.make('-')

# Create the meta_data
# The game name is the name of the file you want to give your model
# The path to final model is the path you want to save the finished table
# The media path is the path you want to save the csv with the cumulative rewards in
game_name = '-'
path_to_final_model = '-'
media_path = '-'

# get the dimensions for the Q-Table
action_space_size = env.action_space.n
observation_space_size = env.observation_space.n

# def hyperparameters
agent_hyperparameters = [hyperparameters['initial_eps'], hyperparameters['final_eps'],
                         hyperparameters['eps_decay_steps'], hyperparameters['gamma'], hyperparameters['learning_rate']]

agent = QLearner(*agent_hyperparameters, action_space=action_space_size, obs_space=observation_space_size)

total_steps = 0

for episode in tqdm(range(hyperparameters['number_of_episodes'])):
    state, _ = env.reset()

    done = False
    reward_for_episode = 0

    for step in range(hyperparameters['max_steps_per_episode']):
        new_state, reward, done, action = agent.act(state, env)
        total_steps += 1
        agent.exploration_decay(total_steps)

        agent.update_q_table(state, action, reward, new_state)

        # update the state
        state = new_state

        # add reward
        reward_for_episode += reward

        if done:
            agent.reward_all_episodes.append(reward_for_episode)
            break

    # information printing
    if episode % (hyperparameters['number_of_episodes'] / NUMBER_OF_LOG_MESSAGES) == 0:
        print(f'\n'
              f'{"~" * 50}\n'
              f'Episode: {episode}\n'
              f'Win Rate: {sum(agent.reward_all_episodes) / (len(agent.reward_all_episodes) + 1)}\n'
              f'total steps done: {total_steps}\n'
              f'{"~" * 50}')

# save final q_table and csv of rewards
pd.DataFrame({'cumulative rewards': agent.reward_all_episodes}).to_csv(f'{media_path}/{game_name}.csv')
pd.DataFrame(agent.q_table).to_csv(f'{path_to_final_model}/{game_name}_qtable.csv')
