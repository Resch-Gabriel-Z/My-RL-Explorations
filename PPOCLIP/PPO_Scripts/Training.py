import gym
import pandas as pd
import torch

from Hyperparameters import hyperparameters
from PPO import ppo_clip, Rewards

parameters = [hyperparameters['number_of_episodes'], hyperparameters['epsilon'], hyperparameters['learning_rate_actor'],
              hyperparameters['learning_rate_critic'], hyperparameters['number_of_epochs'],
              hyperparameters['hidden_layer_size']]

env = gym.make('-')

policy_network, value_network, reward_tracked = ppo_clip(env, *parameters)

rewards_bundled = Rewards(*zip(*reward_tracked))

game_name = '-'
name_final_model = f'{game_name}_final'
path_to_final_model = '-'
media_path = '-'

torch.save(policy_network.state_dict(), f'{path_to_final_model}/{name_final_model}.pt')
df = pd.DataFrame({'rewards': rewards_bundled.reward})
df.to_csv(f'{media_path}/{game_name}.csv')
