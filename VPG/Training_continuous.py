import gym
import pandas as pd
import torch
import torch.optim as optim
from torch.distributions import Normal
from tqdm import tqdm

from Hyperparameters import hyperparameters
from Neural_Networks import VPGContinuous
from Rewards_To_Go import rewards_to_go_calculator
from Trajectory_Collector import trajectory_collector_continuous

NUMBER_OF_LOG_MESSAGES = 100

# Create the metadata
game_name = '-'
path_to_save_model = '-'
name_for_saved_model = '-'

path_to_media = '-'

# Make the environment

env = gym.make(game_name)
action_space = 1
observation_space = 2

# Create the Neural Network

policy = VPGContinuous(in_channels=observation_space, num_actions=action_space)
optimizer = optim.Adam(policy.parameters(), lr=hyperparameters['learning_rate'])

# Save the rewards for an episode

episode_reward = []

for episode in tqdm(range(hyperparameters['number_of_episodes'])):

    # Collect the trajectories

    traject_actions, traject_states, traject_rewards = trajectory_collector_continuous(policy, env, hyperparameters[
        'max_len_trajectory'])

    # Compute rewards to go

    rtg = rewards_to_go_calculator(traject_rewards)

    # Compute the derivative of the objective function according to the VPG pseudocode

    mean, std = policy(traject_states)
    sampler = Normal(mean, std)
    log_probs = -sampler.log_prob(traject_actions)
    loss = torch.sum(log_probs * rtg)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    episode_reward.append(sum(traject_rewards).item())
    if episode % (hyperparameters['number_of_episodes'] / NUMBER_OF_LOG_MESSAGES) == 0:
        print(f'\n'
              f'{"~" * 40}\n'
              f'Episode: {episode}\n'
              f'reward: {sum(episode_reward[episode - 10:episode]) / 10}\n'
              f'{"~" * 40}')

# Save the final model
if path_to_save_model and path_to_media != '-':
    torch.save(policy.state_dict(), f'{path_to_save_model}/{name_for_saved_model}.pt')
    df = pd.DataFrame({'reward per episode': episode_reward})
    df.to_csv(f'{path_to_media}/{name_for_saved_model}.csv')
