import gym
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from tqdm import tqdm

from Agent import Agent
from Agent_Learning import agent_learning
from Hyperparameters import hyperparameters
from Model_Saving_Loading import load_model_dict, save_model_dict, save_final_model
from Neural_Network import DQN
from Replay_Memory import ReplayMemory

NUMBER_OF_LOG_MESSAGES = 100
NUMBER_OF_CHECKPOINTS = 100

# Hyperparameters for the agent
agent_hyperparameters = [hyperparameters['initial_eps'], hyperparameters['final_eps'],
                         hyperparameters['eps_decay_steps']]

# Create the environment
env = gym.make("-", render_mode='rgb_array')

# Create the meta data
game_name = '-'
path_to_model_save = '-'
name_final_model = f'{game_name}_final'
path_to_final_model = '-'

# Create the Agent and the online Network
agent = Agent(*agent_hyperparameters, in_channels=int(np.prod(env.observation_space.shape)),
              num_actions=env.action_space.n)
online_net = DQN(int(np.prod(env.observation_space.shape)), env.action_space.n)

# Initialize the weights of the online net with the policy nets weights
online_net.load_state_dict(agent.policy_net.state_dict())

# Initialize optimizer, loss function
optimizer = optim.Adam(agent.policy_net.parameters(), hyperparameters['learning_rate'])

# Initialize Memory
memory = ReplayMemory(hyperparameters['replay_buffer_size'])

# Initialize starting variables to override
start = 0
total_steps = 0
episode_reward_tracker = []

# Load the model
start, episode_reward_tracker, total_steps, memory = load_model_dict(path=path_to_model_save, name=game_name,
                                                                     policy_net=agent.policy_net,
                                                                     online_net=online_net,
                                                                     optimizer=optimizer,
                                                                     starting_point=start,
                                                                     total_steps=total_steps,
                                                                     episode_reward_tracker=episode_reward_tracker,
                                                                     memory=memory)

# Train the Agent for a number of episodes
for episode in tqdm(range(start, hyperparameters['number_of_episodes'])):
    # First reset the environment and the cumulative reward
    state, _ = env.reset()
    reward_for_episode = 0

    # Then for each step, follow the Pseudocode in the paper
    for step in range(hyperparameters['max_steps_per_episode']):
        state = torch.tensor(state)
        action, new_state, reward, done, *others = agent.act(state, env)
        total_steps += 1
        agent.exploration_decay(total_steps=total_steps)

        # Push the memory
        memory.push(state, action, done, new_state, reward)

        # Update the state
        state = new_state

        # add the reward to the cumulative reward
        reward_for_episode += reward

        # Agent Learning
        agent_learning(hyperparameters['batch_size'], hyperparameters['gamma'], memory=memory, agent=agent,
                       online=online_net, optimizer=optimizer)

        # If a condition arises which makes playing further impossible (such as losing all lives) go to new episode
        if done:
            episode_reward_tracker.append(reward_for_episode)
            break

        # Update the online Network
        if total_steps % hyperparameters['target_update_freq'] == 0:
            online_net.load_state_dict(agent.policy_net.state_dict())

    # Save the Model
    if episode % (hyperparameters['number_of_episodes'] / NUMBER_OF_CHECKPOINTS) == 0:
        save_model_dict(path=path_to_model_save, name=game_name, policy_net=agent.policy_net,
                        online_net=online_net, optimizer=optimizer, starting_point=episode,
                        total_steps=total_steps, memory=memory, episode_reward_tracker=episode_reward_tracker)

    # Print out useful information during Training
    if episode % (hyperparameters['number_of_episodes'] / NUMBER_OF_LOG_MESSAGES) == 0:
        print(f'\n'
              f'{"~" * 40}\n'
              f'Episode: {episode + 1}\n'
              f'average steps since last log: {sum(episode_reward_tracker[-7:]) / 7:.2f}\n'
              f'total steps done: {total_steps}\n'
              f'memory size: {len(memory)}\n'
              f'{"~" * 40}')

# After training, save the models parameters
save_final_model(name=name_final_model, path=path_to_final_model, model=agent.policy_net)
df = pd.DataFrame({'cumulative rewards': episode_reward_tracker})
df.to_csv(f'-/{game_name}.csv')

