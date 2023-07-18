import os

import gym
import numpy as np
import torch

from Neural_Networks import PolicyNetwork

# The game name is the name that the make function requires, it can be looked up on the documentation for the game
game_name = '-'
env = gym.make(game_name, render_mode='human')

# The path of the model is the folder you saved your model in.
# The name of the environment is simply the name you gave the trained model.
path_of_model = '-'
name_of_environment = '-'

number_of_episodes_playing = 100
number_of_max_steps = 2000
episode_reward = 0
# Load the final model
if os.path.exists(f'{path_of_model}/{name_of_environment}_final.pt'):
    agent = PolicyNetwork(input_size=env.observation_space.shape[0], output_size=env.action_space.n, hidden_size=16)
    agent.load_state_dict(
        torch.load(f'{path_of_model}/{name_of_environment}_final.pt'))
    agent.eval()

    # Test the final model
    print(env.reset())
    state, _ = env.reset()
    total_steps = 0
    while number_of_episodes_playing > 0:
        state_tensor = torch.FloatTensor(state)

        action_probs = agent(state_tensor)
        action_probs = action_probs.detach().numpy()
        action = np.random.choice(env.action_space.n, p=action_probs)
        next_state, reward, done, *others = env.step(action)

        total_steps += 1
        episode_reward += reward
        state = next_state

        if done or total_steps > number_of_max_steps:
            number_of_episodes_playing -= 1
            print(f'{"~" * 50}\n'
                  f'Episode: {100 - number_of_episodes_playing}\n'
                  f'reward: {episode_reward}\n')
            episode_reward = 0
            total_steps = 0
            state, _ = env.reset()
