import os

import gym
import torch

from Agent import Agent

# The game name is the name that the make function requires, it can be looked up on the documentation for the game
game_name = '-'
env = gym.make(game_name, render_mode='human')

# The path of the model is the folder you saved your model in.
# The name of the environment is simply the name you gave the trained model.
path_of_model = '-'
name_of_environment = '-'

number_of_episodes_playing = 100
number_of_max_steps = 200
episode_reward = 0
# Load the final model
if os.path.exists(f'{path_of_model}/{name_of_environment}_final.pt'):
    agent = Agent(0.1, 0.1, 0, in_channels=env.observation_space.shape[0], num_actions=env.action_space.n)
    agent.policy_net.load_state_dict(
        torch.load(f'{path_of_model}{name_of_environment}_final.pt'))
    agent.policy_net.eval()

    # Test the final model
    print(env.reset())
    state, _ = env.reset()
    total_steps = 0
    while number_of_episodes_playing > 0:
        state = torch.as_tensor(state)
        action, new_state, reward, done, *others = agent.act(state, env)
        total_steps +=1
        episode_reward += reward
        state = new_state

        if done or total_steps > number_of_max_steps:
            number_of_episodes_playing -= 1
            print(f'{"~" * 50}\n'
                  f'Episode: {100 - number_of_episodes_playing}\n'
                  f'reward: {episode_reward}\n')
            episode_reward = 0
            total_steps = 0
            state, _ = env.reset()
