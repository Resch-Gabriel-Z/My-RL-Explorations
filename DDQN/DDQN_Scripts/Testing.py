import os
import gym
import torch

from Agent import Agent

game_name = 'CartPole-v1'
env = gym.make(game_name, render_mode='human')

path = '/home/gabe/PycharmProjects/RL-Stuff/DDQN/DDQN_trained_model/'
name = 'CartPole' + '_final.pt'

number_of_episodes_playing = 100
episode_reward = 0
# Load the final model
if os.path.exists(path + '/' + name):
    agent = Agent(0.1, 0.1, 0, in_channels=env.observation_space.shape[0], num_actions=env.action_space.n)
    agent.policy_net.load_state_dict(
        torch.load(path + '/' + name))
    agent.policy_net.eval()

    # Test the final model
    print(env.reset())
    state, _ = env.reset()
    while number_of_episodes_playing > 0:
        state = torch.as_tensor(state)
        action, new_state, reward, done, *others = agent.act(state, env)
        episode_reward += reward
        state = new_state

        if done:
            number_of_episodes_playing -= 1
            print(f'{"~"*50}\n'
                  f'Episode: {100 - number_of_episodes_playing}\n'
                  f'reward: {episode_reward}\n')
            episode_reward = 0
            state, _ = env.reset()