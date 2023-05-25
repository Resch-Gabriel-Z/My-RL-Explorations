import gym
import torch

from Neural_Networks import VPGDiscrete, VPGContinuous

discrete = False
game_name = '-'
path_to_model = '-'
MAX_NUM_OF_STEPS = 200

env = gym.make(game_name, render_mode='human')
action_space = 1
observation_space = 2
if discrete:
    policy = VPGDiscrete(in_channels=observation_space, num_actions=action_space)
    policy.load_state_dict(torch.load(f'{path_to_model}'))

    while True:
        state, _ = env.reset()
        cumulative_reward = 0

        for i in range(MAX_NUM_OF_STEPS):
            action = policy.select_action(policy(state))

            new_state, reward, done, *others = env.step(action)

            cumulative_reward += reward

            state = new_state

            if done:
                print(f'reward: {cumulative_reward}')
                break
        print(f'reward: {cumulative_reward}')

else:
    policy = VPGContinuous(in_channels=observation_space, num_actions=action_space)
    policy.load_state_dict(torch.load(f'{path_to_model}'))

    while True:
        state, _ = env.reset()
        cumulative_reward = 0

        for i in range(MAX_NUM_OF_STEPS):
            mean, std = policy(state)
            action = policy.select_action(mean, std)

            new_state, reward, done, *others = env.step(action)

            cumulative_reward += reward

            state = new_state

            if done:
                print(f'reward: {cumulative_reward}')
                break
        print(f'reward: {cumulative_reward}')
