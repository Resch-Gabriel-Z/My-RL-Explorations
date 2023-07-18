import numpy as np
import torch

from Hyperparameters import hyperparameters

MAX_TRAJECTORY_LENGTH = hyperparameters['max_len_trajectory']


def compute_return_advantage(rewards, states, value_network):
    returns = []
    advantages = []
    next_value = 0

    for t in reversed(range(len(rewards))):
        next_value = rewards[t] + next_value
        returns.insert(0, next_value)

        state_tensor = torch.FloatTensor(states[t])
        value = value_network(state_tensor).item()
        advantages.insert(0, returns[0] - value)

    return returns, advantages


def trajectory_collector(env, output_size, policy_network):
    state, _ = env.reset()
    done = False

    states = []
    actions = []
    rewards = []
    steps = 0
    while not done and steps < MAX_TRAJECTORY_LENGTH:
        states.append(state)
        state_tensor = torch.FloatTensor(state)

        action_probs = policy_network(state_tensor)
        action_probs = action_probs.detach().numpy()
        action = np.random.choice(output_size, p=action_probs)
        steps += 1
        next_state, reward, done, *others = env.step(action)

        actions.append(action)
        rewards.append(reward)

        state = next_state
    return states, actions, rewards, action_probs
