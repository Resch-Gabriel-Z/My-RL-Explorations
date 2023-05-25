import numpy as np
import torch


def trajectory_collector_discrete(policy, env, max_len_trajectory):
    """
    A function to collect the trajectories
    Args:
        policy: The Neural Network we are working with
        env: The environment
        max_len_trajectory: the maximum number of actions we do

    Returns: Tensors of states, actions and rewards respectively

    """
    # Reset the environment

    state, _ = env.reset()

    # Create lists to save the trajectory

    trajectory_actions = []
    trajectory_states = []
    trajectory_rewards = []

    # Collect Set of Trajectories

    for _ in range(max_len_trajectory):
        # Calculate the probabilities and select an action

        probs = torch.as_tensor(policy(state))
        action = policy.select_action(probs)

        # Execute the action in the environment

        new_state, reward, done, *others = env.step(action)

        # Append them to the list

        trajectory_actions.append(action)
        trajectory_states.append(state)
        trajectory_rewards.append(reward)

        # Go to the new state

        state = new_state

        if done:
            break
    # Process data to appropriate formats
    trajectory_actions = torch.as_tensor(trajectory_actions)
    trajectory_states = torch.as_tensor(np.array(trajectory_states))
    trajectory_rewards = torch.as_tensor(trajectory_rewards)

    return trajectory_actions, trajectory_states, trajectory_rewards


def trajectory_collector_continuous(policy, env, max_len_trajectory):
    """
    A function to collect the trajectories
    Args:
        policy: The Neural Network we are working with
        env: The environment
        max_len_trajectory: the maximum number of actions we do

    Returns: Tensors of states, actions and rewards respectively

    """
    # Reset the environment

    state, _ = env.reset()

    # Create lists to save the trajectory

    trajectory_actions = []
    trajectory_states = []
    trajectory_rewards = []

    # Collect Set of Trajectories

    for _ in range(max_len_trajectory):
        # Calculate the probabilities and select an action

        mean, std = policy(state)
        action = policy.select_action(mean, std)
        action = action
        # Execute the action in the environment

        new_state, reward, done, *others = env.step(action)

        # Append them to the list

        trajectory_actions.append(action)
        trajectory_states.append(state)
        trajectory_rewards.append(reward)

        # Go to the new state

        state = new_state

        if done:
            break
    # Process data to appropriate formats
    trajectory_actions = torch.as_tensor(trajectory_actions)
    trajectory_states = torch.as_tensor(np.array(trajectory_states))
    trajectory_rewards = torch.as_tensor(trajectory_rewards)

    return trajectory_actions, trajectory_states, trajectory_rewards
