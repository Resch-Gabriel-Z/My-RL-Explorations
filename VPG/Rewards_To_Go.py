import numpy as np
import torch


def rewards_to_go_calculator(rewards):
    """
    Calculate the rewards-to-go using dynamic programming.

    This function efficiently computes the rewards-to-go by leveraging dynamic programming techniques.
    Instead of repeatedly summing the rewards, it traverses the rewards list in reverse order, accumulating
    the rewards as it goes. Each element in the rewards-to-go array is the sum of the current reward and
    the previous rewards-to-go value, resulting in the cumulative sum of rewards from the current position
    to the end of the list.
    Args:
        rewards: A list of rewards

    Returns: A numpy array with the rewards to go

    """

    rewards_to_go = np.zeros_like(rewards)
    for i in reversed(range(len(rewards))):
        rewards_to_go[i] = rewards[i] + (rewards_to_go[i + 1] if i + 1 < len(rewards) else 0)
    return torch.as_tensor(rewards_to_go)
