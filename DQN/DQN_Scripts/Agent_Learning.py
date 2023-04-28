import numpy as np
import torch
import torch.nn as nn

from Replay_Memory import Memory

MIN_SIZE_MEMORY = 100


def agent_learning(batch_size, gamma, memory, agent, optimizer):
    """
    The method described in DQN/DDQN for an agent to learn, this is just an implementation of the algorithm described
    in the paper.
    Args:
        batch_size: the number of memories we want to learn
        gamma: the discount factor
        memory: the memory
        agent: the agent
        optimizer: the optimizer

    """
    if len(memory) < MIN_SIZE_MEMORY:
        return
    sample_memory = memory.sample(batch_size)
    sample_memory_preprocessed = Memory(*zip(*sample_memory))

    sample_actions = torch.as_tensor(sample_memory_preprocessed.action, dtype=torch.int64).unsqueeze(-1)
    sample_rewards = torch.as_tensor(sample_memory_preprocessed.reward, dtype=torch.float32).unsqueeze(-1)
    sample_dones = torch.as_tensor(sample_memory_preprocessed.done, dtype=torch.float32).unsqueeze(-1)
    sample_states = torch.stack(sample_memory_preprocessed.state)
    sample_next_states = torch.as_tensor(np.array(sample_memory_preprocessed.next_state), dtype=torch.float32)

    max_q_values_next, _ = torch.max(agent.policy_net(sample_next_states), dim=1, keepdim=True)
    target = sample_rewards + gamma * max_q_values_next * (1 - sample_dones)

    q_values = agent.policy_net(sample_states)
    actions_q_values = torch.gather(q_values, dim=1, index=sample_actions)

    loss = nn.functional.smooth_l1_loss(input=actions_q_values, target=target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
