import numpy as np
import torch
import torch.nn as nn

from Replay_Memory import Memory

MIN_SIZE_MEMORY = 100


def agent_learning(batch_size, gamma, memory, agent, online, optimizer):
    """
    The method described in DQN/DDQN for an agent to learn, this is just an implementation of the algorithm described
    in the paper.
    Args:
        batch_size: the number of memories we want to learn
        gamma: the discount factor
        memory: the memory
        agent: the agent
        online: the online network
        optimizer: the optimizer

    Remark:
        It might be confusing and challenging at first to implement this. The most important factor for me was to
        understand how pytorch handles calculations and to know what shapes my tensors have.
        I had so much trouble to make it work because the inputs weren't tensors, or the dtype wasn't correct or
        the shape was [batch_size] when it should be [batch_size,1] or something like that.
        So this code is not the first iteration I just throw in, all the functions like unsqueeze/dtype=/keepdim=/etc.
        were implemented by learning how pytorch wants me to work with my inputs, then knowing what they truly are
        and looking for techniques to make sure it is suitable for pytorch.

        So don't be discouraged at first :)
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

    max_q_values_online, _ = torch.max(online(sample_next_states), dim=1, keepdim=True)
    target = sample_rewards + gamma * max_q_values_online * (1 - sample_dones)

    q_values = agent.policy_net(sample_states)
    actions_q_values = torch.gather(q_values, dim=1, index=sample_actions)

    loss = nn.functional.smooth_l1_loss(input=actions_q_values, target=target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
