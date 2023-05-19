import random

import numpy as np


class QLearner:
    """

    """

    def __init__(self, initial_eps, final_eps, eps_decay_steps, gamma,
                 learning_rate, action_space, obs_space):
        """

        Args:
            initial_eps: Initial value for exploring new actions
            final_eps: Final probability to explore instead of exploit
            eps_decay_steps: the steps required to get from the initial value to the final value
            gamma: the discount value
            learning_rate: the learning rate
            action_space: the dimension/number of the action space (the columns of our Q-Table)
            obs_space: the dimensions/number of our dimension space (the rows of our Q-Table)
        """
        self.eps = initial_eps
        self.init_eps = initial_eps
        self.fin_eps = final_eps
        self.eps_decay = eps_decay_steps
        self.gamma = gamma
        self.lr = learning_rate

        self.q_table = np.zeros((obs_space, action_space))

        self.reward_all_episodes = []

    def act(self, state, env):
        """
        A simple method that decides if the agent should explore or exploit, selects the appropriate action (i.e.
        either by chance or by selecting the currently best value for the state) and executes this Args: state: the
        current state we are in env: the current environment we are training on

        Returns: the new state, the reward for our action, that action, and if we are done

        """
        if random.uniform(0, 1) < self.eps:
            action = env.action_space.sample()
        else:
            action = np.argmax(self.q_table[state, :])
        new_state, reward, done, trunct, _ = env.step(action)

        return new_state, reward, done, action

    def update_q_table(self, state, action, reward, new_state):
        """
        The core method for our agent: the method that updates the Q-table according to the Bellman-equation. After
        we executed an action in a given state, we update that value of our table with the equation that considers
        the immediate reward and the reward for the next state. Doing this iteratively to converge towards an optimal
        policy Args: state: the state we were in action: the action we took in that state reward: the reward this
        gave us new_state: the new state we moved towards with our action


        """
        self.q_table[state, action] = self.q_table[state, action] * (1 - self.lr) + self.lr * (
                reward + self.gamma * np.max(self.q_table[new_state, :]))

    def exploration_decay(self, total_steps):
        """
        A simple method to update the exploration probability
        Args:
            total_steps: steps done so far in our entire Training

        Returns:

        """
        self.eps = np.interp(total_steps, [0, self.eps_decay], [self.init_eps, self.fin_eps])
