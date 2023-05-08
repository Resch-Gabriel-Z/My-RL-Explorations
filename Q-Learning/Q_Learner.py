import random

import numpy as np


class QLearner:
    def __init__(self, initial_eps, final_eps, eps_decay_steps, gamma,
                 learning_rate, action_space, obs_space):
        self.eps = initial_eps
        self.init_eps = initial_eps
        self.fin_eps = final_eps
        self.eps_decay = eps_decay_steps
        self.gamma = gamma
        self.lr = learning_rate

        self.q_table = np.zeros((obs_space, action_space))

        self.reward_all_episodes = []

    def act(self, state, env):
        if random.uniform(0, 1) < self.eps:
            action = env.action_space.sample()
        else:
            action = np.argmax(self.q_table[state, :])
        new_state, reward, done, trunct, _ = env.step(action)

        return new_state, reward, done, action

    def update_q_table(self, state, action, reward, new_state):
        self.q_table[state, action] = self.q_table[state, action] * (1 - self.lr) + self.lr * (
                    reward + self.gamma * np.max(self.q_table[new_state, :]))

    def exploration_decay(self, total_steps):
        self.eps = np.interp(total_steps, [0, self.eps_decay], [self.init_eps, self.fin_eps])
