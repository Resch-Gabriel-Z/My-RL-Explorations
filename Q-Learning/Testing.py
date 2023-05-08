import gym
import numpy as np
import pandas as pd

game_name = 'Taxi-v3'
env = gym.make(game_name, render_mode='human')

path_of_model = 'Q_Learning_trained_model/'
name_of_environment = 'Taxi'

number_of_episodes_playing = 100
number_of_max_steps = 200
episode_reward = 0

# Load the q_table
q_table = pd.read_csv(f'{path_of_model}{name_of_environment}_qtable.csv')
q_table = q_table.to_numpy()
q_table = q_table[:,1:]


for episode in range(number_of_episodes_playing):
    state, _ = env.reset()
    total_steps = 0
    done = False
    while number_of_episodes_playing > 0:
        action = np.argmax(q_table[state,:])
        new_state, reward, done, trunct, _ = env.step(action)
        total_steps+=1
        episode_reward+=reward
        state = new_state

        if done or trunct or total_steps > number_of_max_steps:
            number_of_episodes_playing -= 1
            print(f'{"~" * 50}\n'
                  f'Episode: {100 - number_of_episodes_playing}\n'
                  f'reward: {episode_reward}\n')
            episode_reward = 0
            total_steps = 0
            state, _ = env.reset()