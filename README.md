# My Reinforcement Learning Explorations

This repository contains a collection of reinforcement learning algorithms implemented in Python. Each algorithm is
implemented by several scripts, set apart by their purpose, accompanied by trained models and media (such as gifs and
pictures) to demonstrate the Results.

# Introduction

In this repository, I explore various reinforcement learning algorithms and provide their implementations for
educational purposes. The primary goal is to create clean and easily understandable code that might help others as well.

# Algorithms

## Value-based approach

The value-based approach is a family of RL-Algorithms where the agent tries to maximize a _Value-function_.
Which is essentially a function that estimates cumulative rewards, for taking a certain action.

- Q-learning: The most basic algorithm, that also is the fundament of many other Algorithms in this family. It uses the
  Bellmann equation, to update a created table that maps a state to an action with the expected reward. From that we can
  always choose the action with the highest Value.
- DQN: The Core Idea of this algorithm builds upon Q-Learning, but it uses Deep Neural Networks and a Memory to learn
  the optimal Q-Values off-policy
- DDQN: the Double Deep Q-Learning Network is similar to DQN, but it uses a secondary Network to stabilize the Results.

## Policy-Gradient

Algorithms in this family, doesn't try to learn a Q-Value but rather updates the policies directly.

It works by directly computing the gradient of the expected cumulative reward and then performing gradient ascent to
update the Network Parameters towards a higher expected Reward.

Algorithms that belongs to PG typically choose their action therefore not from this Q-Value function, which we doesn't
create, but from the Policy directly, by creating a Probability Distribution that chooses the best action with high
probability.

- VPG: or REINFORCE, is the start-point for many people who want to learn Policy Gradient. It uses a Probability
  Distribution, that is shaped by the Network, to select an action in the environment. With the formula by the policy
  gradient theorem, we update the Network such that the shape of the Probability distribution will get high cumulative
  rewards.
- PPO-Clip: in PPO-Clip (in this implementation with an Actor-Critic Model) we act similar to VPG but also include a
  clip such that the updated Network will not differ to much from the original Network which should result in a more
  stable Network.

# Installation & Usage

Feel free to download this repository and change it however you want, I didn't try to make a software out of it that
requires special UI, or something like that.

Its really just a bunch of scripts in Python. If you just want to Train/Test out the Algorithms for yourself, you can
easily do this by:

1. Clone the repository to your local machine:

```bash
git clone https://github.com/Resch-Gabriel-Z/My-RL-Explorations
```

2. Change the meta-data to appropriate Data
3. Run the Training/Testing Script

# Results

## Value-based approach

### Q-Learning

![CliffWalking.png](Q-Learning%2Fmedia%2FCliffWalking.png)
![FrozenLake.png](Q-Learning%2Fmedia%2FFrozenLake.png)
![Taxi.png](Q-Learning%2Fmedia%2FTaxi.png)

### DQN

![Acrobot.png](DQN%2Fmedia%2FAcrobot.png)
![CartPole.png](DQN%2Fmedia%2FCartPole.png)
![LunarLander.png](DQN%2Fmedia%2FLunarLander.png)
![MountainCar.png](DQN%2Fmedia%2FMountainCar.png)

### DDQN

![Acrobot.png](DDQN%2Fmedia%2FAcrobot.png)
![CartPole.png](DDQN%2Fmedia%2FCartPole.png)
![LunarLander.png](DDQN%2Fmedia%2FLunarLander.png)
![MountainCar.png](DDQN%2Fmedia%2FMountainCar.png)

## Policy Gradient

## VPG

![Acrobot.png](VPG%2Fmedia%2FAcrobot.png)
![CartPole.png](VPG%2Fmedia%2FCartPole.png)
![Pendulum.png](VPG%2Fmedia%2FPendulum.png)

## PPO Clip
![CartPole.png](PPOCLIP%2Fmedia%2FCartPole.png)

