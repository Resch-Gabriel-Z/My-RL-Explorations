import torch
import torch.nn as nn
from torch.distributions import Categorical


class VPGDiscrete(nn.Module):
    """

    """

    def __init__(self, in_channels, num_actions):
        """
        The usual Pytorch NN
        Args:
            in_channels: number of neurons we analyze the state in
            num_actions: number of actions the agent can take
        """
        super(VPGDiscrete, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=128),
            nn.ReLU(),

            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),

            nn.Linear(in_features=128, out_features=num_actions),
        )

    def forward(self, x):
        """
        A usual forward method
        Args:
            x: the state

        Returns: the action_probs

        """
        x = torch.as_tensor(x)
        action_probs = self.fc(x)
        action_probs = nn.functional.softmax(action_probs, dim=-1)
        return action_probs

    def select_action(self, probs):
        """
        A method to select an action with respect to the current probabilities of taking an action according to our
        policy
        Args:
            probs: the probabilities of actions

        Returns: an action

        """
        action_sampler = Categorical(probs)
        action = action_sampler.sample()
        return action.item()


class VPGContinuous(nn.Module):
    """

    """

    def __init__(self, in_channels, num_actions):
        """
        A variation of the NN above, designed for continuous action spaces.
        Since the probability in such an action space for one action is 0, we determine an interval that handles the
        selection of an action.

        As you might know, the probability in continuous spaces is influenced by the standard deviation and mean.
        Therefore, we strive to let the model learn ideal values for both of them.

        Args:
            in_channels: number of neurons we analyze the state in
            num_actions: number of actions the agent can take
        """
        super(VPGContinuous, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=16),
            nn.Tanh(),

            nn.Linear(in_features=16, out_features=32),
            nn.Tanh(),
        )

        # Policy Mean specific Linear Layer
        self.policy_mean_net = nn.Sequential(
            nn.Linear(32, num_actions)
        )

        # Policy Std Dev specific Linear Layer
        self.policy_stddev_net = nn.Sequential(
            nn.Linear(32, num_actions)
        )

    def forward(self, x):
        """
        A usual forward method
        Args:
            x: the state

        Returns: the mean and standard deviation

        """
        x = torch.as_tensor(x)
        action_probs = self.fc(x)
        mean = self.policy_mean_net(action_probs)
        std = torch.exp(self.policy_stddev_net(action_probs))
        return mean, std

    def select_action(self, mean, std):
        """
        A method to determine an action based on the mean and standard deviation we pass in applied to the
        normal distribution.
        It takes a mean and std, and draws a random number from the normal distribution with those parameters.
        As the distribution is influenced by those 2 values, it is more probable that actions are taken that are good.
        Args:
            mean: the current mean of our model
            std : the standard deviation of our model.
        Returns:
            an action

        """
        action = torch.normal(mean, std)
        return action.detach().numpy()
