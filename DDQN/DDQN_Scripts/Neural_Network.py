import torch.nn as nn


class DQN(nn.Module):
    """

    """

    def __init__(self, in_channels, num_actions):
        """
        The usual Pytorch NN
        Args:
            in_channels: number of neurons we analyze the state in
            num_actions: number of actions the agent can take
        """
        super(DQN, self).__init__()
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

        Returns: the action

        """
        action = self.fc(x)
        return action
