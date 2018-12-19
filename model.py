import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return -lim, lim


class Actor(nn.Module):

    def __init__(self, state_size, action_size, use_batch_norm, seed,
                 fc1_units=600, fc2_units=200):
        """
        :param state_size: Dimension of each state
        :param action_size: Dimension of each state
        :param seed: random seed
        :param use_batch_norm: True to use batch norm
        :param fc1_units: number of nodes in 1st hidden layer
        :param fc2_units: number of nodes in 2nd hidden layer
        """
        super(Actor, self).__init__()

        if seed is not None:
            torch.manual_seed(seed)

        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(state_size)
            self.bn2 = nn.BatchNorm1d(fc1_units)
            self.bn3 = nn.BatchNorm1d(fc2_units)

        # batch norm has bias included, disable linear layer bias
        use_bias = not use_batch_norm

        self.use_batch_norm = use_batch_norm
        self.fc1 = nn.Linear(state_size, fc1_units, bias=use_bias)
        self.fc2 = nn.Linear(fc1_units, fc2_units, bias=use_bias)
        self.fc3 = nn.Linear(fc2_units, action_size, bias=use_bias)
        self.reset_parameters()

    def forward(self, state):
        """ map a states to action values
        :param state: shape == (batch, state_size)
        :return: action values
        """

        if self.use_batch_norm:
            x = F.relu(self.fc1(self.bn1(state)))
            x = F.relu(self.fc2(self.bn2(x)))
            return torch.tanh(self.fc3(self.bn3(x)))
        else:
            x = F.leaky_relu(self.fc1(state))
            x = F.leaky_relu(self.fc2(x))
            return torch.tanh(self.fc3(x))

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)


class Critic(nn.Module):

    def __init__(self, state_size, action_size, use_batch_norm, seed,
                 fc1_units=600, fc2_units=200):
        """
        :param duel_network: boolean
        :param state_size: Dimension of each state
        :param action_size: Dimension of each state
        :param seed: random seed
        :param fc1_units: number of nodes in 1st hidden layer
        :param fc2_units: number of nodes in 2nd hidden layer
        """
        super(Critic, self).__init__()

        if seed is not None:
            torch.manual_seed(seed)

        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(state_size)

        # batch norm has bias included, disable linear layer bias
        use_bias = not use_batch_norm

        self.use_batch_norm = use_batch_norm
        self.fc1 = nn.Linear(state_size, fc1_units, bias=use_bias)
        self.fc2 = nn.Linear(fc1_units + action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def forward(self, state, action):
        """ map (states, actions) pairs to Q-values
        :param state: shape == (batch, state_size)
        :param action: shape == (batch, action_size)
        :return: q-values values
        """

        if self.use_batch_norm:
            x = F.relu(self.fc1(self.bn1(state)))
        else:
            x = F.relu(self.fc1(state))

        x = torch.cat((x, action), dim=1)
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)