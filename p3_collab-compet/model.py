"""
Actor (Policy) Model and Critic (Value) Model
The Actor-Critic Architecture is the intersection of
policy-based and value-based methods for deep reinforcement
learning. This is also the basis for both
Deterministic Policy Gradient and Deep Deterministic Policy
Gradient Methods and I plan to first implement the later to
see how that performs.
"""

import numpy as np
import random
import copy
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed=0,
                 fc_units=[400, 300]):
        """
        Initialize parameters and build model.
        Params
        ======
        state_size (int): Dimension of each state
        action_size (int): Dimension of action
        seed (int): Random seed
        fc1_units (int): Number of nodes in first hidden layer
        fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc_units[0])
        self.fc2 = nn.Linear(fc_units[0], fc_units[1])
        self.fc3 = nn.Linear(fc_units[1], action_size)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset parameters
        """
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """
        Build an Actor (Policy) network that maps states -> actions.
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x))
        return x


class Critic(nn.Module):
    """
    Critic (Value) Model.
    """

    def __init__(self, state_size, action_size, seed=0,
                 fc_units=[400, 300]):
        """
        Initialize parameters and build model.
        Params
        ======
        state_size (int): Dimension of each state
        action_size (int): Dimension of each action
        seed (int): Random seed
        fcs1_units (int): Number of nodes in first hidden layer
        fc2_units (int): Number of nodes in second hidden layer
        fc3_units (int): Number of nodes in third hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc_units[0])
        self.fc2 = nn.Linear(fc_units[0] + action_size, fc_units[1])  # because we concat later
        self.fc3 = nn.Linear(fc_units[1], 1)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset parameters
        """
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """
        Build a Critic (Value) Network that maps (state, action) pairs -> Q-values.
        """
        x = F.relu(self.fc1(state))
        x = torch.cat((x, action), dim=1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x