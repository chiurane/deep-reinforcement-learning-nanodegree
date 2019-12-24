"""The Ornstein-Ohlenbeck Process is used by the
Agent as an exploration policy during training.
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

class OUNoise(object):
    """ Ornstein-Uhlenbeck Process."""

    def __init__(self, size, seed=0, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process.
        Params
        =====
        size (int): size of the space
        seed (int): seed value
        mu: mean
        theta: theta
        sigma: sigma
        """
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.size = size
        self.reset()

    def reset(self):
        """ Reset the internal state (= noise) to mean (mu)"""
        self.state = copy.copy(self.mu)

    def sample(self):
        """
        Update internal state and return it as a noise sample
        """
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state