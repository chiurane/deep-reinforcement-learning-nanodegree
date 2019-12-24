"""
The DDPG Agent interacts with the environment and learns from it. It uses the Actor-Critic network architecture,
the OUNoise and Replay Buffer to aid learning
"""

import numpy as np
import random
from replaybuffer import ReplayBuffer
from ounoise import OUNoise
from hyperparams import *
from model import Actor, Critic
import copy
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


class Agent(object):
    """
    The Agent interacts with and learns from the environment.
    """

    def __init__(self, state_size, action_size, num_agents, random_seed=0, params=params):
        """
        Initialize an Agent object.
        Params
        ======
        state_size (int): dimension of each state
        action_size (int): dimension of each action
        num_agents (int): number of agents
        random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = random.seed(random_seed)
        self.params = params

        # Actor (Policy) Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(self.params['DEVICE'])
        self.actor_target = Actor(state_size, action_size, random_seed).to(self.params['DEVICE'])
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.params['LR_ACTOR'])

        # Critic (Value) Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(self.params['DEVICE'])
        self.critic_target = Critic(state_size, action_size, random_seed).to(self.params['DEVICE'])
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.params['LR_CRITIC'],
                                           weight_decay=self.params['WEIGHT_DECAY'])

        # Initialize target and local to same weights
        self.hard_update(self.actor_local, self.actor_target)
        self.hard_update(self.critic_local, self.critic_target)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, self.params['BUFFER_SIZE'], self.params['BATCH_SIZE'], random_seed)

    def hard_update(self, local_model, target_model):
        """
        Hard update model parameters.
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)

    def step(self, states, actions, rewards, next_states, dones):
        """
        Save experiences in replay memory and use random sample from buffer to learn.
        """

        # Save experience / reward, cater for when multiples
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done)

        # Learn if enough samples are available in memory
        if len(self.memory) > self.params['BATCH_SIZE']:
            experiences = self.memory.sample()
            self.learn(experiences, self.params['GAMMA'])

    def act(self, states, add_noise=True):
        """
        Returns actions for a given state as per current policy.
        """
        states = torch.from_numpy(states).float().to(self.params['DEVICE'])
        actions = np.zeros((self.num_agents, self.action_size))
        self.actor_local.eval()
        with torch.no_grad():
            for i, state in enumerate(states):
                actions[i, :] = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            actions += self.noise.sample()
        return np.clip(actions, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma=params['GAMMA']):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Update Critic(Value)
        # Get predicted next-state actions and Q-Values from target Network
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)

        # Compute Q Targe for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Minimise the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(),
                                       1)  # Stabilize learning per bernchmark guidelines
        self.critic_optimizer.step()

        # Update Actor (Policy)
        # Compute Actor Loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        self.soft_update(self.critic_local, self.critic_target, tau=self.params['TAU'])
        self.soft_update(self.actor_local, self.actor_target, tau=self.params['TAU'])

    def soft_update(self, local_model, target_model, tau=params['TAU']):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
