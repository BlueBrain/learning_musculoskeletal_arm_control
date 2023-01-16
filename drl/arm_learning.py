#!/usr/bin/env python

"""arm_learning.py: Source code of the model learning on 2D arm control of musculoskeletal systems

This module demonstrates how to use a gym musculoskeletal environment to learn a model to learn a user defined trajectory mimicking

Example:
    You can directly execute with python command ::
        $ python arm_learning.py

It saves the best models every user-defined steps for comparison
"""

__author__ = "Berat Denizdurduran"
__copyright__ = "Copyright 2022, Berat Denizdurduran"
__license__ = "public, published"
__version__ = "1.0.0"
__email__ = "berat.denizdurduran@alpineintuition.ch"
__status__ = "After-publication"

import math
import random
import sys

import os
from arm_files.arm_musculo import Arm2DVecEnv, Arm2DEnv

import numpy as np
import gym

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import LogNormal

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available()
print(use_cuda)
device   = torch.device("cuda" if use_cuda else "cpu")

from pathlib import Path
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from multiprocessing_env import SubprocVecEnv

num_envs = 16

def make_env():
    def _thunk():
        env = Arm2DVecEnv(visualize=False)
        return env
    return _thunk

envs = [make_env() for i in range(num_envs)]
envs = SubprocVecEnv(envs)

env = Arm2DVecEnv(visualize=False)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.01)
        nn.init.constant_(m.bias, 0.1)

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.PReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.PReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.PReLU(),
            nn.Linear(hidden_size, 1),
        )

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, num_outputs),
            nn.Tanh(),
            nn.Threshold(0.0, 0.0)
        )
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std).data.squeeze()

        self.apply(init_weights)

    def forward(self, x):
        value = self.critic(x)
        mu    = self.actor(x)
        std   = self.log_std.exp().expand_as(mu)
        std = std.to(device)
        dist  = Normal(mu, std*0.1)
        return dist, value

def plot(frame_idx, rewards):
    plt.figure(figsize=(12,8))
    plt.subplot(111)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.savefig("results/arm_ppo_{}".format(frame_idx))
    plt.close()

def test_env(num_steps):
    state = env.reset()
    done = False
    total_reward = 0
    for i in range(num_steps):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, _ = model_musculo(state)
        action = dist.sample().cpu().numpy()[0]
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward -= reward
    envs.reset()
    return total_reward

def compute_gae(next_value, rewards, masks, values, gamma=0.9, tau=0.99):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns

def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
    batch_size = states.size(0)
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]

def ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2):
    for _ in range(ppo_epochs):
        for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
            dist, value = model_musculo(state)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

            actor_loss  = - torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()

            loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

            optimizer_musculo.zero_grad()
            loss.backward()
            optimizer_musculo.step()


num_inputs  = 14#envs.observation_space.shape[0]
num_outputs = 14#envs.action_space.shape[0]

state = envs.reset()

#Hyper params:
hidden_size      = 32
lr               = 3e-4
betas            = (0.9, 0.999)
eps              = 1e-08
weight_decay     = 0.001
num_steps        = 75
mini_batch_size  = 150
ppo_epochs       = 150
threshold_reward = -200

model_musculo = ActorCritic(num_inputs, num_outputs, hidden_size).to(device)
optimizer_musculo = optim.Adam(model_musculo.parameters(), lr=lr)

frame_idx  = 0
test_rewards = []

# To continue learning from user defined checkpoint uncomment following lines

#model_id = 65400
#ppo_model_arm_loaded = torch.load("results/ppo_model_arm_musculo_{}".format(model_id))

#model_musculo.load_state_dict(ppo_model_arm_loaded['model_state_dict'])
#optimizer_musculo.load_state_dict(ppo_model_arm_loaded['optimizer_state_dict'])
#frame_idx = ppo_model_arm_loaded['epoch']
#test_rewards = ppo_model_arm_loaded['loss']


range_steps = 50000

for steps in range(range_steps):

    log_probs = []
    values    = []
    states    = []
    actions   = []
    rewards   = []
    masks     = []
    entropy = 0

    for _ in range(num_steps):
        state = torch.FloatTensor(state).to(device)
        dist, value = model_musculo(state)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        actions.append(action)
        action = action.cpu().numpy()

        next_state, reward, done, _ = envs.step(action)

        entropy += dist.entropy().mean()

        log_probs.append(log_prob)
        values.append(value)
        rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
        masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))
        states.append(state)

        state = next_state
        frame_idx += 1

        if frame_idx % 100 == 0:
            test_reward = np.mean([test_env(num_steps) for _ in range(10)])
            test_rewards.append(test_reward)
            plot(frame_idx, test_rewards)
            print("Iter: {} - Testing, Current Error: {}".format(frame_idx, test_reward))

        if frame_idx % 100 == 0:
            print("Iter: {} - Saving model".format(frame_idx))
            ppo_model_arm_musculo = {
                        'epoch': frame_idx,
                        'model_state_dict': model_musculo.state_dict(),
                        'optimizer_state_dict': optimizer_musculo.state_dict(),
                        'loss': test_rewards}
            torch.save(ppo_model_arm_musculo, "results/ppo_model_arm_musculo_{}".format(frame_idx))

    envs.reset()

    next_state = torch.FloatTensor(next_state).to(device)
    _, next_value = model_musculo(next_state)
    returns = compute_gae(next_value, rewards, masks, values)

    returns   = torch.cat(returns).detach()
    log_probs = torch.cat(log_probs).detach()
    values    = torch.cat(values).detach()
    states    = torch.cat(states)

    actions   = torch.cat(actions)
    advantage = returns - values

    ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantage)

input('Learning finished! Press any key to exit.')
