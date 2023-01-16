#!/usr/bin/env python

"""arm_testing.py: Source code of the model testing on 2D arm control project

This module demonstrates how to use a gym musculoskeletal environment to test a learned model

Example:
    You can directly execute with a python command ::
        $ python arm_testing.py -mi 48000 -c 10 -ns 75

It visulizes the activity of the musculoskeletal system with user defined learned model
Options:
  -mi       --model_id            Model ID
  -c        --counter             Number of Tests
  -ns       --num_steps           Number of Steps
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

import matplotlib.pyplot as plt

import argparse

use_cuda = torch.cuda.is_available()
print(use_cuda)
device   = torch.device("cuda" if use_cuda else "cpu")

from pathlib import Path
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from multiprocessing_env import SubprocVecEnv

num_envs = 1

def make_env():
    def _thunk():
        env = Arm2DVecEnv(visualize=False)
        return env

    return _thunk

envs = [make_env() for i in range(num_envs)]
envs = SubprocVecEnv(envs)

env = Arm2DVecEnv(visualize=True)

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
    plt.savefig("results/arm_ppo_test_{}".format(frame_idx))
    plt.close()


def test_env(num_steps, count):
    state = env.reset()
    target_shoulder =  np.load(os.path.join(os.path.dirname(__file__), "../mpc/results/target_of_elbow_fixed_theta0.npy"))
    target_elbow =  np.load(os.path.join(os.path.dirname(__file__), "../mpc/results/target_of_shoulder_fixed_theta0.npy"))

    state_shoulder = []
    state_elbow = []
    done = False
    total_reward = 0
    total_error = []
    #input("Video")

    for i in range(num_steps):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, _ = model_musculo(state)
        action = dist.sample().cpu().numpy()[0]
        next_state, reward, done, _ = env.step(action)

        positions = env.get_positions()
        state_shoulder.append(positions[0])
        state_elbow.append(positions[1])
        total_error.append(reward)

        state = next_state
        total_reward += reward

    plt.plot(target_elbow)
    plt.plot(state_elbow)
    plt.plot(target_shoulder)
    plt.plot(state_shoulder)
    plt.savefig("results/arm_ppo_states_all_musculo_{}_{}".format(model_id, count))
    plt.close()
    np.save("results/state_elbow_test_{}".format(count), state_elbow)
    np.save("results/state_shoulder_test_{}".format(count), state_shoulder)
    np.save("results/total_error_test_{}".format(count), total_error)
    envs.reset()
    return total_reward



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-mi", "--model_id", type=int, default=300, help="Model ID")
    parser.add_argument("-c", "--counter", type=int, default=1, help="number of tests")
    parser.add_argument("-ns", "--num_steps", type=int, default=75, help="number of steps")
    args = parser.parse_args()

    num_inputs  = 14#envs.observation_space.shape[0]
    num_outputs = 14#envs.action_space.shape[0]

    state = envs.reset()
    #Hyper params:
    hidden_size      = 32
    lr               = 3e-4
    betas            = (0.9, 0.999)
    eps              = 1e-08
    weight_decay     = 0.001
    mini_batch_size  = 200
    ppo_epochs       = 200
    threshold_reward = -200

    model_musculo = ActorCritic(num_inputs, num_outputs, hidden_size).to(device)
    optimizer_musculo = optim.Adam(model_musculo.parameters(), lr=lr)

    model_id = args.model_id
    counter = args.counter
    num_steps = args.num_steps

    ppo_model_arm_musculo_loaded = torch.load("results/ppo_model_arm_musculo_{}".format(model_id), map_location=device)

    model_musculo.load_state_dict(ppo_model_arm_musculo_loaded['model_state_dict'])
    optimizer_musculo.load_state_dict(ppo_model_arm_musculo_loaded['optimizer_state_dict'])

    frame_idx = ppo_model_arm_musculo_loaded['epoch']
    test_rewards = ppo_model_arm_musculo_loaded['loss']

    test_all_rewards = np.array([test_env(num_steps, i) for i in range(counter)])
