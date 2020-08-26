############
# init cwd
############
import os
import sys
ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR)
print("********* cwd {} *********".format(ROOT_DIR))

### bug with MacOS
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import time
import math
import random
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import collections
from collections import namedtuple
import timeit
from datetime import timedelta
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from tensorboardX import SummaryWriter

import pybullet as p

import gym
import uwrtarm_gym
from gym import spaces

from algorithms.DQN import DQN
from utils.ReplayMemory import ReplayMemory
from utils.SelectAction import select_action
from utils.OptimizeModel import optimize_model

def run_training_loop(env, policy_net, target_net, optimizer,
                      memory, BATCHSIZE, n_obs, n_actions,
                      device):
    ####################
    # PARAMS
    ####################
    writer = SummaryWriter()
    total_rewards = []
    ten_rewards = 0
    best_mean_reward = None
    start_time = timeit.default_timer()

    for episode_idx in range(EPISODES):

        ###############################
        # actual and target position
        ###############################
        uwrt_arm_state = env.reset()
        keyboard_position, keyboard_orientation = env.keyboard_position, env.keyboard_orientation

        for timestep_idx in range(TIMESTEPS):
            env.render()
            # get action
            observation = torch.tensor(uwrt_arm_state).view(1, -1, n_obs)
            action = select_action(env, policy_net, observation, n_actions,
                                   EPS_START, EPS_END, EPS_DECAY_LAST_FRAME,
                                   device=device)

            # get reward and next observation
            uwrt_arm_state, reward, done, _ = env.step(np.squeeze(action.numpy()))
            reward = torch.tensor([reward], device=device)
            next_observation = torch.tensor(uwrt_arm_state).view(1, -1, n_obs)

            # Store the transition in memory
            ### ('state', 'action', 'next_state', 'reward')
            memory.push(observation, action, next_observation, reward)

            # Move to the next state
            observation = next_observation

            # Perform one step of the optimization (on the target network)
            optimize_model(policy_net, target_net, optimizer,
                           memory, BATCHSIZE, n_obs, n_actions, GAMMA,
                           device)

            #######################
            if done:
                reward = reward.cpu().numpy().item()
                ten_rewards += reward
                total_rewards.append(reward)
                mean_reward = np.mean(total_rewards[-100:]) * 100
                if (best_mean_reward is None or best_mean_reward < mean_reward) and episode_idx > 100:
                    # For saving the model and possibly resuming training
                    torch.save({
                        'policy_net_state_dict': policy_net.state_dict(),
                        'target_net_state_dict': target_net.state_dict(),
                        'optimizer_policy_net_state_dict': optimizer.state_dict()
                    }, PATH)
                    if best_mean_reward is not None:
                        print("Best mean reward updated %.1f -> %.1f, model saved" % (best_mean_reward, mean_reward))
                    best_mean_reward = mean_reward
                break

            #####################################################################
            # Update the target network, copying all weights and biases in DQN
            #####################################################################
            if episode_idx % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())
            if episode_idx >= 200 and mean_reward > 50:
                print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode_idx + 1, mean_reward))
                break

        ######################
        print('Average Score: {:.2f}'.format(mean_reward))
        elapsed = timeit.default_timer() - start_time
        print("Elapsed time: {}".format(timedelta(seconds=elapsed)))
        writer.close()

def main():

    ####################
    # ENV CONFIG
    ####################
    env = gym.make('UWRTArm-v0', timesteps=TIMESTEPS)

    env.cid = p.connect(p.DIRECT)
    uwrt_arm_state = env.reset()
    keyboard_position, keyboard_orientation = env.keyboard_position, env.keyboard_orientation
    # time.sleep(5) #### display env

    n_obs = env.obs_dim         # = 3 or x,y,z of end effector
    n_actions = env.action_dim  # = 6 or joint angles

    ####################
    # GPU CONFIG
    ####################
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print('Using device:{}'.format(device))

    ####################
    # DQN PARAMS
    ####################
    channels = [1, 32, 64, 512]
    kernel_sizes = [1, 1, 1]
    strides = [1, 1, 1]

    ####################
    # init DQN
    ####################

    policy_net = DQN(in_channel=channels[0], hidden_channels=channels[1:-1],  out_channel=channels[-1],
                     kernel_sizes=kernel_sizes, strides=strides,
                     observation_space_h=1, observation_space_w=n_obs,
                     action_space=n_actions).to(device=device) # move the model parameters to CPU/GPU
    target_net = DQN(in_channel=channels[0], hidden_channels=channels[1:-1],  out_channel=channels[-1],
                     kernel_sizes=kernel_sizes, strides=strides,
                     observation_space_h=1, observation_space_w=n_obs,
                     action_space=n_actions).to(device=device) # move the model parameters to CPU/GPU

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    ####################
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayMemory(REPLAYBUFFER)

    ####################
    # test select action
    ####################
    # observation = torch.tensor(uwrt_arm_state).view(1, -1, n_obs)
    # action = select_action(env, policy_net, observation, n_actions,
    #                       EPS_START, EPS_END, EPS_DECAY_LAST_FRAME,
    #                       device=device)

    ####################
    # training loop
    ####################
    run_training_loop(env, policy_net, target_net, optimizer,
                      memory, BATCHSIZE, n_obs, n_actions,
                      device)

    ####################
    env.close()

if __name__ == '__main__':

    PATH = "../trained_models/policy_dqn.pt"
    #########################
    # HYPER PARAMS
    #########################
    EPISODES = 100
    TIMESTEPS = 1000

    LEARNING_RATE = 1e-3
    BATCHSIZE = 32

    REPLAYBUFFER = 1e4

    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.1
    EPS_DECAY = 200
    EPS_DECAY_LAST_FRAME = 10 ** 4
    TARGET_UPDATE = 1000
    EPS_THRESHOLD = 0

    #########################
    # main
    #########################
    main()


