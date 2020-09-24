############
# init cwd
############
import os
import sys
ROOT_DIR = os.path.abspath("../../")
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

from algorithms.DQN_screens import DQN
from utils.ReplayMemory import ReplayMemory
from utils.SelectAction import select_action
from utils.OptimizeModel import optimize_model

def get_screen(env, device):

    preprocess = T.Compose([T.ToPILImage(),
                            T.Grayscale(num_output_channels=1),
                            T.Resize(40, interpolation=Image.CUBIC),
                            T.ToTensor()])

    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env._get_observation().transpose((2, 0, 1))
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)

    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return preprocess(screen).unsqueeze(0).to(device)

def main():

    ####################
    # ENV CONFIG
    ####################
    env = gym.make('UWRTArm-v0', timesteps=TIMESTEPS, discrete=True, render=True)
    env.cid = p.connect(p.DIRECT)

    n_obs = env.obs_dim         # = 3 or x,y,z of end effector
    n_actions = 6                # = 6 [+/- dx, +/- dy, +/- dz]

    scores_window = collections.deque(maxlen=TIMESTEPS)  # last 100 scores

    ####################
    # GPU CONFIG
    ####################
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print('\n************ Using device:{} ************\n'.format(device))

    ##########################
    # test get_screen()
    ##########################
    # env.reset()
    # plt.figure()
    # plt.imshow(get_screen(env, device).cpu().squeeze(0)[-1].numpy(), cmap='Greys',
    #            interpolation='none')
    # plt.title('Example extracted screen')
    # plt.show()

    init_screen = get_screen(env, device)
    _, _, SCREEN_HEIGHT, SCREEN_WIDTH = init_screen.shape

    ####################
    # DQN PARAMS
    ####################

    channels = [STACK_SIZE, 32, 64, 512]
    kernel_sizes = [8, 4, 3]
    strides = [4, 2, 1]

    ####################
    # init DQN
    ####################

    policy_net = DQN(in_channel=channels[0], hidden_channels=channels[1:-1],  out_channel=channels[-1],
                     kernel_sizes=kernel_sizes, strides=strides,
                     observation_space_h=SCREEN_HEIGHT, observation_space_w=SCREEN_WIDTH,
                     action_space=n_actions).to(device=device) # move the model parameters to CPU/GPU

    # load the model
    checkpoint = torch.load(SAVED_PATH)
    policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    policy_net.eval()

    for param in policy_net.parameters():
        param.requires_grad = False

    ####################
    ####################

    for i_episode in range(EPISODES):

        ###############################
        ###############################
        uwrt_arm_state = env.reset()
        state = get_screen(env, device)
        stacked_states = collections.deque(STACK_SIZE * [state], maxlen=STACK_SIZE)

        for t in range(TIMESTEPS):

            stacked_states_t = torch.cat(tuple(stacked_states), dim=1)
            # Select and perform an action
            action = policy_net(stacked_states_t).max(1)[1].view(1, 1)
            _, reward, done, _ = env.step(action.item())
            # Observe new state
            next_state = get_screen(env, device)
            stacked_states.append(next_state)

            print("Reward: ", reward, "Action: ", action.item(),
                  "\n", policy_net(stacked_states_t))

            if done:
                break
        print("Episode: {0:d}, reward: {1}".format(i_episode + 1, reward), end="\n")

    env.close()

if __name__ == '__main__':

    #########################
    # HYPER PARAMS
    #########################
    EPISODES = 10
    TIMESTEPS = int(2e3)

    STACK_SIZE = 5

    SAVED_PATH = '../trained_models/policy_dqn_screens_1000.pt'
    #########################
    # main
    #########################
    main()


