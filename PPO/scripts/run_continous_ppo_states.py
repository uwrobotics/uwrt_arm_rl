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

import glob

import time
import math
import random
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import collections
from collections import namedtuple
from collections import deque

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

from algorithms.ActorCritic_states import ActorCritic
from utils.CollectTrajectories import collect_trajectories
from utils.CalcReturns import calc_returns

def concat_all(v):
    # print(v.shape)
    if len(v.shape) == 3:  # actions
        return v.reshape([-1, v.shape[-1]])
    if len(v.shape) == 5:  # states
        v = v.reshape([-1, v.shape[-3], v.shape[-2], v.shape[-1]])
        # print(v.shape)
        return v
    return v.reshape([-1])

def run_training_loop(env, policy, optimizer,
                      BATCHSIZE, n_obs, n_actions,
                      EPSILON, BETA,
                      device):
    ####################
    # PARAMS
    ####################
    writer = SummaryWriter()
    best_mean_reward = None
    scores_window = deque(maxlen=100)
    save_scores = []
    start_time = timeit.default_timer()

    for episode_idx in range(EPISODES):
        policy.eval()

        old_probs_lst, states_lst, actions_lst, rewards_lst, values_lst, dones_list = \
            collect_trajectories(env, policy, writer,
                                     n_actions, n_obs, V_MAX,
                                     episode_idx=episode_idx, TIMESTEPS=TIMESTEPS, nrand=5,
                                     device=device)

        episode_score = rewards_lst.sum(dim=0).item()
        scores_window.append(episode_score)
        save_scores.append(episode_score)

        gea, target_value = calc_returns(rewards=rewards_lst,
                                         values=values_lst,
                                         dones=dones_list,
                                         TAU=TAU, DISCOUNT=DISCOUNT,
                                         device=device)
        gea = (gea - gea.mean()) / (gea.std() + 1e-8)

        policy.train()

        # cat all agents
        old_probs_lst = concat_all(old_probs_lst)
        # states_lst = concat_all(states_lst)
        actions_lst = concat_all(actions_lst)
        rewards_lst = concat_all(rewards_lst)
        values_lst = concat_all(values_lst)
        gea = concat_all(gea)
        target_value = concat_all(target_value)

        # gradient ascent step
        n_sample = len(old_probs_lst) // BATCHSIZE
        idx = np.arange(len(old_probs_lst))
        np.random.shuffle(idx)
        for epoch in range(OPT_EPOCH):
            for b in range(n_sample):
                ind = idx[b * BATCHSIZE:(b + 1) * BATCHSIZE]
                g = gea[ind]
                tv = target_value[ind]
                actions = actions_lst[ind]
                old_probs = old_probs_lst[ind]

                action_est, values = policy(states_lst[ind].view(BATCHSIZE, 1, -1))
                sigma = nn.Parameter(torch.zeros(n_actions))
                dist = torch.distributions.Normal(action_est, F.softplus(sigma).to(device))
                log_probs = dist.log_prob(actions)
                log_probs = torch.sum(log_probs, dim=-1)
                entropy = torch.sum(dist.entropy(), dim=-1)

                ratio = torch.exp(log_probs - old_probs)
                ratio_clipped = torch.clamp(ratio, 1 - EPSILON, 1 + EPSILON)
                L_CLIP = torch.mean(torch.min(ratio * g, ratio_clipped * g))
                # entropy bonus
                S = entropy.mean()
                # squared-error value function loss
                L_VF = 0.5 * (tv - values).pow(2).mean()
                # clipped surrogate
                L = -(L_CLIP - L_VF + BETA * S)
                optimizer.zero_grad()
                # This may need retain_graph=True on the backward pass
                # as pytorch automatically frees the computational graph after
                # the backward pass to save memory
                # Without this, the chain of derivative may get lost
                L.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
                optimizer.step()
                del (L)

            # the clipping parameter reduces as time goes on
        EPSILON *= .999

        # the regulation term also reduces
        # this reduces exploration in later runs
        BETA *= .998

        mean_reward = np.mean(scores_window)
        if episode_idx % 10 == 0:
            writer.add_scalar("MEAN REWARD", mean_reward, episode_idx)
            writer.add_scalar("EPSILON", EPSILON, episode_idx)
            writer.add_scalar("BETA", BETA, episode_idx)
        # display some progress every n iterations
        if best_mean_reward is None or best_mean_reward < mean_reward:
            # For saving the model and possibly resuming training
            torch.save({
                'policy_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'EPSILON': EPSILON,
                'BETA': BETA
            }, PATH)
            if best_mean_reward is not None:
                print("Best mean reward updated %.3f -> %.3f, model saved" % (best_mean_reward, mean_reward))
                writer.add_scalar("BEST MEAN REWARD", best_mean_reward, episode_idx)
            best_mean_reward = mean_reward
        if episode_idx >= 25 and mean_reward > 50:
            print('Environment solved in {:d} seasons!\tAverage Score: {:.2f}'.format(episode_idx + 1, mean_reward))
            break

    print('Episode: {}, Ave Reward: {:.2f}'.format(episode_idx, mean_reward))
    elapsed = timeit.default_timer() - start_time
    print("Elapsed time: {}".format(timedelta(seconds=elapsed)))
    writer.close()
    env.close()


def main():

    ####################
    # ENV CONFIG
    ####################
    env = gym.make('UWRTArm-v0', timesteps=TIMESTEPS, discrete=False, render=False)
    env.cid = p.connect(p.DIRECT)
    uwrt_arm_state = env.reset()

    n_obs = env.obs_dim         # = 3 or x,y,z of end effector
    n_actions = env.action_dim  # = 6 or joint angles

    ####################
    # GPU CONFIG
    ####################
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print('\n************ Using device:{} ************\n'.format(device))

    ####################
    # PPO PARAMS
    ####################
    channels = [1, 16, 32]
    kernel_sizes = [1, 1, 1]
    strides = [1, 1, 1]

    ####################
    # init DQN
    ####################

    policy = ActorCritic(in_channel=channels[0], hidden_channels=channels[0:-1],
                         kernel_sizes = kernel_sizes, strides = strides,
                         observation_space_h = 1, observation_space_w = n_obs, action_space = n_actions,
                         shared_layers = [128, 64],
                         critic_hidden_layers = [64],
                         actor_hidden_layers = [64],
                         init_type = 'xavier-uniform',
                         seed = 0).to(device)

    ####################
    optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)

    #####################
    # LOAD CHECKPOINT
    #####################
    if LOAD_PRETRAINED_WEIGHTS:
        checkpoint = torch.load(SAVED_PATH)
        policy.load_state_dict(checkpoint['policy_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        EPSILON = checkpoint['EPSILON']
        BETA = checkpoint['BETA']
    else:
        EPSILON = 0.07
        BETA = .01

    ############################
    # test policy network
    ############################
    # uwrt_arm_state = env.reset()
    # action = env.action_space.sample()
    # uwrt_arm_state, reward, done, _ = env.step(action)
    # observation = torch.tensor(uwrt_arm_state, dtype=torch.float).view(1, -1, n_obs).cuda()
    # action_est, values = policy(observation)

    # ####################
    # # training loop
    # ####################
    run_training_loop(env, policy, optimizer,
                      BATCHSIZE, n_obs, n_actions,
                      EPSILON, BETA,
                      device)

    ####################
    env.close()

if __name__ == '__main__':


    #########################
    # HYPER PARAMS
    #########################

    # env
    EPISODES = int(1e3)
    TIMESTEPS = int(2e4)
    V_MAX = 1

    OPT_EPOCH = 10
    LEARNING_RATE = 2e-4
    BATCHSIZE = 128

    # PPO
    TAU = 0.95
    DISCOUNT = 0.993

    #########################
    # saved weights
    #########################
    weights = sorted(glob.glob(ROOT_DIR + '/PPO/trained_models/policy_ppo_states_*'))
    num_weights = len(weights)
    if num_weights >= 1:
        print("*** Loading Pretrained Weights ***")
        SAVED_PATH = ROOT_DIR + "/PPO/trained_models/policy_ppo_states_" + np.str(num_weights - 1) + "_" + np.str(EPISODES) + ".pt"
        PATH = ROOT_DIR + "/PPO/trained_models/policy_ppo_states_" + np.str(num_weights) + "_" + np.str(EPISODES) + ".pt"
        LOAD_PRETRAINED_WEIGHTS = True
    else:
        PATH = ROOT_DIR + "/PPO/trained_models/policy_ppo_states_" + np.str(num_weights) + "_" + np.str(EPISODES) + ".pt"
        LOAD_PRETRAINED_WEIGHTS = False

    #########################
    # main
    #########################
    main()


