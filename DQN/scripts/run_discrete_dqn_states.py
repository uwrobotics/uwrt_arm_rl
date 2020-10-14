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

from DQN.algorithms.DQN_states import DQN
from DQN.utils.ReplayMemory import ReplayMemory
from DQN.utils.SelectAction import select_action
from DQN.utils.OptimizeModel import optimize_model

def get_discrete_action(env, move_towards_target, continous_action, observation):


    '''
    dx = [0, -dv, dv, 0, 0, 0, 0][action]
    dy = [0, 0, 0, -dv, dv, 0, 0][action]
    dz = -dv
    da = [0, 0, 0, 0, 0, -0.25, 0.25][action]

    action_continuous = [dx, dy, dz, da, 0.3]
    '''

    if move_towards_target:
        # print("*** Moving towards target ***")
        sample_action = np.array(env.action_space.sample(), dtype=np.float64)
        sample_action[0:3] = np.array(env.keyboard_position) - np.squeeze(observation.cpu().numpy())
        discrete_action = sample_action[0:3]
    else:
        dv = 1 # see test_uwrt_arm_env.py
        dx = [-dv, dv, 0, 0, 0, 0][continous_action]
        dy = [0, 0, -dv, dv, 0, 0][continous_action]
        dz = [0, 0, 0, 0, -dv, dv][continous_action]

        discrete_action = np.array([dx, dy, dz])

    print("continous_action: ", continous_action)
    print("discrete_action: ", discrete_action)

    return discrete_action

def run_training_loop(env, policy_net, target_net, optimizer,
                      memory, BATCHSIZE, n_obs, n_actions,
                      device):
    ####################
    # PARAMS
    ####################
    writer = SummaryWriter()
    total_rewards = []
    ten_rewards = 0
    best_mean_reward = -np.inf
    start_time = timeit.default_timer()

    for episode_idx in range(EPISODES):

        it_timesteps = 0
        it_move_towards_target = 0

        # ###############################
        # # actual and target position
        # ###############################
        # uwrt_arm_state = env.reset()
        # state = torch.tensor(uwrt_arm_state, dtype=torch.float).view(1, -1, n_obs)
        # stacked_states = collections.deque(STACK_SIZE * [state], maxlen=STACK_SIZE)
        #
        # for timestep_idx in range(TIMESTEPS):
        #     stacked_states_t = torch.cat(tuple(stacked_states), dim=1)
        #     # select action
        #     action, move_towards_target = select_action(env, policy_net, stacked_states_t, n_actions,
        #                                                 EPS_START, EPS_END, EPS_DECAY_LAST_FRAME, episode_idx,
        #                                                 device=device)
        #     it_move_towards_target += 1 if move_towards_target == True else 0
        #
        #     # get reward and next observation
        #     uwrt_arm_state, reward, done, _ = env.step(action.item())
        #     reward = torch.tensor([reward], device=device)
        #
        #     next_state = torch.tensor(uwrt_arm_state, dtype=torch.float).view(1, -1, n_obs)
        #
        #     if not done:
        #         next_stacked_states = stacked_states
        #         next_stacked_states.append(next_state)
        #         next_stacked_states_t = torch.cat(tuple(next_stacked_states), dim=1)
        #     else:
        #         next_stacked_states_t = None
        #
        #     memory.push(stacked_states_t, action, next_stacked_states_t, reward)
        #
        #     # Move to the next state
        #     stacked_states = next_stacked_states

        ###############################
        # actual and target position
        ###############################
        uwrt_arm_state = env.reset()
        observation = torch.tensor(uwrt_arm_state, dtype=torch.float).view(-1, n_obs)

        for timestep_idx in range(TIMESTEPS):
            # select action
            action, move_towards_target = select_action(env, policy_net, observation, n_actions,
                                                        EPS_START, EPS_END, EPS_DECAY_LAST_FRAME, episode_idx,
                                                        device=device)
            it_move_towards_target += 1 if move_towards_target == True else 0

            # get reward and next observation
            uwrt_arm_state, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)


            if not done:
                next_observation = torch.tensor(uwrt_arm_state, dtype=torch.float).view(-1, n_obs)
            else:
                next_observation = None

            memory.push(observation, action, next_observation, reward)

            # Move to the next state
            observation = next_observation

            ######################
            # optimize_model
            ######################

            if len(memory) > BATCHSIZE:

                transitions = memory.sample(BATCHSIZE)
                # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
                # detailed explanation). This converts batch-array of Transitions
                # to Transition of batch-arrays.
                batch = memory.transition(*zip(*transitions))

                # Compute a mask of non-final states and concatenate the batch elements
                # (a final state would've been the one after which simulation ended)
                non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                        batch.next_state)), device=device, dtype=torch.bool)
                non_final_next_states = torch.cat([s for s in batch.next_state
                                                   if s is not None])
                state_batch = torch.cat(batch.state)
                action_batch = torch.cat(batch.action)
                reward_batch = torch.cat(batch.reward)

                # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
                # columns of actions taken. These are the actions which would've been taken
                # for each batch state according to policy_net
                state_action_values = policy_net(state_batch.cuda()).gather(1, action_batch)

                # Compute V(s_{t+1}) for all next states.
                # Expected values of actions for non_final_next_states are computed based
                # on the "older" target_net; selecting their best reward with max(1)[0].
                # This is merged based on the mask, such that we'll have either the expected
                # state value or 0 in case the state was final.
                next_state_values = torch.zeros(BATCHSIZE, device=device)
                next_state_values[non_final_mask] = target_net(non_final_next_states.cuda()).max(1)[0].detach()

                # Compute the expected Q values
                expected_state_action_values = (next_state_values * GAMMA) + reward_batch

                # next_state_values = torch.zeros(BATCHSIZE, n_actions, device=device)
                # next_state_values[non_final_mask, :] = target_net(non_final_next_states.cuda())

                # Compute Huber loss
                # loss = F.smooth_l1_loss(state_action_values, next_state_values)
                loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
                for param in policy_net.parameters():
                    param.grad.data.clamp_(-1, 1)
                optimizer.step()

            #######################
            if done:
                reward = reward.cpu().numpy().item()
                ten_rewards += reward
                total_rewards.append(reward)
                # if episode_idx > 100:
                #     mean_reward = np.mean(total_rewards[-100:]) * 100
                # else:
                mean_reward = np.mean(total_rewards[-(100):])
                if (best_mean_reward is None or best_mean_reward < mean_reward) and episode_idx > 10:
                    # For saving the model and possibly resuming training
                    torch.save({
                        'policy_net_state_dict': policy_net.state_dict(),
                        'target_net_state_dict': target_net.state_dict(),
                        'optimizer_policy_net_state_dict': optimizer.state_dict()
                    }, PATH)
                    if best_mean_reward is not None:
                        print("\n********************************\n"
                              "Best mean reward updated %.1f -> %.1f, model saved \n"
                              "********************************\n" % (best_mean_reward, mean_reward))
                        writer.add_scalar('Best Mean Reward', mean_reward, episode_idx)
                    best_mean_reward = mean_reward

                break

            # tensorboard
            if ten_rewards != 0 and episode_idx % 10 == 0:
                writer.add_scalar('Averaged Last Ten Rewards', ten_rewards / 10.0, episode_idx)
                ten_rewards = 0

            #####################################################################
            # Update the target network, copying all weights and biases in DQN
            #####################################################################
            if episode_idx % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())
            if episode_idx >= 200 and mean_reward > 1000: # TODO:
                print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode_idx + 1, mean_reward))
                break

        ######################
        print("Move Towards Target: {}/{}, TIMESTEPS: {}".format(it_move_towards_target, timestep_idx, TIMESTEPS))
        print('Reward: {:.2f}, Mean Reward: {:.2f}'.format(reward, mean_reward))
        elapsed = timeit.default_timer() - start_time
        print("Epsiode: {}/{}, Elapsed time: {}\n".format(episode_idx, EPISODES, timedelta(seconds=elapsed)))
        writer.close()

def main():

    ####################
    # ENV CONFIGw
    ####################
    env = gym.make('UWRTArm-v0', timesteps=TIMESTEPS, discrete=True, render=False)
    env.cid = p.connect(p.DIRECT)
    uwrt_arm_state = env.reset()

    n_obs = env.obs_dim         # = 3 or x,y,z of end effector
    n_actions = 6               # = 6 [+/- dx, +/- dy, +/- dz]

    ####################
    # GPU CONFIG
    ####################
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print('\n************ Using device:{} ************\n'.format(device))

    ####################
    # DQN PARAMS
    ####################

    channels = [n_obs, 32, 64, n_actions]
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

    ####################
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayMemory(REPLAYBUFFER)

    #####################
    # LOAD CHECKPOINT
    #####################
    if LOAD_PRETRAINED_WEIGHTS:
        checkpoint = torch.load(SAVED_PATH)
        policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        target_net.load_state_dict(checkpoint['target_net_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_policy_net_state_dict'])
    else:
        target_net.load_state_dict(policy_net.state_dict())
    policy_net.train()
    target_net.eval()

    ##########################
    # test select_action()
    ##########################
    # state = torch.tensor(uwrt_arm_state, dtype=torch.float).view(-1, n_obs)
    # stacked_states = collections.deque(STACK_SIZE * [state], maxlen=STACK_SIZE)
    # for episode_idx in range(10):
    #     stacked_states_t = torch.cat(tuple(stacked_states), dim=1)
    #     action = select_action(env, policy_net, stacked_states_t, n_actions,
    #                           EPS_START, EPS_END, EPS_DECAY_LAST_FRAME, episode_idx=episode_idx,
    #                           device=device)
    #     uwrt_arm_state, reward, done, _ = env.step(action[0].item())
    #     state = torch.tensor(uwrt_arm_state, dtype=torch.float).view(1, -1, n_obs)
    #     stacked_states.append(state)

    ####################
    # training loop
    ####################
    run_training_loop(env, policy_net, target_net, optimizer,
                      memory, BATCHSIZE, n_obs, n_actions,
                      device)

    ####################
    env.close()

if __name__ == '__main__':

    #########################
    # HYPER PARAMS
    #########################
    EPISODES = int(25e3)
    TIMESTEPS = int(2e3)

    LEARNING_RATE = 1e-3
    BATCHSIZE = 32

    REPLAYBUFFER = 1e4

    STACK_SIZE = 5

    GAMMA = 0.99
    EPS_START = 0.3
    EPS_END = 0.1
    EPS_DECAY = 200
    EPS_DECAY_LAST_FRAME = 10 ** 4
    TARGET_UPDATE = 1000
    EPS_THRESHOLD = 0

    #########################
    # saved weights
    #########################
    weights = sorted(glob.glob(ROOT_DIR + '/DQN/trained_models/policy_dqn_states_*'))
    num_weights = len(weights)

    # TODO: clean this up!
    last_pt_file = glob.glob(ROOT_DIR + '/DQN/trained_models/policy_dqn_states_' + str(num_weights - 1) + '*')[0]

    if num_weights >= 1:
        print("*** Loading Pretrained Weights ***")
        SAVED_PATH = ROOT_DIR + "/DQN/trained_models/policy_dqn_states_" + np.str(num_weights - 1) + "_" + np.str(1000) + ".pt"
        PATH = ROOT_DIR + "/DQN/trained_models/policy_dqn_states_" + np.str(num_weights) + "_" + np.str(EPISODES) + ".pt"
        LOAD_PRETRAINED_WEIGHTS = True
    else:
        PATH = ROOT_DIR + "/DQN/trained_models/policy_dqn_states_" + np.str(num_weights) + "_" + np.str(EPISODES) + ".pt"
        LOAD_PRETRAINED_WEIGHTS = False

    #########################
    # main
    #########################
    main()


