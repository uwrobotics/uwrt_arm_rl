import random

import pybullet as p

import numpy as np
import torch

def select_action(env, policy_net, observation, n_actions,
                  EPS_START, EPS_END, EPS_DECAY_LAST_FRAME, episode_idx,
                  device):
    # init
    sample = random.random()
    eps_threshold = max(EPS_END, EPS_START - episode_idx / EPS_DECAY_LAST_FRAME)

    # greedy select
    if sample > eps_threshold:
        with torch.no_grad():
            # print("*** Moving towards Key with DQN ***")
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(observation.cuda()).max(1)[1].view(1, 1), False
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.int64), True
        # return torch.tensor([[n_actions + 1]], device=device, dtype=torch.int64), True