import random
import torch

def select_action(env, policy_net, observation, n_actions,
                  EPS_START, EPS_END, EPS_DECAY_LAST_FRAME,
                  device):
    # init
    sample = random.random()
    eps_threshold = max(EPS_END, EPS_START - observation.size()[0] / EPS_DECAY_LAST_FRAME)

    # greedy select
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(observation)
    else:
        return torch.tensor([env.action_space.sample()])