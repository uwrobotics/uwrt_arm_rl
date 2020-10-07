import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def collect_trajectories(envs, policy, writer,
                         n_actions, n_obs, V_MAX,
                         episode_idx, TIMESTEPS, nrand=5,
                         device="cpu"):
    ten_rewards = 0
    # initialize returning lists and start the game!
    state_list = []
    reward_list = []
    prob_list = []
    action_list = []
    value_list = []
    done_list = []

    uwrt_arm_state = envs.reset()
    uwrt_arm_state = torch.tensor(uwrt_arm_state, dtype=torch.float).view(1, -1, n_obs).cuda()

    # perform nrand random steps
    for _ in range(nrand):
        action = np.random.randn(n_actions)
        action = np.clip(action, -V_MAX, V_MAX)
        _, reward, done, _ = envs.step(action)
        reward = torch.tensor([reward], device=device)

    for t in range(TIMESTEPS):
        action_est, values = policy(uwrt_arm_state)
        sigma = nn.Parameter(torch.zeros(n_actions))
        dist = torch.distributions.Normal(action_est, F.softplus(sigma).to(device))
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        log_probs = torch.sum(log_probs, dim=-1).detach()
        values = values.detach()
        actions = actions.detach()

        env_actions = actions.cpu().numpy()
        uwrt_arm_state, reward, done, _ = envs.step(env_actions[0])
        # print("Reward: ", reward)

        uwrt_arm_state = torch.tensor(uwrt_arm_state, dtype=torch.float).view(1, -1, n_obs).cuda()
        rewards = torch.tensor([reward], device=device)
        dones = torch.tensor([done], device=device)

        state_list.append(uwrt_arm_state.unsqueeze(0))
        prob_list.append(log_probs.unsqueeze(0))
        action_list.append(actions.unsqueeze(0))
        reward_list.append(rewards.unsqueeze(0))
        value_list.append(values.unsqueeze(0))
        done_list.append(dones)

        # if np.any(dones.cpu().numpy()):
        #     ten_rewards += reward
        #     state = envs.reset()
        #     if episode_idx % 10 == 0:
        #         writer.add_scalar('ten episodes average rewards', ten_rewards / 10.0, episode_idx)
        #         ten_rewards = 0

    print("Reward for Traj: {}".format(sum(reward_list.copy()).cpu().numpy()[0][0]))
    state_list = torch.cat(state_list, dim=0)
    prob_list = torch.cat(prob_list, dim=0)
    action_list = torch.cat(action_list, dim=0)
    reward_list = torch.cat(reward_list, dim=0)
    value_list = torch.cat(value_list, dim=0)
    done_list = torch.cat(done_list, dim=0)
    return prob_list, state_list, action_list, reward_list, value_list, done_list