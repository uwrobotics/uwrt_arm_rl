from abc import ABC

import gym
import numpy as np
from gym import spaces

from gym_uwrt_arm.envs.uwrt_arm_env import UWRTArmEnv


class MultiDiscreteToContinuousActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert (isinstance(env, UWRTArmEnv), 'Wrapped Env must be of type UWRTArmEnv')
        self.action_space = spaces.Dict({
            # All joint limit switch states are either STOP[0], MAX_NEGATIVE_VELOCITY[1], MAX_POSITIVE_VELOCITY[2]
            'joint_velocity_commands': spaces.MultiDiscrete(
                np.full(shape=(env.info['arm']['num_joints'],), fill_value=3)),
        })

        # TODO: These are fake numbers. Pull max velocities from env
        self.max_joint_speeds = np.array([0.5, 0.2, 1.5, 0.2, 0.2])

    def action(self, multi_discrete_action):
        assert multi_discrete_action['joint_velocity_commands'].shape == (self.env.info['arm']['num_joints'],)

        joint_velocity_direction = np.array([1 if action == 2 else
                                             0 if action == 0 else
                                             -1 if action == 1 else
                                             None for action in multi_discrete_action['joint_velocity_commands']])

        continuous_action = {
            'joint_velocity_commands': joint_velocity_direction * self.max_joint_speeds
        }

        return continuous_action
