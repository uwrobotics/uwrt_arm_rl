import gym
import numpy as np
from gym import spaces

from gym_uwrt_arm.envs.uwrt_arm_env import UWRTArmEnv


class DiscreteToContinuousDictActionWrapper(gym.ActionWrapper):
    # joint_velocity_commands: STOP[0], MAX_NEGATIVE_VELOCITY[1], MAX_POSITIVE_VELOCITY[2]
    ACTIONS_PER_JOINT = 3

    def __init__(self, env):
        super().__init__(env)
        assert (isinstance(env, UWRTArmEnv), 'Wrapped Env must be of type UWRTArmEnv')

        self.action_space = spaces.Discrete(
            env.info['arm']['num_joints'] * DiscreteToContinuousDictActionWrapper.ACTIONS_PER_JOINT)

        # TODO: These are fake numbers. Pull max velocities from UWRTArmEnv
        self.max_joint_speeds = [0.5, 0.2, 1.5, 0.2, 0.2]

    def action(self, discrete_action):
        joint_num = discrete_action // DiscreteToContinuousDictActionWrapper.ACTIONS_PER_JOINT
        joint_action = discrete_action % DiscreteToContinuousDictActionWrapper.ACTIONS_PER_JOINT

        joint_actions = np.zeros(self.env.info['arm']['num_joints'])

        joint_actions[joint_num] = self.max_joint_speeds[joint_num] * (1 if joint_action == 2 else
                                                                       0 if joint_action == 0 else
                                                                       -1 if joint_action == 1 else None)

        continuous_action = {
            'joint_velocity_commands': joint_actions
        }

        return continuous_action
