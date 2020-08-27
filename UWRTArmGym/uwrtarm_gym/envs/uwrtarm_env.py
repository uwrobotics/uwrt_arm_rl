import gym
from gym import error, spaces, utils
from gym.utils import seeding

import pybullet as p
import pybullet_data
import math
import numpy as np
import random

############
# init cwd
############
import os
import sys

ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
print("********* cwd {} *********".format(ROOT_DIR))

URDF_DIR = "/urdf/uwrt_arm.urdf"

class UWRTArmEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, timesteps):
        p.connect(p.GUI)

        # camera config
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.55, -0.35, 0.2])

        # box space
        self.obs_dim = 3    # x,y,z of end effector
        self.action_dim = 6 # joint angles
        self.observation_space = spaces.Box(np.array([-1] * self.obs_dim), np.array([1] * self.obs_dim))
        self.action_space = spaces.Box(np.array([-1] * self.action_dim), np.array([1] * self.action_dim))

        # config
        self.dv = 0.005
        self.distance_from_target = 0.075
        self.move_tol = 1e-3 # moving towards/away from goal
        self.singularity_tol = 1e-3

        self.t = 0
        self.t_history = 5
        self.t_thres = 250
        # reward function
        self.distance_from_key = np.zeros(shape=timesteps, dtype=np.float)
        self.change_in_distance = np.zeros(shape=timesteps, dtype=np.float)

    def step(self, action):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)

        dv = 0.005
        dx = action[0] * dv
        dy = action[1] * dv
        dz = action[2] * dv

        current_pose = p.getLinkState(self.uwrtarmUid, self.num_joints)
        current_position = current_pose[0] ### in world frame

        target_position = self.keyboard_position
        diff = target_position - np.array(current_position)

        # new_position = [diff[0] + dv, diff[1] + dv, diff[2] + dv] ### hack to move towards key
        new_position = [dx, dy, dz]

        joint_poses = p.calculateInverseKinematics(self.uwrtarmUid, self.num_joints, new_position, self.keyboard_orientation)

        p.setJointMotorControl2(self.uwrtarmUid, 0, p.POSITION_CONTROL, joint_poses[0])
        p.setJointMotorControl2(self.uwrtarmUid, 1, p.POSITION_CONTROL, joint_poses[1])
        # p.setJointMotorControl2(self.uwrtarmUid, 2, p.POSITION_CONTROL, joint_poses[2]) ### fixed shoulder joint
        p.setJointMotorControl2(self.uwrtarmUid, 3, p.POSITION_CONTROL, joint_poses[2])
        p.setJointMotorControl2(self.uwrtarmUid, 4, p.POSITION_CONTROL, joint_poses[3])
        p.setJointMotorControl2(self.uwrtarmUid, 5, p.POSITION_CONTROL, joint_poses[4])

        p.stepSimulation()

        state_keyboard, _ = p.getBasePositionAndOrientation(self.keyboardUid)
        state_robot = p.getLinkState(self.uwrtarmUid, self.num_joints)[0]

        self.distance_from_key[self.t] = np.linalg.norm(diff)
        self.change_in_distance = np.gradient(self.distance_from_key)
        # print("Distance from Target: {:.4f}[m]".format(self.distance_from_key[self.t]))
        # print("Iteration: {}, Gradient: {:.0e}[m]".format(self.t, np.abs(self.change_in_distance[self.t-1])))

        # case 1: we 'hit' the key
        if self.distance_from_key[self.t] < self.distance_from_target:
            reward = 25
            done = True
            self.t = 0
        # case 2: SINGULARITY or we can't reach the goal !!!
        elif self.t > self.t_thres and \
                np.sum(np.abs(self.change_in_distance[self.t-self.t_history:self.t]) < self.singularity_tol) == self.t_history:
            reward = -1e3
            done = True
            self.t = 0
        # case 2: we move closer to the key
        elif self.change_in_distance[self.t-1] < 0 and np.abs(self.change_in_distance[self.t-1]) > self.move_tol:
            reward = 1
            done = False
        # # case 3: we move further to the key
        elif self.change_in_distance[self.t - 1] > 0 and np.abs(self.change_in_distance[self.t-1]) > self.move_tol:
            reward = -1
            done = False
        else:
            reward = 0
            done = False

        # history
        self.t += 1

        info = state_keyboard
        observation = state_robot

        return observation, reward, done, info

    def reset(self):
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)  # we will enable rendering after we loaded everything

        urdfRootPath = pybullet_data.getDataPath()
        p.setGravity(0, 0, -10)

        # # plane
        self.table_position = [0, 0, -0.65]
        # planeUid = p.loadURDF(os.path.join(urdfRootPath, "plane.urdf"), basePosition=self.table_position)
        tableUid = p.loadURDF(os.path.join(urdfRootPath, "table/table.urdf"), basePosition=self.table_position)

        ##############################
        # uwrt arm
        ##############################
        # config
        self.home_position = [0.4, 0.0, 0.4]
        self.home_orientation = p.getQuaternionFromEuler([-np.pi, 0, 0])

        self.uwrtarmUid = p.loadURDF(ROOT_DIR + URDF_DIR, useFixedBase=True)
        self.num_joints = p.getNumJoints(self.uwrtarmUid) - 1  ### fixed shoulder joint
        # p.resetBasePositionAndOrientation(self.uwrtarmUid, [0, 0, 0], [0, 0, 0, 1])

        ##############################
        # "keyboard"
        # cube === key on a keyboard
        ##############################
        # config
        self.keyboard_orientation = p.getQuaternionFromEuler([np.pi, 0, 0])
        self.keyboard_position = [np.random.uniform(0.775, 0.775), np.random.uniform(-0.075, 0.075), np.random.uniform(0.55, 0.65)]

        self.keyboardUid = p.loadURDF(os.path.join(urdfRootPath, "cube_small.urdf"), basePosition=self.keyboard_position, useFixedBase=True)

        current_pose = p.getLinkState(self.uwrtarmUid, self.num_joints, computeForwardKinematics=True)
        current_position = current_pose[0]   ### in world frame
        current_orientation = current_pose[1]

        observation = current_position

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)  # rendering's back on again

        return observation

    def render(self, mode='human'):
        ###########################################################
        # TODO: implement RGB-D into observation & action space
        ###########################################################

        # view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.7, 0, 0.05],
        #                                                   distance=.7,
        #                                                   yaw=90,
        #                                                   pitch=-70,
        #                                                   roll=0,
        #                                                   upAxisIndex=2)
        # proj_matrix = p.computeProjectionMatrixFOV(fov=60,
        #                                            aspect=float(960) / 720,
        #                                            nearVal=0.1,
        #                                            farVal=100.0)
        # (_, _, px, _, _) = p.getCameraImage(width=960,
        #                                     height=720,
        #                                     viewMatrix=view_matrix,
        #                                     projectionMatrix=proj_matrix,
        #                                     renderer=p.ER_BULLET_HARDWARE_OPENGL)
        #
        # rgb_array = np.array(px, dtype=np.uint8)
        # rgb_array = np.reshape(rgb_array, (720, 960, 4))
        #
        # rgb_array = rgb_array[:, :, :3]
        # return rgb_array

        pass

    def close(self):
        p.disconnect()
