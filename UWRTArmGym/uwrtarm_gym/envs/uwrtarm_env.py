import gym
from gym import error, spaces, utils
from gym.utils import seeding

import pybullet as p
import pybullet_data
import math
import numpy as np
import random

from scipy.spatial.transform import Rotation as R

############
# init cwd
############
import os
import sys

ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)
print("********* cwd {} *********".format(ROOT_DIR))

URDF_DIR = "/UWRTArmGym/urdf/uwrt_arm.urdf"

class UWRTArmEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, timesteps, discrete=False, render=True):
        self.is_render = render
        if self.is_render:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        # camera config
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.55, -0.35, 0.2])

        self.discrete = discrete

        # box space
        self.obs_dim = 7    # x,y,z and pose of end effector
        self.action_dim = 6 # joint angles
        self.observation_space = spaces.Box(np.array([-1] * self.obs_dim), np.array([1] * self.obs_dim), dtype=np.float32)
        self.action_space = spaces.Box(np.array([-1] * self.action_dim), np.array([1] * self.action_dim), dtype=np.float32)

        # config
        self.dv = 0.05
        self.x_distance_from_target = 6 / 100 ### [cm]
        self.xy_distance_from_target = 10 / 100  ### [cm]
        self.move_tol = 1e-3 # moving towards/away from goal
        self.singularity_tol = 1e-4

        # timesteps
        self.t = 0
        self.t_history = 5
        self.t_thres = 500
        self.max_timesteps = timesteps

        # reward function
        self.t_scaling = 50 * 1 / self.t_thres
        self.arrived_at_key = 250
        self.missed_key = -1000
        self.distance_from_key = np.zeros(shape=timesteps, dtype=np.float)
        self.change_in_distance = np.zeros(shape=timesteps, dtype=np.float)

    def get_discrete_action(self, continous_action):

        '''
        dx = [0, -dv, dv, 0, 0, 0, 0][action]
        dy = [0, 0, 0, -dv, dv, 0, 0][action]
        dz = -dv
        da = [0, 0, 0, 0, 0, -0.25, 0.25][action]

        action_continuous = [dx, dy, dz, da, 0.3]
        '''

        dv = 1  # see uwrtarm_env.py
        dx = [-dv, dv, 0, 0, 0, 0][continous_action]
        dy = [0, 0, -dv, dv, 0, 0][continous_action]
        dz = [0, 0, 0, 0, -dv, dv][continous_action]

        discrete_action = np.array([dx, dy, dz])

        # print("continous_action: ", continous_action)
        # print("discrete_action: ", discrete_action)

        return discrete_action

    def step(self, action):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)

        if self.discrete:
            action = self.get_discrete_action(action)

        dx = action[0] * self.dv
        dy = action[1] * self.dv
        dz = action[2] * self.dv
        # dR = action[3] * self.dv
        # dP = action[4] * self.dv
        # dY = action[5] * self.dv

        self.current_pose = p.getLinkState(self.uwrtarmUid, self.num_joints)
        self.current_position = self.current_pose[4] ### in world frame to base of wrist link
        current_quartenion = self.current_pose[5]

        ##############################
        # transform allen key
        ##############################

        # rotation
        current_rotation_matrix = R.from_quat([current_quartenion[0], current_quartenion[1], current_quartenion[2], current_quartenion[3]]).as_matrix()
        # transformation
        current_allen_key_transformation = current_rotation_matrix @ self.allen_key
        self.current_position += current_allen_key_transformation.T

        # check
        # np.linalg.norm(self.allen_key) == np.linalg.norm(current_allen_key_transformation) ???

        #########
        target_position = np.array(self.keyboard_position)
        diff = target_position - np.squeeze(self.current_position)
        x_diff = diff[0] - self.x_distance_from_target
        yz_diff = diff[0:-1]

        new_position = [dx, dy, dz]
        ### new_orientation = [dR, dP, dY]

        joint_poses = p.calculateInverseKinematics(self.uwrtarmUid, self.num_joints, new_position)
        # joint_poses = p.calculateInverseKinematics(self.uwrtarmUid, self.num_joints, target_position) ### not sampling actions

        p.setJointMotorControl2(self.uwrtarmUid, 0, p.POSITION_CONTROL, joint_poses[0])
        p.setJointMotorControl2(self.uwrtarmUid, 1, p.POSITION_CONTROL, joint_poses[1])
        # p.setJointMotorControl2(uwrtarmUid, 2, p.POSITION_CONTROL, jointposes[2]) ### fixed shoulder joint
        p.setJointMotorControl2(self.uwrtarmUid, 3, p.POSITION_CONTROL, joint_poses[2])
        p.setJointMotorControl2(self.uwrtarmUid, 4, p.POSITION_CONTROL, joint_poses[3])
        p.setJointMotorControl2(self.uwrtarmUid, 5, p.POSITION_CONTROL, joint_poses[4])

        p.stepSimulation()

        state_keyboard, _ = p.getBasePositionAndOrientation(self.keyboardUid)

        ##############################
        self.current_pose = p.getLinkState(self.uwrtarmUid, self.num_joints, computeForwardKinematics=True)
        self.current_position = np.asarray(self.current_pose[4])  ### in world frame
        self.current_orientation = np.asarray(self.current_pose[5])

        observation = np.append(self.current_position, self.current_orientation)

        self.distance_from_key[self.t] = np.linalg.norm(yz_diff)
        self.change_in_distance = np.gradient(self.distance_from_key)

        # print("X Distance from Keyboard Plane: \t{:.2f}[cm]".format(x_diff * 100))
        # print("YZ Distance from Target:        \t{:.2f}[cm]".format(self.distance_from_key[self.t]*100))
        # print("Iteration: {}, Gradient: {:.0e}[m]".format(self.t, np.abs(self.change_in_distance[self.t-1])))

        # # case 1: we moved passed the key
        # if self.t >= (self.max_timesteps - 1):
        #     print("\n********* Max Time Steps .. *********")
        #     reward = self.missed_key - self.t * self.t_scaling
        #     done = True
        #     self.t = 0
        # elif x_diff < 0:
        #     if np.linalg.norm(diff) < self.xy_distance_from_target:
        #         print("\n********* Hit Key! *********")
        #         reward = self.arrived_at_key - self.t * self.t_scaling
        #         done = True
        #         self.t = 0
        #     else:
        #         print("\n********* Went Beyond Keyboard .. *********")
        #         reward = self.missed_key - self.t * self.t_scaling
        #         done = True
        #         self.t = 0
        # # case 2: we can't reach the goal or SINGULARITY !!!
        # elif self.t > self.t_thres and \
        #         np.sum(np.abs(self.change_in_distance[self.t-self.t_history:self.t]) < self.singularity_tol) == self.t_history:
        #     print("\n********* Singularity .. *********")
        #     reward = self.missed_key - self.t * self.t_scaling
        #     done = True
        #     self.t = 0
        # ## case 2: we move closer to the key
        # elif self.change_in_distance[self.t - 1] < 0 and np.abs(self.change_in_distance[self.t - 1]) > self.move_tol:
        #     reward = 5
        #     done = False
        # ### case 3: we move further to the key
        # elif self.change_in_distance[self.t - 1] > 0 and np.abs(self.change_in_distance[self.t - 1]) > self.move_tol:
        #     reward = -5
        #     done = False
        # else:
        #     reward = 0
        #     done = False

        # case 1: we moved passed the key
        if self.t >= (self.max_timesteps - 1):
            print("\n********* Max Time Steps .. *********")
            reward = -10 - self.t * 1 / 50
            done = True
            self.t = 0
        elif x_diff < 0:
            if np.linalg.norm(diff) < self.xy_distance_from_target:
                print("\n********* Hit Key! *********")
                reward = 25 - self.t * 1 / 50
                done = True
                self.t = 0
            else:
                print("\n********* Went Beyond Keyboard .. *********")
                reward = -10 - self.t * 1 / 50
                done = True
                self.t = 0
        # case 2: we can't reach the goal or SINGULARITY !!!
        elif self.t > self.t_thres and \
                np.sum(np.abs(
                    self.change_in_distance[self.t - self.t_history:self.t]) < self.singularity_tol) == self.t_history:
            print("\n********* Singularity .. *********")
            reward = -10 - self.t * 1 / 50
            done = True
            self.t = 0
        ## case 2: we move closer to the key
        elif self.change_in_distance[self.t - 1] < 0 and np.abs(self.change_in_distance[self.t - 1]) > self.move_tol:
            reward = 1 / 50
            done = False
        ### case 3: we move further to the key
        elif self.change_in_distance[self.t - 1] > 0 and np.abs(self.change_in_distance[self.t - 1]) > self.move_tol:
            reward = - 1 / 50
            done = False
        else:
            reward = 0
            done = False

        # history
        self.t += 1

        info = state_keyboard

        return observation, reward, done, info

    def reset(self):

        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, False)  # we will enable rendering after we loaded everything

        urdfRootPath = pybullet_data.getDataPath()
        p.setGravity(0, 0, -10)

        # # plane
        self.table_position = [0, 0, -0.65]
        tableUid = p.loadURDF(os.path.join(urdfRootPath, "table/table.urdf"), basePosition=self.table_position)

        ##############################
        # uwrt arm
        ##############################
        # config
        self.home_position = [0.1, 0.0, 0.8]
        self.home_orientation = p.getQuaternionFromEuler([0, np.pi/2, 0])

        # self.uwrtarmUid = p.loadURDF(ROOT_DIR + URDF_DIR, useFixedBase=True)
        self.uwrtarmUid = p.loadURDF('/home/akeaveny/git/uwrt_arm_rl/UWRTArmGym/urdf/uwrt_arm.urdf', useFixedBase=True)
        self.num_joints = p.getNumJoints(self.uwrtarmUid) - 1  ### fixed shoulder joint
        # p.resetBasePositionAndOrientation(self.uwrtarmUid, [0, 0, 0], [0, 0, 0, 1])

        ##############################
        # allen key
        ##############################

        self.allen_key_offset = 0.112 / 2
        self.allen_key = np.array([self.allen_key_offset, 0, 0])[np.newaxis].T

        ##############################
        # "keyboard"
        # cube === key on a keyboard
        ##############################
        # config
        self.keyboard_orientation = p.getQuaternionFromEuler([0, 0, 0])
        # self.keyboard_position = [np.random.uniform(0.775, 0.775), np.random.uniform(-0.075, 0.075), np.random.uniform(0.55, 0.65)]
        self.keyboard_position = [np.random.uniform(0.625, 0.625), np.random.uniform(-0.30, 0.30), np.random.uniform(0.65, 0.675)]

        self.keyboardUid = p.loadURDF(os.path.join(urdfRootPath, "cube_small.urdf"),
                                      basePosition=self.keyboard_position, baseOrientation=self.keyboard_orientation,
                                      useFixedBase=True)

        ##############################
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, self.is_render)  # rendering's back on again

        #################
        # home pos
        #################

        for _ in range(10): ### 10 is needed to return to home pos

            joint_poses = p.calculateInverseKinematics(self.uwrtarmUid, endEffectorLinkIndex=self.num_joints,
                                                      targetPosition=self.home_position,
                                                      targetOrientation=self.home_orientation)

            p.setJointMotorControl2(self.uwrtarmUid, 0, p.POSITION_CONTROL, joint_poses[0])
            p.setJointMotorControl2(self.uwrtarmUid, 1, p.POSITION_CONTROL, joint_poses[1])
            # p.setJointMotorControl2(uwrtarmUid, 2, p.POSITION_CONTROL, jointposes[2]) ### fixed shoulder joint
            p.setJointMotorControl2(self.uwrtarmUid, 3, p.POSITION_CONTROL, joint_poses[2])
            p.setJointMotorControl2(self.uwrtarmUid, 4, p.POSITION_CONTROL, joint_poses[3])
            p.setJointMotorControl2(self.uwrtarmUid, 5, p.POSITION_CONTROL, joint_poses[4])

            p.stepSimulation()

        ##############################
        self.current_pose = p.getLinkState(self.uwrtarmUid, self.num_joints, computeForwardKinematics=True)
        self.current_position = np.asarray(self.current_pose[4])  ### in world frame
        self.current_orientation = np.asarray(self.current_pose[5])

        observation = np.append(self.current_position, self.current_orientation)

        return observation

    def render(self, mode='human'):
        ###########################################################
        # TODO: implement RGB-D into observation & action space
        ###########################################################

        # view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=self.keyboard_position,
        #                                                   distance=.35,
        #                                                   roll=0,
        #                                                   pitch=0,
        #                                                   yaw=-90,
        #                                                   upAxisIndex=2)
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=np.array([0, 0, 0.45]),
                                                          distance=1.5,
                                                          roll=0,
                                                          pitch=-45,
                                                          yaw=-90,
                                                          upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                   aspect=float(960) / 720,
                                                   nearVal=0.1,
                                                   farVal=100.0)
        (_, _, px, _, _) = p.getCameraImage(width=960,
                                            height=720,
                                            viewMatrix=view_matrix,
                                            projectionMatrix=proj_matrix,
                                            renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (720, 960, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array

        # pass

    def _get_state(self):
        return self.current_position

    def _get_observation(self):
        return self.render()

    def close(self):
        if self.is_render:
            p.disconnect()
