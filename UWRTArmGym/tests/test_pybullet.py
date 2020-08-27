import pybullet as p
import pybullet_data

import numpy as np
import math

############
# init cwd
############
import os
import sys

ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
print("********* cwd {} *********".format(ROOT_DIR))

URDF_DIR = "/urdf/uwrt_arm.urdf"

################################
# init objects in pybullet
################################
p.connect(p.GUI)
urdfRootPath=pybullet_data.getDataPath()

# plane
# planeUid = p.loadURDF(os.path.join(urdfRootPath, "plane.urdf"), basePosition=[0, 0, -0.65])

# table
tableUid = p.loadURDF(os.path.join(urdfRootPath, "table/table.urdf"), basePosition=[0.5,0,-0.65])

# uwrt arm
uwrtarmUid = p.loadURDF(ROOT_DIR+URDF_DIR, useFixedBase=True)
num_joints = p.getNumJoints(uwrtarmUid) - 1 ### fixed shoulder joint

# home config
homeposition = [0.3, 0.0, 0.8]
homeorientation = p.getQuaternionFromEuler([-np.pi, 0, 0])

##############################
# "keyboard"
# cube === key on a keyboard
##############################
p.setGravity(0,0,-10)
objectorientation = p.getQuaternionFromEuler([np.pi, 0, 0])
objectposition = [np.random.uniform(0.5, 0.7), np.random.uniform(-0.4, 0.4), np.random.uniform(0.5, 0.7)]

objectUid = p.loadURDF(os.path.join(urdfRootPath, "cube_small.urdf"), basePosition=objectposition,
                       useFixedBase=True)


# camera config
p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.55,-0.35,0.2])

################################
#
################################

state_durations = [1,1,1,1]
control_dt = 1./240.
p.setTimestep = control_dt
state_t = 0
current_state = 0

while True:
    state_t += control_dt
    p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)

    #################
    # home pos
    #################

    state_robot = p.getLinkState(uwrtarmUid, num_joints-1)
    jointposes = p.calculateInverseKinematics(uwrtarmUid, num_joints, homeposition, homeorientation)

    if current_state == 0:
        p.setJointMotorControl2(uwrtarmUid, 0, p.POSITION_CONTROL, jointposes[0])
        p.setJointMotorControl2(uwrtarmUid, 1, p.POSITION_CONTROL, jointposes[1])
        # p.setJointMotorControl2(uwrtarmUid, 2, p.POSITION_CONTROL, jointposes[2]) ### fixed shoulder joint
        p.setJointMotorControl2(uwrtarmUid, 3, p.POSITION_CONTROL, jointposes[2])
        p.setJointMotorControl2(uwrtarmUid, 4, p.POSITION_CONTROL, jointposes[3])
        p.setJointMotorControl2(uwrtarmUid, 5, p.POSITION_CONTROL, jointposes[4])

    #################
    # keyboard pos
    #################

    state_robot = p.getLinkState(uwrtarmUid, num_joints)
    jointposes = p.calculateInverseKinematics(uwrtarmUid, num_joints, objectposition, objectorientation)

    if current_state == 1:
        p.setJointMotorControl2(uwrtarmUid, 0, p.POSITION_CONTROL, jointposes[0])
        p.setJointMotorControl2(uwrtarmUid, 1, p.POSITION_CONTROL, jointposes[1])
        # p.setJointMotorControl2(uwrtarmUid, 2, p.POSITION_CONTROL, jointposes[2]) ### fixed shoulder joint
        p.setJointMotorControl2(uwrtarmUid, 3, p.POSITION_CONTROL, jointposes[2])
        p.setJointMotorControl2(uwrtarmUid, 4, p.POSITION_CONTROL, jointposes[3])
        p.setJointMotorControl2(uwrtarmUid, 5, p.POSITION_CONTROL, jointposes[4])

    #####################
    # return to home pos
    #####################

    state_robot = p.getLinkState(uwrtarmUid, num_joints)
    jointposes = p.calculateInverseKinematics(uwrtarmUid, num_joints, homeposition, homeorientation)

    if current_state == 2:
        p.setJointMotorControl2(uwrtarmUid, 0, p.POSITION_CONTROL, jointposes[0])
        p.setJointMotorControl2(uwrtarmUid, 1, p.POSITION_CONTROL, jointposes[1])
        # p.setJointMotorControl2(uwrtarmUid, 2, p.POSITION_CONTROL, jointposes[2]) ### fixed shoulder joint
        p.setJointMotorControl2(uwrtarmUid, 3, p.POSITION_CONTROL, jointposes[2])
        p.setJointMotorControl2(uwrtarmUid, 4, p.POSITION_CONTROL, jointposes[3])
        p.setJointMotorControl2(uwrtarmUid, 5, p.POSITION_CONTROL, jointposes[4])

    if state_t >state_durations[current_state]:
        current_state += 1
        if current_state >= len(state_durations):
            current_state = 0
        state_t = 0
    p.stepSimulation()