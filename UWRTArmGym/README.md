# OpenAI Gym for UWRT GYM with PyBullet
This work is largely based on the following tutorial:
1. [OpenAI Gym Environments with PyBullet](https://www.etedal.net/2020/04/pybullet-panda.html)

## UWRTARM Gym
- Discrete 
- Action Space: 6-d (joint angles)
- Observation Space: 3-d (XYZ of end effector)

![Alt text](tests/images/uwrtarm_gym.png?raw=true "Title")

## Tests
1. python test_pybullet
- This test spawns a 'key' represented by a cube and explores IK in pybullet
2. python test_gym
- This test: 1. randomly samples a action from the action space and 2. explores the engineered "reward" function in env.step