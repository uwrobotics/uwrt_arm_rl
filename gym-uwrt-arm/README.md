# OpenAI Gym for UWRT's Arm using PyBullet
This work was originally based on the following tutorial:
* [OpenAI Gym Environments with PyBullet](https://www.etedal.net/2020/04/pybullet-panda.html)

## UWRT Arm Gym
- Discrete 
- Action Space: 7-d (joint angles)
- Observation Space: 3-d (XYZ of end effector)

![Alt text](images/uwrtarm_gym.png?raw=true "Title")

## Installing
To use the gym environments provided in this package, you must first install it:
```
cd <Repo Root>
conda develop gym-uwrt-arm
```
To uninstall, you can run `conda develop -u gym-uwrt-arm`
## Tests
1. python test_pybullet
- This test spawns a 'key' represented by a cube and explores IK in pybullet
2. python test_gym
- This test: 1. randomly samples a action from the action space and 2. explores the engineered "reward" function in env.step