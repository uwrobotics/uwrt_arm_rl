from gym.envs.registration import register

register(
    id='UWRTArm-v0',
    entry_point='uwrtarm_gym.envs:UWRTArmEnv',
)
