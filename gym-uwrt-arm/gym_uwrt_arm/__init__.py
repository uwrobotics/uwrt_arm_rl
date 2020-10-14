from gym.envs.registration import register

register(
    id='uwrt-arm-v0',
    entry_point='gym_uwrt_arm.envs:UWRTArmEnv',
)
