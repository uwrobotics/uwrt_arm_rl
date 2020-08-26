import time

import gym
import uwrtarm_gym
env = gym.make('UWRTArm-v0', timesteps=1000)
env.reset()


for i_episode in range(20):
    observation = env.reset()
    time.sleep(1)
    for t in range(1000):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print("rewards: ", reward)
        # print("observation: \n", observation)
        # print("action: \n", action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()