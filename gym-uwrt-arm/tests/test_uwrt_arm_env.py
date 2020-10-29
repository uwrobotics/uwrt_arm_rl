import pprint

import gym
import numpy as np
from gym.wrappers import FlattenObservation
from stable_baselines3.common import env_checker

# noinspection PyUnresolvedReferences
import gym_uwrt_arm.envs.uwrt_arm_env
from gym_uwrt_arm.wrappers.discrete_to_continuous_dict_action_wrapper import DiscreteToContinuousDictActionWrapper

class TestClass:
    NUM_EPISODES = 1
    MAX_STEPS = 5000
    KEY_POSITION = np.array([0.6, 0.6, 0.6])
    KEY_ORIENTATION = np.array([0, 0, 0, 1])

    def __run_test(self, env):
        pp = pprint.PrettyPrinter()  # TODO: update to python 3.8 to use sort_dicts = False

        for episode in range(self.NUM_EPISODES):
            initial_observation = env.reset()
            print('Initial Observation:')
            pp.pprint(initial_observation)

            for sim_step in range(self.MAX_STEPS):
                action = env.action_space.sample()
                observation, reward, done, info = env.step(action)

                print()
                print('Action:')
                pp.pprint(action)
                print('Observation:')
                pp.pprint(observation)
                print('Info:')
                pp.pprint(info)
                print('Reward:')
                pp.pprint(reward)

                if done:
                    print()
                    print(f'Episode #{episode} finished after {info["sim"]["steps_executed"]} steps!')
                    print(f'Episode #{episode} exit condition was {info["sim"]["end_condition"]}')
                    print()
                    break

    def test_env(self):
        env = gym.make('uwrt-arm-v0', key_position=self.KEY_POSITION, key_orientation=self.KEY_ORIENTATION,
                       max_steps=self.MAX_STEPS, enable_render=True)
        self.__run_test(env)

        # Implicitly closes the environment
        env_checker.check_env(env=env, warn=True, skip_render_check=False)

    def test_discrete_action_wrapper_env(self):
        env = DiscreteToContinuousDictActionWrapper(
            gym.make('uwrt-arm-v0', key_position=self.KEY_POSITION, key_orientation=self.KEY_ORIENTATION,
                     max_steps=self.MAX_STEPS, enable_render=True))
        self.__run_test(env)

        # Implicitly closes the environment
        env_checker.check_env(env=env, warn=True, skip_render_check=False)

    def test_flatten_observation_wrapper_env(self):
        env = FlattenObservation(
            gym.make('uwrt-arm-v0', key_position=self.KEY_POSITION, key_orientation=self.KEY_ORIENTATION,
                     max_steps=self.MAX_STEPS, enable_render=True))
        self.__run_test(env)

        # TODO: Broken because of dict action. fix upstream in sb3
        # Implicitly closes the environment
        env_checker.check_env(env=env, warn=True, skip_render_check=False)

    def test_gym_api_compliance_for_dqn_wrapper_setup(self):
        env = FlattenObservation(DiscreteToContinuousDictActionWrapper(
            gym.make('uwrt-arm-v0', key_position=self.KEY_POSITION, key_orientation=self.KEY_ORIENTATION,
                     max_steps=self.MAX_STEPS, enable_render=True)))
        self.__run_test(env)

        # TODO: Broken because of dict action. fix upstream in sb3
        # Implicitly closes the environment
        env_checker.check_env(env=env, warn=True, skip_render_check=False)

if __name__ == '__main__':
    test = TestClass()

    # test.test_env()
    # test.test_discrete_action_wrapper_env()
    test.test_flatten_observation_wrapper_env()
    # test.test_gym_api_compliance_for_dqn_wrapper_setup()
