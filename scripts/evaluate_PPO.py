import glob
import os

import argparse

import gym
import numpy as np
from gym.wrappers import FlattenObservation
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecEnv

import config
from gym_uwrt_arm.wrappers.discrete_to_continuous_dict_action_wrapper import DiscreteToContinuousDictActionWrapper

############################################################
#  Parse command line arguments
############################################################
parser = argparse.ArgumentParser(description='Eval DQN with UWRT_ARM_ENV')

parser.add_argument('--vecnormalize', required=False, default=False,
                    type=bool,
                    metavar="Train DQN with normalized obs and rewards")

args = parser.parse_args()

def _load_latest_model(env):
    saved_model_file_mtime = None
    last_modified_checkpoint_file_mtime = None

    saved_model_file_path = SAVE_PATH / f'{GYM_ID}-ppo-trained-model.zip'
    if saved_model_file_path.is_file():
        saved_model_file_mtime = os.path.getmtime(saved_model_file_path)

    checkpoints_dir = SAVE_PATH / config.CHECKPOINTS_DIR
    if checkpoints_dir.is_dir():
        checkpoint_files = glob.glob(f'{checkpoints_dir}/{GYM_ID}-ppo-trained-model_*_steps.zip')

        if checkpoint_files:
            checkpoint_files.sort(key=os.path.getmtime)
            last_modified_checkpoint_file_path = checkpoint_files[-1]
            last_modified_checkpoint_file_mtime = os.path.getmtime(last_modified_checkpoint_file_path)

    if saved_model_file_mtime and last_modified_checkpoint_file_mtime:
        if saved_model_file_mtime >= last_modified_checkpoint_file_mtime:
            choice = 'saved_model'
        else:
            choice = 'saved_checkpoint'
    elif saved_model_file_mtime:
        choice = 'saved_model'
    elif last_modified_checkpoint_file_mtime:
        choice = 'saved_checkpoint'
    else:
        raise

    if choice == 'saved_model':
        print(f'Loading Saved Model from {saved_model_file_path}')
        model = PPO.load(path=saved_model_file_path, env=env, verbose=1)
    elif choice == 'saved_checkpoint':
        print(f'Loading Checkpoint from {last_modified_checkpoint_file_path}')
        model = PPO.load(path=last_modified_checkpoint_file_path, env=env, verbose=1)

    return model


def main(args):
    #################
    # Vec Norm
    #################

    if args.vecnormalize:
        print("Evaluating with Vec Normalize!")
        evaluation_env = DummyVecEnv([lambda: FlattenObservation(
                                                gym.make(GYM_ID, key_position=KEY_POSITION, key_orientation=KEY_ORIENTATION,
                                                max_steps=MAX_STEPS_PER_EPISODE, enable_render=True))])

        saved_vecenv_file_path = (SAVE_PATH / config.STATISTICS_LOG_DIR / 'vec_normalize.pkl')
        assert saved_vecenv_file_path.is_file(), "Do not have a save stats file .."
        evaluation_env = VecNormalize.load(saved_vecenv_file_path, evaluation_env)
        #  do not update them at test time
        evaluation_env.training = False
        # reward normalization is not needed at test time
        evaluation_env.norm_reward = False

    else:
        evaluation_env = FlattenObservation(
                                gym.make(GYM_ID, key_position=KEY_POSITION, key_orientation=KEY_ORIENTATION,
                                max_steps=MAX_STEPS_PER_EPISODE, enable_render=True))

    model = _load_latest_model(evaluation_env)

    while True:
        obs = evaluation_env.reset()
        done, state, total_reward, rewards = False, None, 0, []
        while not done:
            action, state = model.predict(obs, state=state, deterministic=True)
            obs, reward, done, info = evaluation_env.step(action)
            if args.vecnormalize:
                print('reward: {} distance left: {}'.format(reward[0], info[0]["goal"]["distance_to_target"]))
            else:
                print('reward: {} distance left: {}'.format(reward, info["goal"]["distance_to_target"]))
            rewards.append(reward)
            total_reward += reward
        print(f'average_action_reward: {np.mean(rewards)}')
        print(f'total_reward: {total_reward}')
        input()


if __name__ == '__main__':
    # Env params
    GYM_ID = 'uwrt-arm-v0'

    #####################
    # random key loc
    #####################
    # KEY_POSITION = np.array([0.6, 0.6, 0.6])
    # KEY_ORIENTATION = np.array([0, 0, 0, 1])

    KEY_POSITION = np.array([np.random.uniform(0.625, 0.675),
                              np.random.uniform(-0.30, 0.30),
                              np.random.uniform(0.65, 0.675)])
    KEY_ORIENTATION = np.array([0, 0, 0, 1])

    # Training params
    MAX_SIM_SECONDS_PER_EPISODE = 10

    # Other params
    # SAVE_PATH = config.DQN_BASE_SAVE_PATH
    SAVE_PATH = (config.MODELS_DIR_PATH / 'PPO_1').resolve(strict=True)

    '''
    Calculate max sim steps per episode from desired max episode sim time
    '''
    # TODO: remove constants here:
    PYBULLET_STEPS_PER_ENV_STEP = 3  # This is based on a desired_sim_step_duration of 1/100s (100hz control loop)
    PYBULLET_SECONDS_PER_PYBULLET_STEP = 1 / 240

    MAX_STEPS_PER_EPISODE = MAX_SIM_SECONDS_PER_EPISODE / PYBULLET_SECONDS_PER_PYBULLET_STEP / PYBULLET_STEPS_PER_ENV_STEP

    main(args)
