import glob
import os

import gym
import numpy as np
from gym.wrappers import FlattenObservation
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecEnv

import config
from gym_uwrt_arm.wrappers.discrete_to_continuous_dict_action_wrapper import DiscreteToContinuousDictActionWrapper


def _load_latest_model(env):
    saved_model_file_mtime = None
    last_modified_checkpoint_file_mtime = None

    saved_model_file_path = SAVE_PATH / f'{GYM_ID}-dqn-trained-model.zip'
    if saved_model_file_path.is_file():
        saved_model_file_mtime = os.path.getmtime(saved_model_file_path)

    checkpoints_dir = SAVE_PATH / config.CHECKPOINTS_DIR
    if checkpoints_dir.is_dir():
        checkpoint_files = glob.glob(f'{checkpoints_dir}/{GYM_ID}-dqn-trained-model_*_steps.zip')

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
        model = DQN.load(path=saved_model_file_path, env=env, verbose=1)
    elif choice == 'saved_checkpoint':
        print(f'Loading Checkpoint from {last_modified_checkpoint_file_path}')
        model = DQN.load(path=last_modified_checkpoint_file_path, env=env, verbose=1)

    return model


def main():
    evaluation_env = FlattenObservation(DiscreteToContinuousDictActionWrapper(
        gym.make(GYM_ID, key_position=KEY_POSITION, key_orientation=KEY_ORIENTATION,
                 max_steps=MAX_STEPS_PER_EPISODE, enable_render=True)))

    model = _load_latest_model(evaluation_env)

    while True:
        obs = evaluation_env.reset()
        done, state, total_reward, rewards = False, None, 0, []
        while not done:
            action, state = model.predict(obs, state=state, deterministic=True)
            obs, reward, done, info = evaluation_env.step(action)
            print(f'reward: {reward} distance left: {info["goal"]["distance_to_target"]}')
            rewards.append(reward)
            total_reward += reward
        print(f'average_action_reward: {np.mean(rewards)}')
        print(f'total_reward: {total_reward}')
        input()


if __name__ == '__main__':
    # Env params
    GYM_ID = 'uwrt-arm-v0'
    KEY_POSITION = np.array([0.6, 0.6, 0.6])

    KEY_ORIENTATION = np.array([0, 0, 0, 1])

    # Training params
    MAX_SIM_SECONDS_PER_EPISODE = 10

    # Other params
    # SAVE_PATH = config.DQN_BASE_SAVE_PATH
    SAVE_PATH = (config.MODELS_DIR_PATH / 'DQN_3').resolve(strict=True)

    '''
    Calculate max sim steps per episode from desired max episode sim time
    '''
    # TODO: remove constants here:
    PYBULLET_STEPS_PER_ENV_STEP = 3  # This is based on a desired_sim_step_duration of 1/100s (100hz control loop)
    PYBULLET_SECONDS_PER_PYBULLET_STEP = 1 / 240

    MAX_STEPS_PER_EPISODE = MAX_SIM_SECONDS_PER_EPISODE / PYBULLET_SECONDS_PER_PYBULLET_STEP / PYBULLET_STEPS_PER_ENV_STEP

    main()
