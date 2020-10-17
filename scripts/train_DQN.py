import glob
import os

import gym
import numpy as np
from gym.wrappers import FlattenObservation
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.dqn import MlpPolicy

import config
from gym_uwrt_arm.wrappers.discrete_to_continuous_dict_action_wrapper import DiscreteToContinuousDictActionWrapper


def _load_latest_model(training_env):
    '''
    Chooses between the latest checkpoint and the latest save to load. If neither exist, a new model is returned.
    '''

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
        choice = 'new_model'

    if choice == 'saved_model':
        print(f'Loading Checkpoint from {saved_model_file_path}')
        model = DQN.load(path=saved_model_file_path, env=training_env, verbose=1,
                         tensorboard_log=str(SAVE_PATH / config.TENSORBOARD_LOG_DIR))
    elif choice == 'saved_checkpoint':
        print(f'Loading Saved Model from {last_modified_checkpoint_file_path}')
        model = DQN.load(path=last_modified_checkpoint_file_path, env=training_env, verbose=1,
                         tensorboard_log=str(SAVE_PATH / config.TENSORBOARD_LOG_DIR))
    else:
        print('Could not find saved models. Creating new model!')
        model = DQN(MlpPolicy, training_env, verbose=1, tensorboard_log=str(SAVE_PATH / config.TENSORBOARD_LOG_DIR))

    return model


# TODO: normalize rewards https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#pybullet-normalizing-input-features
def main():
    training_env = FlattenObservation(DiscreteToContinuousDictActionWrapper(
        gym.make(GYM_ID, key_position=KEY_POSITION, key_orientation=KEY_ORIENTATION,
                 max_steps=MAX_STEPS_PER_EPISODE, enable_render=False)))

    evaluation_env = FlattenObservation(DiscreteToContinuousDictActionWrapper(
        gym.make(GYM_ID, key_position=KEY_POSITION, key_orientation=KEY_ORIENTATION,
                 max_steps=MAX_STEPS_PER_EPISODE, enable_render=False)))

    model = _load_latest_model(training_env=training_env)

    checkpoint_callback = CheckpointCallback(save_freq=10, save_path=str(SAVE_PATH / config.CHECKPOINTS_DIR),
                                             name_prefix=f'{GYM_ID}-dqn-trained-model')
    evaluation_callback = EvalCallback(eval_env=evaluation_env,
                                       best_model_save_path=str(SAVE_PATH / config.BEST_MODEL_SAVE_DIR),
                                       log_path=str(SAVE_PATH / config.BEST_MODEL_LOG_DIR), eval_freq=50)
    dqn_callbacks = CallbackList([checkpoint_callback, evaluation_callback])

    model.learn(total_timesteps=TOTAL_TRAINING_ENV_STEPS, callback=dqn_callbacks)
    model.save(path=SAVE_PATH / f'{GYM_ID}-dqn-trained-model')

    training_env.close()
    evaluation_env.close()


if __name__ == '__main__':
    GYM_ID = 'uwrt-arm-v0'

    MAX_STEPS_PER_EPISODE = 5000
    KEY_POSITION = np.array([0.6, 0.6, 0.6])
    KEY_ORIENTATION = np.array([0, 0, 0, 1])

    TOTAL_TRAINING_ENV_STEPS = MAX_STEPS_PER_EPISODE * 60 * 8
    SAVE_PATH = config.DQN_BASE_SAVE_PATH

    main()
