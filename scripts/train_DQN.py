import glob
import os

import gym
import numpy as np
from gym.wrappers import FlattenObservation
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
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
        model = DQN(MlpPolicy, env=training_env, verbose=1, tensorboard_log=str(SAVE_PATH / config.TENSORBOARD_LOG_DIR))

    # Clear all unused checkpoints
    if last_modified_checkpoint_file_mtime:
        for file_path in checkpoint_files:
            if file_path != last_modified_checkpoint_file_path:
                os.remove(file_path)

    return model


def _update_model_parameters(model):
    # DQN specific
    model.gamma = 0.99  # Discount factor

    # Model Training
    model.train_freq = 5  # minimum number of env time steps between model training
    model.n_episodes_rollout = -1  # minimum number of episodes between model training
    model.gradient_steps = 1  # number of gradient steps to execute. -1 matches the number of steps in the rollout
    model.learning_rate = 0.001  # learning rate

    # Target network syncing
    model.target_update_interval = 20000  # number of env time steps between target network updates

    # Exploration
    model.exploration_fraction = 0.75  # fraction of entire training period over which the exploration rate is reduced
    model.exploration_initial_eps = 1.0  # initial value of random action probability
    model.exploration_final_eps = 0.1  # final value of random action probability


# TODO: normalize rewards https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#pybullet-normalizing-input-features
def main():
    training_env = Monitor(FlattenObservation(DiscreteToContinuousDictActionWrapper(
        gym.make(GYM_ID, key_position=KEY_POSITION, key_orientation=KEY_ORIENTATION,
                 max_steps=MAX_STEPS_PER_EPISODE, enable_render=False))))

    evaluation_env = FlattenObservation(DiscreteToContinuousDictActionWrapper(
        gym.make(GYM_ID, key_position=KEY_POSITION, key_orientation=KEY_ORIENTATION,
                 max_steps=MAX_STEPS_PER_EPISODE, enable_render=False)))

    # TODO: add tensorboard logs for episode reward, mean episode reward and mean action reward
    model = _load_latest_model(training_env=training_env)
    _update_model_parameters(model)

    checkpoint_callback = CheckpointCallback(save_freq=ESTIMATED_STEPS_PER_MIN_1080TI * 10,
                                             save_path=str(SAVE_PATH / config.CHECKPOINTS_DIR),
                                             name_prefix=f'{GYM_ID}-dqn-trained-model')

    # TODO: tensorboard not getting updated when evalcallback is used
    # evaluation_callback = EvalCallback(eval_env=evaluation_env,
    #                                    best_model_save_path=str(SAVE_PATH / config.BEST_MODEL_SAVE_DIR),
    #                                    log_path=str(SAVE_PATH / config.BEST_MODEL_LOG_DIR),
    #                                    eval_freq=ESTIMATED_STEPS_PER_MIN_1080TI * 10)
    dqn_callbacks = CallbackList([checkpoint_callback])

    model.learn(total_timesteps=TOTAL_TRAINING_ENV_STEPS, callback=dqn_callbacks)
    # TODO: autosave on KEYBOARD_INTERRUPT
    model.save(path=SAVE_PATH / f'{GYM_ID}-dqn-trained-model')

    training_env.close()
    evaluation_env.close()


if __name__ == '__main__':
    # Env params
    GYM_ID = 'uwrt-arm-v0'
    KEY_POSITION = np.array([0.6, 0.6, 0.6])
    KEY_ORIENTATION = np.array([0, 0, 0, 1])

    # Training params
    MAX_SIM_SECONDS_PER_EPISODE = 10
    DESIRED_TRAINING_TIME_HOURS = 24

    # System params
    ESTIMATED_STEPS_PER_SECOND_1080TI = 600

    # Other params
    SAVE_PATH = config.DQN_BASE_SAVE_PATH

    '''
    Calculate max sim steps per episode from desired max episode sim time
    '''
    # TODO: remove constants here:
    PYBULLET_STEPS_PER_ENV_STEP = 3  # This is based on a desired_sim_step_duration of 1/100s (100hz control loop)
    PYBULLET_SECONDS_PER_PYBULLET_STEP = 1 / 240

    MAX_STEPS_PER_EPISODE = MAX_SIM_SECONDS_PER_EPISODE / PYBULLET_SECONDS_PER_PYBULLET_STEP / PYBULLET_STEPS_PER_ENV_STEP

    '''
    Calculate env steps from desired training time
    '''
    DESIRED_TRAINING_TIME_MINS = DESIRED_TRAINING_TIME_HOURS * 60
    ESTIMATED_STEPS_PER_MIN_1080TI = ESTIMATED_STEPS_PER_SECOND_1080TI * 60
    TOTAL_TRAINING_ENV_STEPS = DESIRED_TRAINING_TIME_MINS * ESTIMATED_STEPS_PER_MIN_1080TI

    NUM_TRAINING_EPISODES = TOTAL_TRAINING_ENV_STEPS // MAX_STEPS_PER_EPISODE
    print(f'Beginning to train for about {NUM_TRAINING_EPISODES} episodes ({TOTAL_TRAINING_ENV_STEPS} time steps)')

    main()
