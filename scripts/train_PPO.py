import glob
import os
import sys

import argparse

import gym
import gym_uwrt_arm

import numpy as np

from gym.wrappers import FlattenObservation
from gym_uwrt_arm.wrappers.discrete_to_continuous_dict_action_wrapper import DiscreteToContinuousDictActionWrapper

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor, load_results
from stable_baselines3.ppo import MlpPolicy

import config

############################################################
#  Parse command line arguments
############################################################

parser = argparse.ArgumentParser(description='Train DQN with UWRT_ARM_ENV')

parser.add_argument('--vecnormalize', required=False, default=False,
                    type=bool,
                    metavar="Train DQN with normalized obs and rewards")

args = parser.parse_args()

###########################
###########################

# TODO: add tensorboard logs for episode reward, mean episode reward and mean action reward
class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, monitor_log_dir, save_best_model_path, check_freq=1000, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.monitor_log_dir = monitor_log_dir
        self.check_freq = check_freq
        self.verbose = verbose

        self.save_best_model_path = os.path.join(save_best_model_path, 'best_mean_reward_model')

        self.best_mean_reward = -np.inf
        self.history_for_best_mean_reward = 100

    # from https://stable-baselines.readthedocs.io/en/master/guide/examples.html#using-callback-monitoring-training
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            monitor_csv_dataframe = load_results(self.monitor_log_dir)
            # dataframe is loaded as: [index], [r], [l], [t]
            index = monitor_csv_dataframe['index'].to_numpy()
            rewards = monitor_csv_dataframe['r'].to_numpy()
            episode_lengths = monitor_csv_dataframe['l'].to_numpy()

            last_norm_reward = rewards[-1] / episode_lengths[-1]
            # self.logger.record('Latest Reward', last_norm_reward) # Duplicate of ep_rew_mean
            if self.verbose > 0:
                print('Episode reward:{:.2f} for {}-th Episode'.format(last_norm_reward, index[-1] + 1))

            if len(rewards) > self.history_for_best_mean_reward:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(rewards[-self.history_for_best_mean_reward:]) / \
                              np.mean(episode_lengths[-self.history_for_best_mean_reward:])
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # could save best model
                    # if self.verbose > 0:
                    #     print("Saving new best model to {}".format(self.save_best_model_path))
                    self.model.save(self.save_best_model_path)

                self.logger.record('Best Mean Reward', self.best_mean_reward)
                if self.verbose > 0:
                    print("Best mean reward: {:.2f}\nLast mean reward per episode: {:.2f}"
                            .format(self.best_mean_reward, mean_reward))

        return True

def _load_latest_model(training_env):
    '''
    Chooses between the latest checkpoint and the latest save to load. If neither exist, a new model is returned.
    '''

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

    ################################################
    # TODO (ak): Don't load any checkpoint or model
    ################################################
    print('Training from Scratch!')
    model = PPO(MlpPolicy, env=training_env, verbose=1, tensorboard_log=str(SAVE_PATH / config.TENSORBOARD_LOG_DIR),
                learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, gamma=GAMMA)

    # Clear all unused checkpoints
    if last_modified_checkpoint_file_mtime:
        for file_path in checkpoint_files:
            if file_path != last_modified_checkpoint_file_path:
                os.remove(file_path)

    return model

# TODO: normalize rewards https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#pybullet-normalizing-input-features
def main(args):

    #################
    # Vec Norm
    #################

    if args.vecnormalize:
        print("Normalizing Input Obs and Rewards!")
        # training_env = DummyVecEnv([lambda:
        #                             Monitor(FlattenObservation(DiscreteToContinuousDictActionWrapper(
        #                                     gym.make(GYM_ID, key_position=KEY_POSITION, key_orientation=KEY_ORIENTATION,
        #                                     max_steps=MAX_STEPS_PER_EPISODE, enable_render=False))),
        #                             filename=str(SAVE_PATH))])

        training_env = DummyVecEnv([lambda:
                                    Monitor(FlattenObservation(
                                            gym.make(GYM_ID, key_position=KEY_POSITION, key_orientation=KEY_ORIENTATION,
                                            max_steps=MAX_STEPS_PER_EPISODE, enable_render=False)),
                                    filename=str(SAVE_PATH))])

        # Automatically normalize the input features and reward
        training_env = VecNormalize(training_env, norm_obs=True, norm_reward=True)

    else:
        # training_env = Monitor(FlattenObservation(DiscreteToContinuousDictActionWrapper(
        #                     gym.make(GYM_ID, key_position=KEY_POSITION, key_orientation=KEY_ORIENTATION,
        #                     max_steps=MAX_STEPS_PER_EPISODE, enable_render=False))),
        #                 filename=str(SAVE_PATH))

        training_env = Monitor(FlattenObservation(
                            gym.make(GYM_ID, key_position=KEY_POSITION, key_orientation=KEY_ORIENTATION,
                            max_steps=MAX_STEPS_PER_EPISODE, enable_render=False)),
                        filename=str(SAVE_PATH))

    #################
    #################

    model = _load_latest_model(training_env=training_env)

    ###################
    # callbacks
    ###################

    checkpoint_callback = CheckpointCallback(save_freq=ESTIMATED_STEPS_PER_MIN_1080TI * 5,
                                             save_path=str(SAVE_PATH / config.CHECKPOINTS_DIR),
                                             name_prefix=f'{GYM_ID}-ppo-trained-model')

    # TODO: tensorboard not getting updated when evalcallback is used
    # evaluation_callback = EvalCallback(eval_env=evaluation_env,
    #                                    best_model_save_path=str(SAVE_PATH / config.BEST_MODEL_SAVE_DIR),
    #                                    log_path=str(SAVE_PATH / config.BEST_MODEL_LOG_DIR),
    #                                    eval_freq=ESTIMATED_STEPS_PER_MIN_1080TI * 10)

    custom_callback = TensorboardCallback(monitor_log_dir=str(SAVE_PATH), save_best_model_path=str(SAVE_PATH),
                                            check_freq=1000, verbose=0)

    ppo_callbacks = CallbackList([checkpoint_callback, custom_callback])

    ###################
    ###################

    model.learn(total_timesteps=int(TOTAL_TRAINING_ENV_STEPS), callback=ppo_callbacks)
    # TODO: autosave on KEYBOARD_INTERRUPT
    model.save(path=SAVE_PATH / f'{GYM_ID}-ppo-trained-model')

    #################
    #################

    if args.vecnormalize:
        if not (os.path.exists(SAVE_PATH / config.STATISTICS_LOG_DIR)):
            os.makedirs(SAVE_PATH / config.STATISTICS_LOG_DIR)
        training_env.save(SAVE_PATH / config.STATISTICS_LOG_DIR / 'vec_normalize.pkl')

    training_env.close()


if __name__ == '__main__':
    # Env params
    GYM_ID = 'uwrt-arm-v0'

    #######################
    # RANDOM KEY LOCATION
    #######################

    # KEY_POSITION = np.array([0.6, 0.6, 0.6])
    # KEY_ORIENTATION = np.array([0, 0, 0, 1])

    KEY_POSITION = np.array([np.random.uniform(0.625, 0.675),
                                       np.random.uniform(-0.30, 0.30),
                                       np.random.uniform(0.65, 0.675)])
    KEY_ORIENTATION = np.array([0, 0, 0, 1])

    # Training params
    MAX_SIM_SECONDS_PER_EPISODE = 10
    DESIRED_TRAINING_TIME_HOURS = 0.5

    # System params
    ESTIMATED_STEPS_PER_SECOND_1080TI = 600

    # Other params
    # SAVE_PATH = config.DQN_BASE_SAVE_PATH
    SAVE_PATH = (config.MODELS_DIR_PATH / 'PPO_1').resolve(strict=True)

    '''
    New Model Params
    '''
    # DQN specific
    GAMMA = 0.99  # Discount factor

    # Model Training
    LEARNING_RATE = 0.0001  # learning rate
    BATCH_SIZE = 32

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

    main(args)
