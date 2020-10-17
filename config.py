from pathlib import Path

ROOT_DIR_PATH = Path(__file__).parent.absolute().resolve(strict=True)

MODELS_DIR_PATH = (ROOT_DIR_PATH / 'trained_models').resolve(strict=True)
DQN_BASE_SAVE_PATH = (MODELS_DIR_PATH / 'DQN').resolve(strict=True)

# Relative Paths
CHECKPOINTS_DIR = 'checkpoints'
BEST_MODEL_SAVE_DIR = 'best_model'
BEST_MODEL_LOG_DIR = 'best_model'
TENSORBOARD_LOG_DIR = 'tensorboard_logs'