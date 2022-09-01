# game
GAME               = 'Pong'
ENV_ID             = f'{GAME}NoFrameskip-v4'

# algorithm
DISCOUNT_FACTOR    = 0.99          # deepmind
ALGORITHM_VARIATION={
    'vanilla_dqn'        : False,
    'double_dqn'         : False,
    'dueling_dqn'        : False,
    'double_dueling_dqn' : False,
    'vanilla_dqv'        : False,
    'dueling_dqv'        : True,
}
ALGORITHM          = [key for key, value in ALGORITHM_VARIATION.items() if value][0]

# replay buffer
BUFFER_SIZE        = 200000        # deepmind = 1000000
BUFFER_START_SIZE  = 50000         # deepmind

# exploration
EPSILON_START      = 1.0           # deepmind
EPSILON_END        = 0.1           # deepmind
EPSILON_DECAY      = 200000        # deepmind = 1000000

# neural net parameters
TARGET_UPDATE_FREQ = 2500          # deepmind = 10000
BATCH_SIZE         = 32            # deepmind
LEARNING_RATE      = 0.00005       # deepmind = 0.00025

# run
EXPERIMENT_NAME    = f'{GAME}_{ALGORITHM}_bfr{BUFFER_SIZE:.0e}_eps{EPSILON_DECAY:.0e}_lr{LEARNING_RATE:.0e}_trgt{TARGET_UPDATE_FREQ:.0e}'
TRAINING_STEPS     = 1000000

# saving and logging
SAVE_INTERVAL      = 20000
LOG_INTERVAL       = 2000
SAVE_PATH          = f'models/{GAME}/{EXPERIMENT_NAME}.dat'
LOG_PATH           = f'logs/{GAME}/{EXPERIMENT_NAME}'