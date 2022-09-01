import torch
import random
import numpy as np

from settings import *
from wrap import make_env
from network import DNN

# create game environment
env = make_env(
    ENV_ID, 
    frameskip=4,
    framestack=4,
    render_mode='rgb_array',
    clip_rewards = False)

prediction_net = DNN(env.observation_space.shape, env.action_space.n, device='cpu')
prediction_net.load(f'models/{GAME}/{EXPERIMENT_NAME}.dat')

numGames = 30
total_rewards = []
for i in range(numGames):
    total_game_reward = 0
    done = False
    state = env.reset()
    steps = 0
    noop_steps = random.randint(0,30)
    while not done:
        steps += 1
        if steps < noop_steps:
            action = env.action_space.sample()
        else:
            action = prediction_net.act(state, epsilon=0.05)
        state, reward, done, info = env.step(action)
        total_game_reward += reward
        #vid.capture_frame()
    #print(f'game: {i + 1} total reward: {total_game_reward}')
    total_rewards.append(total_game_reward)

average_reward = np.average(total_rewards)
reward_std = np.std(total_rewards)
print(f'{GAME} with {ALGORITHM}: average reward after {numGames} games: {average_reward} + {reward_std}')

env.close()

# import os
# import gym
# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning) 
# video_path = f'videos/{game}/{run}'
# os.makedirs(os.path.dirname(video_path), exist_ok=True)
# vid = gym.wrappers.monitoring.video_recorder.VideoRecorder(env=env, enabled=True, path=video_path+'.mp4')
# vid.close()
