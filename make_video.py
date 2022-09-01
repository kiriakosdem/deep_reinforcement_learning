import torch
import os
import gym
import warnings

from settings import *
from wrap import make_env
from network import DNN

# create game environment
env = make_env(
    ENV_ID, 
    frameskip=4,
    framestack=4,
    render_mode='human',
    clip_rewards = False)

prediction_net = DNN(env.observation_space.shape, env.action_space.n, device='cpu')
prediction_net.load(f'models/{GAME}/{EXPERIMENT_NAME}.dat')

warnings.filterwarnings("ignore", category=DeprecationWarning) 
video_path = f'videos/{GAME}/{EXPERIMENT_NAME}'
os.makedirs(os.path.dirname(video_path), exist_ok=True)
vid = gym.wrappers.monitoring.video_recorder.VideoRecorder(env=env, enabled=True, path=video_path+'.mp4')

done = False
state = env.reset()

while not done:
    action = prediction_net.act(state, epsilon=0.05)
    state, reward, done, info = env.step(action)
    vid.capture_frame()

env.close()
vid.close()