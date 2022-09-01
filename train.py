import torch
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np
from collections import deque
import warnings

# my imports
from settings import *
from wrap import make_env
from network import DNN

# create game environment
env = make_env(ENV_ID)

# print environment data
print()
print('Environment Name  :', ENV_ID)
print('Basic Wrappers    :', ' - '.join(str(env).split('<')[-4:-1]))
print('Custom Wrappers   :', ' - '.join(str(env).split('<')[1:-4]))
print('Observation Space :', env.observation_space)
print('Action Space      :', env.action_space)
print('Action Meannings  :', ' '.join(env.unwrapped.get_action_meanings()))
print('Reward Range      :', env.reward_range)

# print gpu characteristics (if available)
print()
print('GPU available     :', torch.cuda.is_available())
print('GPU devices num   :', torch.cuda.device_count())

# setting device on GPU if available, else CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using device      :', device)

#Additional Info when using cuda
if device.type == 'cuda':
    print('Device Name       :', torch.cuda.get_device_name(0))

# print hyperparameters
print()
print('Algorithm         :', ALGORITHM)
print('Training Steps    :', TRAINING_STEPS)
print('Start Buffer Size :', BUFFER_START_SIZE)
print('Buffer Size       :', BUFFER_SIZE)
print('Epsilon Decay     :', EPSILON_DECAY)
print('Target Update Freq:', TARGET_UPDATE_FREQ)
print('Learning Rate     :', LEARNING_RATE)

# create prediction and target neural networks and store them in GPU (if available)
if ALGORITHM in ['vanilla_dqn','double_dqn','dueling_dqn','double_dueling_dqn']:
    prediction_net = DNN(env.observation_space.shape, env.action_space.n, device=device).to(device) 
    target_net = DNN(env.observation_space.shape, env.action_space.n, device=device).to(device)
    target_net.load_state_dict(prediction_net.state_dict()) # networks start with the same weights
 
elif ALGORITHM in ['vanilla_dqv', 'dueling_dqv']:
    prediction_net = DNN(env.observation_space.shape, env.action_space.n, device=device).to(device)
    value_net = DNN(env.observation_space.shape, 1, device=device).to(device)
    target_value_net = DNN(env.observation_space.shape, 1, device=device).to(device)
    target_value_net.load_state_dict(value_net.state_dict()) # networks start with the same weights

# select optimizer(s) for backpropagation
q_optimizer = torch.optim.Adam(prediction_net.parameters(), lr=LEARNING_RATE)

if ALGORITHM in ['vanilla_dqv', 'dueling_dqv']:
    v_optimizer = torch.optim.Adam(value_net.parameters(), lr=LEARNING_RATE)

# Initialize Replay Buffer
print()
replay_buffer = deque([], maxlen=BUFFER_SIZE)
state = env.reset()

for i in range(BUFFER_START_SIZE):

    # log progress
    if (i/BUFFER_START_SIZE*100)%10==0:
        print('Initializing Buffer', int(i/BUFFER_START_SIZE*100),'%')

    # select random action
    action = env.action_space.sample()

    # take step in environment based on selected action
    new_state, reward, done, info = env.step(action)
    
    # store transition to replay buffer
    transition = (state, action, reward, done, new_state)
    replay_buffer.append(transition)

    # new state becomes the old one
    state = new_state

    # reset
    if done: state = env.reset()
        
print('Initializing Buffer 100 %')

# main training loop
print()
print(f'Training started...')
warnings.filterwarnings("ignore", category=DeprecationWarning) 

summary_writer = SummaryWriter(LOG_PATH)
current_q_loss = 0
episode_count = 0
episode_rewards = []
epinfos_buffer = deque([], maxlen=100)
state = env.reset()

for step in range(1, TRAINING_STEPS + 1): #for step in itertools.count():

    # Epsilon Decay
    epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])

    # Choose Action 
    action = prediction_net.act(state, epsilon=epsilon)

    # Take Action
    new_state, reward, done, info = env.step(action)
    episode_rewards.append(reward)

    # Add transition to Replay Buffer 
    transition = (state, action, reward, done, new_state)
    replay_buffer.append(transition)

    # new state becomes the old one
    state = new_state

    # if life is lost,
    if done:
        # record stats
        eprew = sum(episode_rewards)
        eplen = len(episode_rewards)
        epinfos_buffer.append({'r': round(eprew, 6), 'l':eplen})
       
        # and reset
        episode_rewards = []
        episode_count += 1
        state = env.reset()

    # Perform Gradient Descent
    transitions = random.sample(replay_buffer, BATCH_SIZE)

    q_loss = prediction_net.compute_loss(transitions, target_net)
    q_optimizer.zero_grad()
    q_loss.backward()
    q_optimizer.step()
    current_q_loss = q_loss.item()

    if ALGORITHM in ['vanilla_dqv', 'dueling_dqv']:
        v_loss = prediction_net.compute_loss(transitions, target_net)
        v_optimizer.zero_grad()
        v_loss.backward()
        v_optimizer.step()
        current_v_loss = v_loss.item()

    # Update Target Network
    if step % TARGET_UPDATE_FREQ == 0:
        if ALGORITHM in ['vanilla_dqn','double_dqn','dueling_dqn','double_dueling_dqn']:
            target_net.load_state_dict(prediction_net.state_dict())
        elif ALGORITHM in ['vanilla_dqv', 'dueling_dqv']:
            target_value_net.load_state_dict(value_net.state_dict())
        
    # Logging
    if step % LOG_INTERVAL == 0:
        rew_mean = np.mean([e['r'] for e in epinfos_buffer])
        len_mean = np.mean([e['l'] for e in epinfos_buffer])
        print(f'Step: {step:6},   Episodes: {episode_count:6},   Avg_Reward: {rew_mean:6.2f},   Avg_Episode_Length: {len_mean:6.2f}, Epsilon: {epsilon:6.4f}')

        summary_writer.add_scalar('Training Metrics/Average Reward', rew_mean, global_step=step)
        summary_writer.add_scalar('Training Metrics/Loss', current_q_loss, global_step=step)

        summary_writer.add_scalar('Episode Metrics/Average Episode Length', len_mean, global_step=step)
        summary_writer.add_scalar('Episode Metrics/Number of Episodes', episode_count, global_step=step)
    
    # Save
    if step % SAVE_INTERVAL == 0 and step != 0:
        print('Saving...')
        prediction_net.save(SAVE_PATH)

print('Training COMPLETED!!!')