import torch
from torch import nn
import random
import numpy as np

from settings import DISCOUNT_FACTOR, ALGORITHM
class DNN(nn.Module):
    def __init__(self, input_shape, output_shape, device='cpu'):
        super(DNN, self).__init__()

        self.device = device
        self.input_shape = input_shape
        self.output_shape = output_shape

        # convolutional layer
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # determine the fully connected input shape after doing a forward pass through the convolutional layer
        conv_out = self.conv(torch.zeros(1, *input_shape))
        self.fully_connected_in = int(np.prod(conv_out.size()))

        # for the dueling architecture, separate the fully connected layer in two streams
        if ALGORITHM in ['dueling_dqn', 'dueling_dqv']:
            self.value_stream = nn.Sequential(
            nn.Linear(self.fully_connected_in, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
            )
            self.advantage_stream = nn.Sequential(
            nn.Linear(self.fully_connected_in, 512),
            nn.ReLU(),
            nn.Linear(512, output_shape)
            )
        else:
            self.fully_connected = nn.Sequential(
            nn.Linear(self.fully_connected_in, 512),
            nn.ReLU(),
            nn.Linear(512, output_shape)
            )

    def forward(self, state_t):
        # pass input through convolutional layer
        conv_out = self.conv(state_t)

        # flatten convolutional layer output
        batch_size = int(np.prod(conv_out.size())/self.fully_connected_in)
        fully_connected_in_size = int(np.prod(conv_out.size())/batch_size)
        fully_connected_in = conv_out.view(batch_size, fully_connected_in_size)

        # get q_values from fully connected layers
        if ALGORITHM in ['dueling_dqn', 'dueling_dqv']:
            value = self.value_stream(fully_connected_in)
            advantages = self.advantage_stream(fully_connected_in)
            mean_advantages = torch.mean(advantages, dim=1).unsqueeze(-1)
            q_values = value + (advantages - mean_advantages)
        else:
            q_values = self.fully_connected(fully_connected_in)

        return q_values

    def act(self, state, epsilon=0.0):
        # epsilon greedy policy
        if random.random() < epsilon:
            action = random.randint(0, self.output_shape - 1)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
            q_values = self.forward(state_tensor)
            max_q_index = torch.argmax(q_values)
            action = max_q_index.detach().tolist()

        return action

    
    def compute_loss(self, transitions, target_net):
        # convert to numpy arrays
        states = np.array([t[0] for t in transitions])
        actions = np.array([t[1] for t in transitions])
        rews = np.array([t[2] for t in transitions])
        dones = np.array([t[3] for t in transitions])
        new_states = np.array([t[4] for t in transitions])
        
        # and then convert to tensors
        states_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(-1)
        rewards_tensor = torch.tensor(rews, dtype=torch.float32, device=self.device).unsqueeze(-1)
        dones_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(-1)
        new_states_tensor = torch.tensor(new_states, dtype=torch.float32, device=self.device)

        # compute targets
        with torch.no_grad():
            if ALGORITHM in ['vanilla_dqn','dueling_dqn']:
                new_q_values_from_target = target_net.forward(new_states_tensor)
                max_new_q_values_from_target = new_q_values_from_target.max(dim=1, keepdim=True)[0]
                y_targets = rewards_tensor + DISCOUNT_FACTOR * max_new_q_values_from_target * (1 - dones_tensor)
            elif ALGORITHM in ['vanilla_dqv','dueling_dqv']:
                new_v_values_from_target = target_net.forward(new_states_tensor)
                y_targets = rewards_tensor + DISCOUNT_FACTOR * new_v_values_from_target * (1 - dones_tensor)
            elif ALGORITHM in ['double_dqn','double_dueling_dqn']:
                new_q_values = self.forward(new_states_tensor)
                best_new_q_values_indices = new_q_values.argmax(dim=1, keepdim=True)
                new_q_values_from_target = target_net.forward(new_states_tensor)
                best_new_q_values_from_target = torch.gather(input=new_q_values_from_target, dim=1, index=best_new_q_values_indices)
                y_targets = rewards_tensor + DISCOUNT_FACTOR * best_new_q_values_from_target * (1 - dones_tensor)
 
        # loss function
        if self.output_shape == 1:
            v_values = self.forward(states_tensor)
            loss = nn.functional.smooth_l1_loss(v_values, y_targets)
        else:
            q_values = self.forward(states_tensor)
            q_values_for_selected_action = torch.gather(input=q_values, dim=1, index=actions_tensor)
            loss = nn.functional.smooth_l1_loss(q_values_for_selected_action, y_targets)

        return loss

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, model):
        self.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))
