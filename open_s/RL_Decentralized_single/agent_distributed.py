import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from config import Config
import random
import torch.optim as optim
import math
from replay_buffer import Priority_replay_buffer, Replay_buffer
from torch.autograd import Variable

'''
    distributed agent common config
'''


class Agent_distributed:
    def __init__(self, config: Config, node_id):
        self.node_id = node_id
        self.node_num = config.NODE_NUMBER

        # input size and output size
        self.time_step = config.WINDOW_LENGTH
        self.distributed_state_dim = config.DISTRIBUTED_STATE_DIM
        self.distributed_action_dim = config.DISTRIBUTED_ACTION_DIM

        # neural setting
        self.lstm_size = config.LSTM_SIZE
        self.batch_size = config.BATCH_SIZE

        self.hold_flow_number = config.HOLD_FLOW_NUMBER

        self.hidden_dim = config.HIDDEN_DIM
        self.fc_hidden_dim = config.FC_HIDDEN_DIM

        self.device = config.DEVICE

        self.board_writer = config.BOARD_WRITER


'''
    distributed agent for dtde
'''


class Agent_distributed_dtde(Agent_distributed):
    def __init__(self, config: Config, node_id):
        super().__init__(config, node_id)
        '''init params for training'''
        self.gamma = config.GAMMA

        '''init the replay buffer'''
        self.replay_buffer = Priority_replay_buffer(config.REPLAY_SIZE)

        '''epsilon init'''
        self.epsilon_frame_idx = 0
        self.epsilon = lambda frame_idx: config.EPSILON_END \
                                         + (config.EPSILON_START - config.EPSILON_END) \
                                         * math.exp(-1 * frame_idx / config.EPSILON_DECAY)

        '''init the DQN'''
        self.current_net = q_network(distributed_state_dim=self.distributed_state_dim,
                                     action_dim=self.distributed_action_dim,
                                     hidden_dim=self.hidden_dim, fc_hidden_dim=self.hidden_dim,
                                     lstm_size=self.lstm_size,
                                     time_step=self.time_step).to(device=self.device)

        self.target_net = q_network(distributed_state_dim=self.distributed_state_dim,
                                    action_dim=self.distributed_action_dim,
                                    hidden_dim=self.hidden_dim,
                                    fc_hidden_dim=self.hidden_dim,
                                    lstm_size=self.lstm_size,
                                    time_step=self.time_step).to(device=self.device)

        for target_para, current_para in zip(self.target_net.parameters(), self.current_net.parameters()):
            target_para.data.copy_(current_para.data)

        self.optimizer = optim.Adam(self.current_net.parameters(), lr=config.LEARNING_RATE)

    ''' get action with epsilon greedy'''

    def e_greedy_action(self, state):
        self.epsilon_frame_idx += 1
        if random.random() > self.epsilon(self.epsilon_frame_idx):
            with torch.no_grad():
                state_1 = torch.tensor([state[0]], device=self.device, dtype=torch.float32)
                state_2 = torch.tensor([state[1]], device=self.device, dtype=torch.float32)
                q_vals = self.current_net(state_1, state_2)
                action = q_vals.max(1)[1].item()
                # logging.info(f"node :{self.node_id} , state_1 : {state_1}, state_2 : {state_2} , action : {action}")
        else:
            action = random.randrange(self.distributed_action_dim)

        return action

    '''get action which have max q value'''

    def greedy_action(self, state):
        with torch.no_grad():
            state_1 = torch.tensor([state[0]], device=self.device, dtype=torch.float32)
            state_2 = torch.tensor([state[1]], device=self.device, dtype=torch.float32)
            q_vals = self.current_net(state_1, state_2)
            action = q_vals.max(1)[1].item()

        return action

    def train_q_network(self):
        if self.replay_buffer.size < self.batch_size:
            return

        state_1_batch, state_2_batch, action_batch, reward_batch, next_state_1_batch, next_state_2_batch, idxs, is_weights \
            = self.replay_buffer.sample_batch(self.batch_size)

        state_1_batch = torch.tensor(state_1_batch, device=self.device, dtype=torch.float)
        state_2_batch = torch.tensor(state_2_batch, device=self.device, dtype=torch.float)
        action_batch = torch.tensor(action_batch, device=self.device, dtype=torch.int64).unsqueeze(1)
        reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float)
        next_state_1_batch = torch.tensor(next_state_1_batch, device=self.device, dtype=torch.float)
        next_state_2_batch = torch.tensor(next_state_2_batch, device=self.device, dtype=torch.float)

        q_values = self.current_net(state_1_batch, state_2_batch).gather(dim=1, index=action_batch)

        next_q_values = self.current_net(next_state_1_batch, next_state_2_batch)
        next_target_values = self.target_net(next_state_1_batch, next_state_2_batch)

        next_target_q_values = next_target_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
        expect_q_values = reward_batch + self.gamma * next_target_q_values
        expect_q_values = expect_q_values.unsqueeze(1)

        errors = torch.abs(q_values - expect_q_values).data.cpu().numpy()
        for i in range(self.batch_size):
            idx = idxs[i]
            self.replay_buffer.update(idx, errors[i])

        self.optimizer.zero_grad()

        loss = (torch.FloatTensor(is_weights).cuda() * F.mse_loss(q_values, expect_q_values)).mean()
        loss.backward()

        for para in self.current_net.parameters():
            para.grad.data.clamp_(-1, 1)

        self.optimizer.step()

    def perceive(self, state, action, reward, next_state):
        action = 0 if action == 0 else action - self.node_id * self.hold_flow_number
        state_1 = torch.tensor([state[0]], device=self.device, dtype=torch.float32)
        state_2 = torch.tensor([state[1]], device=self.device, dtype=torch.float32)

        target = self.current_net(state_1, state_2).data
        old_val = target[0][action]

        next_state_1 = torch.tensor([next_state[0]], device=self.device, dtype=torch.float32)
        next_state_2 = torch.tensor([next_state[1]], device=self.device, dtype=torch.float32)
        target_val = self.target_net(next_state_1, next_state_2).data
        target[0][action] = reward + self.gamma * torch.max(target_val)

        error = abs(old_val - target[0][action])

        self.replay_buffer.add(error.cpu(), (state[0], state[1], action, reward, next_state[0], next_state[1]))

        self.train_q_network()

    def save_current_q_network(self, save_path):
        torch.save(self.target_net.state_dict(), f"{save_path}/node_{self.node_id}_model.pth")

    def load_model(self, save_path):
        self.target_net.load_state_dict(torch.load(f"{save_path}/node_{self.node_id}_model.pth"))
        for target_para, current_para in zip(self.target_net.parameters(), self.current_net.parameters()):
            current_para.data.copy_(target_para.data)


class q_network(nn.Module):
    def __init__(self, distributed_state_dim, action_dim, hidden_dim, fc_hidden_dim, lstm_size, time_step):
        super(q_network, self).__init__()

        self.state_dim = distributed_state_dim[0]
        self.other_channel_state_dim = distributed_state_dim[1]
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.fc_hidden_dim = fc_hidden_dim
        self.lstm_size = lstm_size
        self.time_step = time_step
        self.num_layers = 1

        self.lstm = nn.LSTM(input_size=self.other_channel_state_dim, hidden_size=self.lstm_size,
                            num_layers=self.num_layers,
                            batch_first=True)
        self.fc1 = nn.Linear(self.time_step * self.lstm_size, self.hidden_dim)

        self.fc2 = nn.Linear(self.hidden_dim + self.state_dim, self.fc_hidden_dim)

        self.advantage = nn.Linear(self.fc_hidden_dim, action_dim)

        self.value = nn.Linear(self.fc_hidden_dim, 1)

    def forward(self, observation_state, state):
        # Forward propagate LSTM
        lstm_out, _ = self.lstm(observation_state)
        lstm_out = torch.reshape(lstm_out, [-1, self.time_step * self.lstm_size])
        lstm_out = self.fc1(lstm_out)
        lstm_out = torch.tanh(lstm_out)

        fc2_in = torch.cat((lstm_out, state), dim=1)
        out = self.fc2(fc2_in)
        out = torch.relu(out)

        advantage = self.advantage(out)
        value = self.value(out)

        return value + advantage - advantage.mean()
