import logging
import random
import torch
import numpy as np
from collections import deque
import torch.nn as nn
import torch.nn.functional as F
from config import CentralizedConfig
import torch.optim as optim
from replay_buffer import Priority_replay_buffer


class Agent:
    def __init__(self, config: CentralizedConfig):
        self.initial_epsilon = config.INITIAL_EPSILON
        self.final_epsilon = config.FINAL_EPSILON
        self.decrease_time_epsilon = config.DECREASE_TIME_EPSILON

        self.replace_target_freq = config.REPLACE_TARGET_FREQ

        self.replay_size = config.REPLAY_SIZE
        self.batch_size = config.BATCH_SIZE
        self.gamma = config.GAMMA

        # init experience replay
        self.replay_buffer = Priority_replay_buffer(capacity=self.replay_size)
        # epsilon
        self.epsilon = config.INITIAL_EPSILON

        # output data : select node to send packet or not
        self.action_dim = config.ACTION_DIM

        # input data : node state
        self.state_dim = config.STATE_DIM

        self.device = config.DEVICE

        self.learning_rate = config.LEARNING_RATE

        self.current_net = MLP(self.state_dim, self.action_dim).to(self.device)
        self.target_net = MLP(self.state_dim, self.action_dim).to(self.device)
        for target_para, policy_para in zip(self.target_net.parameters(), self.current_net.parameters()):
            target_para.data.copy_(policy_para.data)

        self.optimizer = optim.Adam(self.current_net.parameters(), lr=self.learning_rate)
        self.fairness_alpha = config.FAIRNESS_ALPHA

    # experience buffer pool
    def perceive(self, state, action, reward, next_state, end, episode):
        state_ = torch.tensor([state], device=self.device, dtype=torch.float)
        target = self.current_net(state_).data
        old_val = target[0][action]

        next_state_ = torch.tensor([next_state], device=self.device, dtype=torch.float)
        target_val = self.target_net(next_state_).data
        target[0][action] = reward + self.gamma * torch.max(target_val)

        error = abs(old_val - target[0][action])
        self.replay_buffer.add(error.cpu(), (state, action, reward, next_state, end))

        self.train_network(episode)

    def train_network(self, episode):
        # Step 1: obtain random mini batch from replay memory
        state_batch, action_batch, reward_batch, next_state_batch, done_batch, idxs, is_weights = self.replay_buffer.sample_batch(self.batch_size)

        state_batch = torch.tensor(state_batch, device=self.device, dtype=torch.float)
        action_batch = torch.tensor(action_batch, device=self.device, dtype=torch.int64).unsqueeze(1)
        reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float)
        next_state_batch = torch.tensor(next_state_batch, device=self.device, dtype=torch.float)
        done_batch = torch.tensor(done_batch, device=self.device, dtype=torch.float)

        # logging.info(f"state batch : \n{state_batch}")
        # logging.info(f"action_batch : \n{action_batch}")
        # logging.info(f"reward_batch : \n{reward_batch}")
        # logging.info(f"next_state_batch : \n{next_state_batch}")
        # logging.info(f"done_batch : \n{done_batch}")

        q_values = self.current_net(state_batch).gather(dim=1, index=action_batch)

        next_q_values = self.current_net(next_state_batch)
        next_target_values = self.target_net(next_state_batch)

        next_target_q_values = next_target_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
        expect_q_values = reward_batch + self.gamma * next_target_q_values * (1 - done_batch)
        expect_q_values = expect_q_values.unsqueeze(1)

        errors = torch.abs(q_values - expect_q_values).cpu().data.numpy()
        for i in range(self.batch_size):
            idx = idxs[i]
            self.replay_buffer.update(idx, errors[i])

        self.optimizer.zero_grad()

        loss = (torch.FloatTensor(is_weights).cuda() * F.mse_loss(q_values, expect_q_values)).mean()
        loss.backward()

        for para in self.current_net.parameters():
            para.grad.data.clamp_(-1, 1)

        self.optimizer.step()

    def update_target_q_network(self, episode):
        # update target Q network
        if episode != 0 and episode % self.replace_target_freq == 0:
            self.target_net.load_state_dict(self.current_net.state_dict())

    # action selection used for training
    def e_greedy_action(self, state, episode):
        self.epsilon = self.get_epsilon(episode=episode)
        if random.random() <= self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            # ！！！！效用函数改变的话，更新的地方！！！！
            # 先将所有Q_value计算f(Q_value)，再选np.argmax(f(Q_value))
            state = torch.tensor([state], device=self.device, dtype=torch.float)
            q_vals = self.current_net(state)
            return q_vals.max(1)[1].item()

    # action selection used for testing
    def greedy_action(self, state):
        state = torch.tensor([state], device=self.device, dtype=torch.float)
        q_vals = self.current_net(state)
        return q_vals.max(1)[1].item()

    # 探索策略更新
    def get_epsilon(self, episode):
        present_epsilon = self.initial_epsilon - (self.initial_epsilon - self.final_epsilon) / \
                          self.decrease_time_epsilon * episode
        return max(self.final_epsilon, present_epsilon)

    def save_current_q_network(self, save_path):
        torch.save(self.target_net.state_dict(), save_path + "/model.pth")

    def load_model(self, path):
        self.current_net.load_state_dict(torch.load(path + "/model.pth"))
        # for target_para, current_para in zip(self.target_net.parameters(), self.current_net.parameters()):
        #     current_para.copy_(target_para)


class MLP(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(state_dim, 20)

        self.advantage = nn.Linear(20, action_dim)

        self.value = nn.Linear(20, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))

        advantage = self.advantage(x)
        value = self.value(x)

        return value + advantage - advantage.mean()

