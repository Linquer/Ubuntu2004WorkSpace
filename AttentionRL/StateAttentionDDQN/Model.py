import torch.nn as nn
import torch
from math import sqrt
import random
import torch.nn.functional as F

class AttentionNetwork(nn.Module):
    # input : batch_size * seq_len * input_dim
    def __init__(self, atten_input_dim, atten_output_dim, noise=False):
        super(AttentionNetwork, self).__init__()
        self.w_k = nn.Linear(atten_input_dim, atten_output_dim)
        self.w_v = nn.Linear(atten_input_dim, atten_output_dim)
        self._norm_fact = 1 / sqrt(atten_output_dim)
        self.noise = noise

    def add_attention_noise(self, atten):
        atten_std = atten.std(dim=-1).detach().view(-1, 1, 1)
        atten_mean = atten.mean(dim=-1).detach().view(-1, 1, 1)
        mask = atten < atten_mean
        atten = (atten + atten_std * 0.005) * mask + (atten) * ~mask
        return atten

    def forward(self, x, Q): # Q: batch_size * 1 * dim_k
        K = self.w_k(x) # K: batch_size * seq_len * dim_k
        V = self.w_v(x) # V: batch_size * seq_len * dim_k

        # K.permute(0, 2, 1): K.T() 就是进行一次转置
        atten = nn.Softmax(dim=-1)(torch.matmul(Q, K.permute(0, 2, 1))) * self._norm_fact # batch_size * 1 * seq_len
        # V(batch, 图片切分数, dim_k)  atten(batch, 1, 图片切分数)
        # 下式的含义：V中每个图片块的内容 * 其对应的atten值，维度依然是V(batch, 图片切分数, dim_k)
        # sum(dim=1)：是将图片块间的值相加，最终的维度为：(batch, dim_k)
        if self.training and self.noise:
            if random.random() > 0.25:
                atten = self.add_attention_noise(atten)
        output = V * atten.permute(0, 2, 1)
        output = output.sum(dim=1)
        return output, atten


class StateNetwork(nn.Module):
    def __init__(self, state_dim, state_cut_dim=16, state_Q_dim=32):
        # input: (batch_size, state_dim)
        # output: (batch_size, block_size, state_cut_dim)
        super(StateNetwork, self).__init__()
        self.state_hidden_dim0 = 32
        self.state_hidden_dim1 = 64
        self.state_hidden_dim2 = 128
        self.state_cut_dim = state_cut_dim
        self.state_Q_dim = state_Q_dim
        self.linaer_layer = nn.Sequential(
            nn.Linear(state_dim, self.state_hidden_dim0),
            nn.LeakyReLU(0.2),
            nn.Linear(self.state_hidden_dim0, self.state_hidden_dim1),
            nn.LeakyReLU(0.2),
            nn.Linear(self.state_hidden_dim1, self.state_hidden_dim2),
            nn.LeakyReLU(0.2)
        )
        self.linear_layer_Q = nn.Sequential(
            nn.Linear(state_dim, self.state_hidden_dim0),
            nn.LeakyReLU(0.2),
            nn.Linear(self.state_hidden_dim0, self.state_hidden_dim1),
            nn.LeakyReLU(0.2),
            nn.Linear(self.state_hidden_dim1, state_Q_dim),
            nn.LeakyReLU(0.2)
        )

    def cut_state(self, state):
        block_size = self.state_hidden_dim2 // self.state_cut_dim
        state = state.view(-1, block_size, self.state_cut_dim)
        return state

    def forward(self, x):
        KV = self.linaer_layer(x)
        KV = self.cut_state(KV)
        Q = self.linear_layer_Q(x).view(-1, 1, self.state_Q_dim)
        return KV, Q

class MLP(nn.Module):
    def __init__(self, n_states,n_actions,cfg,hidden_dim=64):
        """ 初始化q网络，为全连接网络
        """
        super(MLP, self).__init__()
        self.output_layer = nn.Sequential(
            nn.Linear(32, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, n_actions),
        )
        self.state_net = StateNetwork(n_states)
        self.atten_net = AttentionNetwork(16, 32, noise=cfg.atten_noise)
        self.atten = None
        
    def forward(self, x):
        KV, Q = self.state_net(x)
        atten_output, atten = self.atten_net(KV, Q)
        self.atten = atten
        action = self.output_layer(atten_output)
        return action