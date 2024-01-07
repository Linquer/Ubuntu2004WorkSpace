import torch
import torch.nn.functional as F
import torch.nn as nn
from math import sqrt
import random

input_dim_global = 128

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
    def __init__(self, state_dim, state_cut_dim=64, state_Q_dim=128):
        # input: (batch_size, state_dim)
        # output: (batch_size, block_size, state_cut_dim)
        super(StateNetwork, self).__init__()
        self.state_hidden_dim0 = 128
        self.state_hidden_dim1 = 192
        self.state_hidden_dim2 = 256
        self.state_hidden_dim3 = 256
        self.state_hidden_dim4 = 512
        self.state_cut_dim = state_cut_dim
        self.state_Q_dim = state_Q_dim
        self.linaer_layer = nn.Sequential(
            nn.Linear(state_dim, self.state_hidden_dim0),
            nn.LeakyReLU(0.2),
            nn.Linear(self.state_hidden_dim0, self.state_hidden_dim1),
            nn.LeakyReLU(0.2),
            nn.Linear(self.state_hidden_dim1, self.state_hidden_dim2),
            nn.LeakyReLU(0.2),
            nn.Linear(self.state_hidden_dim2, self.state_hidden_dim3),
            nn.LeakyReLU(0.2),
            nn.Linear(self.state_hidden_dim3, self.state_hidden_dim4),
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
        block_size = self.state_hidden_dim4 // self.state_cut_dim
        state = state.view(-1, block_size, self.state_cut_dim)
        return state

    def forward(self, x):
        KV = self.linaer_layer(x)
        KV = self.cut_state(KV)
        Q = self.linear_layer_Q(x).view(-1, 1, self.state_Q_dim)
        return KV, Q

class ActorSoftmax(nn.Module):
    def __init__(self, n_states, n_actions, noise=False):
        super(ActorSoftmax, self).__init__()
        self.n_states = input_dim_global
        self.n_actions = n_actions
        self.atten = None
        self.input_dim = 128
        self.hidden_dim0 = 96
        self.hidden_dim1 = 64
        self.hidden_dim2 = 32
        self.output_layer = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim0),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_dim0, self.hidden_dim1),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_dim1, self.hidden_dim2),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_dim2, self.n_actions),
            nn.LeakyReLU(0.2)
        )
        self.state_net = StateNetwork(self.n_states)
        self.atten_net = AttentionNetwork(64, 128, noise=noise)
    
    def forward(self, x):
        KV, Q = self.state_net(x)
        atten_output, atten = self.atten_net(KV, Q)
        self.atten = atten
        action = self.output_layer(atten_output)
        return F.softmax(action, dim=1)

class Critic(nn.Module):
    def __init__(self,input_dim=128,output_dim=1,hidden_dim=256):
        super(Critic,self).__init__()
        input_dim = input_dim_global
        assert output_dim == 1 # critic must output a single value
        self.mid_hidden_dim = 128
        self.residual_block1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, self.mid_hidden_dim)
        )
        self.residual_block2 = nn.Sequential(
            nn.Linear(self.mid_hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, self.mid_hidden_dim)
        )
        self.residual_block3 = nn.Sequential(
            nn.Linear(self.mid_hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, self.mid_hidden_dim)
        )
        self.residual_block4 = nn.Sequential(
            nn.Linear(self.mid_hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self,x):
        x1 = self.residual_block1(x)
        x1 = F.leaky_relu(x1, 0.5)
        x2 = self.residual_block2(x1)
        x2 = F.leaky_relu(x2, 0.5) + x1
        x3 = self.residual_block3(x2)
        x3 = F.leaky_relu(x3, 0.8) + x2
        x4 = self.residual_block4(x3)
        return x4

