import torch
import torch.nn.functional as F
import torch.nn as nn
from math import sqrt
import random


class ConvNetwork(nn.Module):
    def __init__(self) -> None:
        super(ConvNetwork, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 8, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(8, 4, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(4, 1, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2)
        )
        self.conv_layers_Q = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 8, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(8, 1, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2)
        )
        self.linear_layers_Q = nn.Sequential(
            nn.Linear(540, 300),
            nn.LeakyReLU(0.2),
            nn.Linear(300, 256),
            nn.LeakyReLU(0.2)
        )
        self.flatten = nn.Flatten(start_dim=-2, end_dim=-1)
        self.x_pix_size = 20
        self.y_pix_size = 20

    def cut_image(self, image):
        '''
        input: (batch, channel(1), H, W)
        output: (batch, block_size, h, w)
        '''
        x_pix_size = self.x_pix_size
        y_pix_size = 20
        i_num, j_num = int(image.shape[2] / x_pix_size), int(image.shape[3] / y_pix_size)
        batch = image.size()[0]
        output = torch.zeros(batch, i_num * j_num, y_pix_size, x_pix_size)
        for b in range(batch):
            block_count = 0
            for i in range(i_num):
                for j in range(j_num):
                    sub_img = image[b, 0, i * y_pix_size: (i + 1) * y_pix_size, j * x_pix_size: (j + 1) * x_pix_size]
                    output[b, block_count] = sub_img
                    block_count += 1
        return output
        
    
    def forward(self, X):
        X = self.conv_layers(X)
        X_Q = self.conv_layers_Q(X)
        X_Q = self.flatten(X_Q)
        X_Q = self.linear_layers_Q(X_Q)
        X_KV = ConvNetwork.cut_image(self, X)
        X_KV = self.flatten(X_KV)
        return X_KV, X_Q


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


class ActorSoftmax(nn.Module):
    def __init__(self, n_states, n_actions, noise=False, device='cpu'):
        super(ActorSoftmax, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.atten = None
        self.input_dim = 256
        self.hidden_dim0 = 128
        self.hidden_dim1 = 64
        self.hidden_dim2 = 16
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
        self.device = device
        self.state_net = ConvNetwork().to(device)
        self.atten_net = AttentionNetwork(400, 256, noise=noise).to(device)
    
    def forward(self, x):
        KV, Q = self.state_net(x)
        atten_output, atten = self.atten_net(KV.to(self.device), Q.to(self.device))
        self.atten = atten
        action = self.output_layer(atten_output)
        return F.softmax(action, dim=1)

class Critic(nn.Module):
    def __init__(self,input_dim=128,output_dim=1,hidden_dim=64):
        super(Critic,self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 16, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2)
        )
        self.flatten = nn.Flatten()
        self.linear_layers = nn.Sequential(
            nn.Linear(560, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2)
        )
        assert output_dim == 1 # critic must output a single value
        self.mid_hidden_dim = 32
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
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self,x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.linear_layers(x)
        x1 = self.residual_block1(x)
        x1 = F.relu(x1)
        x2 = self.residual_block2(x1)
        x2 = F.relu(x2) + x1
        x3 = self.residual_block3(x2)
        x3 = F.relu(x3)
        return x3

