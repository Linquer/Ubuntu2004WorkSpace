import matplotlib.pyplot as plt
import seaborn as sns

class Config:
    def __init__(self) -> None:
        self.env_name = "Breakout-v4" # 环境名字
        self.new_step_api = False # 是否用gym的新api
        self.algo_name = "PPO" # 算法名字
        self.mode = "train" # train or test
        self.seed = 1234 # 随机种子
        self.device = "cuda" # device to use
        self.train_eps = 21 # 训练的回合数
        self.test_eps = 10 # 测试的回合数
        self.max_steps = 10000 # 每个回合的最大步数
        self.eval_eps = 3 # 评估的回合数
        self.eval_per_episode = 1 # 评估的频率
        self.epsilon_start = 0.9 # epsilon初始值
        self.epsilon_end = 0.05 # epsilon最终值
        self.epsilon_decay = 500 # epsilon衰减率

        self.train_batch_size = 3 # 每次训练前收集多少轮数的数据
        self.gamma = 0.99 # 折扣因子
        self.k_epochs = 2 # 更新策略网络的次数
        self.actor_lr = 0.002 # actor网络的学习率
        self.critic_lr = 0.0003 # critic网络的学习率
        self.eps_clip = 0.2 # epsilon-clip
        self.entropy_coef = 0.1 # entropy的系数
        self.update_freq = 100 # 更新频率
        self.actor_hidden_dim = 256 # actor网络的隐藏层维度
        self.critic_hidden_dim = 64 # critic网络的隐藏层维度
        self.input_dim = 256
        self.gama_a_s = 0.25
        self.atten_std = 1
        self.atten_noise = False
        self.mid_save = False

def smooth(data, weight=0.75):  
    '''用于平滑曲线，类似于Tensorboard中的smooth曲线
    '''
    last = data[0] 
    smoothed = []
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point  # 计算平滑值
        smoothed.append(smoothed_val)                    
        last = smoothed_val                                
    return smoothed

def plot_rewards(rewards,cfg, tag='train'):
    ''' 画图
    '''
    sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title(f"{tag}ing curve on {cfg.device} of {cfg.algo_name} for {cfg.env_name}")
    plt.xlabel('epsiodes')
    plt.plot(rewards, label='rewards')
    plt.plot(smooth(rewards), label='smoothed')
    plt.legend()