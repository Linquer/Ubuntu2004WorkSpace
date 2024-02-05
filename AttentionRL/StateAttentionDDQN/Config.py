import matplotlib.pyplot as plt
import seaborn as sns
import torch
class Config:
    def __init__(self):
        self.algo_name = 'DoubleDQN' # 算法名称
        self.env_name = 'CartPole-v1' # 环境名称
        self.seed = 1 # 随机种子
        self.train_eps = 100 # 训练回合数
        self.test_eps = 10  # 测试回合数
        self.max_steps = 200 # 每回合最大步数
        self.gamma = 0.99 # 折扣因子
        self.lr = 0.001 # 学习率
        self.epsilon_start = 0.95 # epsilon初始值
        self.epsilon_end = 0.01 # epsilon最终值
        self.epsilon_decay = 500 # epsilon衰减率
        self.buffer_size = 10000 # ReplayBuffer容量
        self.batch_size = 64 # ReplayBuffer中批次大小
        self.target_update = 4 # 目标网络更新频率
        self.hidden_dim = 256 # 神经网络隐藏层维度
        self.atten_noise = False
        self.mid_save = False
        if torch.cuda.is_available(): # 是否使用GPUs
            self.device = 'cuda'
        else:
            self.device = 'cpu'
def smooth(data, weight=0.9):  
    '''用于平滑曲线，类似于Tensorboard中的smooth曲线
    '''
    last = data[0] 
    smoothed = []
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point  # 计算平滑值
        smoothed.append(smoothed_val)                    
        last = smoothed_val                                
    return smoothed

def plot_rewards(rewards,title="learning curve"):
    sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title(f"{title}")
    plt.xlim(0, len(rewards), 10)  # 设置x轴的范围
    plt.xlabel('epsiodes')
    plt.plot(rewards, label='rewards')
    plt.plot(smooth(rewards), label='smoothed')
    plt.legend()

def print_cfgs(cfg):
    ''' 打印参数
    '''
    cfg_dict = vars(cfg)
    print("Hyperparameters:")
    print(''.join(['=']*80))
    tplt = "{:^20}\t{:^20}\t{:^20}"
    print(tplt.format("Name", "Value", "Type"))
    for k,v in cfg_dict.items():
        if v.__class__.__name__ == 'list':
            v = str(v)
        print(tplt.format(k,v,str(type(v))))   
    print(''.join(['=']*80))
