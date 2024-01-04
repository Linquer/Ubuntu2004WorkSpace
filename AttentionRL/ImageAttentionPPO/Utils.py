import copy
import torch
import numpy as np
import cv2 as cv2
import torchvision.transforms as transforms

output_agent = None
trans = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])
def get_image_state(state):
    state = state[20:210] # 切割掉无用的部分
    state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    state = cv2.resize(state, (84, 84))
    image = torch.from_numpy(state).type(torch.float32).unsqueeze(0)
    return trans(image)
def train(cfg, env, agent):
    ''' 训练
    '''
    print("开始训练！")
    count_ = 0
    rewards = []  # 记录所有回合的奖励
    steps = []
    best_ep_reward = -1000 # 记录最大回合奖励
    for i_ep in range(cfg.train_eps):
        ep_reward = 0  # 记录一回合内的奖励
        ep_step = 0
        state = env.reset()  # 重置环境，返回初始状态
        single_count = 0
        temp_reward = 0
        for i in range(cfg.max_steps):
            ep_step += 1
            # if cfg.skip_frame:
            #     if i % cfg.skip_frame != 0:
            #         next_state, reward, done, _ = env.step(action)
            #         temp_reward += reward
            #     else:
            #         state = get_image_state(state)
            #         action = agent.sample_action(state)
            #         next_state, reward, done, _ = env.step(action)
            #         temp_reward += reward
            #         agent.memory.push((state, action, agent.log_probs, temp_reward, done))
            #         temp_reward = 0
            # else:
            #     state = get_image_state(state)
            #     action = agent.sample_action(state)  # 选择动作
            #     next_state, reward, done, _ = env.step(action)  # 更新环境，返回transition
            #     agent.memory.push((state, action, agent.log_probs, reward, done))  # 保存transition
            state = get_image_state(state)
            action = agent.sample_action(state)  # 选择动作
            next_state, reward, done, _ = env.step(action)  # 更新环境，返回transition
            agent.memory.push((state, action, agent.log_probs, reward, done))  # 保存transition
            state = next_state  # 更新下一个状态
            # agent.update()  # 更新智能体
            ep_reward += reward  # 累加奖励
            count_ += 1
            single_count += 1
            if done:
                break
        agent.update()  # 更新智能体
        if (i_ep+1)%cfg.eval_per_episode == 0:
            agent.actor.eval()
            sum_eval_reward = 0
            for _ in range(cfg.eval_eps):
                eval_ep_reward = 0
                state = env.reset()
                for _ in range(cfg.max_steps):
                    state = get_image_state(state)
                    action = agent.predict_action(state)  # 选择动作
                    next_state, reward, done, _ = env.step(action)  # 更新环境，返回transition
                    state = next_state  # 更新下一个状态
                    eval_ep_reward += reward  # 累加奖励
                    if done:
                        break
                sum_eval_reward += eval_ep_reward
            agent.actor.train()
            mean_eval_reward = sum_eval_reward/cfg.eval_eps
            if mean_eval_reward >= best_ep_reward:
                best_ep_reward = mean_eval_reward
                output_agent = copy.deepcopy(agent)
                print(f"回合：{i_ep+1}/{cfg.train_eps}，奖励：{ep_reward:.2f}，评估奖励：{mean_eval_reward:.2f}，最佳评估奖励：{best_ep_reward:.2f}，更新模型！，动作次数：{single_count} {agent.epsilon:.2f}")
            else:
                print(f"回合：{i_ep+1}/{cfg.train_eps}，奖励：{ep_reward:.2f}，评估奖励：{mean_eval_reward:.2f}，最佳评估奖励：{best_ep_reward:.2f}，动作次数：{single_count}{agent.epsilon:.2f}")
            if cfg.mid_save:
                if mean_eval_reward == 200:
                    torch.save(agent, f"./Data/CartPole-v0-StateAttention-None/{i_ep+1}-{cfg.train_eps}.pt")
        steps.append(ep_step)
        rewards.append(ep_reward)
    print("完成训练！")
    print("一共收集 ", count_, " 次数据")
    env.close()
    return output_agent,{'rewards':rewards}

def test(cfg, env, agent):
    print("开始测试！")
    rewards = []  # 记录所有回合的奖励
    steps = []
    agent.actor.eval()
    for i_ep in range(cfg.test_eps):
        ep_reward = 0  # 记录一回合内的奖励
        ep_step = 0
        state = env.reset()  # 重置环境，返回初始状态
        for _ in range(cfg.max_steps):
            ep_step+=1
            state = get_image_state(state)
            action = agent.predict_action(state)  # 选择动作
            next_state, reward, done, _ = env.step(action)  # 更新环境，返回transition
            state = next_state  # 更新下一个状态
            ep_reward += reward  # 累加奖励
            if done:
                break
        steps.append(ep_step)
        rewards.append(ep_reward)
        print(f"回合：{i_ep+1}/{cfg.test_eps}，奖励：{ep_reward:.2f}")
    agent.actor.train()
    print("完成测试")
    env.close()
    return {'rewards':rewards}

def to_tensor(x, cfg):
    return torch.tensor(np.array(x), device=cfg.device, dtype=torch.float32)

import matplotlib.pyplot as plt
from IPython import display
import numpy as np


def use_svg_display():
    # 用矢量图显示
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置matplotlib的轴。 """
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
        ylim=None, xscale='linear', yscale='linear',
        fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """绘制数据点。 """
    if legend is None:
        legend = []
    set_figsize(figsize)
    axes = axes if axes else plt.gca()
    # 如果 `X` 有⼀个轴，输出True
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                    and not hasattr(X[0], "__len__"))
    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)