import os

import matplotlib.pyplot as plt
import numpy as np


# meta agent action reward and the time of meta agent using
def meta_reward_cnt_for_show(x, avg_reward):
    fig = plt.figure()
    ax = fig.add_subplot()

    ax.plot(x, avg_reward, label='avg_reward', color='red')
    ax.set_ylabel("reward")
    ax.legend(loc='upper left')

    plt.title('training reward')
    ax.set_xlabel("eps")

    plt.show()


def meta_reward_cnt_for_save(x, avg_reward, save_path):
    fig = plt.figure()
    ax = fig.add_subplot()

    ax.plot(x, avg_reward, label='avg_reward', color='red')
    ax.set_ylabel("reward")
    ax.legend(loc='upper left')

    plt.title('training reward')
    ax.set_xlabel("eps")

    print(save_path)
    plt.draw()
    plt.savefig(f'{save_path}/meta_reward_cnt.jpg', bbox_inches='tight', dpi=600)
    plt.close()

    np.save(f'{save_path}/without_lifelong_avg_reward.npy', np.array(avg_reward))


if __name__ == '__main__':
    x = [1, 2, 3, 4]
    avg_reward = [0.1, 0.1, 0.5, 0.7]
    meta_reward_cnt_for_show(x=x, avg_reward=avg_reward)
    # meta_reward_cnt_for_save(x, y, z, './gen_model')
