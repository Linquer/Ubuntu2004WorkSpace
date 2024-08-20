import logging
import os
import sys
import torch
import numpy as np
from constant import DYNAMIC_TRAFFIC_MAP


def config_logging(file_name: str, console_level: int = logging.INFO, file_level: int = logging.INFO):
    file_handler = logging.FileHandler(file_name, mode='w', encoding="utf8")
    file_handler.setFormatter(logging.Formatter(
        '%(levelname)s %(asctime)s-%(filename)s[line:%(lineno)d] %(message)s'
    ))
    file_handler.setLevel(file_level)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(
        '%(levelname)s %(asctime)s-%(filename)s[line:%(lineno)d] %(message)s',
    ))
    console_handler.setLevel(console_level)

    logging.basicConfig(
        level=min(console_level, file_level),
        handlers=[file_handler, console_handler],
    )


class CentralizedConfig:

    def __init__(self, node_count, traffic_begin_id, traffic_end_id, traffic_exp_name, traffic_prefix, model_suffix, test=False):

        self.TRAFFIC_EXP_NAME = traffic_exp_name

        self.TRAFFIC_START = traffic_begin_id
        self.TRAFFIC_END = traffic_end_id

        self.FLOW_NUMBER = node_count  # 数据流数量
        self.NODE_PAIR_NUMBER = node_count

        self.STATE_DIM = 7 * self.FLOW_NUMBER  # 状态维度(emergency, arrival indicator)
        self.ACTION_DIM = self.FLOW_NUMBER + 1  # 动作维度(调度某个数据流 + 不调度)

        if not test:
            self.NEURAL_NETWORK_PATH = f'./{traffic_prefix}model/{traffic_exp_name}_{model_suffix}/flow_{node_count}_traffic_{traffic_begin_id}_to_{traffic_end_id}'
        else:
            self.NEURAL_NETWORK_PATH = f'./{traffic_prefix}model/{traffic_exp_name}_{model_suffix}/flow_{node_count}_traffic_0_to_{DYNAMIC_TRAFFIC_MAP[node_count]}'

        os.makedirs(self.NEURAL_NETWORK_PATH, exist_ok=True)

        if test:
            config_logging(self.NEURAL_NETWORK_PATH + '/test_log.log')
        else:
            config_logging(self.NEURAL_NETWORK_PATH + '/train_log.log')

        traffic_path = f'../common/{traffic_prefix}traffic/{traffic_exp_name}'
        traffic_data_set = []
        theory_throughput_path = f'../common/{traffic_prefix}theory_throughput/{traffic_exp_name}'

        logging.info(traffic_path)
        self.THEORY_THROUGHPUT = []
        for traffic_id in range(0, DYNAMIC_TRAFFIC_MAP[node_count] + 100 + 1):
            file_path = traffic_path + f"/traffic_{traffic_id}.npy"

            traffic_info = np.load(file_path)
            traffic_info = np.squeeze(traffic_info, axis=0)
            traffic_data_set.append(traffic_info)

            if os.path.exists(theory_throughput_path + f"/traffic_{traffic_id}.npy"):
                self.THEORY_THROUGHPUT.append(np.load(theory_throughput_path + f"/traffic_{traffic_id}.npy"))
            else:
                self.THEORY_THROUGHPUT.append([0])

        traffic_data_set = np.array(traffic_data_set)
        self.TRAFFICS_COUNT = traffic_data_set.shape[0]
        self.OFFSET = []
        self.INTER_PERIOD = []
        self.DEADLINE = []
        self.ARRIVAL_PROB = []
        self.CHANNEL_PROB = []
        for traffic_id in range(self.TRAFFICS_COUNT):
            single_traffic = traffic_data_set[traffic_id]
            self.OFFSET.append(single_traffic[self.NODE_PAIR_NUMBER * 0:self.NODE_PAIR_NUMBER * 1].astype(np.int64))
            self.INTER_PERIOD.append(
                single_traffic[self.NODE_PAIR_NUMBER * 1:self.NODE_PAIR_NUMBER * 2].astype(np.int64))
            self.DEADLINE.append(single_traffic[self.NODE_PAIR_NUMBER * 2:self.NODE_PAIR_NUMBER * 3].astype(np.int64))
            self.ARRIVAL_PROB.append(single_traffic[self.NODE_PAIR_NUMBER * 3:self.NODE_PAIR_NUMBER * 4])
            self.CHANNEL_PROB.append(single_traffic[self.NODE_PAIR_NUMBER * 4:self.NODE_PAIR_NUMBER * 5])

        # 自定义流量模型
        # self.OFFSET = [[0, 1, 2, 3]]
        # self.INTER_PERIOD = [[2, 4, 2, 4]]
        # self.DEADLINE = [[4, 8, 4, 8]]
        # self.ARRIVAL_PROB = [[1, 1, 1, 1]]
        # self.CHANNEL_PROB = [[0.8, 0.9, 0.8, 0.9]]

        self.LAMBDA_VALUE = 0

        # 仿真时间设置(仿真时间从time slot 1开始)
        self.SIMULATION_TIMES = 0
        self.STEADY_PHASE_TIME = 0
        self.INITIAL_PHASE_TIME = 0

        # self.SIMULATION_TIME_IN_TEST = 100000
        self.SIMULATION_TIME_IN_TEST = 7000

        # ---------------- for training ---------------------
        self.INITIAL_EPSILON = 0.9  # initial epsilon
        self.FINAL_EPSILON = 0.001  # final epsilon
        # self.DECREASE_TIME_EPSILON = 20000  # the number of time from initial epsilon to final epsilon
        self.DECREASE_TIME_EPSILON = 2000000  # the number of time from initial epsilon to final epsilon

        self.REPLAY_SIZE = 150000  # experience replay buffer size
        self.BATCH_SIZE = 128*10  # size of mini batch

        self.GAMMA = 0.95  # before discount rate
        # self.GAMMA = 1.0 # after discount rate
        self.LEARNING_RATE = 0.0001

        # self.EPISODE_TIMES = 10000  # 总训练episode次数
        self.EPISODE_TIMES = 1000  # 总训练episode次数
        self.REPLACE_TARGET_FREQ = 10  # 每隔多少个episode更新目标Q网络参数

        # for testing
        self.TEST_EPISODE_TIMES = 1  # 每隔多少次episode进行测试

        logging.info('the flow information :')
        # logging.info('the offset : ' + str(self.OFFSET))
        # logging.info('the period : ' + str(self.INTER_PERIOD))
        # logging.info('the deadline : ' + str(self.DEADLINE))
        # logging.info('the arrival prob : ' + str(self.ARRIVAL_PROB))
        # logging.info('the channel prob : ' + str(self.CHANNEL_PROB))
        # logging.info('the theory throughput : ' + str(self.THEORY_TIMELY_THROUGHPUT))

        self.DEVICE = torch.device("cuda")
        self.FAIRNESS_ALPHA = 0

        self.FIX_TRAIN_EPS = -1

    # 由流量id重新计算网络仿真时间
    def initialize_simulation_time(self, traffic_id):
        # fine the lcm of all inter periods
        lcm_period = self.LCM_multiply(data=self.INTER_PERIOD[traffic_id])
        # find the max value of offset+deadline
        max_value = self.OFFSET[traffic_id][0] + self.DEADLINE[traffic_id][0]
        for i in range(1, self.FLOW_NUMBER):
            max_value = self.OFFSET[traffic_id][i] + self.DEADLINE[traffic_id][i] if max_value < \
                                                                                     self.OFFSET[traffic_id][i] + \
                                                                                     self.DEADLINE[traffic_id][
                                                                                         i] else max_value
        # fine the minimum L(positive integer), makes l*lcm_value >= max_value
        L = round(max_value / lcm_period) + 1 if max_value % lcm_period != 0 else round(max_value / lcm_period)
        initialize_phase_time = L * lcm_period
        steady_phase_time = lcm_period
        # 设置网络仿真时间
        self.STEADY_PHASE_TIME = steady_phase_time
        self.INITIAL_PHASE_TIME = initialize_phase_time
        self.SIMULATION_TIMES = self.INITIAL_PHASE_TIME + self.STEADY_PHASE_TIME
        # logging.info(f'simulation time : {self.SIMULATION_TIMES}')
        return self.SIMULATION_TIMES

    @classmethod
    def LCM_multiply(cls, data):
        if len(data) == 0:
            return 0
        elif len(data) == 1:
            return data[0]
        else:
            result = cls.LCM_double(data1=data[0], data2=data[1])
            for i in range(2, len(data)):
                result = cls.LCM_double(data1=result, data2=data[i])
        return result

    @classmethod
    def LCM_double(cls, data1, data2):
        if data1 > data2:
            greater = data1
        else:
            greater = data2
        while True:
            if (greater % data1 == 0) and (greater % data2 == 0):
                lcm = greater
                break
            greater += 1
        return lcm
