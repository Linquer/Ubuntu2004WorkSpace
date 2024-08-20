import os

import numpy as np
import logging
import torch
from torch.utils.tensorboard import SummaryWriter


class Config:
    def __init__(self, node_num, flow_num, traffic_id, model_path, traffic_exp_name, exp_prefix):
        self.NODE_NUMBER = node_num
        self.FLOW_NUMBER = flow_num
        self.TRAFFIC_ID = traffic_id
        self.TRAIN_MODE = "dtde"
        self.traffic_exp_name = traffic_exp_name

        '''version 1: flow_num % node_num should equal to zero '''
        if flow_num % node_num != 0:
            logging.error("error ! flow_num % node_num != 0")
            exit()

        self.HOLD_FLOW_NUMBER = int(self.FLOW_NUMBER / self.NODE_NUMBER)

        self.CENTRAL_ACTION_DIM = self.FLOW_NUMBER + 1  # schedule specific flow

        self.CENTRAL_STATE_DIM = self.FLOW_NUMBER * 2  # flows' buffer state(in NX experimental)

        self.DISTRIBUTED_ACTION_DIM = self.HOLD_FLOW_NUMBER + 1

        '''--------------------- traffic config ---------------------------'''
        if os.path.exists(f'../common/{exp_prefix}theory_throughput/{traffic_exp_name}/traffic_{self.TRAFFIC_ID}.npy'):
            self.THEORY_TIMELY_THROUGHPUT = \
                np.load(
                    f'../common/{exp_prefix}theory_throughput/{traffic_exp_name}/traffic_{self.TRAFFIC_ID}.npy')
        else:
            self.THEORY_TIMELY_THROUGHPUT = [0]

        traffic_pattern = \
            np.load(f'../common/{exp_prefix}traffic/{traffic_exp_name}/traffic_{self.TRAFFIC_ID}.npy')

        self.OFFSET = []
        self.INTER_PERIOD = []
        self.DEADLINE = []
        self.ARRIVAL_PROB = []
        self.CHANNEL_PROB = []
        for traffic_id in range(traffic_pattern.shape[0]):
            single_traffic = traffic_pattern[traffic_id]
            self.OFFSET.append(single_traffic[self.FLOW_NUMBER * 0:self.FLOW_NUMBER * 1].astype(np.int64))
            self.INTER_PERIOD.append(
                single_traffic[self.FLOW_NUMBER * 1:self.FLOW_NUMBER * 2].astype(np.int64))
            self.DEADLINE.append(single_traffic[self.FLOW_NUMBER * 2:self.FLOW_NUMBER * 3].astype(np.int64))
            self.ARRIVAL_PROB.append(single_traffic[self.FLOW_NUMBER * 3:self.FLOW_NUMBER * 4])
            self.CHANNEL_PROB.append(single_traffic[self.FLOW_NUMBER * 4:self.FLOW_NUMBER * 5])

        print('(init) the flow information :')
        print('(init) the offset : ' + str(self.OFFSET))
        print('(init) the period : ' + str(self.INTER_PERIOD))
        print('(init) the deadline : ' + str(self.DEADLINE))
        print('(init) the arrival prob : ' + str(self.ARRIVAL_PROB))
        print('(init) the channel prob : ' + str(self.CHANNEL_PROB))

        '''--------------------- model config ---------------------------'''
        self.DISTRIBUTED_LOAD_MODEL_PATH = f'{model_path}/traffic_{self.TRAFFIC_ID}'
        os.makedirs(self.DISTRIBUTED_LOAD_MODEL_PATH, exist_ok=True)

        self.PLOT_SAVE_PATH = f'{self.DISTRIBUTED_LOAD_MODEL_PATH}/plots'
        os.makedirs(self.PLOT_SAVE_PATH, exist_ok=True)

        self.TENSORBOARD_PATH = self.DISTRIBUTED_LOAD_MODEL_PATH + '/tensorboard'
        os.makedirs(self.TENSORBOARD_PATH, exist_ok=True)
        self.BOARD_WRITER = SummaryWriter(log_dir=self.TENSORBOARD_PATH)

        self.SIMULATION_TIMES_IN_TESTING_PROCESS = 3000

        '''(flow buffer state, flow packet arrival indicator)(other channel state) '''
        self.DISTRIBUTED_STATE_DIM = [self.FLOW_NUMBER * 2, 1]

        self.REPLAY_SIZE = 1000

        # self.LSTM_SIZE = 15  # the lstm size(used in node 8 and node 10)
        # self.LSTM_SIZE = 60  # the lstm size, test
        self.LSTM_SIZE = 10  # the lstm size(used in node 4)

        self.BATCH_SIZE = 64  # the batch size
        self.WINDOW_LENGTH = 0  # set after choosing traffic information

        '''for common training'''
        self.EPISODE_TIMES = 2000 if self.THEORY_TIMELY_THROUGHPUT != 0 else 1500
        self.TEST_EPISODE_TIMES = 10

        self.SIMULATION_TIMES = 0
        self.STEADY_PHASE_TIME = 0
        self.INITIAL_PHASE_TIME = 0

        self.HIDDEN_DIM = 30  # used for node 8 and node 10
        self.FC_HIDDEN_DIM = 10
        self.LEARNING_RATE = 0.0001

        '''for dtde training only'''
        self.EPSILON_START = 0.8
        self.EPSILON_END = 0.01
        self.EPSILON_DECAY = 5000
        self.GAMMA = 0.95
        self.SEND_PARAMETER_EPS = 30
        self.SAVE_MODEL_EPS = 30
        self.PARAMETER_UPDATE_DECAY = 0.1
        self.TARGET_UPDATE_EPS = 5
        self.COLLISION_PUNISH_RATIO_BEGIN = 0.2
        self.COLLISION_PUNISH_RATIO_END = 0.05
        self.COLLISION_PUNISH_DECAY = 500000
        self.N_STEP = 3

        self.FIX_TRAINING_EPISODE = 700

        self.DEVICE = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    # 由流量id重新计算网络仿真时间
    def initialize_simulation_time(self):
        # fine the lcm of all inter periods
        lcm_period = self.LCM_multiply(data=self.INTER_PERIOD[0])
        # find the max value of offset+deadline
        max_value = self.OFFSET[0][0] + self.DEADLINE[0][0]
        for i in range(1, self.FLOW_NUMBER):
            max_value = self.OFFSET[0][i] + self.DEADLINE[0][i] \
                if max_value < self.OFFSET[0][i] + self.DEADLINE[0][i] else max_value
        # find the minimum L(positive integer), makes l*lcm_value >= max_value
        L = int(max_value / lcm_period) + 1 \
            if max_value % lcm_period != 0 \
            else int(max_value / lcm_period)
        initialize_phase_time = L * lcm_period
        steady_phase_time = lcm_period
        # 设置网络仿真时间
        self.STEADY_PHASE_TIME = steady_phase_time
        self.INITIAL_PHASE_TIME = initialize_phase_time + self.STEADY_PHASE_TIME
        # self.INITIAL_PHASE_TIME = initialize_phase_time
        self.SIMULATION_TIMES = self.INITIAL_PHASE_TIME + self.STEADY_PHASE_TIME
        # the window length is equal to steady phase time
        self.WINDOW_LENGTH = self.STEADY_PHASE_TIME
        # ---------------------------------------------------------
        logging.info('the flow information :')
        logging.info(f'traffic id : {self.TRAFFIC_ID}')
        logging.info('the offset : ' + str(self.OFFSET[0]))
        logging.info('the period : ' + str(self.INTER_PERIOD[0]))
        logging.info('the deadline : ' + str(self.DEADLINE[0]))
        logging.info('the arrival prob : ' + str(self.ARRIVAL_PROB[0]))
        logging.info('the channel prob : ' + str(self.CHANNEL_PROB[0]))
        logging.info(f'theory throughput : {self.THEORY_TIMELY_THROUGHPUT}')
        logging.info(f'initial phash time : {self.INITIAL_PHASE_TIME} , window length : {self.WINDOW_LENGTH}')
        # ---------------------------------------------------------

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
