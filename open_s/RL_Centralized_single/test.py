import time
from agent import Agent
from config import CentralizedConfig
from environment import Environment
import sys
import logging
import numpy as np
import os
import argparse


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


def parse_args():
    parser = argparse.ArgumentParser(description='Run parameters')
    parser.add_argument('--flow_num', type=int, default=2)
    parser.add_argument('--traffic_id_begin', type=int, default=100)
    parser.add_argument('--traffic_id_end', type=int, default=199)
    parser.add_argument('--model_suffix', type=str, default="local_test")
    parser.add_argument('--traffic_prefix', type=str, default="c_gen_")
    parser.add_argument('--traffic_exp_name', type=str, default="flow_2_p_1")

    return parser.parse_args()


# main：主程序
if __name__ == '__main__':
    args = parse_args()
    print(f"args : {args}")

    flow_num = args.flow_num
    traffic_id_begin = args.traffic_id_begin
    traffic_id_end = args.traffic_id_end
    model_suffix = args.model_suffix
    traffic_exp_name = args.traffic_exp_name
    traffic_prefix = args.traffic_prefix

    model_path = f'./{traffic_prefix}model/{traffic_id_begin}_to_{traffic_id_end}_{model_suffix}/{traffic_exp_name}'
    os.makedirs(model_path, exist_ok=True)
    config_logging(file_name=f'{model_path}/test_log.log')

    logging.info(f"pid : {os.getpid()}")
    testbed_res = []

    for traffic_id in range(traffic_id_begin, traffic_id_end + 1):

        logging.info(f"traffic id : {traffic_id}")
        config = CentralizedConfig(flow_num=flow_num, traffic_id=traffic_id, model_path=model_path, traffic_exp_name=traffic_exp_name,
                                   traffic_prefix=traffic_prefix)
        # 环境
        environment = Environment(config=config)
        # 智能体
        agent = Agent(config=config)
        agent.load_model(config.NEURAL_NETWORK_PATH)
        # 智能体与仿真时间初始化
        config.initialize_simulation_time(traffic_id=0)
        # stats info
        train_begin_time = 0
        train_end_time = 0
        total_train_time = 0
        # best info
        best_train_time = 0
        best_train_episode = 0
        best_timely_throughput = 0

        environment.initialize(config=config, traffic_id=0)
        present_state = environment.get_state(time_slot=1)
        action_cnt = [0 for _ in range(config.ACTION_DIM)]

        # plot var
        plot_state_val = [[] for _ in range(flow_num)]
        plot_send_ind = [[] for _ in range(flow_num)]
        plot_expire_ind = [[] for _ in range(flow_num)]
        plot_collision_ind = [[] for _ in range(flow_num)]
        plot_arrive_ind = [[] for _ in range(flow_num)]
        plot_next_arrive_ind = [[] for _ in range(flow_num)]
        block_cnt = 0
        plot_state = []

        for time_slot in range(1, 10000 + 1):
            # make action
            # logging.info(f"state :{present_state}")
            action = agent.greedy_action(state=present_state)
            action_cnt[action] += 1
            # environment step and give feedback
            plot_state = present_state
            next_state, reward, end, expire_list, generate_list = environment.step(time_slot=time_slot, action=action)
            # update the present observation
            present_state = next_state

            # for idx in range(flow_num):
            #     if idx+1 == action:
            #         plot_send_ind[idx].append(1)
            #     else:
            #         plot_send_ind[idx].append(0)
            #
            #     plot_collision_ind[idx].append(0)
            #     plot_state_val[idx].append(plot_state[idx]/10)
            #     plot_next_arrive_ind[idx].append(plot_state[flow_num + idx])
            #     plot_expire_ind[idx].append(expire_list[idx])
            #     plot_arrive_ind[idx].append(generate_list[idx])
            #
            # if time_slot % 20 == 0 and time_slot >= 300:
            #     print("plot")
            #     scheduling_illustration_for_compare("RL_Cent", block_cnt, plot_state_val, plot_send_ind, plot_expire_ind, plot_collision_ind,
            #                                         plot_arrive_ind, plot_next_arrive_ind)
            #     plot_state_val = [[] for _ in range(flow_num)]
            #     plot_send_ind = [[] for _ in range(flow_num)]
            #     plot_expire_ind = [[] for _ in range(flow_num)]
            #     plot_collision_ind = [[] for _ in range(flow_num)]
            #     plot_arrive_ind = [[] for _ in range(flow_num)]
            #     plot_next_arrive_ind = [[] for _ in range(flow_num)]
            #     block_cnt += 1


        total_send_before_expiration, \
        total_generate_packet_count, \
        mac_delay_from_queue_to_send, \
        mac_delay_from_head_to_send, \
        mac_delay_from_queue_to_send_success, \
        mac_delay_from_head_to_send_success = environment.get_stats()
        present_timely_throughput = total_send_before_expiration / total_generate_packet_count

        testbed_res.append(present_timely_throughput)
        # print('(', episode, ') timely throughput : ', present_timely_throughput,
        #       'train time : ', total_train_time)
        # _success后缀的是指统计成功到达目的节点的数据包，最后用的_success
        logging.info('timely throughput : ' + str(present_timely_throughput) +
                     ',mac delay(queue, total packet) : ' + str(round(mac_delay_from_queue_to_send, 5)) +
                     'mac delay(head, total packet) : ' + str(round(mac_delay_from_head_to_send, 5)) +
                     'mac delay(queue, success packet) : ' + str(round(mac_delay_from_queue_to_send_success, 5)) +
                     'mac delay(head, success packet) : ' + str(round(mac_delay_from_head_to_send_success, 5)) +
                     'action cnt : ' + str(action_cnt))
        logging.info("End")

    os.makedirs("./data/RL_Cent_simu_res", exist_ok=True)
    np.save(f"./data/RL_Cent_simu_res/throughput_node_{flow_num}_simu.npy", testbed_res)
