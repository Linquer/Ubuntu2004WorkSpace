import numpy as np

from agent import Agent
from config import CentralizedConfig
from environment import Environment
import sys
import logging
import os
import argparse
import time


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
    parser.add_argument('--traffic_id_begin', type=int, default=0)
    parser.add_argument('--traffic_id_end', type=int, default=10)
    parser.add_argument('--model_suffix', type=str, default="test")
    parser.add_argument('--traffic_prefix', type=str, default="c_")
    parser.add_argument('--traffic_exp_name', type=str, default="flow_2")

    return parser.parse_args()


# main：主程序
if __name__ == '__main__':

    args = parse_args()

    flow_num = args.flow_num
    traffic_id_begin = args.traffic_id_begin
    traffic_id_end = args.traffic_id_end
    model_suffix = args.model_suffix
    traffic_exp_name = args.traffic_exp_name
    traffic_prefix = args.traffic_prefix

    model_path = f'./{traffic_prefix}model/{traffic_id_begin}_to_{traffic_id_end}_{model_suffix}/{traffic_exp_name}'
    os.makedirs(model_path, exist_ok=True)
    config_logging(file_name=f'{model_path}/log.log')

    logging.info(f'args : {args}')
    logging.info(f"pid : {os.getpid()}")

    for traffic_id in range(traffic_id_begin, traffic_id_end + 1):
        # if not os.path.exists(f'../common/{traffic_prefix}theory_throughput/{traffic_exp_name}/traffic_{traffic_id}.npy'):
        #     logging.error(
        #         f"theory throughput is not exist at path : ../common/{traffic_prefix}theory_throughput/{traffic_exp_name}/traffic_{traffic_id}.npy")
        #     continue

        logging.info(f"flow : {flow_num} , traffic id : {traffic_id}")
        # 配置
        config = CentralizedConfig(flow_num=flow_num, traffic_id=traffic_id, model_path=model_path, traffic_exp_name=traffic_exp_name,
                                   traffic_prefix=traffic_prefix)
        # log config

        # 环境
        environment = Environment(config=config)
        # 智能体
        agent = Agent(config=config, )
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

        get_theory_cnt = 0

        # list for plot
        plot_running_timely_throughput = []

        # training phase
        for episode in range(1, config.EPISODE_TIMES + 1):
            # 记录每个episode的训练开始时间
            train_begin_time = time.time()
            # 环境初始化
            environment.initialize(config=config, traffic_id=0)
            present_state = environment.get_state(time_slot=1)
            '''
             -------------------- training process begin ---------------------
            '''
            for time_slot in range(1, config.SIMULATION_TIMES * 2 + 1):
                # make action
                action = agent.e_greedy_action(state=present_state, episode=episode)
                # environment step and give feedback
                next_state, reward, end, _, _ = environment.step(time_slot=time_slot, action=action)
                # !!!change, before code have no this constraint
                # if time_slot >= config.INITIAL_PHASE_TIME + 1:
                # agent perceive the MDP information(state, action, reward, next_state, done)
                agent.perceive(state=present_state,
                               action=action,
                               reward=reward,
                               next_state=next_state,
                               end=end,
                               episode=episode)
                # update the present observation
                present_state = next_state
            # 记录每个episode的训练结束时间
            train_end_time = time.time()
            # 将单个episode的训练时间添加到总时间
            total_train_time += train_end_time - train_begin_time
            # update the Q target network parameters
            agent.update_target_q_network(episode=episode)
            '''
            ----------------------------- testing phase ---------------------------
            '''
            if episode != 0 and episode % config.TEST_EPISODE_TIMES == 0:
                # 环境初始化
                environment.initialize(config=config, traffic_id=0)
                present_state = environment.get_state(time_slot=1)
                action_cnt = [0 for _ in range(flow_num + 1)]
                for time_slot in range(1, config.SIMULATION_TIME_IN_TEST + 1):
                    # make action
                    action = agent.greedy_action(state=present_state)
                    action_cnt[action] += 1
                    # environment step and give feedback
                    next_state, reward, end, _, _ = environment.step(time_slot=time_slot, action=action)
                    # update the present observation
                    present_state = next_state
                total_send_before_expiration, \
                total_generate_packet_count, \
                mac_delay_from_queue_to_send, \
                mac_delay_from_head_to_send, \
                mac_delay_from_queue_to_send_success, \
                mac_delay_from_head_to_send_success = environment.get_stats()
                present_timely_throughput = total_send_before_expiration / total_generate_packet_count
                # print('(', episode, ') timely throughput : ', present_timely_throughput,
                #       'train time : ', total_train_time)
                # _success后缀的是指统计成功到达目的节点的数据包，最后用的_success
                logging.info('(' + str(episode) + ') timely throughput : ' + str(present_timely_throughput) +
                             'mac delay(queue, total packet) : ' + str(round(mac_delay_from_queue_to_send, 5)) +
                             'mac delay(head, total packet) : ' + str(round(mac_delay_from_head_to_send, 5)) +
                             'mac delay(queue, success packet) : ' + str(round(mac_delay_from_queue_to_send_success, 5)) +
                             'mac delay(head, success packet) : ' + str(round(mac_delay_from_head_to_send_success, 5)) +
                             'action cnt : ' + str(action_cnt))
                plot_running_timely_throughput.append(present_timely_throughput)
                save_Q_parameters = False
                # have label
                if abs(config.THEORY_TIMELY_THROUGHPUT - present_timely_throughput) < 0.01:
                    get_theory_cnt += 1
                # have no label
                if present_timely_throughput > best_timely_throughput:
                    theory_through_put_path = f'../common/{traffic_prefix}theory_throughput/{traffic_exp_name}/traffic_{traffic_id}.npy'
                    logging.info(f"save path : {theory_through_put_path}")
                    os.makedirs(f"../common/{traffic_prefix}theory_throughput/{traffic_exp_name}", exist_ok=True)
                    np.save(theory_through_put_path, [present_timely_throughput])
                    save_Q_parameters = True
                if abs(present_timely_throughput-best_timely_throughput) < 0.02:
                    get_theory_cnt += 1
                # save Q network parameters and update best episode
                if save_Q_parameters:
                    # best info
                    best_train_time = total_train_time
                    best_train_episode = episode
                    best_timely_throughput = present_timely_throughput
                    agent.save_current_q_network(save_path=config.NEURAL_NETWORK_PATH)
                if (config.FIX_TRAIN_EPS == -1 and get_theory_cnt >= 7) or (config.FIX_TRAIN_EPS != -1 and episode >= config.FIX_TRAIN_EPS):
                    # os.makedirs(f"{config.NEURAL_NETWORK_PATH}/plot_res", exist_ok=True)
                    # np.save(f"{config.NEURAL_NETWORK_PATH}/plot_res/running_timely_throughput.npy", np.array(plot_running_timely_throughput))
                    logging.info(f"now episode : {episode}, total time : {total_train_time}, avg time: {total_train_time / episode}")
                    break

        logging.info("End")

    logging.info("All End")
