import time
from agent import Agent
from config import CentralizedConfig
from environment import Environment
import sys
import logging
import os
import argparse
import numpy as np
from constant import TEST_TRAFFIC_LIST


def parse_args():
    parser = argparse.ArgumentParser(description='Run parameters')
    parser.add_argument('--node_count', type=int, default=2)
    parser.add_argument('--traffic_begin_id', type=int, default=0)
    parser.add_argument('--traffic_end_id', type=int, default=99)
    parser.add_argument('--traffic_exp_name', type=str, default="flow_2_p_1")
    parser.add_argument('--traffic_prefix', type=str, default="c_gen_")
    parser.add_argument('--model_suffix', type=str, default="local_test")

    return parser.parse_args()


# main：主程序
if __name__ == '__main__':

    args = parse_args()
    node_count = args.node_count
    traffic_begin_id = args.traffic_begin_id
    traffic_end_id = args.traffic_end_id
    traffic_exp_name = args.traffic_exp_name
    traffic_prefix = args.traffic_prefix
    model_suffix = args.model_suffix

    config = CentralizedConfig(node_count=node_count, traffic_begin_id=traffic_begin_id, traffic_end_id=traffic_end_id,
                               traffic_exp_name=traffic_exp_name, traffic_prefix=traffic_prefix, model_suffix=model_suffix)
    logging.info(f"pid : {os.getpid()}")

    # 配置
    # log config

    # 环境
    environment = Environment(config=config)
    # 智能体
    agent = Agent(config=config, )
    if os.path.exists(f'{config.NEURAL_NETWORK_PATH}/model.pth'):
        agent.load_model(config.NEURAL_NETWORK_PATH)
        logging.info("load exist model success!")

    # 智能体与仿真时间初始化
    simulation_time_list = []
    for id in range(traffic_end_id + 1):
        simulation_time_list.append(config.initialize_simulation_time(traffic_id=id))
    # stats info
    train_begin_time = 0
    train_end_time = 0
    total_train_time = 0
    # best info
    best_train_time = 0
    best_train_episode = 0
    best_timely_throughput = 0

    get_theory_cnt = 0

    logging.info(f"simulation list len : {len(simulation_time_list)}")
    timely_throughput_test_traffic = [[] for _ in range(len(TEST_TRAFFIC_LIST[2]))]
    # training phase
    for episode in range(1, config.EPISODE_TIMES + 1):
        for traffic_id in range(config.TRAFFIC_START, config.TRAFFIC_END + 1):
            # 记录每个episode的训练开始时间
            train_begin_time = time.process_time()
            # 环境初始化
            environment.initialize(config=config, traffic_id=traffic_id)
            present_state = environment.get_state(time_slot=1)
            traffic_information = config.OFFSET[traffic_id].tolist() + \
                                  config.INTER_PERIOD[traffic_id].tolist() + \
                                  config.DEADLINE[traffic_id].tolist() + \
                                  config.ARRIVAL_PROB[traffic_id].tolist() + \
                                  config.CHANNEL_PROB[traffic_id].tolist()
            present_state = traffic_information + present_state
            '''
             -------------------- training process begin ---------------------
            '''
            for time_slot in range(1, simulation_time_list[traffic_id] + 1):
                # make action
                action = agent.e_greedy_action(state=present_state, episode=episode)
                # environment step and give feedback
                next_state, reward, end = environment.step(time_slot=time_slot, action=action)
                # !!!change, before code have no this constraint
                # if time_slot >= config.INITIAL_PHASE_TIME + 1:
                # agent perceive the MDP information(state, action, reward, next_state, done)
                next_state = traffic_information + next_state
                agent.perceive(state=present_state,
                               action=action,
                               reward=reward,
                               next_state=next_state,
                               end=end,
                               episode=episode)
                # update the present observation
                present_state = next_state
            # 记录每个episode的训练结束时间
            train_end_time = time.process_time()
            # 将单个episode的训练时间添加到总时间
            total_train_time += train_end_time - train_begin_time
            # update the Q target network parameters
            agent.update_target_q_network(episode=episode)
            # '''
            # ----------------------------- testing phase ---------------------------
            # '''
        if episode != 0 and episode % config.TEST_EPISODE_TIMES == 0:
            for idx, test_traffic in enumerate(TEST_TRAFFIC_LIST[node_count]):
                # 环境初始化
                logging.info(f"test traffic : {test_traffic}")
                environment.initialize(config=config, traffic_id=test_traffic)
                present_state = environment.get_state(time_slot=1)
                traffic_information = config.OFFSET[test_traffic].tolist() + \
                                      config.INTER_PERIOD[test_traffic].tolist() + \
                                      config.DEADLINE[test_traffic].tolist() + \
                                      config.ARRIVAL_PROB[test_traffic].tolist() + \
                                      config.CHANNEL_PROB[test_traffic].tolist()
                present_state = traffic_information + present_state
                for time_slot in range(1, config.SIMULATION_TIME_IN_TEST + 1):
                    # make action
                    action = agent.greedy_action(state=present_state)
                    # environment step and give feedback
                    next_state, reward, end = environment.step(time_slot=time_slot, action=action)
                    # update the present observation
                    present_state = traffic_information + next_state
                total_send_before_expiration, \
                total_generate_packet_count, \
                mac_delay_from_queue_to_send, \
                mac_delay_from_head_to_send, \
                mac_delay_from_queue_to_send_success, \
                mac_delay_from_head_to_send_success = environment.get_stats()
                present_timely_throughput = total_send_before_expiration / total_generate_packet_count
                timely_throughput_test_traffic[idx].append(present_timely_throughput)
                # logging.info(f"running time throughput list : {plot_running_timely_throughput}")
                # print('(', episode, ') timely throughput : ', present_timely_throughput,
                #       'train time : ', total_train_time)
                # _success后缀的是指统计成功到达目的节点的数据包，最后用的_success
                # print(len(config.THEORY_THROUGHPUT), TEST_TRAFFIC_LIST[test_traffic])
                logging.info('(' + str(episode) + ') timely throughput : ' + str(present_timely_throughput) +
                             'mac delay(queue, total packet) : ' + str(round(mac_delay_from_queue_to_send, 5)) +
                             'mac delay(head, total packet) : ' + str(round(mac_delay_from_head_to_send, 5)) +
                             'mac delay(queue, success packet) : ' + str(round(mac_delay_from_queue_to_send_success, 5)) +
                             'mac delay(head, success packet) : ' + str(round(mac_delay_from_head_to_send_success, 5)) +
                             'theory : ' + str(config.THEORY_THROUGHPUT[test_traffic])
                             )

        logging.info(f"eps : {episode} , node : {node_count}, train time : {total_train_time}")

        if (episode + 1) % 5 == 0:
            agent.save_current_q_network(save_path=config.NEURAL_NETWORK_PATH)
            logging.info("save model --------------=")

        np.save(f"{config.NEURAL_NETWORK_PATH}/training_traffic_test_node_{node_count}_traffic_num_{traffic_end_id-traffic_begin_id+1}.npy", timely_throughput_test_traffic)