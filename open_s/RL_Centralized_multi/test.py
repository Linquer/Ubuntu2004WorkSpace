import time

import numpy as np

from agent import Agent
from config import CentralizedConfig
from environment import Environment
import sys
import logging
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Run parameters')
    parser.add_argument('--node_count', type=int, default=6)
    parser.add_argument('--traffic_begin_id', type=int, default=700)
    parser.add_argument('--traffic_end_id', type=int, default=799)
    parser.add_argument('--traffic_exp_name', type=str, default="flow_6_p_1")
    parser.add_argument('--traffic_prefix', type=str, default="c_gen_")
    parser.add_argument('--model_suffix', type=str, default="hidd_60_2")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(f"args : {args}")
    node_count = args.node_count
    traffic_begin_id = args.traffic_begin_id
    traffic_end_id = args.traffic_end_id
    traffic_exp_name = args.traffic_exp_name
    traffic_prefix = args.traffic_prefix
    model_suffix = args.model_suffix

    config = CentralizedConfig(node_count=node_count, traffic_begin_id=traffic_begin_id, traffic_end_id=traffic_end_id,
                               traffic_exp_name=traffic_exp_name, traffic_prefix=traffic_prefix, model_suffix=model_suffix, test=True)

    logging.info(f"pid : {os.getpid()}")

    environment = Environment(config=config)

    agent = Agent(config=config)
    agent.load_model(config.NEURAL_NETWORK_PATH)

    '''plot var'''
    sum_theory = 0
    sum_ME_D3QN = 0
    plot_D3QN_throughput = []
    plot_theory_throughput = []
    plot_dynamic_throughput = []

    for traffic_id in range(traffic_begin_id, traffic_end_id + 1):
        logging.info(f" ========================= test traffic id : {traffic_id} =========================")
        config.initialize_simulation_time(traffic_id=traffic_id)
        train_begin_time = time.process_time()
        # 环境初始化
        environment.initialize(config=config, traffic_id=traffic_id)
        present_state = environment.get_state(time_slot=1)
        traffic_information = config.OFFSET[traffic_id].tolist() + \
                              config.INTER_PERIOD[traffic_id].tolist() + \
                              config.DEADLINE[traffic_id].tolist() + \
                              config.ARRIVAL_PROB[traffic_id].tolist() + \
                              config.CHANNEL_PROB[traffic_id].tolist()

        logging.info(f"offset : {config.OFFSET[traffic_id].tolist()}")
        logging.info(f"prd : {config.INTER_PERIOD[traffic_id].tolist()}")
        logging.info(f"deadline : {config.DEADLINE[traffic_id].tolist()}")
        logging.info(f"arrival prob : {config.ARRIVAL_PROB[traffic_id].tolist()}")
        logging.info(f"channel prob : {config.CHANNEL_PROB[traffic_id].tolist()}")

        present_state = traffic_information + present_state

        end_time = 5000

        plot_dynamic_throughput_single = []
        for time_slot in range(1, end_time  + 1):
            # make action
            action = agent.greedy_action(state=present_state)
            # environment step and give feedback
            next_state, reward, end = environment.step(time_slot=time_slot, action=action)

            next_state = traffic_information + next_state
            # update the present observation
            present_state = next_state

            if time_slot % 1 == 0:
                total_send_before_expiration, \
                total_generate_packet_count, \
                mac_delay_from_queue_to_send, \
                mac_delay_from_head_to_send, \
                mac_delay_from_queue_to_send_success, \
                mac_delay_from_head_to_send_success = environment.get_stats()
                present_timely_throughput = total_send_before_expiration / total_generate_packet_count if total_generate_packet_count != 0 else 0
                # print('(', episode, ') timely throughput : ', present_timely_throughput,
                #       'train time : ', total_train_time)
                # _success后缀的是指统计成功到达目的节点的数据包，最后用的_success
                plot_dynamic_throughput_single.append(present_timely_throughput)
                if time_slot % 2000 == 0:
                    logging.info('(time = ' + str(time_slot) + ') ' +
                                 'timely throughput : ' + str(present_timely_throughput) +
                                 'theory throughput : ' + str(config.THEORY_THROUGHPUT[traffic_id]) +
                                 'mac delay(queue, total packet) : ' + str(round(mac_delay_from_queue_to_send, 5)) +
                                 'mac delay(head, total packet) : ' + str(round(mac_delay_from_head_to_send, 5)) +
                                 'mac delay(queue, success packet) : ' + str(round(mac_delay_from_queue_to_send_success, 5)) +
                                 'mac delay(head, success packet) : ' + str(round(mac_delay_from_head_to_send_success, 5)))

        total_send_before_expiration, \
        total_generate_packet_count, \
        mac_delay_from_queue_to_send, \
        mac_delay_from_head_to_send, \
        mac_delay_from_queue_to_send_success, \
        mac_delay_from_head_to_send_success = environment.get_stats()
        present_timely_throughput = total_send_before_expiration / total_generate_packet_count

        sum_ME_D3QN += total_send_before_expiration / total_generate_packet_count if total_generate_packet_count != 0 else 0
        sum_theory += config.THEORY_THROUGHPUT[traffic_id][0]

        plot_theory_throughput.append(config.THEORY_THROUGHPUT[traffic_id][0])
        plot_D3QN_throughput.append(total_send_before_expiration / total_generate_packet_count if total_generate_packet_count != 0 else 0)
        plot_dynamic_throughput.append(plot_dynamic_throughput_single)

        logging.info('timely throughput : ' + str(present_timely_throughput) +
                     'theory throughput : ' + str(config.THEORY_THROUGHPUT[traffic_id]) +
                     'mac delay(queue, total packet) : ' + str(round(mac_delay_from_queue_to_send, 5)) +
                     'mac delay(head, total packet) : ' + str(round(mac_delay_from_head_to_send, 5)) +
                     'mac delay(queue, success packet) : ' + str(round(mac_delay_from_queue_to_send_success, 5)) +
                     'mac delay(head, success packet) : ' + str(round(mac_delay_from_head_to_send_success, 5)))

    np.save(f"{config.NEURAL_NETWORK_PATH}/plot_theory_throughput_{node_count}.npy", np.array(plot_theory_throughput))
    np.save(f"{config.NEURAL_NETWORK_PATH}/plot_ME_D3QN_throughput_{node_count}.npy", np.array(plot_D3QN_throughput))
    np.save(f"{config.NEURAL_NETWORK_PATH}/plot_dynamic_throughput_5000_with_{node_count}_node.npy", np.array(plot_dynamic_throughput))

    logging.info("End")
    logging.info(f"avg theory : {sum_theory / 100} , avg D3QN : {sum_ME_D3QN / 100}")