import os
import sys

sys.path.append(os.path.split(os.path.abspath(os.path.dirname(__file__)))[0])

import copy
import argparse
import logging
import time
import sys
from config import Config
from dtde_maddpg import utils
from environment import Environment
from agent_distributed import Agent_distributed_dtde
from plot_res.scheduling_illustration.plot import scheduling_illustration_for_compare


def config_logging(file_name: str, console_level: int = logging.INFO, file_level: int = logging.INFO):
    file_handler = logging.FileHandler(file_name, mode='w', encoding="utf8")
    file_handler.setFormatter(logging.Formatter(
        '%(levelname)s %(filename)s[line:%(lineno)d] %(message)s'
    ))
    file_handler.setLevel(file_level)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(
        '%(levelname)s %(filename)s[line:%(lineno)d] %(message)s',
    ))
    console_handler.setLevel(console_level)

    logging.basicConfig(
        level=min(console_level, file_level),
        handlers=[file_handler, console_handler],
    )


'''
parse run args
'''


def parse_args():
    parser = argparse.ArgumentParser(description='Run parameters')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--node_num', type=int, default=10)
    parser.add_argument('--flow_num', type=int, default=10)
    parser.add_argument('--traffic_id_begin', type=int, default=16)
    parser.add_argument('--traffic_id_end', type=int, default=25)
    parser.add_argument('--model_suffix', type=str, default="platform_traffic_dy")
    parser.add_argument('--traffic_exp_name', type=str, default="manual_flow_10_10_meta/group_1")
    parser.add_argument('--exp_prefix', type=str, default="gen_")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(f"args : {args}")

    node_num = args.node_num
    flow_num = args.flow_num
    traffic_id_begin = args.traffic_id_begin
    traffic_id_end = args.traffic_id_end
    model_suffix = args.model_suffix
    args.train = True
    traffic_exp_name = args.traffic_exp_name
    exp_prefix = args.exp_prefix

    model_path = f'./{exp_prefix}model/{traffic_id_begin}_to_{traffic_id_end}_{model_suffix}/{traffic_exp_name }'
    os.makedirs(model_path, exist_ok=True)
    config_logging(file_name=f'{model_path}/test_log.log')

    logging.info(f"pid : {os.getpid()}")

    for traffic_id in range(traffic_id_begin, traffic_id_end + 1):
        ''' init common object'''
        config = Config(node_num, flow_num, traffic_id, model_path, traffic_exp_name, exp_prefix)

        environment = Environment(config=config)
        '''-----------------------init parse---------------------------'''

        '''stats info'''
        train_begin_time = 0
        train_end_time = 0
        total_train_time = 0
        '''best info'''
        best_train_time = 0
        best_train_episode = 0
        best_timely_throughput = 0

        '''some value for plot'''
        plot_eps = []
        plot_sum_reward_eps_avg_total = []

        '''configuration'''
        config.initialize_simulation_time()
        agent_distributed = [Agent_distributed_dtde(config=config, node_id=i) for i in range(config.NODE_NUMBER)]

        for agent in agent_distributed:
            agent.load_model(config.DISTRIBUTED_LOAD_MODEL_PATH)

        _, present_observations, present_states = environment.initialize(config=config, traffic_id=0)

        idle_time = 0
        action_cnt = [[0 for _ in range(config.DISTRIBUTED_ACTION_DIM)] for _ in range(config.NODE_NUMBER)]

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
            action_distributed = [0 for _ in range(config.NODE_NUMBER)]
            for idx in range(config.NODE_NUMBER):
                action = agent_distributed[idx].greedy_action(
                    present_observations[idx])
                action_cnt[idx][action] += 1
                action_distributed[idx] = 0 if action == 0 else action + idx * config.HOLD_FLOW_NUMBER
                if action_distributed[idx] < 0:
                    logging.error(
                        f'err , action_distributed[idx] < 0 , action:{action}, idx:{idx}, hold flow '
                        f'number : {config.HOLD_FLOW_NUMBER}')

            if sum(action_distributed) == 0:
                idle_time += 1
            '''environment step and give feedback'''
            plot_state = copy.deepcopy(present_observations)
            next_observations, next_states, reward_distributed, expire_list, generate_list = environment.step(time_slot=time_slot,
                                                                                  actions=action_distributed)

            '''update present observations'''
            present_observations = copy.deepcopy(next_observations)
            present_states = copy.deepcopy(next_states)

            # for idx in range(flow_num):
            #     if action_distributed[idx] == 0:
            #         plot_send_ind[idx].append(0)
            #     else:
            #         plot_send_ind[idx].append(1)
            #
            #     plot_collision_ind[idx].append(0)
            #     # logging.info(f"idx : {idx} , state : {plot_state}")
            #     plot_state_val[idx].append(plot_state[idx][1][0] / 10)
            #     plot_next_arrive_ind[idx].append(plot_state[idx][1][1])
            #     plot_expire_ind[idx].append(expire_list[idx])
            #     plot_arrive_ind[idx].append(generate_list[idx])
            #
            # if time_slot % 10 == 0 and time_slot >= 300:
            #     print("plot")
            #     scheduling_illustration_for_compare(f"RL_Decent_{traffic_id}", block_cnt, plot_state_val, plot_send_ind, plot_expire_ind, plot_collision_ind,
            #                                         plot_arrive_ind, plot_next_arrive_ind)
            #     plot_state_val = [[] for _ in range(flow_num)]
            #     plot_send_ind = [[] for _ in range(flow_num)]
            #     plot_expire_ind = [[] for _ in range(flow_num)]
            #     plot_collision_ind = [[] for _ in range(flow_num)]
            #     plot_arrive_ind = [[] for _ in range(flow_num)]
            #     plot_next_arrive_ind = [[] for _ in range(flow_num)]
            #     block_cnt += 1

            if time_slot % 100 == 0:
                total_send_before_expiration, total_generate_packet_count, total_dealloc_for_expiration, collision = environment.get_stats()
                present_timely_throughput = total_send_before_expiration / total_generate_packet_count if total_generate_packet_count != 0 else 0

                logging.info(f"timely throughput : {present_timely_throughput},"
                             f" train time : {total_train_time}, collision : {collision}"
                             f" total time :{config.TEST_EPISODE_TIMES * 5000} , idle time : {idle_time}, "
                             f" total_generate_packet_count : {total_generate_packet_count}, "
                             f" total_dealloc_for_expiration : {total_dealloc_for_expiration},"
                             f" action cnt : {action_cnt}")


