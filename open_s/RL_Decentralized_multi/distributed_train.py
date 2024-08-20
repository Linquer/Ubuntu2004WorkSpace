import os
import sys

import numpy as np

sys.path.append(os.path.split(os.path.abspath(os.path.dirname(__file__)))[0])

import copy
import argparse
import logging
import time
import sys
from config import Config
from environment import Environment
from agent_distributed import Agent_distributed_dtde, MetaAgent


def config_logging(file_name: str, console_level: int = logging.INFO, file_level: int = logging.INFO):
    file_handler = logging.FileHandler(file_name, mode='a', encoding="utf8")
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


'''
parse run args
'''


def parse_args():
    parser = argparse.ArgumentParser(description='Run parameters')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--node_num', type=int, default=10)
    parser.add_argument('--flow_num', type=int, default=10)
    parser.add_argument('--traffic_id_begin', type=int, default=3)
    parser.add_argument('--traffic_id_end', type=int, default=3)
    parser.add_argument('--model_suffix', type=str, default="gen_bug_test")
    parser.add_argument('--traffic_exp_name', type=str, default="manual_flow_10_10_meta/group_1")
    parser.add_argument('--exp_prefix', type=str, default="gen_")
    parser.add_argument('--random_seed', type=int, default=12)
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
    random_seed = args.random_seed

    model_path = f'./{exp_prefix}model/{traffic_id_begin}_to_{traffic_id_end}_{model_suffix}/{traffic_exp_name}'
    os.makedirs(model_path, exist_ok=True)
    config_logging(file_name=f'{model_path}/log.log')

    config = Config(node_num, flow_num, traffic_id_begin, model_path, traffic_exp_name, exp_prefix)
    logging.info("config : ==================")
    for attr in dir(config):
        if not callable(getattr(config, attr)) and not attr.startswith("__"):
            logging.info(f'{attr} : {getattr(config, attr)}')

    config.initialize_simulation_time()
    meta_agents = [MetaAgent(config=config, node_id=i) for i in range(config.NODE_NUMBER)]
    meta_lstm_window_length = config.WINDOW_LENGTH

    logging.info(f"pid : {os.getpid()} =====================================================================================================")

    for traffic_id in range(traffic_id_begin, traffic_id_end + 1):

        if os.path.exists(f"{model_path}/traffic_{traffic_id + 1}"):
            logging.info(f"traffic {traffic_id} has trained, skip!")
            continue

        config = Config(node_num, flow_num, traffic_id, model_path, traffic_exp_name, exp_prefix)

        environment = Environment(config=config)
        '''-----------------------init parse---------------------------'''

        train_begin_time = 0
        train_end_time = 0
        total_train_time = 0
        best_train_time = 0
        best_train_episode = 0
        best_timely_throughput = 0

        plot_eps = []
        plot_sum_reward_eps_avg_total = []
        plot_running_timely_throughput = []
        plot_meta_reward_eps_avg_total = []
        plot_meta_action_cnt_total = []

        config.initialize_simulation_time()
        if config.WINDOW_LENGTH != meta_lstm_window_length:
            logging.warning(f"traffic {traffic_id} window length is not match, skip")
            continue

        if args.train:
            logging.info('training for decentralized training decentralized execution')
            agent_distributed = [Agent_distributed_dtde(config=config, node_id=i) for i in range(config.NODE_NUMBER)]

            arrive_theory_throughput_cnt = 0
            arrive_convergence_cnt = 0
            pre_throughput = 0
            meta_agent_update_cnt = 0

            for episode in range(1, config.EPISODE_TIMES + 1):
                reward_eps_sum = 0
                action_cnt = 0
                meta_reward_eps_sum = 0
                meta_action_cnt = 0

                train_begin_time = time.process_time()
                _, present_observations, present_states = environment.initialize(config=config, traffic_id=0)
                for time_slot in range(1, config.SIMULATION_TIMES * 2 + 1):
                    action_distributed = [0 for _ in range(config.NODE_NUMBER)]

                    meta_explore = False
                    for idx in range(config.NODE_NUMBER):
                        action, meta_explore = agent_distributed[idx].e_greedy_action(
                            state=present_observations[idx],
                            meta_agent=meta_agents[idx],
                            random_seed=random_seed + episode * 1000 + time_slot * 23)
                        action_distributed[idx] = 0 if action == 0 else action + idx * config.HOLD_FLOW_NUMBER
                        if action_distributed[idx] < 0:
                            logging.error(
                                f'err , action_distributed[idx] < 0 , action:{action}, idx:{idx}, hold flow number : {config.HOLD_FLOW_NUMBER}')

                    next_observations, next_states, reward_distributed, _, _ = environment.step(time_slot=time_slot,
                                                                                                actions=action_distributed)

                    if meta_explore:
                        meta_action_cnt += 1
                        meta_reward_eps_sum += reward_distributed[0]
                    action_cnt += 1
                    reward_eps_sum += reward_distributed[0]

                    # logging.info(f"node 1 state 1 : {present_observations[0][0]}")
                    # logging.info(f"node 1 state 2 : {present_observations[0][1]}")
                    # logging.info(f"node 2 state 1 : {present_observations[1][0]}")
                    # logging.info(f"node 2 state 2 : {present_observations[1][1]}")
                    # logging.info(f"action_distributed : {action_distributed}")
                    # logging.info(f"reward_distributed : {reward_distributed}")
                    # logging.info(f"node 1 next state 1 : {next_observations[0][0]}")
                    # logging.info(f"node 1 next state 2 : {next_observations[0][1]}")
                    # logging.info(f"node 2 next state 1 : {next_observations[1][0]}")
                    # logging.info(f"node 2 next state 2 : {next_observations[1][1]}")
                    # logging.info("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")

                    for node_id in range(config.NODE_NUMBER):
                        if time_slot >= config.INITIAL_PHASE_TIME + 1:
                            agent_distributed[node_id].perceive(state=present_observations[node_id],
                                                                action=action_distributed[node_id],
                                                                reward=reward_distributed[node_id],
                                                                next_state=next_observations[node_id],
                                                                will_update=True)

                            meta_agent_update_cnt += 1
                            meta_agents[node_id].perceive(state=present_observations[node_id],
                                                          action=action_distributed[node_id],
                                                          reward=reward_distributed[node_id],
                                                          next_state=next_observations[node_id],
                                                          # will_update=(meta_agent_update_cnt >= config.META_UPDATE_EPISODE),
                                                          will_update=False)

                            if meta_agent_update_cnt >= config.META_UPDATE_EPISODE:
                                meta_agent_update_cnt = 0

                    present_observations = copy.deepcopy(next_observations)
                    present_states = copy.deepcopy(next_states)

                plot_eps.append(episode)
                plot_meta_reward_eps_avg_total.append(0 if meta_action_cnt == 0 else meta_reward_eps_sum / meta_action_cnt)
                plot_meta_action_cnt_total.append(meta_action_cnt)
                plot_sum_reward_eps_avg_total.append(0 if action_cnt == 0 else reward_eps_sum / action_cnt)

                if episode % config.TARGET_UPDATE_EPS == 0:
                    # logging.info("update target param")
                    for idx in range(config.NODE_NUMBER):
                        agent = agent_distributed[idx]
                        agent.target_net.load_state_dict(agent.current_net.state_dict())

                        meta_agent = meta_agents[idx]
                        meta_agent.target_net.load_state_dict(meta_agent.current_net.state_dict())

                train_end_time = time.process_time()
                total_train_time += train_end_time - train_begin_time
                '''-------------------------testing phase-------------------------'''
                if episode % config.TEST_EPISODE_TIMES == 0:
                    _, present_observations, present_states = environment.initialize(config=config, traffic_id=0)

                    idle_time = 0
                    reward_sum_list = [0 for _ in range(config.NODE_NUMBER)]
                    action_cnt = [[0 for _ in range(config.DISTRIBUTED_ACTION_DIM)] for _ in range(config.NODE_NUMBER)]
                    for time_slot in range(1, 5000 + 1):
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
                        next_observations, next_states, reward_distributed, _, _ = environment.step(time_slot=time_slot,
                                                                                                    actions=action_distributed)

                        present_observations = copy.deepcopy(next_observations)
                        present_states = copy.deepcopy(next_states)

                        for idx in range(config.NODE_NUMBER):
                            reward_sum_list[idx] += reward_distributed[idx]

                    for idx in range(config.NODE_NUMBER):
                        config.BOARD_WRITER.add_scalar(f'reward_sum_for_node_{idx}', reward_sum_list[idx], episode)

                    total_send_before_expiration, total_generate_packet_count, total_dealloc_for_expiration, collision = environment.get_stats()
                    present_timely_throughput = total_send_before_expiration / total_generate_packet_count
                    plot_running_timely_throughput.append(present_timely_throughput)

                    logging.info(f"({episode}) timely throughput : {present_timely_throughput},"
                                 f" train time : {total_train_time}, collision : {collision}"
                                 f" total time :{config.TEST_EPISODE_TIMES * 5000} , idle time : {idle_time}, "
                                 f" total_generate_packet_count : {total_generate_packet_count}, "
                                 f" total_dealloc_for_expiration : {total_dealloc_for_expiration},"
                                 f" action cnt : {action_cnt}")

                    if present_timely_throughput > best_timely_throughput:
                        best_timely_throughput = present_timely_throughput
                        best_train_episode = episode
                        best_train_time = total_train_time
                        '''save model parameters'''
                        for node_id in range(config.NODE_NUMBER):
                            agent_distributed[node_id].save_current_q_network(save_path=config.DISTRIBUTED_LOAD_MODEL_PATH)
                            meta_agents[node_id].save_current_q_network(save_path=config.DISTRIBUTED_LOAD_MODEL_PATH)
                    if config.THEORY_TIMELY_THROUGHPUT[0] != 0 and abs(config.THEORY_TIMELY_THROUGHPUT[0] - present_timely_throughput) <= 0.02:
                        arrive_theory_throughput_cnt += 1
                    if episode >= 1500 and abs(config.THEORY_TIMELY_THROUGHPUT[0] - present_timely_throughput) <= 0.03:
                        arrive_theory_throughput_cnt += 1

                    if abs(pre_throughput - present_timely_throughput) <= 0.005 and abs(
                            config.THEORY_TIMELY_THROUGHPUT[0] - present_timely_throughput) <= 0.03:
                        arrive_convergence_cnt += 1
                    else:
                        arrive_convergence_cnt = 0

                    if (config.FIX_TRAINING_EPISODE == -1 and arrive_theory_throughput_cnt >= 3 or arrive_convergence_cnt >= 7) or (
                            config.FIX_TRAINING_EPISODE != -1 and episode >= config.FIX_TRAINING_EPISODE):
                        break

            os.makedirs(f"{config.DISTRIBUTED_LOAD_MODEL_PATH}/plot_res", exist_ok=True)
            np.save(f"{config.DISTRIBUTED_LOAD_MODEL_PATH}/plot_res/running_timely_throughput.npy", np.array(plot_running_timely_throughput))
            logging.info(
                '(Training finished) Traffic : ' + str(traffic_id) + ' best trained timely throughput : ' + str(
                    best_timely_throughput))

        '''-------------------------plot phase-------------------------'''
        polt_utils.meta_reward_cnt_for_save(x=plot_eps,
                                            meta_reward=plot_meta_reward_eps_avg_total,
                                            avg_reward=plot_sum_reward_eps_avg_total,
                                            cnt=plot_meta_action_cnt_total,
                                            save_path=config.PLOT_SAVE_PATH)
