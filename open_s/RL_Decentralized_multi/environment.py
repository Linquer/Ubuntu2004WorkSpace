import copy
import logging
from config import Config
from flow import Flow
from history_information import History_information
from constant import CHANNEL_STATE
import math
import numpy as np


class Environment:

    def __init__(self, config: Config):
        self.flow_number = config.FLOW_NUMBER
        self.hold_flow_number = config.HOLD_FLOW_NUMBER
        self.action_dim = config.DISTRIBUTED_ACTION_DIM
        self.node_number = config.NODE_NUMBER

        self.flows = [Flow(config=config, flow_id=x + 1) for x in range(config.FLOW_NUMBER)]

        self.history_information = [History_information(config=config) for x in range(config.NODE_NUMBER)]
        self.collision = 0

        self.collision_punish_frame_idx = 0
        self.collision_punish_ratio = lambda frame_idx: config.COLLISION_PUNISH_RATIO_END + (
                config.COLLISION_PUNISH_RATIO_BEGIN - config.COLLISION_PUNISH_RATIO_END) * math.exp(
            -1 * frame_idx / config.COLLISION_PUNISH_DECAY)

    def initialize(self, config, traffic_id):
        self.window_length = config.WINDOW_LENGTH
        for flow in self.flows:
            flow.initialize(config=config, traffic_id=traffic_id)
            flow.generate_new_packet(time_slot=1)
        for history_information in self.history_information:
            history_information.initialized(config=config)
        present_observations = self.get_observations(time_slot=1)
        present_states = self.get_states(time_slot=1)
        self.collision = 0
        return self.flows, present_observations, present_states

    def step(self, time_slot, actions):
        send_node_id, send_flow_id = self.get_send_node_id(actions)

        channel_state = CHANNEL_STATE.NO_PACKET
        if send_flow_id != -1:
            packet_have, packet_sent, _, \
                mac_delay_from_queue_to_send, \
                mac_delay_from_head_to_send = self.flows[send_flow_id - 1].remove_buffer_expiration(time_slot=time_slot)
            if packet_have == 1:
                if packet_sent == 1:
                    channel_state = CHANNEL_STATE.ONE_PACKET
                elif packet_sent == 0:
                    channel_state = CHANNEL_STATE.HALF_PACKET
            else:
                channel_state = CHANNEL_STATE.NO_PACKET
        else:
            channel_state = CHANNEL_STATE.NO_PACKET

        expire_list = []
        for flow in self.flows:
            packet_expire, \
                mac_delay_from_queue_to_send, \
                mac_delay_from_head_to_send = flow.check_buffer_expiration(time_slot=time_slot)
            expire_list.append(packet_expire)

        generate_list = []
        for flow in self.flows:
            generate_list.append(flow.generate_new_packet(time_slot=time_slot + 1))

        next_states = self.get_states(time_slot=time_slot + 1)

        reward_distributed = self.get_reward(actions=actions, expire_list=expire_list, time_slot=time_slot + 1,
                                             channel_state=channel_state)

        globle_reward = 0

        self.update_history_observation_information(channel_state=channel_state, send_node_id=send_node_id,
                                                    globle_reward=globle_reward)

        next_observations = self.get_observations(time_slot=time_slot + 1)

        return next_observations, next_states, reward_distributed, expire_list, generate_list

    def update_history_observation_information(self, channel_state, send_node_id, globle_reward):
        for node_id in range(self.node_number):
            for location in range(self.window_length - 1):
                '''move the history information'''
                self.history_information[node_id].history_observation[location][0] = \
                    self.history_information[node_id].history_observation[location + 1][0]

        if channel_state == CHANNEL_STATE.NO_PACKET:
            for idx in range(self.node_number):
                self.history_information[idx].history_observation[-1][0] = CHANNEL_STATE.NO_PACKET + globle_reward
        elif channel_state == CHANNEL_STATE.HALF_PACKET:
            for idx in range(self.node_number):
                if idx == send_node_id:
                    self.history_information[idx].history_observation[-1][0] = CHANNEL_STATE.HALF_PACKET + globle_reward
                else:
                    self.history_information[idx].history_observation[-1][0] = CHANNEL_STATE.NO_PACKET + globle_reward
        elif channel_state == CHANNEL_STATE.ONE_PACKET:
            for idx in range(self.node_number):
                if idx == send_node_id:
                    self.history_information[idx].history_observation[-1][0] = CHANNEL_STATE.ONE_PACKET + globle_reward
                else:
                    self.history_information[idx].history_observation[-1][0] = CHANNEL_STATE.NO_PACKET + globle_reward
        else:
            logging.error("update history observation information!")
            exit()

    def get_observations(self, time_slot):
        node_temporal_information = []
        node_state_information = [[] for _ in range(self.node_number)]
        node_indicator_informataion = [[] for _ in range(self.node_number)]
        other_node_state_information = [[] for _ in range(self.node_number)]
        other_node_indicator_informataion = [[] for _ in range(self.node_number)]
        node_non_temporal_information = []

        for node_idx in range(self.node_number):
            node_temporal_information.append(self.history_information[node_idx].history_observation)

        global_state = []
        global_indicator = []

        for flow in self.flows:
            if (time_slot + 1 - flow.offset - 1 >= 0) and \
                    (time_slot + 1 - flow.offset - 1) % flow.inter_period == 0:
                packet_arrival_indicator = 1
            else:
                packet_arrival_indicator = 0
            global_state.append(flow.get_node_state(time_slot=time_slot))
            global_indicator.append(packet_arrival_indicator)

        # logging.info(f"global state : {global_state}")
        # logging.info(f"global indicator : {global_indicator}")

        for node_id in range(self.node_number):
            node_state_information[node_id] += \
                copy.copy(global_state[node_id * self.hold_flow_number: (node_id + 1) * self.hold_flow_number])
            other_node_state_information[node_id] += \
                copy.copy(global_state[:node_id * self.hold_flow_number]) + copy.copy(
                    global_state[(node_id + 1) * self.hold_flow_number:])

            node_indicator_informataion[node_id] += \
                copy.copy(global_indicator[node_id * self.hold_flow_number: (node_id + 1) * self.hold_flow_number])
            other_node_indicator_informataion[node_id] += \
                copy.copy(global_indicator[:node_id * self.hold_flow_number]) + copy.copy(
                    global_indicator[(node_id + 1) * self.hold_flow_number:])

        observations = []
        for node_idx in range(self.node_number):
            node_non_temporal_information.append(
                node_state_information[node_idx] + node_indicator_informataion[node_idx]
                + other_node_state_information[node_idx] + other_node_indicator_informataion[node_idx])
            observations.append([node_temporal_information[node_idx], node_non_temporal_information[node_idx]])

        return observations

    def get_send_node_id(self, actions):
        send_node_id = -1
        send_flow_id = -1
        for idx in range(self.node_number):
            if actions[idx] > 0:
                if send_flow_id > 0:
                    self.collision += 1
                    return -1, -1
                send_node_id = idx
                send_flow_id = actions[idx]

        return send_node_id, send_flow_id

    def get_states(self, time_slot):
        state_in_buffer = []
        state_in_packet_arrival_indicator = []
        for flow in self.flows:
            state_in_buffer.append(flow.get_node_state(time_slot=time_slot))
            if (time_slot + 1 - flow.offset - 1 >= 0) and \
                    (time_slot + 1 - flow.offset - 1) % flow.inter_period == 0:
                state_in_packet_arrival_indicator.append(1)
            else:
                state_in_packet_arrival_indicator.append(0)
        # return state_in_buffer + state_in_packet_arrival_indicator
        return state_in_buffer + state_in_packet_arrival_indicator

    def get_reward(self, actions, expire_list, time_slot, channel_state):
        send_node_id, send_flow_id = self.get_send_node_id(actions)
        reward_distributed = [0 for _ in range(self.node_number)]

        if send_node_id != -1 and send_flow_id != -1 and channel_state == CHANNEL_STATE.ONE_PACKET:
            for idx in range(self.node_number):
                reward_distributed[idx] += 1.0

        return reward_distributed

    def get_stats(self):
        total_generate_packet_count = 0
        total_send_before_expiration = 0
        total_mac_delay_from_queue_to_send = 0
        total_mac_delay_from_head_to_send = 0
        total_mac_delay_from_queue_to_send_success = 0
        total_mac_delay_from_head_to_send_success = 0
        total_dealloc_for_expiration = 0
        for flow in self.flows:
            total_generate_packet_count += flow.generate_packet_count
            total_send_before_expiration += flow.send_before_expiration
            total_mac_delay_from_queue_to_send += flow.mac_delay_from_queue_to_send
            total_mac_delay_from_head_to_send += flow.mac_delay_from_head_to_send
            total_mac_delay_from_queue_to_send_success += flow.mac_delay_from_queue_to_send_success
            total_mac_delay_from_head_to_send_success += flow.mac_delay_from_head_to_send_success
            total_dealloc_for_expiration += flow.dealloc_for_expiration
        # success
        # average_mac_delay_from_queue_to_send_success = total_mac_delay_from_queue_to_send_success / total_send_before_expiration
        # average_mac_delay_from_head_to_send_success = total_mac_delay_from_head_to_send_success / total_send_before_expiration
        # total
        # average_mac_delay_from_queue_to_send = total_mac_delay_from_queue_to_send / total_generate_packet_count
        # average_mac_delay_from_head_to_send = total_mac_delay_from_head_to_send / total_generate_packet_count
        return total_send_before_expiration, total_generate_packet_count, total_dealloc_for_expiration, self.collision
