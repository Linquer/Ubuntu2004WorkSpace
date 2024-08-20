from constant import CHANNEL_STATE, ACTION
import copy


class History_information:
    def __init__(self, config):
        # history observation
        self.history_observation_size = config.WINDOW_LENGTH
        # (other node packet, other node action)
        # basic_history_observation = [CHANNEL_STATE.NO_PACKET, ACTION.NOT_SEND]
        # (other node packet)
        basic_history_observation = [CHANNEL_STATE.NO_PACKET]
        self.history_observation = [copy.deepcopy(basic_history_observation)
                                    for x in range(self.history_observation_size)]

    def initialized(self, config):
        self.history_observation_size = config.WINDOW_LENGTH
        # basic_history_observation = [CHANNEL_STATE.NO_PACKET, ACTION.NOT_SEND]
        basic_history_observation = [CHANNEL_STATE.NO_PACKET]
        self.history_observation = [copy.deepcopy(basic_history_observation)
                                    for x in range(self.history_observation_size)]