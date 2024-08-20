# channel state
class CHANNEL_STATE:
    # two conditions :
    # -------no packet is sent in channel
    # -------one packet is sent in channel(the packet should be other's node packet)
    NO_PACKET = 0
    ONE_PACKET = 1
    HALF_PACKET = 0.5


# action state
class ACTION:
    SEND = 1  # send packet
    NOT_SEND = 0  # not send packet



