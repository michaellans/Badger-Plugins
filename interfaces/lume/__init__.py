from badger import interface
from typing import Dict
from badger.errors import BadgerInterfaceChannelError


class Interface(interface.Interface):

    name = 'lume'
    # If params not specified, it would be an empty dict

    # Private variables
    _states = {
        'CAMR:IN20:186:R_DIST': 424.00,
        'Pulse_length': 1.86,
        'FBCK:BCI0:1:CHRG_S': 0.25,
        'SOLN:IN20:121:BACT': 0.478,
        'QUAD:IN20:121:BACT': -0.0021,
        'QUAD:IN20:122:BACT': -0.00063,
        'ACCL:IN20:300:L0A_ADES': 58.00,
        'ACCL:IN20:300:L0A_PDES': -9.54,
        'ACCL:IN20:400:L0B_ADES': 70.0,
        'ACCL:IN20:400:L0B_PDES': 9.85,
        'QUAD:IN20:361:BACT': -2.00,
        'QUAD:IN20:371:BACT': 2.00,
        'QUAD:IN20:425:BACT': -1.00,
        'QUAD:IN20:441:BACT': -0.18,
        'QUAD:IN20:511:BACT': 2.85,
        'QUAD:IN20:525:BACT': -3.22,
    }

    _bounds = {
        'QUAD:IN20:361:BACT': [-4.00, -1.00],
        'QUAD:IN20:371:BACT':  [1.00, 4.30],
        'QUAD:IN20:425:BACT': [-7.55, -1.00],
        'QUAD:IN20:441:BACT': [-1.00, 7.55],
        'QUAD:IN20:511:BACT': [-1.00, 7.55],
        'QUAD:IN20:525:BACT': [-7.55, -1.00],
    }

    def __init__(self, **data):
        super().__init__(**data)

        self._states = {}

    def get_values(self, channel_names):
        channel_outputs = {}

        for channel in channel_names:
            try:
                value = self._states[channel]
            except KeyError:
                try:
                    if channel.endswith(".UB"):
                        _channel = channel.split(".")[0]
                        value = self._bounds[_channel][1]
                    elif channel.endswith(".LB"):
                        _channel = channel.split(".")[0]
                        value = self._bounds[_channel][0]
                except KeyError:
                    raise BadgerInterfaceChannelError

            channel_outputs[channel] = value

        return channel_outputs

    def set_values(self, channel_inputs):
        for channel, value in channel_inputs.items():
            self._states[channel] = value

