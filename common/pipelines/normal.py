import logging
from typing import Dict, Set

def parse_devices(device_string):
    colon_position = device_string.find(':')
    if colon_position != -1:
        device_type = device_string[:colon_position]
        if device_type == 'HETERO' or device_type == 'MULTI':
            comma_separated_devices = device_string[colon_position + 1:]
            devices = comma_separated_devices.split(',')
            for device in devices:
                parenthesis_position = device.find(':')
                if parenthesis_position != -1:
                    device = device[:parenthesis_position]
            return devices
    return (device_string,)


def parse_value_per_device(devices: Set[str], values_string: str)-> Dict[str, int]:
    """Format: <device1>:<value1>,<device2>:<value2> or just <value>"""
    values_string_upper = values_string.upper()
    result = {}
    device_value_strings = values_string_upper.split(',')
    for device_value_string in device_value_strings:
        device_value_list = device_value_string.split(':')
        if len(device_value_list) == 2:
            if device_value_list[0] in devices:
                result[device_value_list[0]] = int(device_value_list[1])
        elif len(device_value_list) == 1 and device_value_list[0] != '':
            for device in devices:
                result[device] = int(device_value_list[0])
        elif device_value_list[0] != '':
            raise RuntimeError(f'Unknown string format: {values_string}')
    return result


def get_user_config(flags_d: str, flags_nstreams: str, flags_nthreads: int)-> Dict[str, str]:
    config = {}

    devices = set(parse_devices(flags_d))

    device_nstreams = parse_value_per_device(devices, flags_nstreams)
    for device in devices:
        if device == 'CPU':  # CPU supports a few special performance-oriented keys
            # limit threading for CPU portion of inference
            if flags_nthreads:
                config['CPU_THREADS_NUM'] = str(flags_nthreads)

            config['CPU_BIND_THREAD'] = 'NO'

            # for CPU execution, more throughput-oriented execution via streams
            config['CPU_THROUGHPUT_STREAMS'] = str(device_nstreams[device]) \
                if device in device_nstreams else 'CPU_THROUGHPUT_AUTO'
        elif device == 'GPU':
            config['GPU_THROUGHPUT_STREAMS'] = str(device_nstreams[device]) \
                if device in device_nstreams else 'GPU_THROUGHPUT_AUTO'
            if 'MULTI' in flags_d and 'CPU' in devices:
                # multi-device execution with the CPU + GPU performs best with GPU throttling hint,
                # which releases another CPU thread (that is otherwise used by the GPU driver for active polling)
                config['GPU_PLUGIN_THROTTLE'] = '1'
    return config


class Normal:
    def __init__(self, ie, model, plugin_config, device='CPU'):
        self.model = model
        self.logger = logging.getLogger()

        self.logger.info('Loading network to {} plugin...'.format(device))
        self.exec_net = ie.load_network(network=self.model.net, device_name=device,
                                        config=plugin_config)

    # inference
    def submit_data(self, inputs, id, meta):
        self.res = []
        inputs, preprocessing_meta = self.model.preprocess(inputs)
        infer_res = self.exec_net.infer(inputs=inputs)
        self.res.append(infer_res)
        self.res.append(meta)
        self.res.append(preprocessing_meta)

    def get_result(self, id):
        if self.res:
            raw_result, meta, preprocess_meta = self.res
            return self.model.postprocess(raw_result, preprocess_meta), meta
        return None