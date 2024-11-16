import numpy as np


def map_to_region(traces, subject_region_dict):
    traces_dict = {}
    for region, channel in subject_region_dict.items():
        traces_dict[region] = traces[int(channel), :]

    return traces_dict


def zscore():
    return


def plot_zscore():
    return


def filter():
    return


def scale_voltage(lfp_traces: np.ndarray, voltage_scaling_value: float) -> np.ndarray:
    return lfp_traces * voltage_scaling_value
