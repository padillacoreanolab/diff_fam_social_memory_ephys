import numpy as np
from bidict import bidict
import scipy.stats as stats

SPIKE_GADGETS_MULTIPLIER = 0.6745


def map_to_region(all_traces, subject_region_dict):
    # sort brain regions by channel in incresing order
    sorted_regions = [k for k, v in sorted(subject_region_dict.items(), key=lambda x: x[1])]
    # sort associated selected channels in incresing order
    sorted_channels = [v for k, v in sorted(subject_region_dict.items(), key=lambda x: x[1])]
    # select only desired traces
    traces = all_traces[sorted_channels, ...]
    # create a bidict for brain region to index of new trace array
    brain_region_dict = bidict({region: idx for idx, region in enumerate(sorted_regions)})

    return brain_region_dict, traces


def zscore(traces):
    print(traces.shape)
    mads = stats.median_abs_deviation(traces, axis=1)
    return mads


def plot_zscore():
    return


def filter():
    return


def scale_voltage(lfp_traces: np.ndarray, voltage_scaling_value: float) -> np.ndarray:
    return lfp_traces * voltage_scaling_value
