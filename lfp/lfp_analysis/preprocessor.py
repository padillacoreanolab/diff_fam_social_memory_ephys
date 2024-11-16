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


def median_abs_dev(traces):
    return stats.median_abs_deviation(traces, axis=1)


def zscore(traces):
    mads = median_abs_dev(traces)
    # transpose because of broadcasting rules- trailing axis must be same
    # https://stackoverflow.com/questions/26333005/numpy-subtract-every-row-of-matrix-by-vector
    temp_traces = (traces.transpose() - np.median(traces, axis=1)).transpose()
    zscore_traces = SPIKE_GADGETS_MULTIPLIER * (temp_traces.transpose() / mads).transpose()
    return zscore_traces


def plot_zscore():

    return


def filter():
    return


def scale_voltage(lfp_traces: np.ndarray, voltage_scaling_value: float) -> np.ndarray:
    return lfp_traces * voltage_scaling_value


def root_mean_sqaure(traces):
    return traces / np.sqrt(np.mean(traces**2))
