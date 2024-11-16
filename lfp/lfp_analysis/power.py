import numpy as np
import matplotlib.pyplot as plt
from bidict import bidict
import scipy.stats as stats
from spectral_connectivity import Multitaper, Connectivity


def calculate_power(rms_traces, downsample_rate, halfbandwidth, timewindow, timestep):
    multi_t = Multitaper(
        time_series=rms_traces,
        sampling_frequency=downsample_rate,
        time_halfbandwidth_product=halfbandwidth,
        time_window_duration=timewindow,
        time_window_step=timestep,
    )
    connectivity = Connectivity.from_multitaper(multi_t)
    power = connectivity.power().squeeze()
    frequencies = connectivity.frequencies()
    return (power, frequencies)
