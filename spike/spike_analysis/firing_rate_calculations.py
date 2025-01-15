import numpy as np


def get_spiketrain(timestamp_array, last_timestamp, timebin, sampling_rate=20000):
    """
    creates a spiketrain where each array element is the number of spikes recorded per timebin

    Args (3 total):
        timestamp_array: numpy array, spike timestamp array
        timebin: int, default=1, timebin (ms) of resulting spiketrain
        sampling_rate: int, default=20000, sampling rate
        in Hz of the ephys recording

    Returns (1):
        spiketrain: numpy array, array elements are number
        of spikes per timebin
    """
    hz_to_timebin = int(sampling_rate * 0.001 * timebin)
    bins = np.arange(1, last_timestamp + 2, hz_to_timebin)
    spiketrain = np.histogram(timestamp_array, bins=bins)[0]
    return spiketrain


def get_firing_rate(spiketrain, timebin, smoothing_window, mode="same"):
    """
    calculates firing rate (spikes/second)

    Args (3 total, 1 required):
        spiketrain: numpy array, in timebin (ms) bins
        smoothing_window: int, default=250, smoothing average window (ms)
        timebin: int, default = 1, timebin (ms) of spiketrain
        mode: {"same", "forward", "backward"}
            "same": smoothed firing rates have boundary effects, firing rate array is the same size as the spike train
            "forward": smoothed firing rates do not have boundary effects and have Nans for the first window at the
                beginning of the recording
                this avoids "psychic mice"
            "backward": smoothed firing rate do not have boundary effects and have Nans at the end of recording

    Return (1):
        firing_rate: numpy array of firing rates in timebin sized windows

    """
    if smoothing_window is None:
        firing_rate = spiketrain * 1000 / timebin
    else:
        smoothing_bins = int(smoothing_window / timebin)
        weights = np.ones(smoothing_bins) / smoothing_bins * 1000 / timebin
        if mode == "same":
            firing_rate = np.convolve(spiketrain, weights, mode="same")
        else:
            firing_rate_valid = np.convolve(spiketrain, weights, mode="valid")
            padding_size = len(spiketrain) - len(firing_rate_valid)
            if mode == "forward":
                firing_rate = np.concatenate([np.full(padding_size, np.nan), firing_rate_valid])
            if mode == "backward":
                firing_rate = np.concatenate([firing_rate_valid, np.full(padding_size, np.nan)])
    return firing_rate
