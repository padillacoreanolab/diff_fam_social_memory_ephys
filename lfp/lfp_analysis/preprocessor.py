import numpy as np
import matplotlib.pyplot as plt
from bidict import bidict
import scipy.stats as stats


SPIKE_GADGETS_MULTIPLIER = 0.6745
VOLTAGE_SCALING_VALUE = 0.195


def preprocess(all_traces, subject_region_dict, threshold, scaling, plot=False):
    brain_region_dict, traces = map_to_region(all_traces, subject_region_dict)
    zscored_traces = filter(zscore(scale_voltage(traces, scaling)), threshold)
    if plot:
        plot_zscore(traces, zscored_traces)
    rms_traces = root_mean_sqaure(zscored_traces)
    return rms_traces, brain_region_dict


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


def plot_zscore(processed_traces, zscore_traces, zscore_threshold, file_path=None):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))

    ax1.plot(processed_traces[0])
    ax1.set_title("Processed traces")
    ax1.set_ylabel("Amplitude")

    ax2.plot(zscore_traces[0])
    ax2.set_title("Z-scored Signal")
    ax2.set_ylabel("Z-score")
    ax2.set_xlabel("Time")

    ax3.plot(zscore_threshold[0])
    ax3.set_title("Z-score threshold")
    ax3.set_ylabel("Amplitude")

    # Share y-axis limits between ax2 and ax3
    y_min = min(ax2.get_ylim()[0], ax3.get_ylim()[0])
    y_max = max(ax2.get_ylim()[1], ax3.get_ylim()[1])
    ax2.set_ylim(y_min, y_max)
    ax3.set_ylim(y_min, y_max)

    plt.tight_layout()
    if file_path:
        plt.savefig(file_path)
    else:
        plt.show()
    return


def filter(zscore, threshold):
    return np.where(np.abs(zscore) >= threshold, 0, zscore)


def scale_voltage(lfp_traces: np.ndarray, voltage_scaling_value: float) -> np.ndarray:
    return lfp_traces * voltage_scaling_value


def root_mean_sqaure(traces):
    return traces / np.sqrt(np.mean(traces**2))


if __name__ == "__main__":
    traces = np.loadtxt("tests/test_data/test_traces.csv", delimiter=",")
    SUBJECT_DICT = {"mPFC": 20, "vHPC": 31, "BLA": 30, "NAc": 28, "MD": 29}
    brain_regions, traces = map_to_region(traces, SUBJECT_DICT)
    zscore_traces = zscore(traces)
    plot_zscore(traces, zscore_traces)
