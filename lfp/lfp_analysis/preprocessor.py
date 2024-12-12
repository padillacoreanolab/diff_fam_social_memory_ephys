import numpy as np
import matplotlib.pyplot as plt
from bidict import bidict
import scipy.stats as stats


SPIKE_GADGETS_MULTIPLIER = 0.6745
VOLTAGE_SCALING_VALUE = 0.195


def preprocess(traces, threshold, scaling):
    # brain_region_dict, traces = map_to_region(all_traces, subject_region_dict)
    voltage_scaled_traces = scale_voltage(traces, scaling)
    zscored_traces = zscore(voltage_scaled_traces)
    filtered_traces = filter(zscored_traces, voltage_scaled_traces, threshold)
    rms_traces = root_mean_square(filtered_traces)
    return rms_traces


def map_to_region(subject_region_dict):
    # sort brain regions by channel in incresing order
    sorted_regions = [k for k, v in sorted(subject_region_dict.items(), key=lambda x: x[1])]
    # sort associated selected channels in incresing order
    sorted_channels = [v for k, v in sorted(subject_region_dict.items(), key=lambda x: x[1])]
    # create a bidict for brain region to index of new trace array
    brain_region_dict = bidict({region: idx for idx, region in enumerate(sorted_regions)})
    # return brain_region_dict, traces
    return brain_region_dict, sorted_channels


def median_abs_dev(traces):
    return stats.median_abs_deviation(traces, axis=1)


def zscore(traces):
    mads = median_abs_dev(traces)
    # transpose because of broadcasting rules- trailing axis must be same
    # https://stackoverflow.com/questions/26333005/numpy-subtract-every-row-of-matrix-by-vector
    temp_traces = (traces.transpose() - np.median(traces, axis=1)).transpose()
    zscore_traces = SPIKE_GADGETS_MULTIPLIER * (temp_traces.transpose() / mads).transpose()
    return zscore_traces


def plot_zscore(processed_traces, zscore_traces, thresholded_zscore_traces, file_path=None):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))

    ax1.plot(processed_traces[0])
    ax1.set_title("Processed traces")
    ax1.set_ylabel("Amplitude")

    ax2.plot(zscore_traces[0])
    ax2.set_title("Z-scored Signal")
    ax2.set_ylabel("Z-score")
    ax2.set_xlabel("Time")

    ax3.plot(thresholded_zscore_traces[0])
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


def filter(zscore, voltage_scaled, threshold):
    mask = np.abs(zscore) < threshold
    return np.where(mask, voltage_scaled, np.nan)


def scale_voltage(lfp_traces: np.ndarray, voltage_scaling_value: float) -> np.ndarray:
    return lfp_traces * voltage_scaling_value


def root_mean_square(traces):
    return traces / np.sqrt(np.nanmean(traces**2))


if __name__ == "__main__":
    traces = np.loadtxt("tests/test_data/test_traces.csv", delimiter=",")
    SUBJECT_DICT = {"mPFC": 20, "vHPC": 31, "BLA": 30, "NAc": 28, "MD": 29}
    brain_regions, traces = map_to_region(traces, SUBJECT_DICT)
    zscore_traces = zscore(traces)
    plot_zscore(traces, zscore_traces)
