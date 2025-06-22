# Spike collection Details
# --------------------------------------------------------------------------------------------------------

def save_collection(self, output_path):
    output_data = {
        "metadata": {
            "data_path": self.path,
            "number of recordings": len(self.recordings),
            "sampling rate": self.sampling_rate,
            "total good units": sum(recording.good_neurons for recording in self.recordings),
            "average units per recording": (sum(recording.good_neurons for recording in 
                                                self.recordings) / len(self.recordings))
    }}

# Spike Recording Details
# --------------------------------------------------------------------------------------------------------
'''
    Has event_dict, subj_dict, spike_trains using unit_timestamps, firing_rates, unit_labels | will be used for analyzing the 
    spike data, such as calculating firing rates, z-scores, and other statistics for the recording.
'''

class SpikeRecording:
    """
    A class for an ephys recording after being spike sorted and manually
    curated using phy. Ephys self must have a phy folder.

    Attributes:
        path: str, relative path to the phy folder
            formatted as: './folder/folder/phy'
        subject: str, subject id who was being recorded
        sampling_rate: int, sampling rate of the ephys device
            in Hz, standard in the PC lab is 20,000Hz
        timestamps: numpy array, all spike timestamps
            of good and mua units (no noise unit-generated spikes)
        unit_array: numpy array, unit ids associated with each
            spike in the timestamps
        labels_dict: dict, keys are unit ids (str) and
            values are labels (str)
        unit_timestamps: dict, keys are unit ids (int), and
            values are numpy arrays of timestamps for all spikes
            from "good" units only
        spiketrain: np.array, spiketrain of number of spikes
            in a specified timebin
        unit_spiketrains: dict, spiketrains for each unit
            keys: str, unit ids
            values: np.array, number of spikes per specified timebin
        unit_firing_rates: dict, firing rates per unit
            keys: str, unit ids
            values: np.arrays, firing rate of unit in a specified timebin
                    calculated with a specified smoothing window

    Methods: (all called in __init__)
        unit_labels: creates labels_dict
        spike_specs: creates timestamps and unit_array
        unit_timestamps: creates unit_timestamps dictionary
    """

    def __init__(self, path, sampling_rate=20000):
        """
        constructs all necessary attributes for the Ephysself object
        including creating labels_dict, timestamps, and a unit_timstamps
        dictionary

        Arguments (2 total):
            path: str, relative path to the merged.rec folder containing a phy folder
                formatted as: './folder/folder'
            sampling_rate: int, default=20000; sampling rate of
                the ephys device in Hz
        Returns:
            None
        """
        self.path = path
        self.phy = os.path.join(path, "phy")
        self.name = os.path.basename(path)
        self.sampling_rate = sampling_rate
        self.all_set = False
        self.__unit_labels__()
        self.__spike_specs__()
        if ("good" in self.labels_dict.values()) or ("mua" in self.labels_dict.values()):
            self.__unit_timestamps__()
            self.__freq_dictionary__()


# single-cell notes
# --------------------------------------------------------------------------------------------------------

'''
    Has functions for creating pre-event windows from specified window/offset, calculating event averages - neuron spike rates during events,
    assesment using p-values (needs to be changed to z-score). Also has potentially useful raster plot, needs to be changed to incoroporate
'''

def pre_event_window(event, baseline_window, offset):
    """
    creates an event like object np.array[start(ms), stop(ms)] for
    baseline_window amount of time prior to an event

    Args (2 total):
        event: np.array[start(ms), stop(ms)]
        baseline_window: int, seconds prior to an event

    Returns (1):
        preevent: np.array, [start(ms),stop(ms)] baseline_window(s)
            before event
    """
    preevent = [event[0] - (baseline_window * 1000) - 1, event[0] + (offset * 1000) - 1]
    return np.array(preevent)


def event_avgs(event1_firing_rates, event2_firing_rates):
    unit_averages = {}
    bad_units = []
    for unit in event1_firing_rates.keys():
        try:
            event1_averages = [np.nanmean(event) for event in event1_firing_rates[unit]]
            event2_averages = [np.nanmean(event) for event in event2_firing_rates[unit]]
            unit_averages[unit] = [np.array(event1_averages), np.array(event2_averages)]
        except StatisticsError:
            bad_units.append(unit)
    return unit_averages, bad_units

def w_assessment(p_value, w): # change to z-score assessment
    try:
        if p_value < 0.05:
            if w > 0:
                return "increases"
            else:
                return "decreases"
        else:
            return "not significant"
    except TypeError:
        return "NaN"
    

def plot_raster(spike_collection, event, event_length, pre_window, global_timebin=1000):

    zscore_df = normalization.zscore_global(
        spike_collection, event, event_length, pre_window, global_timebin, plot=False
    )
    zscore_df = zscore_df.drop(columns=["Recording", "Event", "Subject", "original unit id"])
    # Get sorting indices (in descending order, hence the [::-1])
    zscore_array = zscore_df.to_numpy()

    row_means = np.mean(zscore_array, axis=1)
    sort_indices = np.argsort(row_means)[::-1]

    # Reorder the data
    sorted_zscore = zscore_array[sort_indices]
    timebin_s = spike_collection.timebin / 1000  # Convert timebin from ms to seconds
    total_bins = sorted_zscore.shape[1]
    time_axis = np.linspace(-pre_window, (event_length - timebin_s), total_bins)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot heatmap
    im = ax.imshow(
        sorted_zscore, aspect="auto", cmap="viridis", extent=[time_axis[0], time_axis[-1], sorted_zscore.shape[0], 0]
    )

    # Add event rectangle (from 0 to event_length in seconds)
    import matplotlib.patches as patches

    rect = patches.Rectangle(
        (0, 0), event_length / 1000, sorted_zscore.shape[0], linewidth=2, edgecolor="red", facecolor="none", alpha=0.5
    )
    ax.add_patch(rect)

    # Customize plot
    plt.colorbar(im, label="Z-score")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Units (sorted by mean activity)")
    ax.set_title("Neural Activity Aligned to Event Onset")

    # Add vertical line at event onset
    ax.axvline(x=0, color="red", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()


# normalization.py notes
# --------------------------------------------------------------------------------------------------------




# z-score_notes.py
# --------------------------------------------------------------------------------------------------------

"""
Z-Score-Based Neuron Responsiveness Detection

This script implements a statistical pipeline to determine whether individual
neurons are significantly modulated by specific behavioral events using z-scores.

It follows these major steps:
1. Generate event-aligned spike counts.
2. Average responses across trials.
3. Compute pre-event (baseline) and event means.
4. Compute z-scores comparing event firing to baseline.
5. Apply significance thresholds.
6. Aggregate per-unit results.
7. Optionally visualize per-unit z-scored PSTHs.

Adapt this for your own event_dict and spike count infrastructure.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

### ---------------- Step 1: Input Data Preparation ----------------

"""
Assumes you have a DataFrame with the following format:
- One row per unit × trial × event
- Columns include:
    - 'Recording', 'Event name', 'Unit number'
    - 'Pre-event timebin 1..n'
    - 'Event timebin 1..n'
"""

# Replace with your own method of generating this
event_and_pre_event_spikes_df = pd.read_pickle('your_event_dataframe.pkl')

### ---------------- Step 2: Compute Trial-Averaged Firing ----------------

# Identify timebin columns
pre_event_columns = [col for col in event_and_pre_event_spikes_df.columns if 'Pre-event timebin' in col]
event_columns = [col for col in event_and_pre_event_spikes_df.columns if 'Event timebin' in col]

# Compute group-level (averaged across trials) stats
grouped_df = event_and_pre_event_spikes_df.groupby(['Recording', 'Event name', 'Unit number'])

pre_event_means = grouped_df[pre_event_columns].mean().mean(axis=1)
pre_event_sds = grouped_df[pre_event_columns].mean().std(axis=1)
event_means = grouped_df[event_columns].mean().mean(axis=1)

# Compute Z-scores
z_scores = (event_means - pre_event_means) / pre_event_sds

# Combine into DataFrame
event_zscores_df = pd.DataFrame({
    'Recording': pre_event_means.index.get_level_values('Recording'),
    'Event name': pre_event_means.index.get_level_values('Event name'),
    'Unit number': pre_event_means.index.get_level_values('Unit number'),
    'Pre-event M': pre_event_means.values,
    'Pre-event SD': pre_event_sds.values,
    'Event M': event_means.values,
    'Event Z-Score': z_scores.values
})

event_zscores_df.reset_index(drop=True, inplace=True)

### ---------------- Step 3: Apply Significance Threshold ----------------

"""
Apply a one-tailed z-score threshold (e.g., ±1.65) corresponding to a 95% confidence level.
"""

conditions = [
    (event_zscores_df['Event Z-Score'] > 1.65),
    (event_zscores_df['Event Z-Score'] < -1.65)
]
values = ['increase', 'decrease']
event_zscores_df['sig'] = np.select(conditions, values, default='not sig')

### ---------------- Step 4: Combine Unit-Level Results ----------------

"""
Group by neuron and summarize which events showed significant change.
"""

# Optional: Add putative cell type information if available
# umap_df_detail = pd.read_pickle('path/to/umap_df_detail.pkl')
# merged_df = event_zscores_df.merge(umap_df_detail, ...)

# Group by unit
grouped = event_zscores_df.groupby(['Recording', 'Unit number']).agg({
    'Event name': lambda x: list(x),
    'sig': lambda x: list(x)
}).reset_index()

# Process each unit to separate significant and non-significant events
sig_events = []
not_sig_events = []

for _, row in grouped.iterrows():
    sig_list = []
    not_sig_list = []
    for event_name, sig in zip(row['Event name'], row['sig']):
        if sig != 'not sig':
            sig_list.append(event_name)
        else:
            not_sig_list.append(event_name)
    sig_events.append(sig_list)
    not_sig_events.append(not_sig_list)

units_df = pd.DataFrame({
    'Recording': grouped['Recording'],
    'Unit number': grouped['Unit number'],
    'sig events': sig_events,
    'not sig events': not_sig_events
})

### ---------------- Step 5: (Optional) Plot Z-Score Time Series ----------------

def smooth_data(y, sigma):
    """Applies Gaussian smoothing to a 1D array."""
    return gaussian_filter1d(y, sigma=sigma)

def plot_smoothed_zscores(df, unit_rows=5, sigma=2):
    """
    Plots smoothed z-scored time series for a subset of units.

    Parameters:
    - df: DataFrame with pre-event and event timebin columns
    - unit_rows: number of units to plot
    - sigma: smoothing kernel width
    """
    pre_event = [f'Pre-event timebin {i}' for i in range(1, 41)]
    event = [f'Event timebin {i}' for i in range(1, 41)]
    seconds = np.linspace(-10, 10, 80)

    fig, ax = plt.subplots(figsize=(15, 8))
    cmap = plt.get_cmap('tab10')

    for idx, (_, row) in enumerate(df.head(unit_rows).iterrows()):
        full_data = np.array(row[pre_event + event])
        smooth = smooth_data(full_data, sigma=sigma)
        ax.plot(seconds, smooth, label=f'Unit {row["Unit number"]}', color=cmap(idx % 10))

    ax.axvline(x=0, color='red', linestyle='--')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Z-score')
    ax.set_title('Smoothed Peri-Event Z-Scored Firing Rates')
    ax.legend()
    plt.tight_layout()
    plt.show()

# Example usage:
# sig_units = event_zscores_df[event_zscores_df['sig'] != 'not sig']
# plot_smoothed_zscores(sig_units)

### ---------------- Final Notes ----------------

"""
To integrate this into your own analysis:
1. Replace the placeholder DataFrame with your spike-aligned trial data.
2. Match pre-event and event timebin windows.
3. Adjust the number of timebins and smoothing accordingly.
4. Visualize units of interest after z-score labeling.

This pipeline avoids assumptions about firing rate distributions and works well with limited trials.
"""
