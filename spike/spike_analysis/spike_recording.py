import os
import csv
import numpy as np
from collections import defaultdict


class SpikeRecording:
    """
    A class for an ephys recording after being spike sorted and manually
    curated using phy. Ephys recording must have a phy folder.

    Attributes:
        path: str, relative path to the phy folder
            formatted as: './folder/folder/phy'
        subject: str, subject id who was being recorded
        sampling_rate: int, sampling rate of the ephys device
            in Hz, standard in the PC lab is 20,000Hz
        timestamps_var: numpy array, all spike timestamps
            of good and mua units (no noise unit-generated spikes)
        unit_array: numpy array, unit ids associated with each
            spike in the timestamps_var
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
        get_unit_labels: creates labels_dict
        get_spike_specs: creates timestamps_var and unit_array
        get_unit_timestamps: creates unit_timestamps dictionary
    """

    def __init__(self, path, sampling_rate=20000):
        """
        constructs all necessary attributes for the EphysRecording object
        including creating labels_dict, timestamps_var, and a unit_timstamps
        dictionary

        Arguments (2 total):
            path: str, relative path to the phy folder
                formatted as: './folder/folder/phy'
            sampling_rate: int, default=20000; sampling rate of
                the ephys device in Hz
        Returns:
            None
        """
        self.path = path
        self.name = os.path.basename(path).split("/")[-1]
        self.sampling_rate = sampling_rate
        self.zscored_events = {}
        self.wilcox_dfs = {}
        self.get_unit_labels()
        self.get_spike_specs()
        self.get_unit_timestamps()

    def get_unit_labels(self):
        """
        assigns self.labels_dicts as a dictionary
        with unit id (str) as key and label as values (str)
        labels: 'good', 'mua', 'noise'

        Arguments:
            None

        Returns:
            None
        """
        labels = "cluster_group.tsv"
        with open(os.path.join(self.path, labels), "r") as f:
            reader = csv.DictReader(f, delimiter="\t")
            self.labels_dict = {row["cluster_id"]: row["group"] for row in reader}

    def get_spike_specs(self):
        """
        imports spike_time and spike_unit from phy folder
        deletes spikes from units labeled noise in unit and timestamp array
        and assigns self.timstamps_var (numpy array)
        as the remaining timestamps and assigns self.unit_array
        (numpy array) as the unit ids associated with each spike

        Args:
            None

        Returns:
            None
        """
        timestamps = "spike_times.npy"
        unit = "spike_clusters.npy"
        timestamps_var = np.load(os.path.join(self.path, timestamps))
        unit_array = np.load(os.path.join(self.path, unit))
        spikes_to_delete = []
        unsorted_clusters = {}
        for spike in range(len(timestamps_var)):
            try:
                if self.labels_dict[unit_array[spike].astype(str)] == "noise":
                    spikes_to_delete.append(spike)
            except KeyError:
                spikes_to_delete.append(spike)
                if unit_array[spike] in unsorted_clusters.keys():
                    total_spikes = unsorted_clusters[unit_array[spike]]
                    total_spikes = total_spikes + 1
                    unsorted_clusters[unit_array[spike]] = total_spikes
                else:
                    unsorted_clusters[unit_array[spike]] = 1
        for unit, no_spike in unsorted_clusters.items():
            print(f"Unit {unit} is unsorted & has {no_spike} spikes")
            print(f"Unit {unit} will be deleted")
        self.timestamps_var = np.delete(timestamps_var, spikes_to_delete)
        self.unit_array = np.delete(unit_array, spikes_to_delete)

    def get_unit_timestamps(self):
        """
        Creates a dictionary of units to spike timestamps.
        Keys are unit ids (int) and values are spike timestamps for that unit (numpy arrays),
        and assigns dictionary to self.unit_timestamps.

        Args:
            None

        Return:
            None
        """

        # Initialize a defaultdict for holding lists
        unit_timestamps = defaultdict(list)

        # Loop through each spike only once
        for spike, unit in enumerate(self.unit_array):
            # Append the timestamp to the list for the corresponding unit
            unit_timestamps[str(unit)].append(self.timestamps_var[spike])
        # convert lists to numpy arrays once complete
        for unit, timestamps in unit_timestamps.items():
            unit_timestamps[str(unit)] = np.array(timestamps)
        self.unit_timestamps = unit_timestamps
