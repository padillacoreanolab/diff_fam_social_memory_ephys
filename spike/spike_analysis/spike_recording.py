import os
import csv
import numpy as np
from collections import defaultdict
import spike_analysis.firing_rate_calculations as fr


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
        unit_labels: creates labels_dict
        spike_specs: creates timestamps_var and unit_array
        unit_timestamps: creates unit_timestamps dictionary
    """

    def __init__(self, path, sampling_rate=20000):
        """
        constructs all necessary attributes for the Ephysself object
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
        self.name = os.path.dirname(path).split(os.sep)[-1]
        self.sampling_rate = sampling_rate
        self.all_set = False
        self.unit_labels()
        self.spike_specs()
        self.unit_timestamps()

    def check(self):
        missing = []
        attributes = ["timebin", "subject", "event_dict"]
        for attr in attributes:
            if not hasattr(self, attr):
                missing.append(attr)
        if len(missing) > 0:
            print(f"Cannot execute:{self.name} is missing the following attributes:")
            print(missing)
        else:
            self.all_set = True

    def unit_labels(self):
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

    def spike_specs(self):
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

    def unit_timestamps(self):
        """
        Creates a dictionary of units to spike timestamps.
        Keys are unit ids (int) and values are spike timestamps for that unit (numpy arrays),
        and assigns dictionary to self.unit_timestamps.
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

    def analyze(self, timebin, ignore_freq=0.1, smoothing_window=None, mode="same"):
        self.timebin = timebin
        self.ignore_freq = ignore_freq
        self.smoothing_window = smoothing_window
        self.mode = mode
        self.check()
        self.__freq_dictionary__()
        self.__whole_spiketrain__()
        self.__unit_spiketrains__()
        self.__unit_firing_rates__()

    def __freq_dictionary__(self):
        sampling_rate = self.sampling_rate
        last_timestamp = self.timestamps_var[-1]
        freq_dict = {}
        for unit in self.unit_timestamps.keys():
            if self.labels_dict[str(unit)] == "good":
                no_spikes = len(self.unit_timestamps[unit])
                unit_freq = no_spikes / last_timestamp * sampling_rate
                freq_dict[unit] = unit_freq
        self.freq_dict = freq_dict

    def __whole_spiketrain__(self):
        """
        creates a spiketrain for each self where each array element is the number of spikes per timebin
        and assigns as .spiketrain for each self
        """
        last_timestamp = self.timestamps_var[-1]
        self.spiketrain = fr.get_spiketrain(self.timestamps_var, last_timestamp, self.timebin, self.sampling_rate)

    def __unit_spiketrains__(self):
        """
        Creates a dictionary and assigns it as self.unit_spiketrains for each self.
        Only 'good' unit ids (not 'mua') with firing rates > ignore_freq are included.
        Keys are unit ids (ints) and values are numpy arrays of spiketrains in timebin-sized bins
        """
        sampling_rate = self.sampling_rate
        last_timestamp = self.timestamps_var[-1]
        unit_spiketrains = {}
        for unit in self.freq_dict.keys():
            if self.freq_dict[unit] > self.ignore_freq:
                unit_spiketrains[unit] = fr.get_spiketrain(
                    self.unit_timestamps[unit],
                    last_timestamp,
                    self.timebin,
                    sampling_rate,
                )
        self.unit_spiketrains = unit_spiketrains

    def __unit_firing_rates__(self):
        """
        Calculates firing rates per unit, creates a dictionary and assigns it as self.unit_firing_rates
        Keys are unit ids (int) and values are numpy arrays of firing rates (Hz) in timebin sized bins
        Calculated using smoothing_window for averaging
        Creates a multi dimensional array as self.unit_firing_rate_array of timebins x units
        """
        unit_firing_rates = {}
        for unit in self.unit_spiketrains.keys():
            unit_firing_rates[unit] = fr.get_firing_rate(
                self.unit_spiketrains[unit], self.timebin, self.smoothing_window, self.mode
            )
        self.unit_firing_rates = unit_firing_rates
        self.unit_firing_rate_array = np.array([unit_firing_rates[key] for key in unit_firing_rates]).T

    def __event_snippets__(self, event, whole_self, event_length, pre_window=0, post_window=0):
        """
        takes snippets of spiketrains or firing rates for events with optional pre-event and post-event windows (s)
        all events must be of equal length (extends snippet lengths for events shorter then event_length and trims those
        that are longer)

        Args (6 total, 4 required):
            self: Spikeself instance, self to get snippets
            event: str, event type of which ephys snippets happen during
            whole_self: numpy array, spiketrain or firing rates for the whole self
            event_length: float, length (s) of events used through padding and trimming events
            pre_window: int, default=0, seconds prior to start of event
            post_window: int, default=0, seconds after end of event

        Returns (1):
            event_snippets: a list of lists, where each list is a list of
                firing rates or spiketrains during an event including
                pre_window & post_windows, accounting for event_length and
                timebins for a single unit or for the population returning
                a list of numpy arrays
        """
        if self.all_set:
            if type(event) is str:
                events = self.event_dict[event]
            else:
                events = event
            event_snippets = []
            pre_window = round(pre_window * 1000)
            post_window = round(post_window * 1000)
            event_length = event_length * 1000
            event_len = int((event_length + pre_window + post_window) / self.timebin)
            print("event len", event_len)
            for i in range(events.shape[0]):
                pre_event = int((events[i][0] - pre_window) / self.timebin)
                post_event = pre_event + event_len
                if len(whole_self.shape) == 1:
                    event_snippet = whole_self[pre_event:post_event]
                    print("single unit start and stop", pre_event, post_event)
                    # drop events that start before the beginning of the self
                    # given a long prewindow
                    if pre_event >= 0:
                        # drop events that go beyond the end of the self
                        if post_event < whole_self.shape[0]:

                            event_snippets.append(event_snippet)
                else:
                    event_snippet = whole_self[pre_event:post_event, ...]
                    print("self array start and stop", pre_event, post_event)
                    # drop events that start before the beginning of the self
                    # given a long prewindow
                    if pre_event >= 0:
                        # drop events that go beyond the end of the self
                        if post_event < whole_self.shape[0]:
                            event_snippets.append(event_snippet)

            print(np.array(event_snippets).shape)
            # event_snippets = [trial, timebins, units] or [trial, timebin] per unit
            return event_snippets
        else:
            self.check()
            return None

    def __unit_event_firing_rates__(self, event, event_length, pre_window=0, post_window=0):
        """
        returns firing rates for events per unit

        Args (5 total, 3 required):
            self: Spikeself instance, self for firing rates
            event: str, event type of which ehpys snippets happen during
            event_length: float, length (s) of events used by padding or trimming events
            pre_window: int, default=0, seconds prior to start of event
            post_window: int, default=0, seconds after end of event

        Return (1):
            unit_event_firing_rates: dict, keys are unit ids (???),
            values are lsts of numpy arrays of firing rates per event
        """
        if self.all_set():
            unit_event_firing_rates = {}
            for unit in self.unit_firing_rates.keys():
                unit_event_firing_rates[unit] = self.__event_snippets__(
                    event,
                    self.unit_firing_rates[unit],
                    event_length,
                    pre_window,
                    post_window,
                )
            return unit_event_firing_rates
        else:
            self.check()
            return None

    def __event_firing_rates__(self, event, event_length, pre_window=0, post_window=0):
        """
        Grabs event firing rates from a whole self through the selfs
        unit firing rate array (units by time bins)

        Args (5 total, 3 required):
            self: Spikeself instance, self for firing rates
            event: str, event type of which ehpys snippets happen during
            event_length: float, length (s) of events used by padding or trimming events
            pre_window: int, default=0, seconds prior to start of event
            post_window: int, default=0, seconds after end of event

        Returns (1):
            event_firing_rates: list of arrays, where each array
            is timebins x units and list is len(no of events)
        """
        if self.all_set:
            event_firing_rates = self.__event_snippets__(
                event, self.unit_firing_rate_array, event_length, pre_window, post_window
            )
            return event_firing_rates
        else:
            self.check()
            return None
