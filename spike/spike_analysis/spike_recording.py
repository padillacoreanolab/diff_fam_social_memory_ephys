import os
import csv
import numpy as np
from collections import defaultdict
import spike.spike_analysis.firing_rate_calculations as fr
import json
import h5py


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

    def __check__(self):
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

    def __unit_labels__(self):
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
        with open(os.path.join(self.phy, labels), "r") as f:
            reader = csv.DictReader(f, delimiter="\t")
            self.labels_dict = {row["cluster_id"]: row["group"] for row in reader}

    def __spike_specs__(self):
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
        timestamps = np.load(os.path.join(self.phy, timestamps))
        unit_array = np.load(os.path.join(self.phy, unit))
        spikes_to_delete = []
        unsorted_clusters = {}
        for spike in range(len(timestamps)):
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
        self.timestamps = np.delete(timestamps, spikes_to_delete)
        self.unit_array = np.delete(unit_array, spikes_to_delete)

    def __unit_timestamps__(self):
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
            unit_timestamps[str(unit)].append(self.timestamps[spike])
        # convert lists to numpy arrays once complete
        for unit, timestamps in unit_timestamps.items():
            unit_timestamps[str(unit)] = np.array(timestamps)
        self.unit_timestamps = unit_timestamps

    def analyze(self, timebin, ignore_freq=0.1, smoothing_window=None, mode="same"):
        self.timebin = timebin
        self.ignore_freq = ignore_freq
        self.smoothing_window = smoothing_window
        self.mode = mode
        self.__check__()
        self.__whole_spiketrain__()
        self.__unit_spiketrains__()
        self.__unit_firing_rates__()

    def __freq_dictionary__(self):
        sampling_rate = self.sampling_rate
        last_timestamp = self.timestamps[-1]
        freq_dict = {}
        for unit in self.unit_timestamps.keys():
            if self.labels_dict[str(unit)] == "good":
                no_spikes = len(self.unit_timestamps[unit])
                unit_freq = no_spikes / last_timestamp * sampling_rate
                freq_dict[unit] = unit_freq
        self.freq_dict = freq_dict
        self.good_neurons = len(freq_dict.keys())

    def __whole_spiketrain__(self):
        """
        creates a spiketrain for each self where each array element is the number of spikes per timebin
        and assigns as .spiketrain for each self
        """
        last_timestamp = self.timestamps[-1]
        self.spiketrain = fr.get_spiketrain(self.timestamps, last_timestamp, self.timebin, self.sampling_rate)

    def __unit_spiketrains__(self):
        """
        Creates a dictionary and assigns it as self.unit_spiketrains for each self.
        Only 'good' unit ids (not 'mua') with firing rates > ignore_freq are included.
        Keys are unit ids (ints) and values are numpy arrays of spiketrains in timebin-sized bins
        """
        sampling_rate = self.sampling_rate
        last_timestamp = self.timestamps[-1]
        unit_spiketrains = {}
        i = 0
        for unit in self.freq_dict.keys():
            if self.freq_dict[unit] > self.ignore_freq:
                i += 1
                unit_spiketrains[unit] = fr.get_spiketrain(
                    self.unit_timestamps[unit],
                    last_timestamp,
                    self.timebin,
                    sampling_rate,
                )
        self.unit_spiketrains = unit_spiketrains
        self.analyzed_neurons = i

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

    def set_subject(self, subject: str):
        """
        Sets the subject attribute for the SpikeRecording object
        """
        self.subject = subject

    def set_event_dict(self, event_dict: dict):
        """
        Sets the event_dict attribute for the SpikeRecording object
        """
        self.event_dict = event_dict

    def event_snippets(self, event, whole_self, event_length, pre_window=0, post_window=0):
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
            for i in range(events.shape[0]):
                pre_event = int((events[i][0] - pre_window) / self.timebin)
                post_event = pre_event + event_len
                if len(whole_self.shape) == 1:
                    event_snippet = whole_self[pre_event:post_event]
                    # drop events that start before the beginning of the self
                    # given a long prewindow
                    if pre_event >= 0:
                        # drop events that go beyond the end of the self
                        if post_event < whole_self.shape[0]:

                            event_snippets.append(event_snippet)
                else:
                    event_snippet = whole_self[pre_event:post_event, ...]
                    # drop events that start before the beginning of the self
                    # given a long prewindow
                    if pre_event >= 0:
                        # drop events that go beyond the end of the self
                        if post_event < whole_self.shape[0]:
                            event_snippets.append(event_snippet)
            # event_snippets = [trial, timebins, units] or [trial, timebin] per unit
            return np.array(event_snippets)
        else:
            self.__check__()
            return None

    def unit_event_firing_rates(self, event, event_length, pre_window=0, post_window=0):
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
        if self.all_set:
            unit_event_firing_rates = {}
            for unit in self.unit_firing_rates.keys():
                unit_event_firing_rates[unit] = self.event_snippets(
                    event,
                    self.unit_firing_rates[unit],
                    event_length,
                    pre_window,
                    post_window,
                )
            return unit_event_firing_rates
        else:
            self.__check__()
            return None

    def event_firing_rates(self, event, event_length, pre_window=0, post_window=0):
        """
        Grabs event firing rates from a whole recording through the recording
        unit firing rate array (units by time bins)

        Args (5 total, 3 required):
            self: SpikeRecording instance, self for firing rates
            event: str, event type of which ehpys snippets happen during
            event_length: float, length (s) of events used by padding or trimming events
            pre_window: int, default=0, seconds prior to start of event
            post_window: int, default=0, seconds after end of event

        Returns (1):
            event_firing_rates: list of arrays, where each array
            is timebins x units and list is len(no of events)
        """
        if self.all_set:
            event_firing_rates = self.event_snippets(
                event, self.unit_firing_rate_array, event_length, pre_window, post_window
            )
            # event_snippets = [trial, timebins, units]
            return event_firing_rates
        else:
            self.__check__()
            return None

    def __str__(self):
        """
        Returns a string representation of the SpikeRecording object with details about
        the number of good units, number of MUAs, subject name, event_dict assignment,
        recording length in minutes, and analysis parameters if set.
        """
        # Calculate the length of the recording in minutes
        recording_length_minutes = self.timestamps[-1] / self.sampling_rate / 60

        # Count the number of MUAs
        mua_count = sum(1 for label in self.labels_dict.values() if label == "mua")

        # Count the number of good units (labeled as "good")
        good_unit_count = sum(1 for label in self.labels_dict.values() if label == "good")

        # Check if event_dict is assigned
        has_event_dict = hasattr(self, "event_dict")

        # Prepare the analysis parameters if they are set
        analysis_params = ""
        if hasattr(self, "timebin") and hasattr(self, "ignore_freq") and hasattr(self, "smoothing_window"):
            if hasattr(self, "freq_dict"):
                above_ignore_freq = sum(1 for freq in self.freq_dict.values() if freq > self.ignore_freq)
            else:
                above_ignore_freq = 0
            analysis_params = (
                f"  Good units above ignore frequency: {above_ignore_freq}\n"
                f"     \n"
                f"Analysis Parameters:\n"
                f"  Timebin: {self.timebin}s\n"
                f"  Ignore Frequency: {self.ignore_freq}Hz\n"
                f"  Smoothing Window: {self.smoothing_window}\n"
                f""
            )

        # Build the string representation
        # Prepare the behavioral events if event_dict is assigned
        behavioral_events = ""
        if has_event_dict:
            behavioral_events = "Event Overview:\n"
            for event, events in self.event_dict.items():
                behavioral_events += f"  {event}: {len(events)} events\n"

        return (
            f"SpikeRecording Summary:\n"
            f"  Recording: {self.name}\n"
            f"  Subject: {getattr(self, 'subject', None)}\n"
            f"  Event Dict Assigned: {has_event_dict}\n"
            f"  Recording Length: {recording_length_minutes:.2f} minutes\n"
            f"     \n"
            f"Unit Overvew:\n"
            f"  Number of Good Units: {good_unit_count}\n"
            f"  Number of MUAs: {mua_count}\n"
            f"{analysis_params}"
            f"\n"
            f"{behavioral_events}"
        )

    @staticmethod
    def save_rec_to_h5(recording, rec_path):
        """
        Save SpikeRecording object to H5 file, excluding analysis-specific attributes.

        Parameters:
        -----------
        recording : SpikeRecording
            The SpikeRecording object to save
        rec_path : str
            Base path for saving (without extension)
        """
        h5_path = rec_path + ".h5"

        with h5py.File(h5_path, "w") as f:
            # === CORE DATA GROUP ===
            data_group = f.create_group("data")

            # Save core spike data
            data_group.create_dataset("timestamps", data=recording.timestamps, compression="gzip", compression_opts=9)
            data_group.create_dataset("unit_array", data=recording.unit_array, compression="gzip", compression_opts=9)

            # Save unit timestamps dictionary
            unit_timestamps_group = data_group.create_group("unit_timestamps")
            for unit_id, timestamps in recording.unit_timestamps.items():
                unit_timestamps_group.create_dataset(
                    str(unit_id), data=timestamps, compression="gzip", compression_opts=9
                )

            # === METADATA GROUP ===
            metadata = f.create_group("metadata")

            # Core recording metadata
            metadata.attrs["name"] = recording.name
            metadata.attrs["path"] = recording.path
            metadata.attrs["phy_path"] = recording.phy
            metadata.attrs["sampling_rate"] = recording.sampling_rate
            metadata.attrs["good_neurons"] = recording.good_neurons
            if hasattr(recording, "first_timestamp"):
                metadata.attrs["first_timestamp"] = recording.first_timestamp

            # Optional subject
            if hasattr(recording, "subject"):
                metadata.attrs["subject"] = recording.subject

            # Recording length in minutes
            recording_length_minutes = recording.timestamps[-1] / recording.sampling_rate / 60
            metadata.attrs["recording_length_minutes"] = recording_length_minutes

            # === LABELS DICTIONARY ===
            labels_group = f.create_group("labels")
            for unit_id, label in recording.labels_dict.items():
                labels_group.attrs[str(unit_id)] = label

            # === FREQUENCY DICTIONARY ===
            freq_group = f.create_group("frequencies")
            for unit_id, freq in recording.freq_dict.items():
                freq_group.attrs[str(unit_id)] = freq

            # === EVENT DICTIONARY (if exists) ===
            if hasattr(recording, "event_dict") and recording.event_dict is not None:
                event_group = f.create_group("events")
                for event_name, event_data in recording.event_dict.items():
                    if isinstance(event_data, (np.ndarray, list)):
                        event_group.create_dataset(
                            event_name, data=np.array(event_data), compression="gzip", compression_opts=9
                        )
                    else:
                        event_group.attrs[event_name] = str(event_data)

            # === ANALYSIS FLAGS (what's available, not the data itself) ===
            analysis_flags = f.create_group("analysis_flags")
            analysis_flags.attrs["has_been_analyzed"] = hasattr(recording, "timebin")
            analysis_flags.attrs["has_event_dict"] = (
                hasattr(recording, "event_dict") and recording.event_dict is not None
            )
            analysis_flags.attrs["all_set"] = getattr(recording, "all_set", False)

    @staticmethod
    def save_metadata_to_json(recording, json_path):
        """
        Save SpikeRecording metadata to JSON file.

        Parameters:
        -----------
        recording : SpikeRecording
            The SpikeRecording object
        json_path : str
            Path where JSON should be saved
        """
        json_path = os.path.join(json_path, ".json")

        # Calculate derived metrics
        recording_length_minutes = recording.timestamps[-1] / recording.sampling_rate / 60
        mua_count = sum(1 for label in recording.labels_dict.values() if label == "mua")
        good_unit_count = sum(1 for label in recording.labels_dict.values() if label == "good")
        noise_unit_count = sum(1 for label in recording.labels_dict.values() if label == "noise")

        metadata = {
            # Core recording info
            "name": recording.name,
            "path": recording.path,
            "phy_path": recording.phy,
            "sampling_rate": recording.sampling_rate,
            "recording_length_minutes": round(recording_length_minutes, 2),
            # Subject info
            "subject": getattr(recording, "subject", None),
            # Unit summary
            "total_units": len(recording.labels_dict),
            "good_units": good_unit_count,
            "mua_units": mua_count,
            "noise_units": noise_unit_count,
            "good_neurons": recording.good_neurons,
            # Spike summary
            "total_spikes": len(recording.timestamps),
            "first_timestamp": int(recording.timestamps[0]),
            "last_timestamp": int(recording.timestamps[-1]),
            # Frequency statistics
            "unit_frequencies": {unit: float(freq) for unit, freq in recording.freq_dict.items()},
            "mean_firing_rate": float(np.mean(list(recording.freq_dict.values()))) if recording.freq_dict else 0,
            "max_firing_rate": float(np.max(list(recording.freq_dict.values()))) if recording.freq_dict else 0,
            # Event summary
            "has_events": hasattr(recording, "event_dict") and recording.event_dict is not None,
            "event_summary": {},
            # Analysis status
            "has_been_analyzed": hasattr(recording, "timebin"),
            "all_set": getattr(recording, "all_set", False),
        }

        # Add event summary if events exist
        if hasattr(recording, "event_dict") and recording.event_dict is not None:
            for event_name, event_data in recording.event_dict.items():
                if isinstance(event_data, (np.ndarray, list)):
                    metadata["event_summary"][event_name] = len(event_data)
                else:
                    metadata["event_summary"][event_name] = str(event_data)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(json_path), exist_ok=True)

        # Save to JSON
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=4)

        return json_path

    @staticmethod
    def load_rec_from_h5(h5_path):
        """
        Load a SpikeRecording object from H5 file.

        Parameters:
        -----------
        h5_path : str
            Path to the H5 file

        Returns:
        --------
        recording : SpikeRecording
            Reconstructed SpikeRecording object
        """
        with h5py.File(h5_path, "r") as f:
            # Extract metadata
            metadata = f["metadata"]

            # Create new SpikeRecording object with minimal initialization
            # We'll bypass the normal __init__ process and rebuild manually
            recording = object.__new__(SpikeRecording)

            # Restore core attributes
            recording.name = metadata.attrs["name"]
            recording.path = metadata.attrs["path"]
            recording.phy = metadata.attrs["phy_path"]
            recording.sampling_rate = metadata.attrs["sampling_rate"]
            recording.good_neurons = metadata.attrs["good_neurons"]

            # Restore subject if it exists
            if "subject" in metadata.attrs:
                recording.subject = metadata.attrs["subject"]

            # Restore core data
            data_group = f["data"]
            recording.timestamps = data_group["timestamps"][:]
            recording.unit_array = data_group["unit_array"][:]

            # Restore unit_timestamps dictionary
            recording.unit_timestamps = {}
            unit_timestamps_group = data_group["unit_timestamps"]
            for unit_id in unit_timestamps_group.keys():
                recording.unit_timestamps[unit_id] = unit_timestamps_group[unit_id][:]

            # Restore labels_dict
            recording.labels_dict = {}
            labels_group = f["labels"]
            for unit_id, label in labels_group.attrs.items():
                recording.labels_dict[unit_id] = label

            # Restore freq_dict
            recording.freq_dict = {}
            freq_group = f["frequencies"]
            for unit_id, freq in freq_group.attrs.items():
                recording.freq_dict[unit_id] = freq

            # Restore event_dict if it exists
            if "events" in f:
                recording.event_dict = {}
                event_group = f["events"]
                # Load event datasets
                for event_name in event_group.keys():
                    recording.event_dict[event_name] = event_group[event_name][:]
                # Load event attributes
                for event_name, event_value in event_group.attrs.items():
                    recording.event_dict[event_name] = event_value

        return recording
