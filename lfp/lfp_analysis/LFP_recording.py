import spikeinterface.extractors as se
import spikeinterface.preprocessing as sp
import lfp.lfp_analysis.preprocessor as preprocessor
import lfp.lfp_analysis.connectivity_wrapper as connectivity_wrapper
import lfp.trodes.read_exported as trodes
import os
from pathlib import Path
import h5py
import numpy as np
import json
from bidict import bidict

LFP_FREQ_MIN = 0.5
LFP_FREQ_MAX = 300
ELECTRIC_NOISE_FREQ = 60
LFP_SAMPLING_RATE = 1000
EPHYS_SAMPLING_RATE = 20000


class LFPRecording:
    def __init__(
        self,
        subject: str,
        channel_dict: dict,
        merged_rec_path: str,
        behavior_dict=None,
        trodes_directory=None,
        threshold=None,
        elec_noise_freq=60,
        sampling_rate=20000,
        min_freq=0.5,
        max_freq=300,
        resample_rate=1000,
        voltage_scaling=0.195,
        halfbandwidth=2,
        timewindow=1,
        timestep=0.5,
        load=False,
    ):
        self.merged_rec_path = merged_rec_path
        self.sampling_rate = sampling_rate
        self.recording_name = os.path.basename(merged_rec_path).split("/")[-1]
        self.subject = subject
        self.behavior_dict = behavior_dict
        self.channel_dict = channel_dict
        self.voltage_scaling = voltage_scaling
        self.elec_noise_freq = elec_noise_freq
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.resample_rate = resample_rate
        self.threshold = threshold
        self.halfbandwidth = halfbandwidth
        self.timewindow = timewindow
        self.timestep = timestep
        self.trodes_directory = trodes_directory
        self.rec_path = os.path.dirname(merged_rec_path)
        if not load:
            recording = self._read_trodes()
            self.traces = self._get_selected_traces(recording)
            self.rec_length = self.traces.shape[0] / 1000 / 60

    def _read_trodes(self):
        print(f"Processing {self.recording_name}")
        recording = se.read_spikegadgets(self.merged_rec_path, stream_id="trodes")
        recording = sp.notch_filter(recording, freq=self.elec_noise_freq)
        recording = sp.bandpass_filter(recording, freq_min=self.min_freq, freq_max=self.max_freq)
        recording = sp.resample(recording, resample_rate=self.resample_rate)
        return recording

    def _get_selected_traces(self, recording):
        start_frame = self.find_start_recording_time()
        self.brain_region_dict, sorted_channels = preprocessor.map_to_region(self.channel_dict)
        sorted_channels = [str(channel) for channel in sorted_channels]
        # Channel ids are the "names" of the channels as strings
        traces = recording.get_traces(channel_ids=sorted_channels, start_frame=start_frame)
        return traces

    def plot_to_find_threshold(self, threshold, file_path=None):
        scaled_traces = preprocessor.scale_voltage(self.traces, voltage_scaling_value=self.voltage_scaling)
        zscore_traces = preprocessor.zscore(scaled_traces)
        thresholded_traces = preprocessor.filter(zscore_traces, threshold)
        preprocessor.plot_zscore(scaled_traces, zscore_traces, thresholded_traces, file_path)

    def process(self, threshold=None):
        print(f"processing {self.recording_name}")
        if (threshold is None) and (self.threshold is None):
            print("Please choose a threshold")
            raise ValueError("Threshold is not set")

        if threshold is None:
            threshold = self.threshold

        self.rms_traces = preprocessor.preprocess(self.traces, threshold, self.voltage_scaling)
        print("RMS Traces calculated")
        self.connectivity, self.frequencies, self.power, self.coherence, self.grangers = (
            connectivity_wrapper.connectivity_wrapper(
                self.rms_traces, self.resample_rate, self.halfbandwidth, self.timewindow, self.timestep
            )
        )

    def export_trodes_timestamps(self, trodes_directory):
        if trodes is None:
            print("You need trodes directory to extract timestamps.")
        trodes.trodes_extract_single_file(trodes_directory, self.merged_rec_path, mode="-time")
        # need to go to merged.time folder and read merged.timestamps.dat file

    def find_start_recording_time(self):
        """If the timestamps file is found at self.merged_rec_path, then use it. Otherwise ask user to create it"""
        timestamps_path = str(Path(self.merged_rec_path).with_suffix(".time"))
        if os.path.exists(timestamps_path):
            for file in os.listdir(timestamps_path):
                if file.endswith(".timestamps.dat"):
                    timestamps_file_path = os.path.join(timestamps_path, file)
                    timestamps = trodes.read_trodes_extracted_data_file(timestamps_file_path)
                    self.first_timestamp = int(timestamps["first_timestamp"])
                    print("Found first timestamp")
        else:
            self.export_trodes_timestamps(self.trodes_directory)
            for file in os.listdir(timestamps_path):
                if file.endswith(".timestamps.dat"):
                    timestamps_file_path = os.path.join(timestamps_path, file)
                    timestamps = trodes.read_trodes_extracted_data_file(timestamps_file_path)
                    self.first_timestamp = int(timestamps["first_timestamp"])
                    print("Extracted first timestamp")
        return

    @staticmethod
    def save_rec_to_h5(recording, rec_path):
        h5_path = rec_path + ".h5"
        json_path = rec_path + ".json"
        LFPRecording.save_metadata_to_json(recording, json_path)
        with h5py.File(h5_path, "w") as f:
            # Save channel dictionary
            channel_group = f.create_group("channels")
            for key, value in recording.channel_dict.items():
                channel_group.attrs[key] = str(value)
            brain_region_dict = f.create_group("brain region dict")
            for key, value in recording.brain_region_dict.items():
                brain_region_dict.attrs[key] = str(value)
            data_group = f.create_group("data")
            data_group.create_dataset("traces", data=recording.traces, compression="gzip", compression_opts=9)

            # Save RMS traces if they exist
            if hasattr(recording, "rms_traces"):
                data_group.create_dataset(
                    "rms_traces", data=recording.rms_traces, compression="gzip", compression_opts=9
                )

            # Save connectivity analysis results if they exist
            if hasattr(recording, "coherence"):
                data_group.create_dataset("coherence", data=recording.coherence, compression="gzip", compression_opts=9)

            if hasattr(recording, "frequencies"):
                data_group.create_dataset(
                    "frequencies", data=recording.frequencies, compression="gzip", compression_opts=9
                )

            if hasattr(recording, "grangers"):
                data_group.create_dataset("grangers", data=recording.grangers, compression="gzip", compression_opts=9)

            if hasattr(recording, "power"):
                data_group.create_dataset("power", data=recording.power, compression="gzip", compression_opts=9)
            if recording.behavior_dict is not None:
                behavior_group = f.create_group("behavior")
                for key, value in recording.behavior_dict.items():
                    if isinstance(value, (np.ndarray, list)):
                        behavior_group.create_dataset(key, data=np.array(value), compression="gzip")
                    else:
                        behavior_group.attrs[key] = str(value)
            metadata = f.create_group("metadata")
            # Save recording metadata
            metadata.attrs["subject"] = recording.subject
            metadata.attrs["merged_rec_path"] = str(recording.merged_rec_path)
            metadata.attrs["first_timestamp"] = recording.first_timestamp
            metadata.attrs["name"] = recording.recording_name
            metadata.attrs["electrical noise frequency"] = recording.elec_noise_freq
            metadata.attrs["sampling rate"] = recording.sampling_rate
            metadata.attrs["min_freq"] = recording.min_freq
            metadata.attrs["max_freq"] = recording.max_freq
            metadata.attrs["recording length"] = recording.rec_length
            metadata.attrs["resample_rate"] = recording.resample_rate
            metadata.attrs["voltage"] = recording.voltage_scaling
            metadata.attrs["half bandwidth product"] = recording.halfbandwidth
            metadata.attrs["window duration"] = recording.timewindow
            metadata.attrs["window step"] = recording.timestep
            # Save behavior dictionary
            if recording.threshold is not None:
                metadata.attrs["zscore theshold"] = recording.threshold

    @staticmethod
    def save_metadata_to_json(recording, json_path):
        """
        Save recording metadata to a JSON file.

        Parameters:
        -----------
        recording : Recording object
            The recording object containing the metadata
        output_path : str or Path
            Path where the JSON file should be saved

        Returns:
        --------
        str
            Path to the saved JSON file
        """
        metadata = {
            # Required metadata
            "subject": recording.subject,
            "name": recording.recording_name,
            "merged_rec_path": str(recording.merged_rec_path),
            "number of channels": int(len(recording.channel_dict.keys())),
            "traces shape": recording.traces.shape,
            "recording length": f"{recording.rec_length} min",
            "resample_rate": recording.resample_rate,
            "min_freq": recording.min_freq,
            "max_freq": recording.max_freq,
            "half_bandwidth_product": recording.halfbandwidth,
            "window_duration": recording.timewindow,
            "window_step": recording.timestep,
            # Optional zscore threshold
            "zscore_threshold": recording.threshold if hasattr(recording, "threshold") else None,
            # Data availability flags
            "has_behavior": hasattr(recording, "behavior_dict") and recording.behavior_dict is not None,
            "has_rms_traces": hasattr(recording, "rms_traces"),
            "has_power": hasattr(recording, "power"),
            "has_granger": hasattr(recording, "grangers"),
            "has_coherence": hasattr(reccording, "coherence")
        }

        # Ensure output directory exists
        os.makedirs(os.path.dirname(json_path), exist_ok=True)

        # Save to JSON file
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=4)

    @staticmethod
    def load_rec_from_h5(h5_path):
        """
        Load a recording object from an H5 file.

        Parameters:
        -----------
        h5_path : str or Path
            Path to the H5 file

        Returns:
        --------
        recording : Recording
            Reconstructed recording object
        """

        with h5py.File(h5_path, "r") as f:
            # Extract metadata
            metadata = f["metadata"]

            # Get channel map
            channel_dict = {}
            channel_group = f["channels"]
            for key, value in channel_group.attrs.items():
                channel_dict[key] = int(value)
            brain_region_dict = bidict()
            brain_regs = f["brain region dict"]
            for key, value in brain_regs.attrs.items():
                brain_region_dict[key] = int(value)
            # Extract behavior dict if it exists
            behavior_dict = None
            if "behavior" in f:
                behavior_dict = {}
                behavior_group = f["behavior"]
                # Load behavior datasets
                for key in behavior_group.keys():
                    behavior_dict[key] = behavior_group[key][:]
                # Load behavior attributes
                for key, value in behavior_group.attrs.items():
                    behavior_dict[key] = value

            # Extract threshold if it exists
            threshold = None
            if "zscore theshold" in metadata.attrs:
                threshold = metadata.attrs["zscore theshold"]
            # Create object with all initialization parameters
            recording = LFPRecording(
                subject=metadata.attrs["subject"],
                channel_dict=channel_dict,
                merged_rec_path=metadata.attrs["merged_rec_path"],
                behavior_dict=behavior_dict,
                elec_noise_freq=metadata.attrs["electrical noise frequency"],
                sampling_rate=metadata.attrs["sampling rate"],
                min_freq=metadata.attrs["min_freq"],
                max_freq=metadata.attrs["max_freq"],
                resample_rate=metadata.attrs["resample_rate"],
                voltage_scaling=metadata.attrs["voltage"],
                threshold=threshold,
                halfbandwidth=metadata.attrs["half bandwidth product"],
                timewindow=metadata.attrs["window duration"],
                timestep=metadata.attrs["window step"],
                load=True,
            )

            # Load additional attributes that aren't part of initialization
            recording.first_timestamp = metadata.attrs["first_timestamp"]
            recording.recording_name = metadata.attrs["name"]
            recording.rec_length = metadata.attrs["recording length"]
            recording.brain_region_dict = brain_region_dict
            # Load data arrays
            data_group = f["data"]
            recording.traces = data_group["traces"][:]

            # Load optional data if they exist
            if "rms_traces" in data_group:
                recording.rms_traces = data_group["rms_traces"][:]
            if "coherence" in data_group:
                recording.coherence = data_group["coherence"][:]
            if "frequencies" in data_group:
                recording.frequencies = data_group["frequencies"][:]
            if "grangers" in data_group:
                recording.grangers = data_group["grangers"][:]
            if "power" in data_group:
                recording.power = data_group["power"][:]
                recording.connectivity, frequencies = connectivity_wrapper.calculate_multitaper(recording.rms_traces, recording.resample_rate, recording.halfbandwidth, recording.timewindow, recording.timestep)

        return recording
