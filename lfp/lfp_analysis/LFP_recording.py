import spikeinterface.extractors as se
import spikeinterface.preprocessing as sp
import lfp_analysis.preprocessor as preprocessor
import lfp_analysis.connectivity_wrapper as connectivity_wrapper
import trodes.read_exported as trodes
import os
from pathlib import Path

LFP_FREQ_MIN = 0.5
LFP_FREQ_MAX = 300
ELECTRIC_NOISE_FREQ = 60
LFP_SAMPLING_RATE = 1000
EPHYS_SAMPLING_RATE = 20000


class LFPRecording:
    def __init__(
        self,
        subject,
        behavior_dict,
        channel_dict,
        merged_rec_path,
        elec_noise_freq=60,
        sampling_rate=20000,
        min_freq=0.5,
        max_freq=300,
        resample_rate=1000,
        voltage_scaling=0.195,
        spike_gadgets_multiplier=0.675,
        threshold=None,
        halfbandwidth=2,
        timewindow=1,
        timestep=0.5,
    ):
        self.merged_rec_path = merged_rec_path
        self.sampling_rate = sampling_rate
        self.subject = subject
        self.behavior_dict = behavior_dict
        self.channel_map = channel_dict
        self.voltage_scaling = voltage_scaling
        self.spike_gadgets_multiplier = spike_gadgets_multiplier
        self.elec_noise_freq = elec_noise_freq
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.resample_rate = resample_rate
        self.threshold = threshold
        self.halfbandwidth = halfbandwidth
        self.timewindow = timewindow
        self.timestep = timestep
        self.rec_path = os.path.dirname(merged_rec_path)
        self._read_trodes()

    def _read_trodes(self):
        recording = se.read_spikegadgets(self.merged_rec_path, stream_id="trodes")
        recording = se.read_spikegadgets(self.merged_rec_path, stream_id="trodes")
        recording = sp.notch_filter(recording, freq=self.elec_noise_freq)
        recording = sp.bandpass_filter(recording, freq_min=self.min_freq, freq_max=self.max_freq)
        recording = sp.resample(recording, resample_rate=self.resample_rate)
        self.recording = recording

    def _get_selected_traces(self):
        self.brain_region_dict, sorted_channels = preprocessor.map_to_region(self.channel_map)
        sorted_channels = [str(channel) for channel in sorted_channels]
        # Channel ids are the "names" of the channels as strings
        self.traces = self.recording.get_traces(channel_ids=sorted_channels).T
        return self.traces

    def plot_to_find_threshold(self, threshold, file_path=None):
        zscore_traces = preprocessor.zscore(self.traces, threshold, scaling=self.voltage_scaling)
        preprocessor.plot_zscore(self.traces, zscore_traces, file_path)

    def process(self, threshold=None):
        if (threshold is None) & (self.threshold is None):
            print("Please choose a threshold")
        else:
            if (threshold is None) & (self.threshold is not None):
                threshold = self.threshold
            self.rms_traces = preprocessor.preprocess(self.traces, threshold, self.voltage_scaling, plot=False)
            self.connectivity, self.frequencies, self.power, self.coherence, self.granger = (
                connectivity_wrapper.connectivity_wrapper(
                    self.rms_traces, self.sampling_rate, self.halfbandwidth, self.timewindow, self.timestep
                )
            )

    def export_trodes_timestamps(self, trodes_directory):
        trodes.trodes_extract_single_file(trodes_directory, self.merged_rec_path, mode="-time")
        # need to go to merged.time folder and read merged.timestamps.dat file

    def find_start_recording_time(self):
        timestamps_path = str(Path(self.merged_rec_path).with_suffix(".time"))
        if os.path.exists(timestamps_path):
            for file in os.listdir(timestamps_path):
                if file.endswith(".timestamps.dat"):
                    timestamps_file_path = os.path.join(timestamps_path, file)
                    timestamps = trodes.read_trodes_extracted_data_file(timestamps_file_path)
                    self.first_timestamp = timestamps["first_timestamp"]
        else:
            print("Please run export_trodes_timestamps()")
            print("Trodes installation necessary for this step")
        return
