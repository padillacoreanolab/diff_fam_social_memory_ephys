from pathlib import Path
from tqdm import tqdm
from lfp_analysis.LFP_recording import LFPRecording

DEFAULT_KWARGS = {
    "sampling_rate": 20000,
    "voltage_scaling": 0.195,
    "spike_gadgets_multiplier": 0.675,
    "elec_noise_freq": 60,
    "min_freq": 0.5,
    "max_freq": 300,
    "resample_rate": 1000,
    "halfbandwidth": 2,
    "timewindow": 1,
    "timestep": 0.5,
}


class LFPCollection:
    def __init__(
        self,
        recording_to_behavior_dict: dict,
        subject_to_channel_dict: dict,
        data_path: str,
        recording_to_subject_dict: dict,
        threshold: int,
        trodes_directory: str,
        **kwargs
    ):
        """Initialize LFPCollection object."""
        # Required parameters
        self.data_path = data_path
        self.recording_to_behavior_dict = recording_to_behavior_dict
        self.subject_to_channel_dict = subject_to_channel_dict
        self.recording_to_subject_dict = recording_to_subject_dict
        self.trodes_directory = trodes_directory
        self.kwargs = {}
        for key, default_value in DEFAULT_KWARGS.items():
            self.kwargs[key] = kwargs.get(key, default_value)

        self.kwargs["threshold"] = threshold
        self.threshold = threshold
        # Initialize recordings
        self.lfp_recordings = self._make_recordings()

    def _make_recordings(self):
        lfp_recordings = []
        for data_directory in Path(self.data_path).glob("*"):
            if data_directory.is_dir():
                for rec_file in data_directory.glob("*merged.rec"):
                    subject = self.recording_to_subject_dict[rec_file.name]
                    behavior_dict = self.recording_to_behavior_dict[rec_file.name]
                    channel_dict = self.subject_to_channel_dict[subject]
                    lfp_rec = LFPRecording(
                        subject, behavior_dict, channel_dict, rec_file, self.trodes_directory, **self.kwargs
                    )
                    lfp_recordings.append(lfp_rec)

        return lfp_recordings

    def process(self):
        for recording in tqdm(self.lfp_recordings):
            recording.process(self.threshold)
