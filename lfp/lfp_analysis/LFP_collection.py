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
        **kwargs,
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


# TO DO
def combine_collections(list_of_collections):
    attr_match = check_attributes_match(list_of_collections)
    complete_recordings = []
    if attr_match:
        for collection in list_of_collections:
            complete_recordings.extend(collection.collection)
        list_of_collections[0].collection = complete_recordings
        return list_of_collections[0]


def check_attributes_match(instances):
    """
    Check if specified attributes match across all instances.
    Issues warnings for any mismatched attributes.

    Args:
        instances: List of class instances to compare
        attributes: List of attribute names to check

    Returns:
        bool: True if all specified attributes match across instances, False otherwise
    """
    attributes = [
        "timebin",
        "sampling_rate",
        "voltage_scaling",
        "spike_gadgets_multiplier",
        "elec_noise_freq",
        "min_freq",
        "max_freq",
        "resample_rate",
        "halfbandwidth",
        "timewindow",
        "timestep",
        "threshold",
    ]

    if not instances or len(instances) < 2:
        return True

    all_match = True
    first_instance = instances[0]

    for attr in attributes:
        # Get the value from first instance
        try:
            reference_value = getattr(first_instance, attr)
        except AttributeError:
            print(f"Warning: Attribute '{attr}' not found in {type(first_instance).__name__}")
            all_match = False
            continue

        # Compare with all other instances
        for i, instance in enumerate(instances[1:], 2):  # Start enum at 2 for clearer warnings
            try:
                current_value = getattr(instance, attr)
                if current_value != reference_value:
                    print(f"Warning: {attr} mismatch detected:")
                    print(f"  Instance 1: {reference_value}")
                    print(f"  Instance {i}: {current_value}")
                    all_match = False
            except AttributeError:
                print(f"Warning: Attribute '{attr}' not found in instance {i}")
                all_match = False
    return all_match
