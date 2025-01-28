from pathlib import Path
from tqdm import tqdm
from lfp.lfp_analysis.LFP_recording import LFPRecording
import os
import numpy as np
import json
import glob

DEFAULT_KWARGS = {
    "sampling_rate": 20000,
    "voltage_scaling": 0.195,
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
        subject_to_channel_dict: dict,
        data_path: str,
        recording_to_subject_dict: dict,
        threshold: int,
        recording_to_behavior_dict=None,
        trodes_directory=None,
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
                        subject=subject,
                        channel_dict=channel_dict,
                        merged_rec_path=rec_file,
                        behavior_dict=behavior_dict,
                        trodes_directory=self.trodes_directory,
                        **self.kwargs,
                    )
                    lfp_recordings.append(lfp_rec)
        return lfp_recordings

    def process(self):
        for recording in tqdm(self.lfp_recordings):
            recording.process(self.threshold)

    def save_to_json(collection, output_path):
        """Save LFP collection metadata to JSON and individual recordings to H5 files.

        Parameters
        ----------
        collection : LFPCollection
            Collection object containing recordings and metadata
        output_path : str or Path
            Path to save the JSON metadata file
        """
        # Prepare metadata dictionary
        output_data = {
            "metadata": {
                "data_path": collection.data_path,
                "trodes_directory": collection.trodes_directory,
                "threshold": collection.threshold,
                "number of recordings": len(collection.lfp_recordings),
            },
            "kwargs": collection.kwargs,
            "dictionaries": {
                "recording_to_behavior": collection.recording_to_behavior_dict,
                "subject_to_channel": collection.subject_to_channel_dict,
                "recording_to_subject": collection.recording_to_subject_dict,
            },
        }

        # Convert numpy arrays to lists in recording_to_behavior_dict
        if collection.recording_to_behavior_dict is not None:
            for recording_name, behavior_dict in output_data["dictionaries"]["recording_to_behavior"].items():
                for key, value in behavior_dict.items():
                    if isinstance(value, np.ndarray):
                        behavior_dict[key] = value.tolist()

        # Create directory for JSON
        collection_path = os.path.join(output_path, "lfp_collection.json")
        os.makedirs(output_path, exist_ok=True)

        # Save metadata to JSON
        with open(collection_path, "w") as f:
            json.dump(output_data, f, indent=4, default=str)

        # Create and save recordings to separate directory
        recordings_dir = os.path.join(output_path, "recordings")
        os.makedirs(recordings_dir, exist_ok=True)

        for rec in collection.lfp_recordings:
            rec_path = os.path.join(recordings_dir, f"{rec.recording_name}")  # Use recording_name attribute
            LFPRecording.save_rec_to_h5(rec, rec_path)

    @staticmethod
    def load_collection(json_path):
        """Load collection from JSON metadata and H5 recordings.

        Parameters
        ----------
        json_path : str or Path
            Path to the JSON metadata file

        Returns
        -------
        LFPCollection
            Loaded collection object
        """
        json_path = Path(json_path)

        # Load JSON metadata
        with open(json_path, "r") as f:
            data = json.load(f)

        # Extract metadata with defaults for backward compatibility
        metadata = data["metadata"]
        # Create collection instance
        collection = LFPCollection(
            subject_to_channel_dict=data["dictionaries"]["subject_to_channel"],
            data_path=metadata["data_path"],
            recording_to_subject_dict=data["dictionaries"]["recording_to_subject"],
            threshold=metadata["threshold"],
            recording_to_behavior_dict=data["dictionaries"]["recording_to_behavior"],
            trodes_directory=metadata["trodes_directory"],
            **data["kwargs"],
        )

        # Load recordings from H5 files
        json_dir = os.path.dirname(json_path)
        recordings_dir = os.path.join(json_dir, "recordings")
        if not os.path.exists(recordings_dir):
            raise FileNotFoundError(f"Recordings directory not found at {recordings_dir}")

        collection.lfp_recordings = []
        for h5_file in Path(recordings_dir).glob("*.h5"):  # Sort for consistent loading order
            try:
                recording = LFPRecording.load_rec_from_h5(h5_file)
                collection.lfp_recordings.append(recording)
            except Exception as e:
                raise RuntimeError(f"Failed to load recording {h5_file}: {str(e)}")

        return collection
