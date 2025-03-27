from pathlib import Path
from tqdm import tqdm
from lfp.lfp_analysis.LFP_recording import LFPRecording
import os
import numpy as np
import json
from bidict import bidict


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
        recording_to_event_dict=None,
        trodes_directory=None,
        json_path=None,
        **kwargs,
    ):
        """Initialize LFPCollection object."""
        # Required parameters
        self.data_path = data_path
        self.recording_to_event_dict = recording_to_event_dict
        self.subject_to_channel_dict = subject_to_channel_dict
        self.recording_to_subject_dict = recording_to_subject_dict
        self.trodes_directory = trodes_directory
        self.threshold = threshold
        self.kwargs = {}
        for key, default_value in DEFAULT_KWARGS.items():
            self.kwargs[key] = kwargs.get(key, default_value)

        # Initialize recordings
        if json_path is not None:
            self.load_recordings(json_path)
        else:
            self.lfp_recordings = self._make_recordings()

    def _make_recordings(self):
        lfp_recordings = []
        for data_directory in Path(self.data_path).glob("*"):
            if data_directory.is_dir():
                for rec_file in data_directory.glob("*merged.rec"):
                    subject = self.recording_to_subject_dict[rec_file.name]
                    channel_dict = self.subject_to_channel_dict[subject]
                    if self.recording_to_event_dict is not None:
                        event_dict = self.recording_to_event_dict[rec_file.name]
                    else:
                        event_dict = None
                    lfp_rec = LFPRecording(
                        subject=subject,
                        channel_dict=channel_dict,
                        merged_rec_path=rec_file,
                        event_dict=event_dict,
                        trodes_directory=self.trodes_directory,
                        threshold=self.threshold,
                        **self.kwargs,
                    )
                    lfp_recordings.append(lfp_rec)
        return lfp_recordings

    def process(self):
        is_first = True
        for recording in tqdm(self.lfp_recordings):
            recording.process(self.threshold)
            if is_first:
                self.frequencies = recording.frequencies
                self.brain_region_dict = recording.brain_region_dict
                is_first = False

    def save_to_json(collection, output_path, notes=""):
        """Save LFP collection metadata to JSON and individual recordings to H5 files.

        Parameters
        ----------
        collection : LFPCollection
            Collection object containing recordings and metadata
        output_path : str or Path
            Path to save the JSON metadata file
        notes: opt, str
        """
        # Prepare metadata dictionary
        output_data = {
            "metadata": {
                "data_path": collection.data_path,
                "number of recordings": len(collection.lfp_recordings),
                "brain regions": list(collection.brain_region_dict.keys()),
                "threshold": collection.threshold,
                "frequencies": collection.frequencies,
                "trodes_directory": collection.trodes_directory,
                "Notes": notes,
            },
            "kwargs": collection.kwargs,
            "dictionaries": {
                "recording_to_event": collection.recording_to_event_dict,
                "subject_to_channel": collection.subject_to_channel_dict,
                "recording_to_subject": collection.recording_to_subject_dict,
                "brain_region_dict": dict(collection.brain_region_dict),
            },
        }

        # Convert numpy arrays to lists in recording_to_event_dict
        if collection.recording_to_event_dict is not None:
            for recording_name, event_dict in output_data["dictionaries"]["recording_to_event"].items():
                for key, value in event_dict.items():
                    if isinstance(value, np.ndarray):
                        event_dict[key] = value.tolist()

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
            rec_path = os.path.join(recordings_dir, f"{rec.name}")
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
            #recording_to_event_dict=data["dictionaries"]["recording_to_event"],
            trodes_directory=metadata["trodes_directory"],
            json_path=json_path,
            **data["kwargs"],
        )
        collection.frequencies = metadata["frequencies"]
    
        #collection.brain_region_dict = bidict(data["dictionaries"]["brain_region_dict"])
    
        return collection

    def load_recordings(self, json_path):
        json_dir = os.path.dirname(json_path)
        recordings_dir = os.path.join(json_dir, "recordings")
        if not os.path.exists(recordings_dir):
            raise FileNotFoundError(f"Recordings directory not found at {recordings_dir}")
        self.lfp_recordings = []
        for h5_file in Path(recordings_dir).glob("*.h5"):  # Sort for consistent loading order
            try:
                recording = LFPRecording.load_rec_from_h5(h5_file)
                self.lfp_recordings.append(recording)
        
            except Exception as e:
                raise RuntimeError(f"Failed to load recording {h5_file}: {str(e)}")
