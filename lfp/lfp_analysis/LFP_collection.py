from pathlib import Path
from tqdm import tqdm
from lfp.lfp_analysis.LFP_recording import LFPRecording
import lfp.lfp_analysis.preprocessor as preprocessor
import os
import numpy as np
import json
from bidict import bidict
import matplotlib.pyplot as plt


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
    """
    LFPCollection is a collection of LFPRecording objects. For channel mapping, the orientation of the head stage matters. When you open trodes, the channel should match the trodes label minus 1 since trodes is 1 indexed and we want 0 indexed.
    
    Args (4 required, 4 optional):
        subject_to_channel_dict: dict, subject names to brain regions to channel       
            mapping.
            key: str, subject name (must match with recording_to_subject_dict)
            value: dict, brian region to channel numbers  
                key: str, brain region, must match exactly across subjects
                value: int, channel number (0 indexed) 
        data_path: str, path to data folder containing your .rec folders with merged.rec files in them. Type an r in front of your path ie r"path/to/data"
        recording_to_subject_dict: dict, exact names of your merged.rec files to the subject name
            key: str, merged.rec file name exactly as it appears in your data folder
            value: str, subject name matching exactly what is in your subject to channel dict
        threshold: int, default=None, the number of median absolute deviations (MADs) to filter out of the LFP traces, similar to standard deviations (SDs)
        recording_to_event_dict: dict, optional, default=None
            key: str, merged.rec file name exactly as it appears in your data folder
            value: dict, event names to start and stop time arrays
                key: str, event name, must be consistent across recordings
                value: numpy array, shape (n,2) where n is the number of events and 2 is the start and stop time of each event 
        target_confirmation_dict: dict, optional, default=None
            key: str, subject name, matching exactly with subject_to_channel and recording_to_subject dicts
            value: list, a list of strings of brain regions (must match with subject_to_channel_dict) of brain regions to exclude per subject
        trodes_directory: str, path to trodes folder with an r outside the string, ex. r'path/to/trodes/fodler'
        json_path: str, path to json folder for collection in same folder with folder of rec jsons and h5s if you are loading a pre-initialized LFPCollection object, put an r outside of the path string. ex. r'path/to/json'
        
        
        
    """
    def __init__(
        self,
        subject_to_channel_dict: dict,
        data_path: str,
        recording_to_subject_dict: dict,
        threshold: None,
        recording_to_event_dict=None,
        target_confirmation_dict=None,
        #{subject: [list of bad brain regions], each subject needs a list!
        # subjects with all good targets needs to be assigned to an empty list i.e. {subject: []}
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
        # if json path is given, read in h5 files instead of making new recording objects
        if json_path is not None:
            self.load_recordings(json_path)
        else:
            self.recordings = self._make_recordings()
            self.brain_region_dict = self.recordings[0].brain_region_dict
        if target_confirmation_dict is not None:
            self.exclude_regions(target_confirmation_dict)
        
    def _make_recordings(self):
        recordings = []
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
                    recordings.append(lfp_rec)
        return recordings
    
    def diagnostic_plots_channel_finder(self):
        """
        Plots raw traces and filtered traces side by side for each brain region.
        Skip regions that contain only NaNs.

        Args (1 requires):
            threshold: int, the number of median absolute deviations (MADs) to filter out of the LFP traces, similar to standard deviations (SDs)
            
        Returns: 
            none
        """
        for recording in self.recordings:
            recording.plot_all_channels()

    def diagnostic_plots(self, threshold):
        """
        Plots raw traces and filtered traces side by side for each brain region.
        Skip regions that contain only NaNs.

        Args (1 requires):
            threshold: int, the number of median absolute deviations (MADs) to filter out of the LFP traces, similar to standard deviations (SDs)
            
        Returns: 
            none
        """
        for recording in self.recordings:
            # Find valid regions (not all NaNs)
            scaled_traces = preprocessor.scale_voltage(recording.traces, recording.voltage_scaling)
            zscore_traces = preprocessor.zscore(scaled_traces)
            filtered_traces = preprocessor.zscore_filter(zscore_traces, scaled_traces, threshold)
            brain_region_dict = recording.brain_region_dict
            valid_regions = []
            for region_idx in range(scaled_traces.shape[1]):
                if not np.isnan(scaled_traces[:, region_idx]).all() and region_idx in brain_region_dict.inverse:
                    valid_regions.append(region_idx)

            if not valid_regions:
                print("No valid regions to plot.")
                return

            # Create figure with appropriate number of rows
            n_rows = len(valid_regions)
            fig, axes = plt.subplots(n_rows, 2, figsize=(12, 2 * n_rows))

            # Handle case where there's only one row
            if n_rows == 1:
                axes = axes.reshape(1, 2)

            # Plot each region
            num_samples = scaled_traces.shape[0]
            time_in_seconds = np.arange(num_samples) / recording.resample_rate
            time_in_minutes = time_in_seconds / 60
            for i, region_idx in enumerate(valid_regions):
                region_name = brain_region_dict.inverse[region_idx]

                # Plot raw trace on the left
                axes[i, 0].plot(time_in_minutes, scaled_traces[:, region_idx])
                axes[i, 0].set_title(f"{region_name} - Scaled")
                axes[i, 0].set_ylabel("Amplitude (uV)")

                # Plot filtered trace on the right
                axes[i, 1].plot(time_in_minutes, filtered_traces[:, region_idx])
                axes[i, 1].set_title(f"{region_name} - Filtered")
                axes[i, 1].set_ylabel("Amplitude (uV)")

            # Set common x-label for bottom plots
            if n_rows > 0:
                axes[n_rows-1, 0].set_xlabel("Time (min)")
                axes[n_rows-1, 1].set_xlabel("Time (min)")

            plt.tight_layout()
            plt.suptitle(f'{recording.name}', y = 1, fontsize = 20)
            plt.show()

    def preprocess(self, threshold=None):
        """
        Calculates rms traces for all recordings in the collection. 
        """
        if threshold is None and self.threshold is not None:
            threhsold = self.threshold
        if threshold is None and self.threshold is None:
            print('No threshold has been chosen, LFP signals will not be filtered.')
            print('Using a threshold of 0.')
            self.threshold = 0
        for recording in tqdm(self.recordings):
            recording.preprocess(self.threshold)
        
    
    def calculate_all(self):
        for recording in tqdm(self.recordings):
            recording.calculate_all()
        self.frequencies = self.recordings[0].frequencies

    def calculate_power(self):
        for recording in tqdm(self.recordings):
            recording.calculate_power()
        self.frequencies = self.recordings[0].frequencies
        
    def calculate_coherence(self):
        for recording in tqdm(self.recordings):
            recording.calculate_coherence()
        self.frequencies = self.recordings[0].frequencies
        
    def calculate_granger_causality(self):
        for recording in tqdm(self.recordings):
            recording.calculate_granger_causality()
        self.frequencies = self.recordings[0].frequencies
        
        
    def exclude_regions(self, target_confirmation_dict):
        for recording in self.recordings:
            bad_regions = target_confirmation_dict[recording.subject]
            #check to see if target confirmation exclusion has already been done 
            if hasattr(recording, 'excluded_regions'):
                pass
            else:
                recording.exclude_regions(bad_regions)
    
    def interpolate(self, modes = 'all', kind = 'linear'):
        for recording in self.recordings:
            if modes == 'all':
                recording.interpolate_power(kind)
                recording.interpolate_coherence(kind)
                recording.interpolate_granger(kind)
            else: 
                for mode in modes:
                    if mode == 'power':
                        recording.interpolate_power(kind)
                    elif mode == 'coherence':
                        recording.interpolate_coherence(kind)
                    elif mode == 'granger':
                        recording.interpolate_granger(kind)
                    else:
                        raise ValueError("Invalid mode. Choose 'all', 'power', 'coherence', or 'granger'.")    
           
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
                "number of recordings": len(collection.recordings),
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

        for rec in collection.recordings:
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
    
        collection.brain_region_dict = bidict(data["dictionaries"]["brain_region_dict"])
    
        return collection

    def load_recordings(self, json_path):
        json_dir = os.path.dirname(json_path)
        recordings_dir = os.path.join(json_dir, "recordings")
        if not os.path.exists(recordings_dir):
            raise FileNotFoundError(f"Recordings directory not found at {recordings_dir}")
        self.recordings = []
        for h5_file in Path(recordings_dir).glob("*.h5"):  # Sort for consistent loading order
            try:
                recording = LFPRecording.load_rec_from_h5(h5_file)
                self.recordings.append(recording)
        
            except Exception as e:
                raise RuntimeError(f"Failed to load recording {h5_file}: {str(e)}")
