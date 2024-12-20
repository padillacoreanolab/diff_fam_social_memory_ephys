import os
from spike_analysis.spike_recording import SpikeRecording
import spike_analysis.firing_rate_calculations as fr
import numpy as np


class SpikeCollection:
    """
    This class initializes and reads in phy folders as EphysRecording
    instances.

    Attributes:
        path: str, relative path to the folder of merged.rec files
            for each reacording
        sampling_rate: int, default=20000 sampling rate of ephys device in Hz
    """

    def __init__(self, path, event_dict={}, subject_dict={}, sampling_rate=20000):
        self.sampling_rate = sampling_rate
        self.path = path
        self.event_dict = event_dict
        self.subject_dict = subject_dict

        self.make_collection()
        if not event_dict:
            print("Please assign event dictionaries to each recording")
            print("as recording.event_dict")
            print("event_dict = {event name(str): np.array[[start(ms), stop(ms)]...]")
        else:
            self.event_dict = event_dict
        if not subject_dict:
            print("Please assign subjects to each recording as recording.subject")
        else:
            self.subject_dict = subject_dict

    def make_collection(self):
        collection = []
        for root, dirs, files in os.walk(self.path):
            for directory in dirs:
                if directory.endswith("merged.rec"):
                    print("loading ", directory)
                    recording = SpikeRecording(
                        os.path.join(self.path, directory, "phy"),
                        self.sampling_rate,
                    )
                    if "good" not in recording.labels_dict.values():
                        print(f"{directory} has no good units")
                        print("and will not be included in the collection")

                    else:
                        if self.subject_dict:
                            try:
                                recording.subject = self.subject_dict[directory]
                            except KeyError:
                                print(f"{directory} not found in subject dict")
                        if self.event_dict:
                            try:
                                recording.subject = self.event_dict[directory]
                            except KeyError:
                                print(f"{directory} not found in event dict")
                        collection.append(recording)
        self.collection = collection

    def analyze(self, timebin, ignore_freq=0.1, smoothing_window=None, mode="same"):
        self.timebin = timebin
        self.ignore_freq = ignore_freq
        self.smoothing_window = smoothing_window
        self.mode = mode
        for recording in self.collection:
            recording.analyze(timebin, ignore_freq, smoothing_window, mode)
        self.__all_set__()

    def __all_set__(self):
        """
        double checks that all SpikeRecordings in the collection have attributes: subject & event_dict and that
        each event_dict has the same keys. Warns users which recordings are missing subjects or event_dicts.
        If all set, prints "All set to analyze" and calculates spiketrains and firing rates.
        """
        is_first = True
        is_good = True
        missing_events = []
        missing_subject = []
        event_dicts_same = True
        event_type = False
        for recording in self.collection:
            if not hasattr(recording, "event_dict"):
                missing_events.append(recording.name)
            else:
                if is_first:
                    last_recording_events = recording.event_dict.keys()
                    is_first = False
                else:
                    if recording.event_dict.keys() != last_recording_events:
                        event_dicts_same = False
                for value in recording.event_dict.values():
                    if type(value) is np.ndarray:
                        if (value.ndim == 2) & (value.shape[1] == 2):
                            event_type = True
            if not hasattr(recording, "subject"):
                missing_subject.append(recording.name)
        if len(missing_events) > 0:
            print("These recordings are missing event dictionaries:")
            print(f"{missing_events}")
            is_good = False
        else:
            if not event_dicts_same:
                print("Your event dictionary keys are different across recordings.")
                print("Please double check them:")
                for recording in self.collection:
                    print(recording.name, "keys:", recording.event_dict.keys())
                is_good = False
        if len(missing_subject) > 0:
            print(f"These recordings are missing subjects: {missing_subject}")
            is_good = False
        if not event_type:
            print("Event arrays are not 2 dimensional numpy arrays of shape (n x 2).")
            print("Please fix.")
        if is_good:
            for recording in self.collection:
                recording.all_set = True
            print("All set to analyze")
