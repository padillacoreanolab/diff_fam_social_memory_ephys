import os
from spike.spike_analysis.spike_recording import SpikeRecording
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
                        os.path.join(self.path, directory),
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
                                recording.event_dict = self.event_dict[directory]
                            except KeyError:
                                print(f"{directory} not found in event dict")
                        collection.append(recording)
        self.recordings = collection

    def analyze(self, timebin, ignore_freq=0.1, smoothing_window=None, mode="same"):
        self.timebin = timebin
        self.ignore_freq = ignore_freq
        self.smoothing_window = smoothing_window
        self.mode = mode
        analyzed_neurons = 0
        good_neurons = 0
        for recording in self.recordings:
            recording.analyze(timebin, ignore_freq, smoothing_window, mode)
            analyzed_neurons += recording.analyzed_neurons
            good_neurons += recording.good_neurons
        self.good_neurons = good_neurons
        self.analyzed_neurons = analyzed_neurons
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
        for recording in self.recordings:
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
                for recording in self.recordings:
                    print(recording.name, "keys:", recording.event_dict.keys())
                is_good = False
        if len(missing_subject) > 0:
            print(f"These recordings are missing subjects: {missing_subject}")
            is_good = False
        if not event_type:
            print("Event arrays are not 2 dimensional numpy arrays of shape (n x 2).")
            print("Please fix.")
        if is_good:
            for recording in self.recordings:
                recording.all_set = True
            print("All set to analyze")

    def __str__(self):
        """
        Returns a summary of the SpikeCollection object, including:
        - Number of recordings
        - Average number of good units
        - Average number of events per event type (if event_dicts are present)
        - Number of unique subjects
        """
        num_recordings = len(self.recordings)
        avg_good_units = (
            sum(recording.good_neurons for recording in self.recordings) / num_recordings if num_recordings > 0 else 0
        )

        # Calculate average number of events per event type
        event_counts = {}
        for recording in self.recordings:
            if hasattr(recording, "event_dict"):
                for event, events in recording.event_dict.items():
                    event_counts[event] = event_counts.get(event, 0) + len(events)

        avg_events_per_type = (
            {event: count / num_recordings for event, count in event_counts.items()} if event_counts else "N/A"
        )

        # Get the number of unique subjects
        missing_subjects = [
            recording.name for recording in self.recordings if getattr(recording, "subject", None) is None
        ]
        if missing_subjects:
            subject_info = f"Missing Subjects for Recordings: {missing_subjects}"
        else:
            unique_subjects = len(set(recording.subject for recording in self.recordings))
            subject_info = f"Number of Unique Subjects: {unique_subjects}"

        return (
            f"SpikeCollection Summary:\n"
            f"  Number of Recordings: {num_recordings}\n"
            f"  Average Number of Good Units: {avg_good_units:.2f}\n"
            f"  Average Number of Events per Event Type: {avg_events_per_type}\n"
            f"  {subject_info}\n"
            f"\n"
        )

    def recording_details(self):
        details = []
        for recording in self.recordings:
            subject = getattr(recording, "subject", "Unknown")
            good_units = getattr(recording, "good_neurons", 0)
            recording_length = recording.timestamps_var[-1] / recording.sampling_rate / 60  # in minutes

            # Get the number of events per event type
            event_counts = {}
            if hasattr(recording, "event_dict"):
                for event, events in recording.event_dict.items():
                    event_counts[event] = len(events)

            details.append(
                f"\n"
                f"Recording: {recording.name}\n"
                f"  Subject: {subject}\n"
                f"  Number of Good Units: {good_units}\n"
                f"  Recording Length: {recording_length:.2f} minutes\n"
                f"  Events per Event Type: {event_counts}\n"
            )
        print(f"Recording Details:\n" f"{''.join(details)}")
        return None
