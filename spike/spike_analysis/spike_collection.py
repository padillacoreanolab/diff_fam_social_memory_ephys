import os
from spike_recording import SpikeRecording

# to do need to not add recordings that have no good neurons


class SpikeCollection:
    """
    This class initializes and reads in phy folders as EphysRecording
    instances.

    Attributes:
        path: str, relative path to the folder of merged.rec files
            for each reacording
        sampling_rate: int, default=20000 sampling rate of ephys device in Hz
        wilcox_dfs: dict
            keys: str, '{event1 } vs {event2} ({equalize}s)' or
            '{equalize}s {event} vs {baseline_window}s baseline'
            values: df, of wilcoxon stats, p values, recording name, subject,
            and event type
        zscored_events: dict
        PCA_dfs:
        fishers_exact: dict

    """

    def __init__(self, path, sampling_rate=20000):
        self.sampling_rate = sampling_rate
        self.path = path
        self.wilcox_dfs = {}
        self.zscored_events = {}
        self.PCA_dfs = {}
        self.fishers_exact = {}
        self.make_collection()
        print("Please assign event dictionaries to each recording")
        print("as recording.event_dict")
        print("event_dict = {event name(str): np.array[[start(ms), stop(ms)]...]")
        print("Please assign subjects to each recording as recording.subject")

    def make_collection(self):
        collection = {}
        for root, dirs, files in os.walk(self.path):
            for directory in dirs:
                if directory.endswith("merged.rec"):
                    print("loading ", directory)
                    tempobject = SpikeRecording(
                        os.path.join(self.path, directory, "phy"),
                        self.sampling_rate,
                    )
                if "good" not in tempobject.labels_dict:
                    print(f"{directory} has no good units")
                    print("and will not be included in the collection")
                else:
                    collection[directory] = tempobject
        self.collection = collection
