# TODO:
#   - test trial_decoder result roc_auc values are between 0 and 1
#   - test LOO decoder produces N folds equal to number of recordings
#   - test LOO roc_auc shape matches (T, N_recordings)
#   - test condition_dict path in trial_PCA labels events correctly

import unittest
import numpy as np
from pathlib import Path
import spike.spike_analysis.decoders as decoders
from spike.spike_analysis.spike_collection import SpikeCollection


TEST_DATA_PATH = str(Path(__file__).parent / "test_data")


def make_test_collection(timebin=50, ignore_freq=0.1):
    collection = SpikeCollection(TEST_DATA_PATH)
    events_a = np.array([[5000, 6000], [10000, 11000], [15000, 16000]])
    events_b = np.array([[20000, 21000], [25000, 26000], [30000, 31000]])
    for recording in collection.recordings:
        recording.subject = recording.name
        recording.event_dict = {"event_a": events_a, "event_b": events_b}
    collection.analyze(timebin=timebin, ignore_freq=ignore_freq)
    return collection


def make_synthetic_decoder_data(trials_per_event, T, n_PCs):
    """Build a decoder_data dict with known shapes for unit testing helpers.

    Args:
        trials_per_event: dict mapping event name -> number of trials
        T: timebins per trial
        n_PCs: number of PCs per trial
    """
    return {
        event: [np.random.rand(T, n_PCs) for _ in range(n_trials)]
        for event, n_trials in trials_per_event.items()
    }



class TestPrepData(unittest.TestCase):
    """Unit tests for __prep_data__ using synthetic decoder_data.

    Setup: 2 recordings × 4 event_a trials + 2 recordings × 6 event_b trials
      → 8 positive (event_a), 12 negative (event_b), 20 total
    """

    def setUp(self):
        self.events = ["event_a", "event_b"]
        # 2 recordings × 4 event_a trials = 8 total; 2 recordings × 6 event_b = 12 total
        self.trials_per_event = {"event_a": 8, "event_b": 12}
        self.T = 20
        self.n_PCs = 5
        self.decoder_data = make_synthetic_decoder_data(
            self.trials_per_event, self.T, self.n_PCs
        )

    def test_data_shape(self):
        """data should be (total_trials, n_PCs, T) after transposing."""
        data, _ = decoders.__prep_data__(self.decoder_data, self.events, "event_a")
        self.assertEqual(data.shape, (20, self.n_PCs, self.T))

    def test_labels_sum_and_length(self):
        """8 positive (event_a) labels, 12 negative (event_b) labels, 20 total."""
        _, labels = decoders.__prep_data__(self.decoder_data, self.events, "event_a")
        self.assertEqual(len(labels), 20)
        self.assertEqual(int(labels.sum()), 8)
        self.assertEqual(int((labels == 0).sum()), 12)


class TestPrepDataLOO(unittest.TestCase):
    """Unit tests for __prep_data_loo__ — same shape/label checks plus recording label alignment.

    Setup: 2 recordings × 4 event_a trials + 2 recordings × 6 event_b trials
      → 8 positive (event_a), 12 negative (event_b), 20 total
    The train/test split on recordings happens later in _trial_decoder_LOO,
    so __prep_data_loo__ should contain all recordings.
    """

    def setUp(self):
        self.events = ["event_a", "event_b"]
        self.recordings = ["rec_0", "rec_1"]
        # 2 recordings × 4 event_a = 8 total; 2 recordings × 6 event_b = 12 total
        self.trials_per_event = {"event_a": 8, "event_b": 12}
        self.T = 20
        self.n_PCs = 5
        self.decoder_data = make_synthetic_decoder_data(
            self.trials_per_event, self.T, self.n_PCs
        )
        # 4 trials from rec_0 and 4 from rec_1 for event_a; 6 each for event_b
        self.recording_labels = {
            "event_a": ["rec_0"] * 4 + ["rec_1"] * 4,
            "event_b": ["rec_0"] * 6 + ["rec_1"] * 6,
        }

    def test_data_shape(self):
        data, _, _ = decoders.__prep_data_loo__(
            self.decoder_data, self.recording_labels, self.events, "event_a"
        )
        self.assertEqual(data.shape, (20, self.n_PCs, self.T))

    def test_labels_sum_and_length(self):
        """8 positive (event_a) labels, 12 negative (event_b) labels, 20 total."""
        _, labels, _ = decoders.__prep_data_loo__(
            self.decoder_data, self.recording_labels, self.events, "event_a"
        )
        self.assertEqual(len(labels), 20)
        self.assertEqual(int(labels.sum()), 8)
        self.assertEqual(int((labels == 0).sum()), 12)

    def test_recording_label_array_aligned_with_labels(self):
        """rec_label_arr must have the same length as labels so LOO masking is valid."""
        _, labels, rec_label_arr = decoders.__prep_data_loo__(
            self.decoder_data, self.recording_labels, self.events, "event_a"
        )
        self.assertEqual(len(rec_label_arr), len(labels))

    def test_all_recordings_present(self):
        """All recordings should be present — the train/test split happens in _trial_decoder_LOO, not here."""
        _, _, rec_label_arr = decoders.__prep_data_loo__(
            self.decoder_data, self.recording_labels, self.events, "event_a"
        )
        self.assertEqual(set(rec_label_arr), set(self.recordings))


class TestTrialPCA(unittest.TestCase):
    """Tests for trial_PCA decoder_data output structure."""

    def setUp(self):
        self.events = ["event_a", "event_b"]
        self.event_length = 1.0
        self.pre_window = 0
        self.post_window = 0
        self.timebin = 50
        self.no_PCs = 3
        self.collection = make_test_collection(timebin=self.timebin)
        self.decoder_data = decoders.trial_PCA(
            self.collection, self.event_length, self.pre_window,
            self.post_window, self.no_PCs, self.events,
        )

    def test_decoder_data_has_correct_event_keys(self):
        self.assertEqual(set(self.decoder_data.keys()), set(self.events))

    def test_each_trial_shape(self):
        """Each trial should be (T, no_PCs)."""
        T = int((self.event_length + self.pre_window + self.post_window) * 1000 / self.timebin)
        for event in self.events:
            for trial in self.decoder_data[event]:
                self.assertEqual(trial.shape, (T, self.no_PCs))

    def test_recording_labels_returned_when_requested(self):
        decoder_data, recording_labels = decoders.trial_PCA(
            self.collection, self.event_length, self.pre_window,
            self.post_window, self.no_PCs, self.events,
            return_recording_labels=True,
        )
        self.assertEqual(set(recording_labels.keys()), set(self.events))
        for event in self.events:
            self.assertEqual(len(recording_labels[event]), len(decoder_data[event]))


class TestTrialDecoder(unittest.TestCase):
    """Integration tests for trial_decoder result object."""

    def setUp(self):
        self.events = ["event_a", "event_b"]
        self.event_length = 1.0
        self.pre_window = 0
        self.post_window = 0
        self.timebin = 50
        self.no_PCs = 3
        self.num_fold = 2
        self.collection = make_test_collection(timebin=self.timebin)
        self.result = decoders.trial_decoder(
            self.collection,
            num_fold=self.num_fold,
            no_PCs=self.no_PCs,
            events=self.events,
            event_length=self.event_length,
            pre_window=self.pre_window,
            post_window=self.post_window,
        )

    def test_result_has_correct_events(self):
        # with 2 events the decoder is a binary problem — it runs once and breaks,
        # so result.events contains only the first event (positive class)
        self.assertEqual(self.result.events, [self.events[0]])

    def test_roc_auc_shape(self):
        """roc_auc should be (T, num_fold) for each event."""
        T = int((self.event_length + self.pre_window + self.post_window) * 1000 / self.timebin)
        for event in self.result.events:
            self.assertEqual(self.result.results[event].roc_auc.shape, (T, self.num_fold))

    def test_shuffle_roc_auc_shape(self):
        """Shuffle roc_auc should match real roc_auc shape."""
        for event in self.result.events:
            self.assertEqual(
                self.result.results[event].roc_auc_shuffle.shape,
                self.result.results[event].roc_auc.shape,
            )


if __name__ == "__main__":
    unittest.main()
