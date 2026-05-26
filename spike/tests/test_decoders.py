# TODO:
#   - test trial_decoder result roc_auc values are between 0 and 1
#   - test LOO decoder produces N folds equal to number of recordings
#   - test LOO roc_auc shape matches (T, N_recordings)
#   - test condition_dict path in trial_PCA labels events correctly

import unittest
import numpy as np
from pathlib import Path
from unittest.mock import patch, call
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
        self.decoder_data, _ = decoders.trial_PCA(
            self.collection, self.event_length, self.pre_window,
            self.post_window, events=self.events, no_PCs=self.no_PCs,
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
        decoder_data, recording_labels, _ = decoders.trial_PCA(
            self.collection, self.event_length, self.pre_window,
            self.post_window, events=self.events, no_PCs=self.no_PCs,
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


class TestPrepDataFlat(unittest.TestCase):
    """Verify __prep_data_flat__ produces (trials, T*n_PCs) instead of (trials, n_PCs, T)."""

    def setUp(self):
        self.events = ["event_a", "event_b"]
        self.trials_per_event = {"event_a": 8, "event_b": 12}
        self.T = 20
        self.n_PCs = 5
        self.decoder_data = make_synthetic_decoder_data(
            self.trials_per_event, self.T, self.n_PCs
        )

    def test_data_is_2d(self):
        """Flat prep should return a 2D array, not 3D."""
        data, _ = decoders.__prep_data_flat__(self.decoder_data, self.events, "event_a")
        self.assertEqual(data.ndim, 2)

    def test_data_shape(self):
        """Shape should be (total_trials, T*n_PCs)."""
        data, _ = decoders.__prep_data_flat__(self.decoder_data, self.events, "event_a")
        self.assertEqual(data.shape, (20, self.T * self.n_PCs))

    def test_labels_sum_and_length(self):
        """Same label counts as normal prep — flattening shouldn't change trial counts."""
        _, labels = decoders.__prep_data_flat__(self.decoder_data, self.events, "event_a")
        self.assertEqual(len(labels), 20)
        self.assertEqual(int(labels.sum()), 8)
        self.assertEqual(int((labels == 0).sum()), 12)

    def test_flat_vs_normal_same_trial_count(self):
        """Flat and normal prep should have the same number of trials (rows)."""
        data_normal, _ = decoders.__prep_data__(self.decoder_data, self.events, "event_a")
        data_flat, _ = decoders.__prep_data_flat__(self.decoder_data, self.events, "event_a")
        self.assertEqual(data_normal.shape[0], data_flat.shape[0])

    def test_flat_has_more_features_than_normal_timebin_slice(self):
        """Flat features (T*n_PCs) should be T times more than a single timebin slice (n_PCs)."""
        data_normal, _ = decoders.__prep_data__(self.decoder_data, self.events, "event_a")
        data_flat, _ = decoders.__prep_data_flat__(self.decoder_data, self.events, "event_a")
        # normal timebin slice has n_PCs features; flat has T*n_PCs
        self.assertEqual(data_flat.shape[1], data_normal.shape[1] * self.T)


class TestPrepDataFlatLOO(unittest.TestCase):
    """Verify __prep_data_flat_loo__ produces (trials, T*n_PCs) with recording labels."""

    def setUp(self):
        self.events = ["event_a", "event_b"]
        self.recordings = ["rec_0", "rec_1"]
        self.trials_per_event = {"event_a": 8, "event_b": 12}
        self.T = 20
        self.n_PCs = 5
        self.decoder_data = make_synthetic_decoder_data(
            self.trials_per_event, self.T, self.n_PCs
        )
        self.recording_labels = {
            "event_a": ["rec_0"] * 4 + ["rec_1"] * 4,
            "event_b": ["rec_0"] * 6 + ["rec_1"] * 6,
        }

    def test_data_is_2d(self):
        data, _, _ = decoders.__prep_data_flat_loo__(
            self.decoder_data, self.recording_labels, self.events, "event_a"
        )
        self.assertEqual(data.ndim, 2)

    def test_data_shape(self):
        data, _, _ = decoders.__prep_data_flat_loo__(
            self.decoder_data, self.recording_labels, self.events, "event_a"
        )
        self.assertEqual(data.shape, (20, self.T * self.n_PCs))

    def test_rec_label_arr_aligned(self):
        data, labels, rec_label_arr = decoders.__prep_data_flat_loo__(
            self.decoder_data, self.recording_labels, self.events, "event_a"
        )
        self.assertEqual(len(rec_label_arr), len(labels))
        self.assertEqual(len(rec_label_arr), data.shape[0])

    def test_all_recordings_present(self):
        _, _, rec_label_arr = decoders.__prep_data_flat_loo__(
            self.decoder_data, self.recording_labels, self.events, "event_a"
        )
        self.assertEqual(set(rec_label_arr), set(self.recordings))


class TestInputRouting(unittest.TestCase):
    """Verify input='full_trial' routes to flat prep and input='timebin' routes to normal prep."""

    def setUp(self):
        self.events = ["event_a", "event_b"]
        self.trials_per_event = {"event_a": 8, "event_b": 8}
        self.T = 4
        self.n_PCs = 3
        self.num_fold = 2
        self.decoder_data = make_synthetic_decoder_data(
            self.trials_per_event, self.T, self.n_PCs
        )

    def _fake_results(self):
        roc = np.array([0.7, 0.7])
        return (
            {"test_roc_auc": roc, "estimator": [], "probabilities": {"probabilities": [], "labels": []}},
            {"test_roc_auc": roc},
        )

    def test_full_trial_calls_prep_data_flat(self):
        """input='full_trial' should call __prep_data_flat__, not __prep_data__."""
        with patch("spike.spike_analysis.decoders.__prep_data_flat__", wraps=decoders.__prep_data_flat__) as mock_flat, \
             patch("spike.spike_analysis.decoders.__prep_data__", wraps=decoders.__prep_data__) as mock_normal, \
             patch("spike.spike_analysis.decoders._random_forest", return_value=self._fake_results()):
            decoders.trial_decoder(
                None, num_fold=self.num_fold, no_PCs=self.n_PCs,
                events=self.events, event_length=1.0,
                input="full_trial", decoder_data=self.decoder_data,
            )
        self.assertTrue(mock_flat.called)
        mock_normal.assert_not_called()

    def test_timebin_calls_prep_data_normal(self):
        """input='timebin' (default) should call __prep_data__, not __prep_data_flat__."""
        with patch("spike.spike_analysis.decoders.__prep_data__", wraps=decoders.__prep_data__) as mock_normal, \
             patch("spike.spike_analysis.decoders.__prep_data_flat__", wraps=decoders.__prep_data_flat__) as mock_flat, \
             patch("spike.spike_analysis.decoders._random_forest", return_value=self._fake_results()):
            decoders.trial_decoder(
                None, num_fold=self.num_fold, no_PCs=self.n_PCs,
                events=self.events, event_length=1.0,
                input="timebin", decoder_data=self.decoder_data,
            )
        self.assertTrue(mock_normal.called)
        mock_flat.assert_not_called()

    def test_full_trial_result_has_one_timebin(self):
        """full_trial mode runs one classifier total, so roc_auc shape is (1, num_fold)."""
        result = decoders.trial_decoder(
            None, num_fold=self.num_fold, no_PCs=self.n_PCs,
            events=self.events, event_length=1.0,
            input="full_trial", decoder_data=self.decoder_data,
        )
        for event in result.events:
            self.assertEqual(result.results[event].roc_auc.shape[0], 1)

    def test_timebin_result_has_T_timebins(self):
        """timebin mode runs T classifiers, so roc_auc shape[0] == T."""
        result = decoders.trial_decoder(
            None, num_fold=self.num_fold, no_PCs=self.n_PCs,
            events=self.events, event_length=1.0,
            input="timebin", decoder_data=self.decoder_data,
        )
        for event in result.events:
            self.assertEqual(result.results[event].roc_auc.shape[0], self.T)


class TestClassifierRouting(unittest.TestCase):
    """Verify that classifier_type routes to the correct helper functions."""

    def setUp(self):
        self.events = ["event_a", "event_b"]
        self.trials_per_event = {"event_a": 8, "event_b": 8}
        self.T = 4
        self.n_PCs = 3
        self.num_fold = 2
        self.decoder_data = make_synthetic_decoder_data(
            self.trials_per_event, self.T, self.n_PCs
        )

    def _fake_results(self):
        """Minimal return value that satisfies trial_decoder's result-building logic."""
        roc = np.array([0.7, 0.7])
        return (
            {"test_roc_auc": roc, "estimator": [], "probabilities": {"probabilities": [], "labels": []}},
            {"test_roc_auc": roc},
        )

    def test_rf_classifier_calls_random_forest(self):
        """classifier_type='RF' should call _random_forest, never _linear."""
        with patch("spike.spike_analysis.decoders._random_forest", return_value=self._fake_results()) as mock_rf, \
             patch("spike.spike_analysis.decoders._linear") as mock_linear:
            decoders.trial_decoder(
                None, num_fold=self.num_fold, no_PCs=self.n_PCs,
                events=self.events, event_length=1.0,
                classifier_type="RF", decoder_data=self.decoder_data,
            )
        self.assertTrue(mock_rf.called)
        mock_linear.assert_not_called()

    def test_linear_classifier_calls_linear(self):
        """classifier_type='linear' should call _linear, never _random_forest."""
        with patch("spike.spike_analysis.decoders._linear", return_value=self._fake_results()) as mock_linear, \
             patch("spike.spike_analysis.decoders._random_forest") as mock_rf:
            decoders.trial_decoder(
                None, num_fold=self.num_fold, no_PCs=self.n_PCs,
                events=self.events, event_length=1.0,
                classifier_type="linear", decoder_data=self.decoder_data,
            )
        self.assertTrue(mock_linear.called)
        mock_rf.assert_not_called()

    def test_linear_passes_C_to_linear_helper(self):
        """C kwarg should be forwarded to _linear."""
        with patch("spike.spike_analysis.decoders._linear", return_value=self._fake_results()) as mock_linear:
            decoders.trial_decoder(
                None, num_fold=self.num_fold, no_PCs=self.n_PCs,
                events=self.events, event_length=1.0,
                classifier_type="linear", decoder_data=self.decoder_data,
                C=0.1,
            )
        _, call_kwargs = mock_linear.call_args
        self.assertEqual(call_kwargs.get("C"), 0.1)

    def test_rf_is_default_classifier(self):
        """Omitting classifier_type should default to RF."""
        with patch("spike.spike_analysis.decoders._random_forest", return_value=self._fake_results()) as mock_rf, \
             patch("spike.spike_analysis.decoders._linear") as mock_linear:
            decoders.trial_decoder(
                None, num_fold=self.num_fold, no_PCs=self.n_PCs,
                events=self.events, event_length=1.0,
                decoder_data=self.decoder_data,
            )
        self.assertTrue(mock_rf.called)
        mock_linear.assert_not_called()


# ---------------------------------------------------------------------------
# Cross-generalization helper tests
# ---------------------------------------------------------------------------

def make_constant_event_folds(T, n_PCs, num_fold, trials_per_event_per_fold, fill_value):
    """Make event_folds where every trial matrix is filled with fill_value.

    Args:
        trials_per_event_per_fold : list of ints, one per fold (can be uneven)
        fill_value                : scalar to fill every element of every trial
    """
    return [
        [np.full((T, n_PCs), fill_value) for _ in range(n)]
        for n in trials_per_event_per_fold
    ]


class TestBuildTrainFold(unittest.TestCase):
    """Unit tests for __build_train_fold__.

    A trials = all 1.0  (2 per fold × 5 folds = 10 total)
    B trials = all 2.0  (3 per fold × 5 folds = 15 total)

    Hold out fold k=0:
      train_A = 4 folds × 2 = 8 A-trials  → label 0, values 1.0
      train_B = 4 folds × 3 = 12 B-trials → label 1, values 2.0
      X_train shape: (20, n_PCs, T)
    """

    def setUp(self):
        self.T, self.n_PCs = 10, 4
        self.num_fold = 5
        self.n_A_per_fold = 2
        self.n_B_per_fold = 3
        self.n_train_A = (self.num_fold - 1) * self.n_A_per_fold   # 8
        self.n_train_B = (self.num_fold - 1) * self.n_B_per_fold   # 12
        self.n_train_total = self.n_train_A + self.n_train_B        # 20
        self.event_folds = {
            "A": make_constant_event_folds(self.T, self.n_PCs, self.num_fold, [self.n_A_per_fold]*self.num_fold, 1.0),
            "B": make_constant_event_folds(self.T, self.n_PCs, self.num_fold, [self.n_B_per_fold]*self.num_fold, 2.0),
        }

    def test_X_train_shape(self):
        X_train, _ = decoders.__build_train_fold__(0, self.event_folds, "A", "B")
        self.assertEqual(X_train.shape, (self.n_train_total, self.n_PCs, self.T))

    def test_y_train_length_matches_X_train(self):
        X_train, y_train = decoders.__build_train_fold__(0, self.event_folds, "A", "B")
        self.assertEqual(len(y_train), X_train.shape[0])

    def test_y_train_label_counts(self):
        """n_train_A zeros (A) and n_train_B ones (B)."""
        _, y_train = decoders.__build_train_fold__(0, self.event_folds, "A", "B")
        self.assertEqual(int((y_train == 0).sum()), self.n_train_A)
        self.assertEqual(int((y_train == 1).sum()), self.n_train_B)

    def test_X_train_A_trials_contain_ones(self):
        """First n_train_A rows (e1 trials) should all be 1.0."""
        X_train, _ = decoders.__build_train_fold__(0, self.event_folds, "A", "B")
        np.testing.assert_array_equal(X_train[:self.n_train_A], np.ones((self.n_train_A, self.n_PCs, self.T)))

    def test_X_train_B_trials_contain_twos(self):
        """Last n_train_B rows (e2 trials) should all be 2.0."""
        X_train, _ = decoders.__build_train_fold__(0, self.event_folds, "A", "B")
        np.testing.assert_array_equal(X_train[self.n_train_A:], np.full((self.n_train_B, self.n_PCs, self.T), 2.0))

    def test_held_out_fold_excluded_for_all_k(self):
        """Regardless of which fold is held out, train size stays n_train_total."""
        for k in range(self.num_fold):
            X_train, _ = decoders.__build_train_fold__(k, self.event_folds, "A", "B")
            self.assertEqual(X_train.shape[0], self.n_train_total)

    def test_X_train_last_two_dims_are_n_PCs_and_T(self):
        X_train, _ = decoders.__build_train_fold__(0, self.event_folds, "A", "B")
        self.assertEqual(X_train.shape[1], self.n_PCs)
        self.assertEqual(X_train.shape[2], self.T)


class TestBuildTestFold(unittest.TestCase):
    """Unit tests for __build_test_fold__.

    A trials = all 1.0  (2 per fold × 5 folds = 10 total)
    B trials = all 2.0  (3 per fold × 5 folds = 15 total)
    C trials = all 3.0  (4 per fold × 5 folds = 20 total)

    test_pairs for training A vs B (A=0, B=1):
      A_C: A keeps label 0, C gets B's label (1) → fold 0: 2 A + 4 C = 6 trials
      B_C: B keeps label 1, C gets A's label (0) → fold 0: 3 B + 4 C = 7 trials
    """

    def setUp(self):
        self.T, self.n_PCs = 10, 4
        self.num_fold = 5
        self.n_A_per_fold = 2
        self.n_B_per_fold = 3
        self.n_C_per_fold = 4
        self.n_test_AC = self.n_A_per_fold + self.n_C_per_fold  # 6
        self.n_test_BC = self.n_B_per_fold + self.n_C_per_fold  # 7
        self.event_folds = {
            "A": make_constant_event_folds(self.T, self.n_PCs, self.num_fold, [self.n_A_per_fold]*self.num_fold, 1.0),
            "B": make_constant_event_folds(self.T, self.n_PCs, self.num_fold, [self.n_B_per_fold]*self.num_fold, 2.0),
            "C": make_constant_event_folds(self.T, self.n_PCs, self.num_fold, [self.n_C_per_fold]*self.num_fold, 3.0),
        }
        self.test_pairs = [
            ("A", "C", 0, 1),
            ("B", "C", 1, 0),
        ]

    def test_returns_both_test_keys(self):
        X_test, _ = decoders.__build_test_fold__(0, self.event_folds, self.test_pairs)
        self.assertIn("A_C", X_test)
        self.assertIn("B_C", X_test)

    def test_X_test_shape_A_C(self):
        X_test, _ = decoders.__build_test_fold__(0, self.event_folds, self.test_pairs)
        self.assertEqual(X_test["A_C"].shape, (self.n_test_AC, self.n_PCs, self.T))

    def test_X_test_shape_B_C(self):
        X_test, _ = decoders.__build_test_fold__(0, self.event_folds, self.test_pairs)
        self.assertEqual(X_test["B_C"].shape, (self.n_test_BC, self.n_PCs, self.T))

    def test_X_test_A_C_content(self):
        """First n_A_per_fold rows are A (1.0), last n_C_per_fold rows are C (3.0)."""
        X_test, _ = decoders.__build_test_fold__(0, self.event_folds, self.test_pairs)
        np.testing.assert_array_equal(X_test["A_C"][:self.n_A_per_fold], np.ones((self.n_A_per_fold, self.n_PCs, self.T)))
        np.testing.assert_array_equal(X_test["A_C"][self.n_A_per_fold:], np.full((self.n_C_per_fold, self.n_PCs, self.T), 3.0))

    def test_X_test_B_C_content(self):
        """First n_B_per_fold rows are B (2.0), last n_C_per_fold rows are C (3.0)."""
        X_test, _ = decoders.__build_test_fold__(0, self.event_folds, self.test_pairs)
        np.testing.assert_array_equal(X_test["B_C"][:self.n_B_per_fold], np.full((self.n_B_per_fold, self.n_PCs, self.T), 2.0))
        np.testing.assert_array_equal(X_test["B_C"][self.n_B_per_fold:], np.full((self.n_C_per_fold, self.n_PCs, self.T), 3.0))

    def test_y_test_label_values_A_C(self):
        """n_A_per_fold zeros (A keeps 0), n_C_per_fold ones (C gets B's label 1)."""
        _, y_test = decoders.__build_test_fold__(0, self.event_folds, self.test_pairs)
        expected = np.array([0]*self.n_A_per_fold + [1]*self.n_C_per_fold)
        np.testing.assert_array_equal(y_test["A_C"], expected)

    def test_y_test_label_values_B_C(self):
        """n_B_per_fold ones (B keeps 1), n_C_per_fold zeros (C gets A's label 0)."""
        _, y_test = decoders.__build_test_fold__(0, self.event_folds, self.test_pairs)
        expected = np.array([1]*self.n_B_per_fold + [0]*self.n_C_per_fold)
        np.testing.assert_array_equal(y_test["B_C"], expected)

    def test_y_test_length_matches_X_test(self):
        X_test, y_test = decoders.__build_test_fold__(0, self.event_folds, self.test_pairs)
        for key in X_test:
            self.assertEqual(len(y_test[key]), X_test[key].shape[0])


class TestCrossGenDecoder(unittest.TestCase):
    """Structure and shape tests for _cross_gen_decoder.

    A trials = all 1.0  (9 total)
    B trials = all 2.0  (12 total)
    C trials = all 3.0  (15 total)
    3 folds → A: 3/fold, B: 4/fold, C: 5/fold
    """

    def setUp(self):
        self.T, self.n_PCs = 6, 3
        self.num_fold = 3
        self.events = ["A", "B", "C"]
        flat_data = {
            "A": [np.ones((self.T, self.n_PCs)) for _ in range(9)],
            "B": [np.full((self.T, self.n_PCs), 2.0) for _ in range(12)],
            "C": [np.full((self.T, self.n_PCs), 3.0) for _ in range(15)],
        }
        # _cross_gen_decoder expects {train_key: decoder_data} for each pair
        self.decoder_data_by_pair = {
            "A_B": flat_data, "A_C": flat_data, "B_C": flat_data,
        }

    def test_returns_three_training_pair_keys(self):
        raw = decoders._cross_gen_decoder(self.decoder_data_by_pair, self.events, self.num_fold, "RF")
        self.assertEqual(set(raw.keys()), {"A_B", "A_C", "B_C"})

    def test_each_training_pair_has_two_generalization_test_keys(self):
        """Each train key should have exactly 2 generalization test keys (containing the held-out event)."""
        raw = decoders._cross_gen_decoder(self.decoder_data_by_pair, self.events, self.num_fold, "RF")
        for train_key, test_pairs in raw.items():
            train_events = set(train_key.split("_"))
            held_out = (set(self.events) - train_events).pop()
            gen_keys = [k for k in test_pairs if held_out in k.split("_")]
            self.assertEqual(len(gen_keys), 2, f"{train_key} should have 2 gen test keys, got {list(test_pairs.keys())}")

    def test_timebin_list_length_equals_T(self):
        raw = decoders._cross_gen_decoder(self.decoder_data_by_pair, self.events, self.num_fold, "RF")
        for train_key, test_pairs in raw.items():
            for test_key, timebin_list in test_pairs.items():
                self.assertEqual(len(timebin_list), self.T, f"{train_key}→{test_key}")

    def test_roc_auc_array_length_equals_num_fold(self):
        raw = decoders._cross_gen_decoder(self.decoder_data_by_pair, self.events, self.num_fold, "RF")
        for train_key, test_pairs in raw.items():
            for test_key, timebin_list in test_pairs.items():
                for t_dict in timebin_list:
                    self.assertEqual(len(t_dict["test_roc_auc"]), self.num_fold)

    def test_all_roc_auc_values_between_0_and_1(self):
        raw = decoders._cross_gen_decoder(self.decoder_data_by_pair, self.events, self.num_fold, "RF")
        for train_key, test_pairs in raw.items():
            for test_key, timebin_list in test_pairs.items():
                for t_dict in timebin_list:
                    for val in t_dict["test_roc_auc"]:
                        if not np.isnan(val):
                            self.assertGreaterEqual(val, 0.0)
                            self.assertLessEqual(val, 1.0)

    def test_test_keys_contain_held_out_event(self):
        """Generalization test keys must include the held-out (third) event."""
        raw = decoders._cross_gen_decoder(self.decoder_data_by_pair, self.events, self.num_fold, "RF")
        for train_key in raw:
            train_events = set(train_key.split("_"))
            held_out = (set(self.events) - train_events).pop()
            gen_keys = [k for k in raw[train_key] if held_out in k.split("_")]
            self.assertGreater(len(gen_keys), 0, f"No gen keys containing {held_out} in {train_key}")


class TestCrossGenResults(unittest.TestCase):
    """Integration tests for cross_gen_results object."""

    def setUp(self):
        self.T, self.n_PCs = 6, 3
        self.num_fold = 3
        self.events = ["A", "B", "C"]
        flat_data = {
            "A": [np.ones((self.T, self.n_PCs)) for _ in range(9)],
            "B": [np.full((self.T, self.n_PCs), 2.0) for _ in range(12)],
            "C": [np.full((self.T, self.n_PCs), 3.0) for _ in range(15)],
        }
        decoder_data_by_pair = {
            "A_B": flat_data, "A_C": flat_data, "B_C": flat_data,
        }
        raw = decoders._cross_gen_decoder(decoder_data_by_pair, self.events, self.num_fold, "RF")
        self.result = decoders.cross_gen_results(raw, self.num_fold, 1.0, 0, 0)

    def test_roc_auc_scores_has_three_train_keys(self):
        self.assertEqual(set(self.result.roc_auc_scores.keys()), {"A_B", "A_C", "B_C"})

    def test_each_train_key_has_two_generalization_test_keys(self):
        """Each train key should have 2 generalization test keys (containing held-out event)."""
        for train_key, test_pairs in self.result.roc_auc_scores.items():
            train_events = set(train_key.split("_"))
            held_out = (set(self.events) - train_events).pop()
            gen_keys = [k for k in test_pairs if held_out in k.split("_")]
            self.assertEqual(len(gen_keys), 2, f"{train_key}: {list(test_pairs.keys())}")

    def test_nested_model_result_roc_auc_shape(self):
        """Each nested_model_result should have roc_auc shape (T, num_fold)."""
        for train_key, test_pairs in self.result.roc_auc_scores.items():
            for test_key, nmr in test_pairs.items():
                self.assertEqual(nmr.roc_auc.shape, (self.T, self.num_fold))

    def test_repr_runs_without_error(self):
        try:
            repr(self.result)
        except Exception as e:
            self.fail(f"__repr__ raised {e}")


class TestSplitIntoFolds(unittest.TestCase):
    """Unit tests for __split_into_folds__."""

    def setUp(self):
        self.T, self.n_PCs = 10, 4
        self.num_fold = 5

    def _make_trials(self, n):
        return [np.random.rand(self.T, self.n_PCs) for _ in range(n)]

    def test_returns_correct_number_of_folds(self):
        trials = self._make_trials(20)
        folds = decoders.__split_into_folds__(trials, self.num_fold)
        self.assertEqual(len(folds), self.num_fold)

    def test_total_trials_preserved(self):
        """All trials across all folds should sum to the original count."""
        trials = self._make_trials(20)
        folds = decoders.__split_into_folds__(trials, self.num_fold)
        self.assertEqual(sum(len(f) for f in folds), 20)

    def test_no_trial_appears_twice(self):
        """Each trial should appear in exactly one fold (no duplicates by identity)."""
        trials = self._make_trials(20)
        folds = decoders.__split_into_folds__(trials, self.num_fold)
        all_ids = [id(t) for fold in folds for t in fold]
        self.assertEqual(len(all_ids), len(set(all_ids)))

    def test_each_fold_is_a_list(self):
        trials = self._make_trials(20)
        folds = decoders.__split_into_folds__(trials, self.num_fold)
        for fold in folds:
            self.assertIsInstance(fold, list)

    def test_each_trial_shape_preserved(self):
        """Splitting should not alter the shape of individual trial matrices."""
        trials = self._make_trials(20)
        folds = decoders.__split_into_folds__(trials, self.num_fold)
        for fold in folds:
            for trial in fold:
                self.assertEqual(trial.shape, (self.T, self.n_PCs))

    def test_uneven_split_still_covers_all_trials(self):
        """17 trials into 5 folds — numpy array_split handles remainders gracefully."""
        trials = self._make_trials(17)
        folds = decoders.__split_into_folds__(trials, self.num_fold)
        self.assertEqual(sum(len(f) for f in folds), 17)

    def test_single_fold(self):
        """num_fold=1 should put everything in one fold."""
        trials = self._make_trials(10)
        folds = decoders.__split_into_folds__(trials, 1)
        self.assertEqual(len(folds), 1)
        self.assertEqual(len(folds[0]), 10)


class TestFitClfSingle(unittest.TestCase):
    """Unit tests for __fit_clf_single__."""

    def setUp(self):
        np.random.seed(0)
        self.X = np.random.rand(20, 5)
        self.y = np.array([0] * 10 + [1] * 10)

    def test_rf_returns_fitted_classifier(self):
        clf = decoders.__fit_clf_single__(self.X, self.y, "RF")
        self.assertTrue(hasattr(clf, "predict"))
        # sklearn fitted estimators expose estimators_ (BaggingClassifier)
        self.assertTrue(hasattr(clf, "estimators_"))

    def test_linear_returns_fitted_classifier(self):
        clf = decoders.__fit_clf_single__(self.X, self.y, "linear")
        self.assertTrue(hasattr(clf, "predict"))
        self.assertTrue(hasattr(clf, "coef_"))

    def test_linear_accepts_C_kwarg(self):
        clf = decoders.__fit_clf_single__(self.X, self.y, "linear", C=0.1)
        self.assertEqual(clf.C, 0.1)


class TestScoreTest(unittest.TestCase):
    """Unit tests for __score_test__."""

    def setUp(self):
        np.random.seed(42)
        X_train = np.random.rand(40, 5)
        y_train = np.array([0] * 20 + [1] * 20)
        self.clf_rf = decoders.__fit_clf_single__(X_train, y_train, "RF")
        self.clf_lin = decoders.__fit_clf_single__(X_train, y_train, "linear")
        self.X_test = np.random.rand(10, 5)
        self.y_test = np.array([0] * 5 + [1] * 5)

    def test_rf_returns_float_between_0_and_1(self):
        auc = decoders.__score_test__(self.clf_rf, self.X_test, self.y_test, "RF")
        self.assertIsInstance(auc, float)
        self.assertGreaterEqual(auc, 0.0)
        self.assertLessEqual(auc, 1.0)

    def test_linear_returns_float_between_0_and_1(self):
        auc = decoders.__score_test__(self.clf_lin, self.X_test, self.y_test, "linear")
        self.assertIsInstance(auc, float)
        self.assertGreaterEqual(auc, 0.0)
        self.assertLessEqual(auc, 1.0)

    def test_single_class_in_test_returns_nan(self):
        """If test set has only one class, roc_auc is undefined — should return nan."""
        y_one_class = np.zeros(10)
        auc = decoders.__score_test__(self.clf_rf, self.X_test, y_one_class, "RF")
        self.assertTrue(np.isnan(auc))


# ---------------------------------------------------------------------------
# LOO cross-gen tests
# ---------------------------------------------------------------------------

class TestCrossGenDecoderLOOHoldout(unittest.TestCase):
    """Verify the actual LOO invariant: held-out recording absent from X_train,
    and each X_test contains data from exactly one recording.

    Each recording gets a unique constant fill value (10, 20, 30) across all events,
    so we can identify which recording any row came from by its value.
    input='full_trial' is used so there is exactly one __fit_clf_single__ call per fold
    per training pair (rather than T calls), making fold order predictable.
    """

    def setUp(self):
        self.T, self.n_PCs = 2, 2
        self.events = ["A", "B", "C"]
        self.recs = ["rec_0", "rec_1", "rec_2"]
        self.rec_values = [10.0, 20.0, 30.0]
        n_per_rec = 3

        flat_data = {ev: [] for ev in self.events}
        rec_labels = {ev: [] for ev in self.events}
        for ev in self.events:
            for rec, val in zip(self.recs, self.rec_values):
                for _ in range(n_per_rec):
                    flat_data[ev].append(np.full((self.T, self.n_PCs), val))
                    rec_labels[ev].append(rec)

        self.decoder_data_by_pair = {"A_B": flat_data, "A_C": flat_data, "B_C": flat_data}
        self.recording_labels_by_pair = {"A_B": rec_labels, "A_C": rec_labels, "B_C": rec_labels}

    def _run_capturing_fit(self):
        """Run LOO decoder and capture every X_train passed to __fit_clf_single__."""
        captured = []
        original = decoders.__fit_clf_single__

        def capturing_fit(X_train, y_train, classifier_type, **kwargs):
            captured.append(X_train.copy())
            return original(X_train, y_train, classifier_type, **kwargs)

        with patch("spike.spike_analysis.decoders.__fit_clf_single__", side_effect=capturing_fit):
            decoders._cross_gen_decoder(
                self.decoder_data_by_pair, self.events, num_fold=5,
                classifier_type="linear", input="full_trial",
                recording_labels_by_pair=self.recording_labels_by_pair,
            )
        return captured

    def _run_capturing_score(self):
        """Run LOO decoder and capture every X_test passed to __score_test__."""
        captured = []
        original = decoders.__score_test__

        def capturing_score(clf, X_test, y_test, classifier_type):
            captured.append(X_test.copy())
            return original(clf, X_test, y_test, classifier_type)

        with patch("spike.spike_analysis.decoders.__score_test__", side_effect=capturing_score):
            decoders._cross_gen_decoder(
                self.decoder_data_by_pair, self.events, num_fold=5,
                classifier_type="linear", input="full_trial",
                recording_labels_by_pair=self.recording_labels_by_pair,
            )
        return captured

    def test_held_out_recording_absent_from_X_train(self):
        """For each LOO fold, the held-out recording's value must not appear in X_train.
        In full_trial mode: 3 pairs × 3 folds × 2 fits (clf + clf_shuf) = 18 captures.
        Folds iterate in rec order within each pair, with 2 consecutive captures per fold.
        """
        captured = self._run_capturing_fit()

        n_pairs = 3  # combinations of 3 events
        n_folds = len(self.recs)
        for pair_idx in range(n_pairs):
            for fold_idx, (rec, val) in enumerate(zip(self.recs, self.rec_values)):
                base = pair_idx * n_folds * 2 + fold_idx * 2
                X_train = captured[base]
                self.assertFalse(
                    np.any(X_train == val),
                    f"pair {pair_idx}, held-out {rec}: value {val} found in X_train"
                )
                # sanity check: other recordings ARE present
                for other_val in self.rec_values:
                    if other_val != val:
                        self.assertTrue(
                            np.any(X_train == other_val),
                            f"pair {pair_idx}, held-out {rec}: expected value {other_val} absent from X_train"
                        )

    def test_X_test_contains_only_held_out_recording(self):
        """Each X_test passed to __score_test__ should contain values from exactly
        one recording (the held-out one). Verified by checking that each X_test's
        unique values all belong to a single recording's fill value.
        """
        captured = self._run_capturing_score()
        rec_value_set = set(self.rec_values)

        for i, X_test in enumerate(captured):
            unique_vals = set(np.unique(X_test))
            vals_from_recs = unique_vals & rec_value_set
            self.assertEqual(
                len(vals_from_recs), 1,
                f"X_test #{i} contains values from {len(vals_from_recs)} recordings: {vals_from_recs}"
            )


class TestCrossGenDecoderLOO(unittest.TestCase):
    """Tests for _cross_gen_decoder when recording_labels_by_pair is provided (LOO path).

    Setup: 3 recordings × 3 trials each × 3 events = 9 trials per event.
    LOO folds = 3 (one per recording). num_fold=5 is passed but should be ignored.
    """

    def setUp(self):
        self.T, self.n_PCs = 6, 3
        self.events = ["A", "B", "C"]
        self.recs = ["rec_0", "rec_1", "rec_2"]
        self.N = len(self.recs)
        n_total = self.N * 3  # 9 trials per event
        rec_labels = {ev: [r for r in self.recs for _ in range(3)] for ev in self.events}
        flat_data = {
            "A": [np.ones((self.T, self.n_PCs)) for _ in range(n_total)],
            "B": [np.full((self.T, self.n_PCs), 2.0) for _ in range(n_total)],
            "C": [np.full((self.T, self.n_PCs), 3.0) for _ in range(n_total)],
        }
        self.decoder_data_by_pair = {"A_B": flat_data, "A_C": flat_data, "B_C": flat_data}
        self.recording_labels_by_pair = {"A_B": rec_labels, "A_C": rec_labels, "B_C": rec_labels}

    def _run(self, input="timebin"):
        return decoders._cross_gen_decoder(
            self.decoder_data_by_pair, self.events, num_fold=5,
            classifier_type="RF", input=input,
            recording_labels_by_pair=self.recording_labels_by_pair,
        )

    def test_roc_auc_length_equals_N_not_num_fold(self):
        """LOO folds = N recordings (3), not num_fold (5)."""
        raw = self._run()
        for train_key, test_pairs in raw.items():
            for test_key, timebin_list in test_pairs.items():
                if test_key == "_models":
                    continue
                for t_dict in timebin_list:
                    self.assertEqual(len(t_dict["test_roc_auc"]), self.N,
                                     f"{train_key}→{test_key}: expected {self.N} LOO folds")

    def test_timebin_list_length_equals_T(self):
        raw = self._run()
        for train_key, test_pairs in raw.items():
            for test_key, timebin_list in test_pairs.items():
                if test_key == "_models":
                    continue
                self.assertEqual(len(timebin_list), self.T, f"{train_key}→{test_key}")

    def test_all_roc_auc_between_0_and_1(self):
        raw = self._run()
        for train_key, test_pairs in raw.items():
            for test_key, timebin_list in test_pairs.items():
                if test_key == "_models":
                    continue
                for t_dict in timebin_list:
                    for val in t_dict["test_roc_auc"]:
                        if not np.isnan(val):
                            self.assertGreaterEqual(val, 0.0)
                            self.assertLessEqual(val, 1.0)

    def test_full_trial_returns_one_timebin_per_test_pair(self):
        raw = self._run(input="full_trial")
        for train_key, test_pairs in raw.items():
            for test_key, timebin_list in test_pairs.items():
                if test_key == "_models":
                    continue
                self.assertEqual(len(timebin_list), 1, f"{train_key}→{test_key}")

    def test_full_trial_roc_auc_length_equals_N(self):
        raw = self._run(input="full_trial")
        for train_key, test_pairs in raw.items():
            for test_key, timebin_list in test_pairs.items():
                if test_key == "_models":
                    continue
                self.assertEqual(len(timebin_list[0]["test_roc_auc"]), self.N)

    def test_three_train_keys(self):
        raw = self._run()
        self.assertEqual(set(k for k in raw.keys()), {"A_B", "A_C", "B_C"})


class TestCrossGenResultsLOO(unittest.TestCase):
    """Verifies cross_gen_results built from LOO output has shape (T, N_recordings)."""

    def setUp(self):
        self.T, self.n_PCs = 6, 3
        self.events = ["A", "B", "C"]
        self.N = 3
        n_total = self.N * 3
        rec_labels = {ev: [r for r in ["rec_0", "rec_1", "rec_2"] for _ in range(3)] for ev in self.events}
        flat_data = {
            ev: [np.ones((self.T, self.n_PCs)) * i for i in range(n_total)]
            for ev in self.events
        }
        decoder_data_by_pair = {"A_B": flat_data, "A_C": flat_data, "B_C": flat_data}
        recording_labels_by_pair = {"A_B": rec_labels, "A_C": rec_labels, "B_C": rec_labels}
        raw = decoders._cross_gen_decoder(
            decoder_data_by_pair, self.events, num_fold=5,
            classifier_type="RF",
            recording_labels_by_pair=recording_labels_by_pair,
        )
        self.result = decoders.cross_gen_results(raw, self.N, 1.0, 0, 0)

    def test_num_fold_is_N(self):
        self.assertEqual(self.result.num_fold, self.N)

    def test_roc_auc_shape_is_T_by_N(self):
        for train_key, test_pairs in self.result.roc_auc_scores.items():
            for test_key, nmr in test_pairs.items():
                self.assertEqual(nmr.roc_auc.shape, (self.T, self.N),
                                 f"{train_key}→{test_key}: got {nmr.roc_auc.shape}")

    def test_repr_runs_without_error(self):
        try:
            repr(self.result)
        except Exception as e:
            self.fail(f"__repr__ raised {e}")


class TestTrialPCAHoldoutRecordingLabels(unittest.TestCase):
    """Verifies _trial_PCA_holdout returns recording labels aligned with decoder_data."""

    def setUp(self):
        self.events = ["event_a", "event_b"]
        self.event_length = 1.0
        self.collection = make_test_collection(timebin=50)

    def test_returns_three_tuple(self):
        result = decoders._trial_PCA_holdout(
            self.collection, self.event_length, 0, 0,
            pca_events=self.events, project_events=self.events,
            no_PCs=3, return_recording_labels=True,
        )
        self.assertEqual(len(result), 3)

    def test_recording_labels_keys_match_project_events(self):
        decoder_data, num_pcs, rec_labels = decoders._trial_PCA_holdout(
            self.collection, self.event_length, 0, 0,
            pca_events=self.events, project_events=self.events,
            no_PCs=3, return_recording_labels=True,
        )
        self.assertEqual(set(rec_labels.keys()), set(self.events))

    def test_recording_labels_aligned_with_decoder_data(self):
        """Each event's label list must have the same length as its trial list."""
        decoder_data, num_pcs, rec_labels = decoders._trial_PCA_holdout(
            self.collection, self.event_length, 0, 0,
            pca_events=self.events, project_events=self.events,
            no_PCs=3, return_recording_labels=True,
        )
        for event in self.events:
            self.assertEqual(len(rec_labels[event]), len(decoder_data[event]),
                             f"Mismatch for {event}")

    def test_recording_labels_are_recording_names(self):
        """Labels should be actual recording names from the collection."""
        decoder_data, num_pcs, rec_labels = decoders._trial_PCA_holdout(
            self.collection, self.event_length, 0, 0,
            pca_events=self.events, project_events=self.events,
            no_PCs=3, return_recording_labels=True,
        )
        rec_names = {r.name for r in self.collection.recordings}
        for event in self.events:
            for label in rec_labels[event]:
                self.assertIn(label, rec_names)

    def test_without_flag_returns_two_tuple(self):
        result = decoders._trial_PCA_holdout(
            self.collection, self.event_length, 0, 0,
            pca_events=self.events, project_events=self.events,
            no_PCs=3, return_recording_labels=False,
        )
        self.assertEqual(len(result), 2)


# ---------------------------------------------------------------------------
# 2x2 LOO holdout tests
# ---------------------------------------------------------------------------

class TestCrossGenDecoder2x2LOOHoldout(unittest.TestCase):
    """Verify the LOO invariant for _cross_gen_decoder_2x2: held-out recording
    absent from X_train, and each X_test contains data from exactly one recording.

    4 events (a, b, c, d), 3 recordings each with a unique constant fill value (10, 20, 30).
    input='full_trial' gives one __fit_clf_single__ call per fold per direction.
    """

    def setUp(self):
        self.T, self.n_PCs = 2, 2
        self.a, self.b, self.c, self.d = "a", "b", "c", "d"
        self.all_events = [self.a, self.b, self.c, self.d]
        self.recs = ["rec_0", "rec_1", "rec_2"]
        self.rec_values = [10.0, 20.0, 30.0]
        n_per_rec = 3

        flat_data = {ev: [] for ev in self.all_events}
        rec_labels = {ev: [] for ev in self.all_events}
        for ev in self.all_events:
            for rec, val in zip(self.recs, self.rec_values):
                for _ in range(n_per_rec):
                    flat_data[ev].append(np.full((self.T, self.n_PCs), val))
                    rec_labels[ev].append(rec)

        self.train_test_pairs = [
            ((self.a, self.b), (self.c, self.d)),
            ((self.c, self.d), (self.a, self.b)),
            ((self.a, self.c), (self.b, self.d)),
            ((self.b, self.d), (self.a, self.c)),
        ]
        train_keys = [f"{e1}_{e2}" for (e1, e2), _ in self.train_test_pairs]
        self.decoder_data_by_pair = {k: flat_data for k in train_keys}
        self.recording_labels_by_pair = {k: rec_labels for k in train_keys}

    def _run_capturing_fit(self):
        captured = []
        original = decoders.__fit_clf_single__

        def capturing_fit(X_train, y_train, classifier_type, **kwargs):
            captured.append(X_train.copy())
            return original(X_train, y_train, classifier_type, **kwargs)

        with patch("spike.spike_analysis.decoders.__fit_clf_single__", side_effect=capturing_fit):
            decoders._cross_gen_decoder_2x2(
                self.decoder_data_by_pair, self.train_test_pairs, num_fold=5,
                classifier_type="linear", input="full_trial",
                recording_labels_by_pair=self.recording_labels_by_pair,
            )
        return captured

    def _run_capturing_score(self):
        captured = []
        original = decoders.__score_test__

        def capturing_score(clf, X_test, y_test, classifier_type):
            captured.append(X_test.copy())
            return original(clf, X_test, y_test, classifier_type)

        with patch("spike.spike_analysis.decoders.__score_test__", side_effect=capturing_score):
            decoders._cross_gen_decoder_2x2(
                self.decoder_data_by_pair, self.train_test_pairs, num_fold=5,
                classifier_type="linear", input="full_trial",
                recording_labels_by_pair=self.recording_labels_by_pair,
            )
        return captured

    def test_held_out_recording_absent_from_X_train(self):
        """For each LOO fold, the held-out recording's value must not appear in X_train.
        In full_trial mode: 4 directions × 3 folds × 2 fits (clf + clf_shuf) = 24 captures.
        """
        captured = self._run_capturing_fit()

        n_directions = len(self.train_test_pairs)
        n_folds = len(self.recs)
        for dir_idx in range(n_directions):
            for fold_idx, (rec, val) in enumerate(zip(self.recs, self.rec_values)):
                base = dir_idx * n_folds * 2 + fold_idx * 2
                X_train = captured[base]
                self.assertFalse(
                    np.any(X_train == val),
                    f"direction {dir_idx}, held-out {rec}: value {val} found in X_train"
                )
                for other_val in self.rec_values:
                    if other_val != val:
                        self.assertTrue(
                            np.any(X_train == other_val),
                            f"direction {dir_idx}, held-out {rec}: expected value {other_val} absent from X_train"
                        )

    def test_X_test_contains_only_held_out_recording(self):
        """Each X_test passed to __score_test__ should contain values from exactly one recording."""
        captured = self._run_capturing_score()
        rec_value_set = set(self.rec_values)

        for i, X_test in enumerate(captured):
            unique_vals = set(np.unique(X_test))
            vals_from_recs = unique_vals & rec_value_set
            self.assertEqual(
                len(vals_from_recs), 1,
                f"X_test #{i} contains values from {len(vals_from_recs)} recordings: {vals_from_recs}"
            )


if __name__ == "__main__":
    unittest.main()
