# TODO: get_indices
#   - test that indices are correct for a list with repeated items
#   - test that the last group is captured correctly
#
# TODO: condition_pca
#   - test that condition_pca produces one transformed_data entry per condition
#   - test that recordings not in condition_dict are excluded from that condition's transform
#   - test that condition_pca and avg_trajectory_matrix produce the same row count
#
# TODO: PCAResult object formatting
#   - test that __str__ runs without error and contains key summary fields
#   - test that recording_overview has one row per recording
#   - test that cumulative_variance is monotonically increasing and ends at 1.0
#   - test that explained_variance sums to 1.0
#   - test the "more features than samples" warning path sets transformed_data to None

import unittest
import numpy as np
from pathlib import Path
from spike.spike_analysis.spike_collection import SpikeCollection
import spike.spike_analysis.pca_trajectories as pca_traj


TEST_DATA_PATH = str(Path(__file__).parent / "test_data")

EXPECTED_RECORDINGS = {
    "test_recording_merged.rec",
    "test_rec2_merged.rec",
    "test_rec_fewgoodunits_merged.rec",
}


def make_test_collection(timebin=50, ignore_freq=0.1):
    """Helper: build a fully analyzed test collection with two events."""
    collection = SpikeCollection(TEST_DATA_PATH)
    events_a = np.array([[5000, 6000], [10000, 11000]])   # 2 events, 1s each, at t=5s & t=10s
    events_b = np.array([[20000, 21000], [30000, 31000]])  # 2 events, 1s each, at t=20s & t=30s
    for recording in collection.recordings:
        recording.subject = recording.name
        recording.event_dict = {"event_a": events_a, "event_b": events_b}
    collection.analyze(timebin=timebin, ignore_freq=ignore_freq)
    return collection


class TestAvgTrajectoryMatrix(unittest.TestCase):
    """Tests for avg_trajectory_matrix (average mode PCA)."""

    def setUp(self):
        self.events = ["event_a", "event_b"]
        self.event_length = 1.0   # seconds
        self.pre_window = 0
        self.post_window = 0
        self.timebin = 50          # ms
        self.collection = make_test_collection(timebin=self.timebin)
        self.pca_result = pca_traj.avg_trajectory_matrix(
            self.collection,
            self.event_length,
            pre_window=self.pre_window,
            post_window=self.post_window,
            events=self.events,
        )

    # ---------- dimension tests ----------

    def test_matrix_row_dimension(self):
        """
        Rows = len(events) * num_points
        num_points = int((event_length + pre_window + post_window) * 1000 / timebin)
        The *1000 converts event lengths from seconds to ms so the units match timebin.
        """
        num_points = int(
            (self.event_length + self.pre_window + self.post_window) * 1000 / self.timebin
        )
        expected_rows = len(self.events) * num_points
        self.assertEqual(self.pca_result.matrix_df.shape[0], expected_rows)

    def test_matrix_row_dimension_with_windows(self):
        """Row count grows correctly when pre/post windows are added."""
        pre, post = 0.5, 0.5
        result = pca_traj.avg_trajectory_matrix(
            self.collection,
            self.event_length,
            pre_window=pre,
            post_window=post,
            events=self.events,
        )
        num_points = int((self.event_length + pre + post) * 1000 / self.timebin)
        expected_rows = len(self.events) * num_points
        self.assertEqual(result.matrix_df.shape[0], expected_rows)

    # ---------- key tests ----------

    def test_recording_keys(self):
        """
        Each recording name should appear exactly analyzed_neurons times in the columns,
        in the same order as recordings in the collection.
        """
        expected = []
        for recording in self.collection.recordings:
            expected.extend([recording.name] * recording.analyzed_neurons)
        self.assertEqual(self.pca_result.matrix_df.columns.to_list(), expected)

    def test_event_keys(self):
        """
        Index should be each event label repeated num_points times, in event order.
        """
        num_points = int(
            (self.event_length + self.pre_window + self.post_window) * 1000 / self.timebin
        )
        expected = [event for event in self.events for _ in range(num_points)]
        self.assertEqual(self.pca_result.matrix_df.index.to_list(), expected)

    def test_pca_result_is_not_none(self):
        self.assertIsNotNone(self.pca_result)

    def test_pca_result_has_coefficients(self):
        """PCA coefficients should exist when samples >= features."""
        self.assertIsNotNone(self.pca_result.coefficients)


class TestAvgTrajHelper(unittest.TestCase):
    """Unit tests for avg_traj — no collection needed."""

    def setUp(self):
        self.n_trials = 5
        self.num_points = 20
        self.n_neurons = 10
        self.events = ["event_a", "event_b"]
        # shape: (n_trials, num_points, n_neurons)
        self.firing_rates = np.random.rand(self.n_trials, self.num_points, self.n_neurons)

    def test_event_averages_shape(self):
        """Averaging across trials (axis 0) should give (num_points, n_neurons)."""
        event_averages, _ = pca_traj.avg_traj(self.firing_rates, self.num_points, self.events)
        self.assertEqual(event_averages.shape, (self.num_points, self.n_neurons))

    def test_event_keys_length(self):
        """event_keys should have one label per timebin per event."""
        _, event_keys = pca_traj.avg_traj(self.firing_rates, self.num_points, self.events)
        self.assertEqual(len(event_keys), len(self.events) * self.num_points)

    def test_event_keys_order(self):
        """event_keys should repeat each event label num_points times, in event order."""
        _, event_keys = pca_traj.avg_traj(self.firing_rates, self.num_points, self.events)
        expected = [event for event in self.events for _ in range(self.num_points)]
        self.assertEqual(event_keys, expected)


class TestTrialTrajHelper(unittest.TestCase):
    """Unit tests for trial_traj — no collection needed."""

    def setUp(self):
        self.n_trials = 5
        self.num_points = 20
        self.n_neurons = 10
        self.min_event = 3  # use only the first 3 trials
        # shape: (n_trials, num_points, n_neurons)
        self.firing_rates = np.random.rand(self.n_trials, self.num_points, self.n_neurons)

    def test_concatenated_shape(self):
        """Trials are flattened into rows: shape should be (min_event * num_points, n_neurons)."""
        event_firing_rates_conc, _ = pca_traj.trial_traj(
            self.firing_rates, self.num_points, self.min_event
        )
        self.assertEqual(
            event_firing_rates_conc.shape, (self.min_event * self.num_points, self.n_neurons)
        )

    def test_num_data_ps(self):
        """num_data_ps should equal min_event * num_points."""
        _, num_data_ps = pca_traj.trial_traj(
            self.firing_rates, self.num_points, self.min_event
        )
        self.assertEqual(num_data_ps, self.min_event * self.num_points)

    def test_only_min_event_trials_used(self):
        """trial_traj should only use the first min_event trials."""
        event_firing_rates_conc, _ = pca_traj.trial_traj(
            self.firing_rates, self.num_points, self.min_event
        )
        expected = self.firing_rates[: self.min_event].reshape(
            self.min_event * self.num_points, self.n_neurons
        )
        np.testing.assert_array_equal(event_firing_rates_conc, expected)


class MockRecording:
    """Minimal stand-in for SpikeRecording — needs analyzed_neurons, name, event_dict."""
    def __init__(self, name, analyzed_neurons, event_dict):
        self.name = name
        self.analyzed_neurons = analyzed_neurons
        self.event_dict = event_dict


class TestCheckRecording(unittest.TestCase):

    def test_passes_when_analyzed_neurons_meet_threshold(self):
        rec = MockRecording("rec", analyzed_neurons=5, event_dict={
            "event_a": np.array([[1000, 2000], [3000, 4000]])
        })
        self.assertTrue(pca_traj.check_recording(rec, min_neurons=5, events=["event_a"], to_print=False))

    def test_fails_when_analyzed_neurons_below_threshold(self):
        """analyzed_neurons (post ignore_freq) is what enters PCA — that is what min_neurons gates."""
        rec = MockRecording("rec", analyzed_neurons=2, event_dict={
            "event_a": np.array([[1000, 2000], [3000, 4000]])
        })
        self.assertFalse(pca_traj.check_recording(rec, min_neurons=5, events=["event_a"], to_print=False))

    def test_fails_when_single_event_has_zero_duration(self):
        """A placeholder row of [[start, start]] signals no real events — should be excluded."""
        rec = MockRecording("rec", analyzed_neurons=5, event_dict={
            "event_a": np.array([[500, 500]])   # start == stop → zero duration
        })
        self.assertFalse(pca_traj.check_recording(rec, min_neurons=0, events=["event_a"], to_print=False))

    def test_fails_when_event_is_empty(self):
        """An empty event array should return False without raising an error."""
        rec = MockRecording("rec", analyzed_neurons=5, event_dict={
            "event_a": np.array([])
        })
        self.assertFalse(pca_traj.check_recording(rec, min_neurons=0, events=["event_a"], to_print=False))

    def test_passes_when_single_event_has_nonzero_duration(self):
        """One real event with non-zero duration should pass."""
        rec = MockRecording("rec", analyzed_neurons=5, event_dict={
            "event_a": np.array([[1000, 2000]])
        })
        self.assertTrue(pca_traj.check_recording(rec, min_neurons=0, events=["event_a"], to_print=False))

    def test_min_neurons_filters_recordings_in_pca(self):
        """Recordings with too few analyzed_neurons should be excluded from the PCA matrix."""
        collection = make_test_collection()
        few_units_rec = next(
            r for r in collection.recordings if r.name == "test_rec_fewgoodunits_merged.rec"
        )
        # set threshold just above that recording's analyzed_neurons
        threshold = few_units_rec.analyzed_neurons + 1

        result_no_filter = pca_traj.avg_trajectory_matrix(
            collection, event_length=1.0, pre_window=0, post_window=0,
            events=["event_a", "event_b"], min_neurons=0,
        )
        result_filtered = pca_traj.avg_trajectory_matrix(
            collection, event_length=1.0, pre_window=0, post_window=0,
            events=["event_a", "event_b"], min_neurons=threshold,
        )
        self.assertLess(result_filtered.matrix_df.shape[1], result_no_filter.matrix_df.shape[1])
        self.assertNotIn("test_rec_fewgoodunits_merged.rec", result_filtered.matrix_df.columns)


class TestEventNumbers(unittest.TestCase):
    """Tests for event_numbers — verifies it returns the per-event trial minimum."""

    def setUp(self):
        self.events = ["event_a", "event_b"]
        self.collection = make_test_collection()
        # 2 recordings get 5 event_a bouts, 1 recording gets only 3 — min across recordings = 3
        events_a_many = np.array([[5000, 6000], [10000, 11000], [15000, 16000],
                                  [20000, 21000], [25000, 26000]])
        events_a_few  = np.array([[5000, 6000], [10000, 11000], [15000, 16000]])
        events_b = np.array([[30000, 31000], [35000, 36000], [40000, 41000],
                             [45000, 46000], [50000, 51000]])
        for i, recording in enumerate(self.collection.recordings):
            recording.event_dict["event_a"] = events_a_few if i == 0 else events_a_many
            recording.event_dict["event_b"] = events_b

    def test_event_numbers_returns_minimum_across_recordings(self):
        """event_numbers should return the minimum trial count across all recordings per event."""
        result = pca_traj.event_numbers(self.collection, self.events, min_neurons=0)
        # one recording has 3 event_a bouts, two have 5 — minimum is 3
        self.assertEqual(result["event_a"], 3)
        # all recordings have 5 event_b bouts
        self.assertEqual(result["event_b"], 5)


class TestTrialTrajectoryMatrix(unittest.TestCase):
    """Integration tests for trial_trajectory_matrix vs avg_trajectory_matrix."""

    def setUp(self):
        self.events = ["event_a", "event_b"]
        self.event_length = 1.0
        self.pre_window = 0
        self.post_window = 0
        self.timebin = 50
        self.collection = make_test_collection(timebin=self.timebin)
        # each event has 2 trials per recording, so min_event = 2
        self.min_event = 2

    def test_avg_mode_matrix_rows(self):
        """avg_trajectory_matrix rows = len(events) * num_points (trials averaged away)."""
        result = pca_traj.avg_trajectory_matrix(
            self.collection, self.event_length,
            pre_window=self.pre_window, post_window=self.post_window,
            events=self.events,
        )
        num_points = int((self.event_length + self.pre_window + self.post_window) * 1000 / self.timebin)
        self.assertEqual(result.matrix_df.shape[0], len(self.events) * num_points)

    def test_trial_mode_matrix_rows(self):
        """trial_trajectory_matrix rows = len(events) * min_event * num_points (trials kept)."""
        result = pca_traj.trial_trajectory_matrix(
            self.collection, self.event_length,
            pre_window=self.pre_window, post_window=self.post_window,
            events=self.events,
        )
        num_points = int((self.event_length + self.pre_window + self.post_window) * 1000 / self.timebin)
        self.assertEqual(result.matrix_df.shape[0], len(self.events) * self.min_event * num_points)

    def test_trial_mode_has_more_rows_than_avg_mode(self):
        """trial mode should have more rows than avg mode when there are multiple trials."""
        avg_result = pca_traj.avg_trajectory_matrix(
            self.collection, self.event_length,
            pre_window=self.pre_window, post_window=self.post_window,
            events=self.events,
        )
        trial_result = pca_traj.trial_trajectory_matrix(
            self.collection, self.event_length,
            pre_window=self.pre_window, post_window=self.post_window,
            events=self.events,
        )
        self.assertGreater(trial_result.matrix_df.shape[0], avg_result.matrix_df.shape[0])


if __name__ == "__main__":
    unittest.main()
