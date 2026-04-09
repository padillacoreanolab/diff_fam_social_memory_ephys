import io
import math
import unittest
import numpy as np
from bidict import bidict
from unittest.mock import patch
from lfp.lfp_analysis.LFP_recording import LFPRecording
import lfp.lfp_analysis.event_extraction as ee


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_fake_recording(n_timebins=40, n_freqs=10, n_regions=3, n_events=5, seed=0):
    """
    Recording with synthetic LFP arrays and a correctly-timed event dict.

    Timing: timestep=0.5s → 500ms per timebin.
    Events are spaced evenly across the recording, each 1000ms (2 timebins) long.
    A 'baseline' key uses the first 2 events.
    """
    np.random.seed(seed)
    rec = LFPRecording(
        subject="test_subject",
        channel_dict={"mPFC": 5, "vHPC": 1, "BLA": 3},
        merged_rec_path="/fake/path/fake_merged.rec",
        load=True,
    )
    rec.name = "fake_merged.rec"
    rec.subject = "test_subject"
    rec.brain_region_dict = bidict({"vHPC": 0, "BLA": 1, "mPFC": 2})
    rec.timestep = 0.5  # seconds → 500ms per timebin

    rec.power = np.random.rand(n_timebins, n_freqs, n_regions)
    rec.coherence = np.random.rand(n_timebins, n_freqs, n_regions, n_regions)
    rec.granger = np.random.rand(n_timebins, n_freqs, n_regions, n_regions)

    # timestep=0.5s → freq_timebin=500ms, total=n_timebins*500ms
    freq_timebin_ms = rec.timestep * 1000
    total_ms = n_timebins * freq_timebin_ms
    event_duration_ms = freq_timebin_ms * 2  # 2 timebins long
    spacing = total_ms / (n_events + 1)

    events = []
    for i in range(n_events):
        start = spacing * (i + 1)
        stop = start + event_duration_ms
        if stop < total_ms:
            events.append([start, stop])

    rec.event_dict = {
        "test_event": np.array(events),
        "baseline": np.array(events[:2]),
    }
    return rec


def make_fake_collection(n_recordings=3, n_timebins=40, n_freqs=10, n_regions=3, n_events=5):
    """Minimal collection-like object that passes the duck-typing check."""
    class FakeCollection:
        pass

    col = FakeCollection()
    col.recordings = [
        make_fake_recording(n_timebins, n_freqs, n_regions, n_events, seed=i)
        for i in range(n_recordings)
    ]
    col.brain_region_dict = bidict({"vHPC": 0, "BLA": 1, "mPFC": 2})
    return col


# ---------------------------------------------------------------------------
# all_set
# ---------------------------------------------------------------------------

class TestAllSet(unittest.TestCase):
    def _make_collection(self, recordings):
        class FakeCol:
            pass
        col = FakeCol()
        col.recordings = recordings
        return col

    def _make_recording(self, name, subject=None, event_dict=None):
        class FakeRec:
            pass
        rec = FakeRec()
        rec.name = name
        if subject is not None:
            rec.subject = subject
        if event_dict is not None:
            rec.event_dict = event_dict
        return rec

    def test_prints_all_set_when_everything_valid(self):
        recs = [
            self._make_recording("rec0", subject="s0", event_dict={"sniff": [], "explore": []}),
            self._make_recording("rec1", subject="s1", event_dict={"sniff": [], "explore": []}),
        ]
        with patch("sys.stdout", new=io.StringIO()) as mock_out:
            ee.all_set(self._make_collection(recs))
        self.assertIn("All set to analyze", mock_out.getvalue())

    def test_warns_when_recording_missing_event_dict(self):
        recs = [
            self._make_recording("rec0", subject="s0", event_dict={"sniff": []}),
            self._make_recording("rec1", subject="s1"),  # no event_dict
        ]
        with patch("sys.stdout", new=io.StringIO()) as mock_out:
            ee.all_set(self._make_collection(recs))
        self.assertIn("missing event dictionaries", mock_out.getvalue())

    def test_warns_when_event_dict_keys_differ(self):
        recs = [
            self._make_recording("rec0", subject="s0", event_dict={"sniff": []}),
            self._make_recording("rec1", subject="s1", event_dict={"explore": []}),
        ]
        with patch("sys.stdout", new=io.StringIO()) as mock_out:
            ee.all_set(self._make_collection(recs))
        self.assertIn("different", mock_out.getvalue())

    def test_warns_when_subject_missing(self):
        recs = [
            self._make_recording("rec0", event_dict={"sniff": []}),  # no subject
        ]
        with patch("sys.stdout", new=io.StringIO()) as mock_out:
            ee.all_set(self._make_collection(recs))
        self.assertIn("missing subjects", mock_out.getvalue())

    def test_does_not_print_all_set_when_issues_exist(self):
        recs = [self._make_recording("rec0")]  # no subject, no event_dict
        with patch("sys.stdout", new=io.StringIO()) as mock_out:
            ee.all_set(self._make_collection(recs))
        self.assertNotIn("All set to analyze", mock_out.getvalue())


# ---------------------------------------------------------------------------
# get_events
# ---------------------------------------------------------------------------

class TestGetEvents(unittest.TestCase):
    def setUp(self):
        self.rec = make_fake_recording(n_timebins=40, n_freqs=10, n_regions=3, n_events=5)

    def test_returns_list(self):
        result = ee.get_events(self.rec, "test_event", "power", None, 0, 0)
        self.assertIsInstance(result, list)

    def test_returns_non_empty_list(self):
        result = ee.get_events(self.rec, "test_event", "power", None, 0, 0)
        self.assertGreater(len(result), 0)

    def test_power_averaged_shape(self):
        result = ee.get_events(self.rec, "test_event", "power", None, 0, 0, average=True)
        for snippet in result:
            self.assertEqual(snippet.shape, (10, 3))  # [n_freqs, n_regions]

    def test_power_not_averaged_has_time_dimension(self):
        result = ee.get_events(self.rec, "test_event", "power", None, 0, 0, average=False)
        for snippet in result:
            self.assertEqual(snippet.ndim, 3)           # [time, n_freqs, n_regions]
            self.assertEqual(snippet.shape[1], 10)
            self.assertEqual(snippet.shape[2], 3)

    def test_coherence_averaged_shape(self):
        result = ee.get_events(self.rec, "test_event", "coherence", None, 0, 0, average=True)
        for snippet in result:
            self.assertEqual(snippet.shape, (10, 3, 3))

    def test_granger_averaged_shape(self):
        result = ee.get_events(self.rec, "test_event", "granger", None, 0, 0, average=True)
        for snippet in result:
            self.assertEqual(snippet.shape, (10, 3, 3))

    def test_events_beyond_recording_are_excluded(self):
        # Add one event that runs past the end of the recording
        freq_timebin_ms = self.rec.timestep * 1000   # 500ms
        n_timebins = self.rec.power.shape[0]          # 40
        total_ms = n_timebins * freq_timebin_ms       # 20000ms
        valid_event = [2000, 3000]
        out_of_range_event = [total_ms - 100, total_ms + 500]  # post_event will exceed n_timebins
        self.rec.event_dict["oob_test"] = np.array([valid_event, out_of_range_event])
        result = ee.get_events(self.rec, "oob_test", "power", None, 0, 0)
        self.assertEqual(len(result), 1)

    def test_pre_window_extends_snippet(self):
        no_window = ee.get_events(self.rec, "test_event", "power", None, 0, 0, average=False)
        with_pre = ee.get_events(self.rec, "test_event", "power", None, 0.5, 0, average=False)
        # pre_window=0.5s=500ms=1 extra timebin at start
        self.assertGreater(with_pre[0].shape[0], no_window[0].shape[0])

    def test_post_window_extends_snippet(self):
        no_window = ee.get_events(self.rec, "test_event", "power", None, 0, 0, average=False)
        with_post = ee.get_events(self.rec, "test_event", "power", None, 0, 0.5, average=False)
        self.assertGreater(with_post[0].shape[0], no_window[0].shape[0])

    def test_event_len_fixes_snippet_length(self):
        # With event_len, all snippets should end at the same number of timebins
        result = ee.get_events(self.rec, "test_event", "power", 1.0, 0, 0, average=False)
        lengths = [s.shape[0] for s in result]
        self.assertEqual(len(set(lengths)), 1)  # all the same length

    def test_count_matches_valid_events(self):
        # All 5 events in the fixture are within range
        result = ee.get_events(self.rec, "test_event", "power", None, 0, 0)
        self.assertEqual(len(result), len(self.rec.event_dict["test_event"]))


# ---------------------------------------------------------------------------
# average_events
# ---------------------------------------------------------------------------

class TestAverageEvents(unittest.TestCase):
    def setUp(self):
        self.collection = make_fake_collection(n_recordings=3, n_freqs=10, n_regions=3)

    def test_returns_dict_with_event_as_key(self):
        result = ee.average_events(self.collection, ["test_event"], "power")
        self.assertIn("test_event", result)

    def test_value_is_list_of_length_n_recordings(self):
        result = ee.average_events(self.collection, ["test_event"], "power")
        self.assertEqual(len(result["test_event"]), 3)

    def test_per_recording_average_shape_power(self):
        result = ee.average_events(self.collection, ["test_event"], "power")
        for rec_avg in result["test_event"]:
            self.assertEqual(rec_avg.shape, (10, 3))  # [n_freqs, n_regions]

    def test_per_recording_average_shape_granger(self):
        result = ee.average_events(self.collection, ["test_event"], "granger")
        for rec_avg in result["test_event"]:
            self.assertEqual(rec_avg.shape, (10, 3, 3))

    def test_multiple_events_all_in_dict(self):
        result = ee.average_events(self.collection, ["test_event", "baseline"], "power")
        self.assertIn("test_event", result)
        self.assertIn("baseline", result)

    def test_single_recording_input(self):
        rec = make_fake_recording()
        result = ee.average_events(rec, ["test_event"], "power")
        self.assertIn("test_event", result)
        self.assertEqual(len(result["test_event"]), 1)

    def test_baseline_changes_values(self):
        raw = ee.average_events(self.collection, ["test_event"], "power")
        baselined = ee.average_events(
            self.collection, ["test_event"], "power", baseline="baseline"
        )
        # Baseline correction should change the values
        self.assertFalse(
            np.allclose(raw["test_event"][0], baselined["test_event"][0])
        )

    def test_single_baseline_applies_to_all_events(self):
        # One baseline for multiple events — should not raise
        result = ee.average_events(
            self.collection, ["test_event", "baseline"], "power", baseline="baseline"
        )
        self.assertIn("test_event", result)
        self.assertIn("baseline", result)


# ---------------------------------------------------------------------------
# baselined_events
# ---------------------------------------------------------------------------

class TestBaslinedEvents(unittest.TestCase):
    def setUp(self):
        self.collection = make_fake_collection(n_recordings=3, n_freqs=10, n_regions=3, n_events=5)

    def test_returns_dict_with_event_as_key(self):
        result = ee.baselined_events(self.collection, ["test_event"], "power")
        self.assertIn("test_event", result)

    def test_value_is_list_of_length_n_recordings(self):
        result = ee.baselined_events(self.collection, ["test_event"], "power")
        self.assertEqual(len(result["test_event"]), 3)

    def test_per_recording_entry_is_array(self):
        result = ee.baselined_events(self.collection, ["test_event"], "power")
        for arr in result["test_event"]:
            self.assertIsInstance(arr, np.ndarray)

    def test_per_recording_shape_power(self):
        result = ee.baselined_events(self.collection, ["test_event"], "power")
        for arr in result["test_event"]:
            self.assertEqual(arr.ndim, 3)         # [n_trials, n_freqs, n_regions]
            self.assertEqual(arr.shape[1], 10)
            self.assertEqual(arr.shape[2], 3)

    def test_per_recording_shape_granger(self):
        result = ee.baselined_events(self.collection, ["test_event"], "granger")
        for arr in result["test_event"]:
            self.assertEqual(arr.ndim, 4)         # [n_trials, n_freqs, n_regions, n_regions]
            self.assertEqual(arr.shape[1], 10)
            self.assertEqual(arr.shape[2], 3)
            self.assertEqual(arr.shape[3], 3)

    def test_different_trial_counts_per_recording_do_not_crash(self):
        # Give recordings different numbers of events
        col = make_fake_collection(n_recordings=2, n_freqs=10, n_regions=3)
        col.recordings[0].event_dict["test_event"] = col.recordings[0].event_dict["test_event"][:3]
        col.recordings[1].event_dict["test_event"] = col.recordings[1].event_dict["test_event"][:5]
        result = ee.baselined_events(col, ["test_event"], "power")
        self.assertEqual(result["test_event"][0].shape[0], 3)
        self.assertEqual(result["test_event"][1].shape[0], 5)

    def test_baseline_correction_changes_values(self):
        raw = ee.baselined_events(self.collection, ["test_event"], "power")
        baselined = ee.baselined_events(
            self.collection, ["test_event"], "power", baseline="baseline"
        )
        self.assertFalse(np.allclose(raw["test_event"][0], baselined["test_event"][0]))

    def test_single_recording_input(self):
        rec = make_fake_recording()
        result = ee.baselined_events(rec, ["test_event"], "power")
        self.assertIn("test_event", result)
        self.assertEqual(len(result["test_event"]), 1)


# ---------------------------------------------------------------------------
# event_difference
# ---------------------------------------------------------------------------

class TestEventDifference(unittest.TestCase):
    def setUp(self):
        self.collection = make_fake_collection(n_recordings=3, n_freqs=10, n_regions=3)

    def test_returns_dict_with_correct_key(self):
        result = ee.event_difference(self.collection, "test_event", "baseline", "power")
        self.assertIn("test_event vs baseline", result)

    def test_output_shape_power(self):
        result = ee.event_difference(self.collection, "test_event", "baseline", "power")
        arr = result["test_event vs baseline"]
        self.assertEqual(arr.shape, (3, 10, 3))  # [n_recordings, n_freqs, n_regions]

    def test_output_shape_granger(self):
        result = ee.event_difference(self.collection, "test_event", "baseline", "granger")
        arr = result["test_event vs baseline"]
        self.assertEqual(arr.shape, (3, 10, 3, 3))

    def test_output_shape_uses_actual_n_freqs_not_hardcoded(self):
        # Use a collection with n_freqs=7 — would break if 500 were still hardcoded
        col = make_fake_collection(n_recordings=2, n_freqs=7, n_regions=3)
        result = ee.event_difference(col, "test_event", "baseline", "power")
        arr = result["test_event vs baseline"]
        self.assertEqual(arr.shape[1], 7)

    def test_identical_events_give_zero_difference(self):
        # If both events are the same, (e1-e2)/(e1+e2) should be 0 everywhere
        result = ee.event_difference(self.collection, "test_event", "test_event", "power")
        arr = result["test_event vs test_event"]
        # Where denominator is non-zero, difference should be 0
        nonzero_mask = ~np.isnan(arr) & (arr != 0)
        # All non-nan values should be 0 (or very close, due to float arithmetic)
        np.testing.assert_array_almost_equal(arr[~np.isnan(arr)], 0.0)

    def test_with_baselines(self):
        # Should not raise with baselines provided
        result = ee.event_difference(
            self.collection, "test_event", "baseline", "power",
            baseline1="baseline", baseline2="baseline"
        )
        self.assertIn("test_event vs baseline", result)


# ---------------------------------------------------------------------------
# Indexing correctness
# ---------------------------------------------------------------------------

class TestGetEventsIndexing(unittest.TestCase):
    """Verify that snippets contain values from the correct timepoints."""

    def _make_sentinel_recording(self, event_start_ms, event_stop_ms, sentinel=77.0):
        """
        Recording where the timepoints that the event maps to are filled with
        a sentinel value and everything else is 0. Makes it easy to verify
        the right rows were extracted.

        With timestep=0.5s → freq_timebin=500ms:
            pre_event  = ceil(event_start_ms / 500)
            post_event = ceil(event_stop_ms  / 500)
        """
        rec = make_fake_recording(n_timebins=40, n_freqs=10, n_regions=3, n_events=1)
        rec.power[:] = 0.0
        freq_timebin = rec.timestep * 1000  # 500ms
        pre_event  = math.ceil(event_start_ms / freq_timebin)
        post_event = math.ceil(event_stop_ms  / freq_timebin)
        rec.power[pre_event:post_event, :, :] = sentinel
        rec.event_dict["sentinel_event"] = np.array([[event_start_ms, event_stop_ms]])
        return rec, sentinel

    def test_snippet_values_match_correct_timepoints(self):
        rec, sentinel = self._make_sentinel_recording(2000, 3000)
        result = ee.get_events(rec, "sentinel_event", "power", None, 0, 0, average=True)
        self.assertEqual(len(result), 1)
        np.testing.assert_array_almost_equal(result[0], sentinel)

    def test_pre_window_extends_by_correct_number_of_timebins(self):
        rec, _ = self._make_sentinel_recording(2000, 3000)
        no_window  = ee.get_events(rec, "sentinel_event", "power", None, 0,   0,   average=False)
        with_pre   = ee.get_events(rec, "sentinel_event", "power", None, 1.0, 0,   average=False)
        # pre_window=1.0s=1000ms → 1000/500 = 2 extra timebins at the start
        expected_extra = math.ceil(1.0 * 1000 / (rec.timestep * 1000))
        self.assertEqual(with_pre[0].shape[0] - no_window[0].shape[0], expected_extra)

    def test_post_window_extends_by_correct_number_of_timebins(self):
        rec, _ = self._make_sentinel_recording(2000, 3000)
        no_window  = ee.get_events(rec, "sentinel_event", "power", None, 0,   0,   average=False)
        with_post  = ee.get_events(rec, "sentinel_event", "power", None, 0,   0.5, average=False)
        # post_window=0.5s=500ms → 500/500 = 1 extra timebin at the end
        expected_extra = math.ceil(0.5 * 1000 / (rec.timestep * 1000))
        self.assertEqual(with_post[0].shape[0] - no_window[0].shape[0], expected_extra)

    def test_event_len_snippet_has_correct_length(self):
        rec, _ = self._make_sentinel_recording(2000, 3000)
        freq_timebin = rec.timestep * 1000  # 500ms
        event_len = 2.0  # seconds → 2000ms
        # post_event = ceil((start + event_len_ms) / timebin) = ceil((2000+2000)/500) = 8
        # pre_event  = ceil(2000/500) = 4  →  length = 8-4 = 4 timebins
        expected_len = math.ceil((2000 + event_len * 1000) / freq_timebin) - math.ceil(2000 / freq_timebin)
        result = ee.get_events(rec, "sentinel_event", "power", event_len, 0, 0, average=False)
        self.assertEqual(result[0].shape[0], expected_len)

    def test_different_events_index_different_timepoints(self):
        rec = make_fake_recording(n_timebins=40, n_freqs=10, n_regions=3)
        rec.power[:] = 0.0
        # event A at timebins 4:6 → value 1.0
        # event B at timebins 14:16 → value 2.0
        rec.power[4:6, :, :] = 1.0
        rec.power[14:16, :, :] = 2.0
        rec.event_dict["event_a"] = np.array([[2000, 3000]])   # → timebins 4:6
        rec.event_dict["event_b"] = np.array([[7000, 8000]])   # → timebins 14:16
        result_a = ee.get_events(rec, "event_a", "power", None, 0, 0, average=True)
        result_b = ee.get_events(rec, "event_b", "power", None, 0, 0, average=True)
        np.testing.assert_array_almost_equal(result_a[0], 1.0)
        np.testing.assert_array_almost_equal(result_b[0], 2.0)


# ---------------------------------------------------------------------------
# Baseline math
# ---------------------------------------------------------------------------

class TestBaselineMath(unittest.TestCase):
    """
    Verify the baseline correction formula:
        ((event - baseline) / (baseline + 0.00001)) * 100
    """

    def _make_known_recording(self, event_val, baseline_val):
        """
        Recording where event timebins have a constant value of event_val
        and baseline timebins have a constant value of baseline_val.

        Event    → [2000, 3000] ms → timebins 4:6
        Baseline → [5000, 6000] ms → timebins 10:12
        """
        rec = make_fake_recording(n_timebins=40, n_freqs=10, n_regions=3, n_events=1)
        rec.power[:] = 0.0
        rec.power[4:6, :, :]   = event_val
        rec.power[10:12, :, :] = baseline_val
        rec.event_dict["event"]    = np.array([[2000, 3000]])
        rec.event_dict["baseline"] = np.array([[5000, 6000]])
        return rec

    def test_baseline_formula_known_values(self):
        event_val, baseline_val = 10.0, 5.0
        rec = self._make_known_recording(event_val, baseline_val)
        result = ee.average_events(rec, ["event"], "power", baseline="baseline")
        expected = ((event_val - baseline_val) / (baseline_val + 0.00001)) * 100
        np.testing.assert_array_almost_equal(result["event"][0], expected, decimal=3)

    def test_baseline_equal_to_event_gives_zero(self):
        rec = self._make_known_recording(event_val=7.0, baseline_val=7.0)
        result = ee.average_events(rec, ["event"], "power", baseline="baseline")
        np.testing.assert_array_almost_equal(result["event"][0], 0.0, decimal=3)

    def test_event_double_baseline_gives_approx_100(self):
        # event=10, baseline=5 → ((10-5)/5)*100 = 100
        rec = self._make_known_recording(event_val=10.0, baseline_val=5.0)
        result = ee.average_events(rec, ["event"], "power", baseline="baseline")
        np.testing.assert_array_almost_equal(result["event"][0], 100.0, decimal=2)

    def test_event_difference_formula_known_values(self):
        # event1=8, event2=4 → (8-4)/(8+4)*100 = 4/12*100 ≈ 33.33
        # event   → [2000, 3000] ms → timebins 4:6
        # baseline → [5000, 6000] ms → timebins 10:12
        col = make_fake_collection(n_recordings=2, n_freqs=10, n_regions=3)
        for rec in col.recordings:
            rec.power[:] = 0.0
            rec.power[4:6, :, :]   = 8.0
            rec.power[10:12, :, :] = 4.0
            rec.event_dict["test_event"] = np.array([[2000, 3000]])
            rec.event_dict["baseline"]   = np.array([[5000, 6000]])
        result = ee.event_difference(col, "test_event", "baseline", "power")
        expected = (8.0 - 4.0) / (8.0 + 4.0) * 100
        arr = result["test_event vs baseline"]
        np.testing.assert_array_almost_equal(arr, expected, decimal=2)


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------

class TestNaNHandling(unittest.TestCase):
    """
    Verify NaN propagates correctly and is only created where expected.
    These tests are designed to help diagnose unexpected NaN in real data.
    """

    def _make_coherence_recording(self, n_regions=5):
        rec = make_fake_recording(n_timebins=40, n_freqs=10, n_regions=n_regions)
        # Replace with fully controlled data (no accidental NaN)
        rec.coherence[:] = np.random.rand(*rec.coherence.shape)
        return rec

    def test_no_nan_in_source_means_no_nan_in_output(self):
        rec = self._make_coherence_recording()
        # No NaN anywhere in coherence
        result = ee.get_events(rec, "test_event", "coherence", None, 0, 0, average=True)
        for snippet in result:
            self.assertFalse(np.any(np.isnan(snippet)))

    def test_diagonal_only_nan_stays_diagonal_only(self):
        # Spectral connectivity always returns NaN on the diagonal (self-coherence)
        # After extraction, only the diagonal should be NaN — not off-diagonal entries
        n_regions = 5
        rec = self._make_coherence_recording(n_regions=n_regions)
        for i in range(n_regions):
            rec.coherence[:, :, i, i] = np.nan
        result = ee.get_events(rec, "test_event", "coherence", None, 0, 0, average=True)
        for snippet in result:
            for i in range(n_regions):
                for j in range(n_regions):
                    if i == j:
                        self.assertTrue(np.all(np.isnan(snippet[:, i, j])),
                                        f"diagonal [{i},{j}] should be NaN")
                    else:
                        self.assertFalse(np.any(np.isnan(snippet[:, i, j])),
                                         f"off-diagonal [{i},{j}] should not be NaN")

    def test_nan_fraction_with_diagonal_only_equals_one_over_n_regions(self):
        # For n_regions with only diagonal NaN: NaN fraction = n_regions / n_regions^2 = 1/n_regions
        n_regions = 5
        rec = self._make_coherence_recording(n_regions=n_regions)
        for i in range(n_regions):
            rec.coherence[:, :, i, i] = np.nan
        result = ee.average_events(rec, ["test_event"], "coherence")
        avg = result["test_event"][0]  # [n_freqs, n_regions, n_regions]
        nan_fraction = np.mean(np.isnan(avg))
        expected = 1.0 / n_regions
        self.assertAlmostEqual(nan_fraction, expected, places=5)

    def test_excluded_region_nan_propagates_to_row_and_col(self):
        # Simulates what the data looks like after exclude_regions is called:
        # row and col of excluded region should be NaN in output
        n_regions = 5
        rec = self._make_coherence_recording(n_regions=n_regions)
        excluded_idx = 0
        rec.coherence[:, :, excluded_idx, :] = np.nan
        rec.coherence[:, :, :, excluded_idx] = np.nan
        result = ee.get_events(rec, "test_event", "coherence", None, 0, 0, average=True)
        for snippet in result:
            self.assertTrue(np.all(np.isnan(snippet[:, excluded_idx, :])),
                            "excluded region row should be all NaN")
            self.assertTrue(np.all(np.isnan(snippet[:, :, excluded_idx])),
                            "excluded region col should be all NaN")

    def test_non_excluded_regions_not_nan_after_exclusion(self):
        n_regions = 5
        rec = self._make_coherence_recording(n_regions=n_regions)
        excluded_idx = 0
        rec.coherence[:, :, excluded_idx, :] = np.nan
        rec.coherence[:, :, :, excluded_idx] = np.nan
        result = ee.get_events(rec, "test_event", "coherence", None, 0, 0, average=True)
        for snippet in result:
            # Off-diagonal pairs that don't involve the excluded region should be fine
            self.assertFalse(np.any(np.isnan(snippet[:, 1, 2])))
            self.assertFalse(np.any(np.isnan(snippet[:, 2, 1])))
            self.assertFalse(np.any(np.isnan(snippet[:, 3, 4])))

    def test_nan_fraction_with_one_excluded_region(self):
        # For n=5 regions with 1 excluded: row(5) + col(4) + remaining diag(4) = 13/25 = 52%
        n_regions = 5
        rec = self._make_coherence_recording(n_regions=n_regions)
        excluded_idx = 0
        rec.coherence[:, :, excluded_idx, :] = np.nan
        rec.coherence[:, :, :, excluded_idx] = np.nan
        for i in range(1, n_regions):
            rec.coherence[:, :, i, i] = np.nan  # remaining diagonal
        result = ee.average_events(rec, ["test_event"], "coherence")
        avg = result["test_event"][0]
        nan_fraction = np.mean(np.isnan(avg))
        expected = (n_regions + (n_regions - 1) + (n_regions - 1)) / n_regions**2
        self.assertAlmostEqual(nan_fraction, expected, places=5)

    def test_power_nan_in_excluded_region_propagates(self):
        rec = make_fake_recording(n_timebins=40, n_freqs=10, n_regions=3)
        rec.power[:, :, 1] = np.nan  # region 1 all NaN
        result = ee.get_events(rec, "test_event", "power", None, 0, 0, average=True)
        for snippet in result:
            self.assertTrue(np.all(np.isnan(snippet[:, 1])))
            self.assertFalse(np.any(np.isnan(snippet[:, 0])))
            self.assertFalse(np.any(np.isnan(snippet[:, 2])))


# ---------------------------------------------------------------------------
# band_calcs
# ---------------------------------------------------------------------------

class TestBandCalcs(unittest.TestCase):
    """
    Band boundaries (0-indexed frequency bins):
        Delta      0:4
        Theta      4:13
        Beta       13:31
        Low gamma  31:71
        High gamma 71:100

    freq_axis is relative to the individual per-recording array:
        average_events   → per-rec [f, b]           → freq_axis=0 (default)
        baselined_events → per-rec [n_trials, f, b] → freq_axis=1
    """

    BANDS = ["Delta", "Theta", "Beta", "Low gamma", "High gamma"]

    def _avg_events_values(self, n_recs=3, n_freqs=100, n_regions=3, seed=0):
        """Mimics average_events output: {event: list of [f, b] per rec}"""
        np.random.seed(seed)
        return {
            "event_a": [np.random.rand(n_freqs, n_regions) for _ in range(n_recs)],
            "event_b": [np.random.rand(n_freqs, n_regions) for _ in range(n_recs)],
        }

    def _baselined_events_values(self, n_recs=3, n_trials=4, n_freqs=100, n_regions=3, seed=0):
        """Mimics baselined_events output: {event: list of [n_trials, f, b] per rec}"""
        np.random.seed(seed)
        return {
            "event_a": [np.random.rand(n_trials, n_freqs, n_regions) for _ in range(n_recs)],
            "event_b": [np.random.rand(n_trials, n_freqs, n_regions) for _ in range(n_recs)],
        }

    # --- outer_dict='agent' structure ---

    def test_agent_outer_keys_are_event_names(self):
        values = self._avg_events_values()
        result = ee.band_calcs(values, outer_dict='agent')
        self.assertEqual(set(result.keys()), set(values.keys()))

    def test_agent_inner_keys_are_band_names(self):
        values = self._avg_events_values()
        result = ee.band_calcs(values, outer_dict='agent')
        for event in values:
            self.assertEqual(set(result[event].keys()), set(self.BANDS))

    def test_agent_band_value_is_list_of_per_rec_arrays(self):
        n_recs = 3
        values = self._avg_events_values(n_recs=n_recs)
        result = ee.band_calcs(values, outer_dict='agent')
        for event in values:
            for band in self.BANDS:
                self.assertEqual(len(result[event][band]), n_recs)

    # --- outer_dict='band' structure ---

    def test_band_outer_keys_are_band_names(self):
        values = self._avg_events_values()
        result = ee.band_calcs(values, outer_dict='band')
        self.assertEqual(set(result.keys()), set(self.BANDS))

    def test_band_inner_keys_are_event_names(self):
        values = self._avg_events_values()
        result = ee.band_calcs(values, outer_dict='band')
        for band in self.BANDS:
            self.assertEqual(set(result[band].keys()), set(values.keys()))

    def test_band_values_match_agent_values(self):
        values = self._avg_events_values()
        agent_result = ee.band_calcs(values, outer_dict='agent')
        band_result  = ee.band_calcs(values, outer_dict='band')
        for event in values:
            for band in self.BANDS:
                np.testing.assert_array_equal(
                    agent_result[event][band],
                    band_result[band][event]
                )

    # --- freq_axis=0: average_events pipeline ---

    def test_freq_axis0_shape_power(self):
        # per-rec [f, b] → band mean drops f → [b]
        n_recs, n_regions = 3, 3
        values = self._avg_events_values(n_recs=n_recs, n_regions=n_regions)
        result = ee.band_calcs(values, outer_dict='agent', freq_axis=0)
        for event in values:
            for band in self.BANDS:
                for arr in result[event][band]:
                    self.assertEqual(arr.shape, (n_regions,))

    def test_freq_axis0_known_delta_value(self):
        # Fill bins 0:4 with 3.0, rest 0.0 → Delta mean = 3.0
        n_recs, n_regions = 2, 3
        arr = np.zeros((100, n_regions))
        arr[0:4, :] = 3.0
        values = {"ev": [arr] * n_recs}
        result = ee.band_calcs(values, outer_dict='agent', freq_axis=0)
        for per_rec in result["ev"]["Delta"]:
            np.testing.assert_array_almost_equal(per_rec, 3.0)

    def test_freq_axis0_uniform_all_bands_equal(self):
        n_recs, n_regions = 2, 3
        arr = np.full((100, n_regions), 7.0)
        values = {"ev": [arr] * n_recs}
        result = ee.band_calcs(values, outer_dict='agent', freq_axis=0)
        for band in self.BANDS:
            for per_rec in result["ev"][band]:
                np.testing.assert_array_almost_equal(per_rec, 7.0)

    # --- freq_axis=1: baselined_events pipeline ---

    def test_freq_axis1_shape_power(self):
        # per-rec [n_trials, f, b] → band mean drops f → [n_trials, b]
        n_recs, n_trials, n_regions = 3, 4, 3
        values = self._baselined_events_values(n_recs=n_recs, n_trials=n_trials, n_regions=n_regions)
        result = ee.band_calcs(values, outer_dict='agent', freq_axis=1)
        for event in values:
            for band in self.BANDS:
                for arr in result[event][band]:
                    self.assertEqual(arr.shape, (n_trials, n_regions))

    def test_freq_axis1_known_delta_value(self):
        n_trials, n_regions = 4, 3
        arr = np.zeros((n_trials, 100, n_regions))
        arr[:, 0:4, :] = 5.0
        values = {"ev": [arr]}
        result = ee.band_calcs(values, outer_dict='agent', freq_axis=1)
        np.testing.assert_array_almost_equal(result["ev"]["Delta"][0], 5.0)

    def test_freq_axis1_variable_trial_counts_do_not_crash(self):
        # baselined_events can produce different n_trials per recording
        values = {
            "ev": [
                np.random.rand(3, 100, 3),  # rec0: 3 trials
                np.random.rand(7, 100, 3),  # rec1: 7 trials
            ]
        }
        result = ee.band_calcs(values, outer_dict='agent', freq_axis=1)
        self.assertEqual(result["ev"]["Delta"][0].shape[0], 3)
        self.assertEqual(result["ev"]["Delta"][1].shape[0], 7)

    # --- granger/coherence (4-D per-rec arrays) ---

    def test_freq_axis0_shape_granger(self):
        # per-rec [f, b, b] → band mean → [b, b]
        n_recs, n_regions = 2, 4
        values = {"ev": [np.random.rand(100, n_regions, n_regions) for _ in range(n_recs)]}
        result = ee.band_calcs(values, outer_dict='agent', freq_axis=0)
        for per_rec in result["ev"]["Delta"]:
            self.assertEqual(per_rec.shape, (n_regions, n_regions))

    def test_freq_axis1_shape_granger_with_trials(self):
        # per-rec [n_trials, f, b, b] → band mean → [n_trials, b, b]
        n_recs, n_trials, n_regions = 2, 5, 4
        values = {"ev": [np.random.rand(n_trials, 100, n_regions, n_regions) for _ in range(n_recs)]}
        result = ee.band_calcs(values, outer_dict='agent', freq_axis=1)
        for per_rec in result["ev"]["Beta"]:
            self.assertEqual(per_rec.shape, (n_trials, n_regions, n_regions))

    # --- NaN handling ---

    def test_nan_in_freq_bins_propagates_to_band(self):
        n_regions = 3
        arr = np.ones((100, n_regions))
        arr[0:4, 1] = np.nan  # NaN in delta bins for region 1
        values = {"ev": [arr]}
        result = ee.band_calcs(values, outer_dict='agent', freq_axis=0)
        self.assertTrue(np.isnan(result["ev"]["Delta"][0][1]))
        self.assertFalse(np.isnan(result["ev"]["Delta"][0][0]))

    # --- default and invalid ---

    def test_default_freq_axis_is_0(self):
        values = self._avg_events_values()
        default = ee.band_calcs(values, outer_dict='agent')
        explicit = ee.band_calcs(values, outer_dict='agent', freq_axis=0)
        for event in values:
            for band in self.BANDS:
                for a, b in zip(default[event][band], explicit[event][band]):
                    np.testing.assert_array_equal(a, b)

    def test_default_outer_dict_is_agent(self):
        values = self._avg_events_values()
        self.assertEqual(
            set(ee.band_calcs(values).keys()),
            set(values.keys())
        )

    def test_invalid_outer_dict_raises_value_error(self):
        values = self._avg_events_values()
        with self.assertRaises(ValueError):
            ee.band_calcs(values, outer_dict='event')


if __name__ == "__main__":
    unittest.main()
