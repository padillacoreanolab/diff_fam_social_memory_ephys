import unittest
import os
import shutil
import json
import numpy as np
import h5py
from bidict import bidict
from unittest.mock import patch
from lfp.lfp_analysis.LFP_recording import LFPRecording
from lfp.tests.utils import EXAMPLE_RECORDING_FILEPATH
import lfp.lfp_analysis.connectivity_wrapper as connectivity_wrapper


CHANNEL_DICT = {"mPFC": 1, "vHPC": 9, "BLA": 11, "NAc": 27, "MD": 3}
OUTPUT_DIR = os.path.join("lfp", "tests", "output")
REC_PATH = os.path.join(OUTPUT_DIR, "test_rec")
H5_PATH = REC_PATH + ".h5"
JSON_PATH = REC_PATH + ".json"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def make_fake_recording(n_timebins=30, n_freqs=10, n_regions=3, seed=0):
    """LFPRecording with synthetic arrays — no .rec file needed."""
    np.random.seed(seed)
    channel_dict = {"mPFC": 5, "vHPC": 1, "BLA": 3}
    rec = LFPRecording(
        subject="test_subject",
        channel_dict=channel_dict,
        merged_rec_path="/fake/path/fake_merged.rec",
        load=True,
    )
    rec.brain_region_dict = bidict({"vHPC": 0, "BLA": 1, "mPFC": 2})
    rec.traces = np.random.rand(500, n_regions)
    rec.rms_traces = np.random.rand(500, n_regions)
    rec.power = np.random.rand(n_timebins, n_freqs, n_regions)
    rec.coherence = np.random.rand(n_timebins, n_freqs, n_regions, n_regions)
    rec.granger = np.random.rand(n_timebins, n_freqs, n_regions, n_regions)
    rec.rec_length = 1.0
    return rec


# ---------------------------------------------------------------------------
# exclude_regions
# ---------------------------------------------------------------------------

class TestExcludeRegions(unittest.TestCase):
    def setUp(self):
        self.rec = make_fake_recording()
        # vHPC is index 0, BLA is index 1, mPFC is index 2
        self.excluded_idx = 0  # vHPC

    def test_rms_traces_excluded_column_is_nan(self):
        self.rec.exclude_regions(["vHPC"])
        self.assertTrue(np.all(np.isnan(self.rec.rms_traces[:, self.excluded_idx])))

    def test_power_excluded_region_is_nan(self):
        self.rec.exclude_regions(["vHPC"])
        self.assertTrue(np.all(np.isnan(self.rec.power[:, :, self.excluded_idx])))

    def test_coherence_excluded_row_is_nan(self):
        self.rec.exclude_regions(["vHPC"])
        self.assertTrue(np.all(np.isnan(self.rec.coherence[:, :, self.excluded_idx, :])))

    def test_coherence_excluded_col_is_nan(self):
        self.rec.exclude_regions(["vHPC"])
        self.assertTrue(np.all(np.isnan(self.rec.coherence[:, :, :, self.excluded_idx])))

    def test_granger_excluded_row_is_nan(self):
        self.rec.exclude_regions(["vHPC"])
        self.assertTrue(np.all(np.isnan(self.rec.granger[:, :, self.excluded_idx, :])))

    def test_granger_excluded_col_is_nan(self):
        self.rec.exclude_regions(["vHPC"])
        self.assertTrue(np.all(np.isnan(self.rec.granger[:, :, :, self.excluded_idx])))

    def test_non_excluded_rms_traces_unchanged(self):
        original = self.rec.rms_traces.copy()
        self.rec.exclude_regions(["vHPC"])
        np.testing.assert_array_equal(self.rec.rms_traces[:, 1], original[:, 1])
        np.testing.assert_array_equal(self.rec.rms_traces[:, 2], original[:, 2])

    def test_non_excluded_power_unchanged(self):
        original = self.rec.power.copy()
        self.rec.exclude_regions(["vHPC"])
        np.testing.assert_array_equal(self.rec.power[:, :, 1], original[:, :, 1])
        np.testing.assert_array_equal(self.rec.power[:, :, 2], original[:, :, 2])

    def test_excluded_regions_attribute_is_set(self):
        self.rec.exclude_regions(["vHPC"])
        self.assertEqual(self.rec.excluded_regions, ["vHPC"])

    def test_empty_list_does_nothing(self):
        original_rms = self.rec.rms_traces.copy()
        self.rec.exclude_regions([])
        np.testing.assert_array_equal(self.rec.rms_traces, original_rms)
        self.assertEqual(self.rec.excluded_regions, [])

    def test_works_without_power_coherence_granger(self):
        rec = make_fake_recording()
        del rec.power
        del rec.coherence
        del rec.granger
        # Should not raise
        rec.exclude_regions(["vHPC"])
        self.assertTrue(np.all(np.isnan(rec.rms_traces[:, 0])))

    def test_multiple_regions_excluded(self):
        self.rec.exclude_regions(["vHPC", "BLA"])
        self.assertTrue(np.all(np.isnan(self.rec.rms_traces[:, 0])))
        self.assertTrue(np.all(np.isnan(self.rec.rms_traces[:, 1])))
        # mPFC (index 2) unchanged
        self.assertFalse(np.any(np.isnan(self.rec.rms_traces[:, 2])))


# ---------------------------------------------------------------------------
# interpolate_power
# ---------------------------------------------------------------------------

class TestInterpolatePower(unittest.TestCase):
    def setUp(self):
        self.rec = make_fake_recording(n_timebins=50, n_freqs=10, n_regions=3)

    def test_nan_values_in_valid_region_are_filled(self):
        # Inject NaN at a few interior timepoints in region 1
        self.rec.power[10:15, :, 1] = np.nan
        self.rec.interpolate_power()
        self.assertFalse(np.any(np.isnan(self.rec.power[10:15, :, 1])))

    def test_all_nan_region_stays_nan(self):
        self.rec.power[:, :, 2] = np.nan
        self.rec.interpolate_power()
        self.assertTrue(np.all(np.isnan(self.rec.power[:, :, 2])))

    def test_non_nan_values_are_unchanged(self):
        self.rec.power[10:15, :, 1] = np.nan
        original = self.rec.power.copy()  # snapshot after NaN injection
        self.rec.interpolate_power()
        # Positions that were not NaN before interpolation should be exactly preserved
        mask = ~np.isnan(original)
        np.testing.assert_array_equal(self.rec.power[mask], original[mask])

    def test_shape_preserved(self):
        original_shape = self.rec.power.shape
        self.rec.power[10:15, :, 1] = np.nan
        self.rec.interpolate_power()
        self.assertEqual(self.rec.power.shape, original_shape)

    def test_no_nans_returns_early_without_modifying(self):
        original = self.rec.power.copy()
        result = self.rec.interpolate_power()
        self.assertIsNone(result)
        np.testing.assert_array_equal(self.rec.power, original)

    def test_other_regions_not_affected(self):
        original = self.rec.power.copy()
        self.rec.power[10:15, :, 1] = np.nan
        self.rec.interpolate_power()
        # Regions 0 and 2 should be completely unchanged
        np.testing.assert_array_equal(self.rec.power[:, :, 0], original[:, :, 0])
        np.testing.assert_array_equal(self.rec.power[:, :, 2], original[:, :, 2])


# ---------------------------------------------------------------------------
# interpolate_granger (tests __interpolate_coherence_granger__ via the public method)
# ---------------------------------------------------------------------------

class TestInterpolateGranger(unittest.TestCase):
    def setUp(self):
        self.rec = make_fake_recording(n_timebins=50, n_freqs=10, n_regions=3)

    def test_nan_values_in_valid_pair_are_filled(self):
        # Inject NaN at interior timepoints for off-diagonal pair (0, 1)
        self.rec.granger[10:15, :, 0, 1] = np.nan
        self.rec.interpolate_granger()
        self.assertFalse(np.any(np.isnan(self.rec.granger[10:15, :, 0, 1])))

    def test_all_nan_pair_stays_nan(self):
        # An entire off-diagonal pair that is all NaN should stay NaN
        self.rec.granger[:, :, 0, 1] = np.nan
        self.rec.interpolate_granger()
        self.assertTrue(np.all(np.isnan(self.rec.granger[:, :, 0, 1])))

    def test_diagonal_is_set_to_zero(self):
        self.rec.granger[10:15, :, 0, 1] = np.nan  # trigger interpolation
        self.rec.interpolate_granger()
        for i in range(self.rec.granger.shape[2]):
            np.testing.assert_array_equal(self.rec.granger[:, :, i, i], 0)

    def test_non_nan_off_diagonal_values_unchanged(self):
        self.rec.granger[10:15, :, 0, 1] = np.nan
        original = self.rec.granger.copy()  # snapshot after NaN injection
        self.rec.interpolate_granger()
        # Non-NaN off-diagonal positions should be exactly preserved
        for j in range(3):
            for k in range(3):
                if j != k:
                    mask = ~np.isnan(original[:, :, j, k])
                    np.testing.assert_array_equal(
                        self.rec.granger[:, :, j, k][mask],
                        original[:, :, j, k][mask]
                    )

    def test_shape_preserved(self):
        original_shape = self.rec.granger.shape
        self.rec.granger[10:15, :, 0, 1] = np.nan
        self.rec.interpolate_granger()
        self.assertEqual(self.rec.granger.shape, original_shape)

    def test_no_nans_returns_array_unchanged(self):
        original = self.rec.granger.copy()
        self.rec.interpolate_granger()
        # No NaN → returns values unchanged (except diagonal which is still set to 0)
        self.assertEqual(self.rec.granger.shape, original.shape)


# ---------------------------------------------------------------------------
# H5 save/load roundtrip (synthetic data, no .rec file needed)
# ---------------------------------------------------------------------------

class TestH5Roundtrip(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)

    def _saved_recording(self):
        rec = make_fake_recording()
        LFPRecording.save_rec_to_h5(rec, REC_PATH)
        return rec

    def test_save_creates_h5_file(self):
        self._saved_recording()
        self.assertTrue(os.path.exists(H5_PATH))

    def test_save_creates_json_file(self):
        self._saved_recording()
        self.assertTrue(os.path.exists(JSON_PATH))

    def test_json_is_valid(self):
        self._saved_recording()
        with open(JSON_PATH) as f:
            data = json.load(f)
        self.assertIsInstance(data, dict)

    def test_load_returns_lfp_recording(self):
        self._saved_recording()
        loaded = LFPRecording.load_rec_from_h5(H5_PATH)
        self.assertIsInstance(loaded, LFPRecording)

    def test_subject_preserved(self):
        self._saved_recording()
        loaded = LFPRecording.load_rec_from_h5(H5_PATH)
        self.assertEqual(loaded.subject, "test_subject")

    def test_rms_traces_preserved(self):
        original = self._saved_recording()
        loaded = LFPRecording.load_rec_from_h5(H5_PATH)
        np.testing.assert_array_equal(loaded.rms_traces, original.rms_traces)

    def test_granger_preserved(self):
        original = self._saved_recording()
        loaded = LFPRecording.load_rec_from_h5(H5_PATH)
        np.testing.assert_array_equal(loaded.granger, original.granger)

    def test_coherence_preserved(self):
        original = self._saved_recording()
        loaded = LFPRecording.load_rec_from_h5(H5_PATH)
        np.testing.assert_array_equal(loaded.coherence, original.coherence)

    def test_brain_region_dict_preserved(self):
        self._saved_recording()
        loaded = LFPRecording.load_rec_from_h5(H5_PATH)
        self.assertEqual(dict(loaded.brain_region_dict), {"vHPC": 0, "BLA": 1, "mPFC": 2})

    def test_recording_without_power_has_no_power_after_load(self):
        rec = make_fake_recording()
        del rec.power
        LFPRecording.save_rec_to_h5(rec, REC_PATH)
        loaded = LFPRecording.load_rec_from_h5(H5_PATH)
        self.assertFalse(hasattr(loaded, "power"))

    def test_event_dict_preserved(self):
        rec = make_fake_recording()
        rec.event_dict = {"sniff": np.array([[100, 200], [300, 400]]),
                          "explore": np.array([[500, 600]])}
        LFPRecording.save_rec_to_h5(rec, REC_PATH)
        loaded = LFPRecording.load_rec_from_h5(H5_PATH)
        np.testing.assert_array_equal(loaded.event_dict["sniff"], rec.event_dict["sniff"])
        np.testing.assert_array_equal(loaded.event_dict["explore"], rec.event_dict["explore"])


# ---------------------------------------------------------------------------
# File-dependent tests (require Example_Recording to be downloaded)
# ---------------------------------------------------------------------------

@unittest.skipUnless(
    os.path.exists(EXAMPLE_RECORDING_FILEPATH),
    "Example recording not found — run: python -m lfp.tests.utils download_test_data"
)
class TestLFPRecordingFileIO(unittest.TestCase):
    def test_read_trodes_returns_recording(self):
        rec = LFPRecording("test", CHANNEL_DICT, EXAMPLE_RECORDING_FILEPATH)
        recording = rec._read_trodes()
        self.assertIsNotNone(recording)

    def test_traces_have_correct_number_of_regions(self):
        rec = LFPRecording("test", CHANNEL_DICT, EXAMPLE_RECORDING_FILEPATH)
        self.assertEqual(rec.traces.shape[1], len(CHANNEL_DICT))

    def test_traces_resampled_to_1khz(self):
        rec = LFPRecording("test", CHANNEL_DICT, EXAMPLE_RECORDING_FILEPATH)
        # At 1kHz, 1 second = 1000 samples — check rate is sensible
        self.assertEqual(rec.resample_rate, 1000)

    def test_channel_order_is_consistent(self):
        # BLA traces should be identical regardless of what other channels are loaded alongside
        rec_full = LFPRecording("s1", {"mPFC": 1, "BLA": 7, "vHPC": 31}, EXAMPLE_RECORDING_FILEPATH)
        rec_bla_only = LFPRecording("s2", {"BLA": 7}, EXAMPLE_RECORDING_FILEPATH)
        bla_idx_full = rec_full.brain_region_dict["BLA"]
        np.testing.assert_array_equal(
            rec_full.traces[:, bla_idx_full],
            rec_bla_only.traces[:, 0]
        )


if __name__ == "__main__":
    unittest.main()
