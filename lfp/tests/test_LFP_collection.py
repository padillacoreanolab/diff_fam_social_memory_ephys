import unittest
import os
import shutil
import json
import numpy as np
from bidict import bidict
from unittest.mock import patch, MagicMock
from lfp.lfp_analysis.LFP_collection import LFPCollection
from lfp.lfp_analysis.LFP_recording import LFPRecording


OUTPUT_DIR = os.path.join("lfp", "tests", "output")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_fake_recording(name, subject, n_timebins=30, n_freqs=10, n_regions=3, seed=0):
    np.random.seed(seed)
    channel_dict = {"mPFC": 5, "vHPC": 1, "BLA": 3}
    rec = LFPRecording(
        subject=subject,
        channel_dict=channel_dict,
        merged_rec_path=f"/fake/path/{name}",
        load=True,
    )
    rec.name = name
    rec.brain_region_dict = bidict({"vHPC": 0, "BLA": 1, "mPFC": 2})
    rec.traces = np.random.rand(500, n_regions)
    rec.rms_traces = np.random.rand(500, n_regions)
    rec.coherence = np.random.rand(n_timebins, n_freqs, n_regions, n_regions)
    rec.granger = np.random.rand(n_timebins, n_freqs, n_regions, n_regions)
    rec.rec_length = 1.0
    return rec


def make_fake_collection(n_recordings=3):
    """LFPCollection with synthetic recordings, bypassing all file I/O."""
    subjects = [f"subject{i}" for i in range(n_recordings)]
    rec_names = [f"rec{i}_merged.rec" for i in range(n_recordings)]
    channel_dict = {"mPFC": 5, "vHPC": 1, "BLA": 3}

    fake_recordings = [
        make_fake_recording(rec_names[i], subjects[i], seed=i)
        for i in range(n_recordings)
    ]

    with patch.object(LFPCollection, "_make_recordings", return_value=fake_recordings):
        collection = LFPCollection(
            subject_to_channel_dict={s: channel_dict for s in subjects},
            data_path="/fake/data",
            recording_to_subject_dict={rec_names[i]: subjects[i] for i in range(n_recordings)},
            threshold=4,
        )

    collection.frequencies = list(np.linspace(0, 100, 50))
    return collection


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestCollectionInit(unittest.TestCase):
    def test_recordings_list_is_populated(self):
        collection = make_fake_collection(n_recordings=3)
        self.assertEqual(len(collection.recordings), 3)

    def test_brain_region_dict_set_from_first_recording(self):
        collection = make_fake_collection()
        self.assertEqual(dict(collection.brain_region_dict), {"vHPC": 0, "BLA": 1, "mPFC": 2})

    def test_threshold_stored(self):
        collection = make_fake_collection()
        self.assertEqual(collection.threshold, 4)

    def test_kwargs_filled_with_defaults_for_missing_keys(self):
        collection = make_fake_collection()
        self.assertIn("sampling_rate", collection.kwargs)
        self.assertIn("halfbandwidth", collection.kwargs)
        self.assertIn("timestep", collection.kwargs)

    def test_custom_kwargs_override_defaults(self):
        subjects = ["s0"]
        rec_names = ["rec0_merged.rec"]
        channel_dict = {"mPFC": 5, "vHPC": 1, "BLA": 3}
        fake_recs = [make_fake_recording(rec_names[0], subjects[0])]
        with patch.object(LFPCollection, "_make_recordings", return_value=fake_recs):
            collection = LFPCollection(
                subject_to_channel_dict={"s0": channel_dict},
                data_path="/fake",
                recording_to_subject_dict={"rec0_merged.rec": "s0"},
                threshold=4,
                halfbandwidth=5,
                timestep=0.25,
            )
        self.assertEqual(collection.kwargs["halfbandwidth"], 5)
        self.assertEqual(collection.kwargs["timestep"], 0.25)

    def test_target_confirmation_dict_triggers_exclude_regions(self):
        collection = make_fake_collection(n_recordings=2)
        # All subjects should have excluded_regions set after init with target_confirmation_dict
        target_dict = {rec.subject: ["vHPC"] for rec in collection.recordings}
        with patch.object(LFPCollection, "_make_recordings",
                          return_value=collection.recordings):
            collection2 = LFPCollection(
                subject_to_channel_dict={"subject0": {"mPFC": 5, "vHPC": 1, "BLA": 3},
                                          "subject1": {"mPFC": 5, "vHPC": 1, "BLA": 3}},
                data_path="/fake",
                recording_to_subject_dict={"rec0_merged.rec": "subject0",
                                            "rec1_merged.rec": "subject1"},
                threshold=4,
                target_confirmation_dict=target_dict,
            )
        for rec in collection2.recordings:
            self.assertTrue(np.all(np.isnan(rec.rms_traces[:, 0])))  # vHPC = index 0


# ---------------------------------------------------------------------------
# exclude_regions
# ---------------------------------------------------------------------------

class TestExcludeRegions(unittest.TestCase):
    def setUp(self):
        self.collection = make_fake_collection(n_recordings=3)
        self.target_dict = {rec.subject: ["vHPC"] for rec in self.collection.recordings}

    def test_exclude_called_on_all_recordings(self):
        self.collection.exclude_regions(self.target_dict)
        for rec in self.collection.recordings:
            self.assertTrue(np.all(np.isnan(rec.rms_traces[:, 0])))

    def test_excluded_regions_attribute_set_on_each_recording(self):
        self.collection.exclude_regions(self.target_dict)
        for rec in self.collection.recordings:
            self.assertEqual(rec.excluded_regions, ["vHPC"])

    def test_already_excluded_recording_is_skipped(self):
        self.collection.exclude_regions(self.target_dict)
        original_rms = [rec.rms_traces.copy() for rec in self.collection.recordings]
        # Calling again with same dict should skip all (already excluded)
        self.collection.exclude_regions(self.target_dict)
        for i, rec in enumerate(self.collection.recordings):
            np.testing.assert_array_equal(rec.rms_traces, original_rms[i])

    def test_different_subjects_can_have_different_exclusions(self):
        target_dict = {
            self.collection.recordings[0].subject: ["vHPC"],
            self.collection.recordings[1].subject: ["BLA"],
            self.collection.recordings[2].subject: [],
        }
        self.collection.exclude_regions(target_dict)
        # rec0: vHPC (idx 0) excluded
        self.assertTrue(np.all(np.isnan(self.collection.recordings[0].rms_traces[:, 0])))
        self.assertFalse(np.any(np.isnan(self.collection.recordings[0].rms_traces[:, 1])))
        # rec1: BLA (idx 1) excluded
        self.assertTrue(np.all(np.isnan(self.collection.recordings[1].rms_traces[:, 1])))
        self.assertFalse(np.any(np.isnan(self.collection.recordings[1].rms_traces[:, 0])))
        # rec2: nothing excluded
        self.assertFalse(np.any(np.isnan(self.collection.recordings[2].rms_traces)))


# ---------------------------------------------------------------------------
# get_recording
# ---------------------------------------------------------------------------

class TestGetRecording(unittest.TestCase):
    def setUp(self):
        self.collection = make_fake_collection(n_recordings=3)

    def test_returns_correct_recording_by_name(self):
        target = self.collection.recordings[1]
        result = self.collection.get_recording(target.name)
        self.assertIs(result, target)

    def test_raises_value_error_for_unknown_name(self):
        with self.assertRaises(ValueError):
            self.collection.get_recording("does_not_exist_merged.rec")

    def test_error_message_contains_name(self):
        with self.assertRaises(ValueError) as ctx:
            self.collection.get_recording("missing_merged.rec")
        self.assertIn("missing_merged.rec", str(ctx.exception))


# ---------------------------------------------------------------------------
# remove_recording
# ---------------------------------------------------------------------------

class TestRemoveRecording(unittest.TestCase):
    def setUp(self):
        self.collection = make_fake_collection(n_recordings=3)

    def test_remove_by_name_reduces_count(self):
        name = self.collection.recordings[0].name
        self.collection.remove_recording(name)
        self.assertEqual(len(self.collection.recordings), 2)

    def test_remove_by_name_removes_correct_recording(self):
        target = self.collection.recordings[1]
        self.collection.remove_recording(target.name)
        self.assertNotIn(target, self.collection.recordings)

    def test_remove_by_object_reduces_count(self):
        target = self.collection.recordings[0]
        self.collection.remove_recording(target)
        self.assertEqual(len(self.collection.recordings), 2)

    def test_remove_by_object_removes_correct_recording(self):
        target = self.collection.recordings[2]
        self.collection.remove_recording(target)
        self.assertNotIn(target, self.collection.recordings)

    def test_remove_by_unknown_name_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.collection.remove_recording("nonexistent_merged.rec")

    def test_remove_unknown_object_raises_value_error(self):
        outsider = make_fake_recording("outsider_merged.rec", "stranger")
        with self.assertRaises(ValueError):
            self.collection.remove_recording(outsider)

    def test_remaining_recordings_are_intact(self):
        target = self.collection.recordings[0]
        remaining = self.collection.recordings[1:]
        self.collection.remove_recording(target)
        self.assertEqual(self.collection.recordings, remaining)


# ---------------------------------------------------------------------------
# interpolate
# ---------------------------------------------------------------------------

class TestInterpolate(unittest.TestCase):
    def setUp(self):
        self.collection = make_fake_collection(n_recordings=2)

    def test_all_mode_calls_all_three_methods_on_each_recording(self):
        for rec in self.collection.recordings:
            rec.interpolate_power = MagicMock()
            rec.interpolate_coherence = MagicMock()
            rec.interpolate_granger = MagicMock()
        self.collection.interpolate(modes="all")
        for rec in self.collection.recordings:
            rec.interpolate_power.assert_called_once_with("linear")
            rec.interpolate_coherence.assert_called_once_with("linear")
            rec.interpolate_granger.assert_called_once_with("linear")

    def test_power_mode_only_calls_interpolate_power(self):
        for rec in self.collection.recordings:
            rec.interpolate_power = MagicMock()
            rec.interpolate_coherence = MagicMock()
            rec.interpolate_granger = MagicMock()
        self.collection.interpolate(modes=["power"])
        for rec in self.collection.recordings:
            rec.interpolate_power.assert_called_once()
            rec.interpolate_coherence.assert_not_called()
            rec.interpolate_granger.assert_not_called()

    def test_granger_mode_only_calls_interpolate_granger(self):
        for rec in self.collection.recordings:
            rec.interpolate_power = MagicMock()
            rec.interpolate_coherence = MagicMock()
            rec.interpolate_granger = MagicMock()
        self.collection.interpolate(modes=["granger"])
        for rec in self.collection.recordings:
            rec.interpolate_granger.assert_called_once()
            rec.interpolate_power.assert_not_called()
            rec.interpolate_coherence.assert_not_called()

    def test_invalid_mode_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.collection.interpolate(modes=["spectral_granger"])

    def test_custom_kind_is_passed_through(self):
        for rec in self.collection.recordings:
            rec.interpolate_power = MagicMock()
            rec.interpolate_coherence = MagicMock()
            rec.interpolate_granger = MagicMock()
        self.collection.interpolate(modes="all", kind="cubic")
        for rec in self.collection.recordings:
            rec.interpolate_power.assert_called_once_with("cubic")


# ---------------------------------------------------------------------------
# calculate_all
# ---------------------------------------------------------------------------

class TestCalculateAll(unittest.TestCase):
    def setUp(self):
        self.collection = make_fake_collection(n_recordings=2)

    def test_batched_granger_without_output_dir_raises(self):
        with self.assertRaises(ValueError):
            self.collection.calculate_all(batched_granger=True)


# ---------------------------------------------------------------------------
# preprocess — NOTE: exposes bug where threshold param is silently ignored
# ---------------------------------------------------------------------------

class TestPreprocess(unittest.TestCase):
    def setUp(self):
        self.collection = make_fake_collection(n_recordings=2)
        for rec in self.collection.recordings:
            rec.preprocess = MagicMock()

    def test_preprocess_called_on_all_recordings(self):
        self.collection.preprocess()
        for rec in self.collection.recordings:
            rec.preprocess.assert_called_once()

    def test_preprocess_uses_collection_threshold(self):
        # collection.threshold = 4, so each recording should get preprocess(4)
        self.collection.preprocess()
        for rec in self.collection.recordings:
            rec.preprocess.assert_called_once_with(4)

    def test_threshold_argument_is_used(self):
        # BUG: currently the threshold param is ignored due to typo ('threhsold')
        # This test documents the correct expected behavior: passing threshold=7
        # should call recording.preprocess(7), not recording.preprocess(4)
        self.collection.preprocess(threshold=7)
        for rec in self.collection.recordings:
            rec.preprocess.assert_called_once_with(7)


# ---------------------------------------------------------------------------
# save / load roundtrip
# ---------------------------------------------------------------------------

class TestCollectionH5Roundtrip(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        cls.collection = make_fake_collection(n_recordings=2)
        # Don't save power — load_rec_from_h5 would try to recalculate multitaper
        for rec in cls.collection.recordings:
            if hasattr(rec, "power"):
                del rec.power
        LFPCollection.save_to_json(cls.collection, OUTPUT_DIR)
        cls.json_path = os.path.join(OUTPUT_DIR, "lfp_collection.json")

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)

    def test_save_creates_collection_json(self):
        self.assertTrue(os.path.exists(self.json_path))

    def test_save_creates_recordings_directory(self):
        recordings_dir = os.path.join(OUTPUT_DIR, "recordings")
        self.assertTrue(os.path.exists(recordings_dir))

    def test_save_creates_one_h5_per_recording(self):
        recordings_dir = os.path.join(OUTPUT_DIR, "recordings")
        h5_files = list(os.scandir(recordings_dir))
        h5_files = [f for f in h5_files if f.name.endswith(".h5")]
        self.assertEqual(len(h5_files), len(self.collection.recordings))

    def test_load_returns_lfp_collection(self):
        loaded = LFPCollection.load_collection(self.json_path)
        self.assertIsInstance(loaded, LFPCollection)

    def test_loaded_collection_has_correct_number_of_recordings(self):
        loaded = LFPCollection.load_collection(self.json_path)
        self.assertEqual(len(loaded.recordings), len(self.collection.recordings))

    def test_brain_region_dict_preserved(self):
        loaded = LFPCollection.load_collection(self.json_path)
        self.assertEqual(dict(loaded.brain_region_dict), {"vHPC": 0, "BLA": 1, "mPFC": 2})

    def test_threshold_preserved(self):
        loaded = LFPCollection.load_collection(self.json_path)
        self.assertEqual(loaded.threshold, self.collection.threshold)

    def test_recording_subjects_preserved(self):
        loaded = LFPCollection.load_collection(self.json_path)
        original_subjects = {rec.subject for rec in self.collection.recordings}
        loaded_subjects = {rec.subject for rec in loaded.recordings}
        self.assertEqual(original_subjects, loaded_subjects)

    def test_rms_traces_preserved_per_recording(self):
        loaded = LFPCollection.load_collection(self.json_path)
        original_by_name = {rec.name: rec for rec in self.collection.recordings}
        for loaded_rec in loaded.recordings:
            original_rec = original_by_name[loaded_rec.name]
            np.testing.assert_array_equal(loaded_rec.rms_traces, original_rec.rms_traces)


if __name__ == "__main__":
    unittest.main()
