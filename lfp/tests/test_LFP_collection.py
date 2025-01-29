import unittest
from lfp.lfp_analysis.LFP_collection import LFPCollection
from lfp.tests.utils import TEST_DATA_DIR
import shutil
import os
import json
import numpy as np
from pathlib import Path

OUTPUT_DIR = os.path.join("lfp", "tests", "output")


class TestLFPCollection(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.recording_to_behavior_dict = {
            "example_recording_merged.rec": {
                "behavior1": [[1, 2], [4, 9], [11, 15]],
                "behavior2": [[19, 22], [27, 32]],
            },
            "example_recording2_merged.rec": {
                "behavior1": [[1, 2], [4, 9], [11, 15]],
                "behavior2": [[19, 22], [27, 32]],
            },
        }
        self.subject_to_channel_dict = {
            "subject1": {"channel1": 1},
            "subject2": {"channel1": 1},
        }
        self.recording_to_subject_dict = {
            "example_recording_merged.rec": "subject1",
            "example_recording2_merged.rec": "subject2",
        }
        self.data_path = TEST_DATA_DIR

    def test_initialization_custom_parameters(self):
        """Test LFPCollection initialization with custom parameters"""
        custom_params = {
            "sampling_rate": 10000,
            "voltage_scaling": 0.5,
            "spike_gadgets_multiplier": 1.0,
            "elec_noise_freq": 50,
            "min_freq": 1.0,
            "max_freq": 200,
            "resample_rate": 500,
            "halfbandwidth": 3,
            "timewindow": 2,
            "timestep": 0.25,
        }

        lfp_collection = LFPCollection(
            subject_to_channel_dict=self.subject_to_channel_dict,
            data_path=self.data_path,
            recording_to_subject_dict=self.recording_to_subject_dict,
            threshold=0.75,
            recording_to_behavior_dict=self.recording_to_behavior_dict,
            **custom_params,
        )

        self.assertIsNotNone(lfp_collection.kwargs)

    def test_invalid_input(self):
        """Test LFPCollection initialization with invalid input"""
        with self.assertRaises(TypeError):
            LFPCollection(None, None, None)


class TestH5FileCollection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)
        os.mkdir(OUTPUT_DIR)

    def setUp(self):
        """Set up test fixtures before each test method"""
        self.recording_to_behavior_dict = {
            "example_recording_merged.rec": {
                "behavior1": [[1, 2], [4, 9], [11, 15]],
                "behavior2": [[19, 22], [27, 32]],
            },
            "example_recording2_merged.rec": {
                "behavior1": [[1, 2], [4, 9], [11, 15]],
                "behavior2": [[19, 22], [27, 32]],
            },
        }
        self.subject_to_channel_dict = {
            "subject1": {"channel1": 1},
            "subject2": {"channel1": 1},
        }
        self.recording_to_subject_dict = {
            "example_recording_merged.rec": "subject1",
            "example_recording2_merged.rec": "subject2",
        }
        self.data_path = TEST_DATA_DIR

    def test_1_save_collection(self):
        lfp_collection = LFPCollection(
            subject_to_channel_dict=self.subject_to_channel_dict,
            data_path=self.data_path,
            recording_to_subject_dict=self.recording_to_subject_dict,
            threshold=0.75,
            recording_to_behavior_dict=self.recording_to_behavior_dict,
        )
        LFPCollection.save_to_json(lfp_collection, OUTPUT_DIR)
        collection_path = os.path.join(OUTPUT_DIR, "lfp_collection.json")
        self.assertTrue(os.path.exists(collection_path))
        try:
            with open(collection_path, "r") as f:
                json_data = json.load(f)
                self.assertTrue(isinstance(json_data, dict), "File is not a valid JSON file")
        except (json.JSONDecodeError, OSError) as e:
            self.fail(f"Could not verify JSON file format: {str(e)}")

    def test_2_load_collection(self):
        collection_path = os.path.join(OUTPUT_DIR, "lfp_collection.json")
        instance1 = LFPCollection.load_collection(collection_path)
        self.assertTrue(isinstance(instance1, LFPCollection))

        self.assertFalse(hasattr(instance1, "power"))
        instance2 = LFPCollection(
            subject_to_channel_dict=self.subject_to_channel_dict,
            data_path=self.data_path,
            recording_to_subject_dict=self.recording_to_subject_dict,
            threshold=0.75,
            recording_to_behavior_dict=self.recording_to_behavior_dict,
        )
        attrs1 = vars(instance1)
        attrs2 = vars(instance2)
        self.assertEqual(set(attrs1.keys()), set(attrs2.keys()), "Instances have different attributes")
        for attr in attrs1:
            if isinstance(attrs1[attr], np.ndarray):
                np.testing.assert_array_equal(attrs1[attr], attrs2[attr], f"Values differ for attribute {attr}")
            elif isinstance(attrs1[attr], dict):
                self.assertDictEqual(attrs1[attr], attrs2[attr], f"Dictionary values differ for attribute {attr}")
            elif attr == "lfp_recordings":
                # Create dictionaries of recordings by name for each collection
                recs1 = {rec.name: rec for rec in attrs1[attr]}
                recs2 = {rec.name: rec for rec in attrs2[attr]}

                # Check that they have the same recording names
                self.assertEqual(set(recs1.keys()), set(recs2.keys()), "Collections have different recording names")

                # Compare each matching pair of recordings
                for name in recs1:
                    rec1 = recs1[name]
                    rec2 = recs2[name]
                    self.rec_tests(rec1, rec2)
            else:
                self.assertEqual(attrs1[attr], attrs2[attr], f"Values differ for attribute {attr}")

    def rec_tests(self, rec1, rec2):
        attrs1 = vars(rec1)
        attrs2 = vars(rec2)
        self.assertEqual(set(attrs1.keys()), set(attrs2.keys()), "Instances have different attributes")
        for attr in attrs1:
            if isinstance(attrs1[attr], np.ndarray):
                np.testing.assert_array_equal(attrs1[attr], attrs2[attr], f"Values differ for attribute {attr}")
            elif isinstance(attrs1[attr], dict):
                try:
                    self.assertDictEqual(attrs1[attr], attrs2[attr], f"Dictionary values differ for attribute {attr}")
                except ValueError:
                    self.compare_dicts_with_arrays(attrs1[attr], attrs2[attr], attr)
            else:
                if attr == "merged_rec_path":
                    self.assertEqual(Path(attrs1[attr]), Path(attrs2[attr]), f"Values differ for attribute {attr}")
                else:
                    self.assertEqual(attrs1[attr], attrs2[attr], f"Values differ for attribute {attr}")

    def compare_dicts_with_arrays(self, dict1, dict2, dict_name=""):
        """Compare dictionaries that might contain numpy arrays as values."""
        # Check keys match

        self.assertEqual(set(dict1.keys()), set(dict2.keys()), f"Keys differ in {dict_name}")

        # Compare values
        for key in dict1:
            val1 = dict1[key]
            val2 = dict2[key]

            if isinstance(val1, np.ndarray):
                np.testing.assert_array_equal(val1, val2, f"Array values differ for key {key} in {dict_name}")
            elif isinstance(val1, dict):
                # Recursive call for nested dictionaries
                self.compare_dicts_with_arrays(val1, val2, f"{dict_name}[{key}]")
            else:
                self.assertEqual(val1, val2, f"Values differ for key {key} in {dict_name}")


if __name__ == "__main__":
    unittest.main()
