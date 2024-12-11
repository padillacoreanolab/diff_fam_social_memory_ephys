import unittest
import numpy as np
import io
from unittest.mock import patch
from spike_analysis.spike_recording import EphysRecording


class TestSpikeRecording(unittest.TestCase):
    def test_cluster_dict(self):
        data_path = r"tests/test_data/test_recording_merged.rec/phy"
        test_recording = EphysRecording(data_path)
        try:
            test_recording.labels_dict
            self.assertIsInstance(test_recording.labels_dict, dict)
        except NameError:
            self.fail("Dictionary does not exist")
        test_dict = {"key": "value"}
        self.assertGreater(len(test_dict), 0)
        for key in test_recording.labels_dict.keys():
            self.assertIsInstance(key, str, f"Key {key} is not a string")
        self.assertEqual(test_recording.labels_dict["2"], "good")
        self.assertEqual(test_recording.labels_dict["234"], "mua")
        self.assertEqual(test_recording.labels_dict["105"], "noise")

    def test_delete_noise(self):
        data_path = r"tests/test_data/test_recording_merged.rec/phy"
        test_recording = EphysRecording(data_path)
        self.assertIsInstance(test_recording.unit_array, np.ndarray)
        self.assertNotIn("224", test_recording.unit_array)

    def test_unsorted_clusters(self):
        data_path = r"tests/test_data/test_recording_merged.rec/phy"
        test_recording = EphysRecording(data_path)
        self.assertNotIn("169", test_recording.labels_dict)
        with patch("sys.stdout", new=io.StringIO()) as fake_stdout:
            test_recording.get_spike_specs()
            # Get what was printed
            printed_output = fake_stdout.getvalue()
            # Assert the expected message is in the output
            expected_message = "Unit 169 is unsorted & has 88 spikes"
            self.assertIn(expected_message, printed_output)
            self.assertIn("Unit 169 will be deleted", printed_output)

    def test_unit_dict(self):
        data_path = r"tests/test_data/test_recording_merged.rec/phy"
        test_recording = EphysRecording(data_path)
        self.assertEqual(len(test_recording.unit_timestamps.keys()), 26)
        for key, value in test_recording.unit_timestamps.items():
            self.assertIsInstance(key, int, f"Key {key} is not an integer")
            self.assertIsInstance(value, np.ndarray, f"Key {key} is not an integer")
