import unittest
import numpy as np
import io
from unittest.mock import patch
from spike_analysis.spike_recording import SpikeRecording


class TestSpikeRecording(unittest.TestCase):
    def test_cluster_dict(self):
        with patch("sys.stdout", new=io.StringIO()):
            data_path = r"tests/test_data/test_recording_merged.rec/phy"
            test_recording = SpikeRecording(data_path)
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
        with patch("sys.stdout", new=io.StringIO()):
            data_path = r"tests/test_data/test_recording_merged.rec/phy"
            test_recording = SpikeRecording(data_path)
            self.assertIsInstance(test_recording.unit_array, np.ndarray)
            self.assertNotIn("224", test_recording.unit_array)

    def test_unsorted_clusters(self):
        with patch("sys.stdout", new=io.StringIO()) as fake_stdout:
            data_path = r"tests/test_data/test_recording_merged.rec/phy"
            test_recording = SpikeRecording(data_path)
            self.assertNotIn("169", test_recording.labels_dict)
            test_recording.spike_specs()
            # Get what was printed
            printed_output = fake_stdout.getvalue()
            # Assert the expected message is in the output
            expected_message = "Unit 169 is unsorted & has 88 spikes"
            self.assertIn(expected_message, printed_output)
            self.assertIn("Unit 169 will be deleted", printed_output)

    def test_unit_dict(self):
        with patch("sys.stdout", new=io.StringIO()):
            data_path = r"tests/test_data/test_recording_merged.rec/phy"
            test_recording = SpikeRecording(data_path)
            self.assertEqual(len(test_recording.unit_timestamps.keys()), 26)
            for key, value in test_recording.unit_timestamps.items():
                self.assertIsInstance(key, str, f"Key {key} is not a string")
                self.assertIsInstance(value, np.ndarray, f"Key {key} is not a numpy array")
            self.assertEqual(len(test_recording.unit_timestamps["23"]), 25339)

    # def test_analyze(self):
    # def test_check(self):
    # rewrite for recordings not for collection
    # def test_unit_keys(self):
    #     with patch("sys.stdout", new=io.StringIO()) as fake_stdout:
    #         test_collection = SpikeCollection(r"tests/test_data")
    #         i = 0
    #         for recording in test_collection.collection.values():
    #             recording.event_dict = {"event": [i, i + 1]}
    #             recording.subject = i
    #             i += 1
    #         test_collection.analyze(timebin=50, ignore_freq=0.5)
    #         # Get what was printed
    #         printed_output = fake_stdout.getvalue()
    #         # Assert the expected message is in the output
    #         self.assertIn("All set to analyze", printed_output)

    # def test_freq_dict(self):
    #     with patch("sys.stdout", new=io.StringIO()):
    #         test_collection = SpikeCollection(r"tests/test_data")
    #         i = 0
    #         for name, recording in test_collection.collection.items():
    #             recording.event_dict = {"event": [i, i + 1]}
    #             recording.subject = i
    #             i += 1
    #         test_collection.analyze(timebin=50, ignore_freq=0.5)
    #         for key in recording.freq_dict.keys():
    #             self.assertIsInstance(key, str, f"Key {key} is not a string in freq_dict")
    #             if name == "test_recording_fewgoodunits_merged.rec":
    #                 self.assertEqual(len(recording.freq_dict.keys()), 2)
    #                 self.assertEqual(recording.freq_dict["3"], 2.74)

    # def test_whole_spiketrain(self):
    #     with patch("sys.stdout", new=io.StringIO()):
    #         test_collection = SpikeCollection(r"tests/test_data")
    #         i = 0
    #         for name, recording in test_collection.collection.items():
    #             recording.event_dict = {"event": [i, i + 1]}
    #             recording.subject = i
    #             i += 1
    #         test_collection.analyze(timebin=50, ignore_freq=0.5)
    #         self.assertEqual(len(recording.spiketrain), 46833)

    # def test_unit_spiketrain(self):
    #     with patch("sys.stdout", new=io.StringIO()):
    #         test_collection = SpikeCollection(r"tests/test_data")
    #         i = 0
    #         for name, recording in test_collection.collection.items():
    #             recording.event_dict = {"event": [i, i + 1]}
    #             recording.subject = i
    #             i += 1
    #         test_collection.analyze(timebin=50, ignore_freq=0.5)
    #         for key, value in recording.unit_spiketrains.items():
    #             self.assertIsInstance(key, str, f"Key {key} is not a string in unit_spiketrains")
    #             self.assertIsInstance(value, np.ndarray)
    #             if name == "test_recording_fewgoodunits_merged.rec":
    #                 self.assertEqual(len(recording.unit_spiketrains.keys()), 2)
    #                 self.assertEqual(len(recording.unit_spiketrains["3"], 46833))

    # def test_unit_firingrates(self):
    #     with patch("sys.stdout", new=io.StringIO()):
    #         test_collection = SpikeCollection(r"tests/test_data")
    #         i = 0
    #         for name, recording in test_collection.collection.items():
    #             recording.event_dict = {"event": [i, i + 1]}
    #             recording.subject = i
    #             i += 1
    #         test_collection.analyze(timebin=50, ignore_freq=0.5)
    #         for key, value in recording.unit_firing_rates.items():
    #             self.assertIsInstance(key, str, f"Key {key} is not a string in unit_firing_rates")
    #             self.assertIsInstance(value, np.ndarray)
    #             if name == "test_recording_fewgoodunits_merged.rec":
    #                 self.assertEqual(len(recording.unit_firing_rates.keys()), 2)
    #                 self.assertEqual(len(recording.unit_firing_rates["3"]), 46833)
    #                 self.assertEqual(recording.unit_firing_rates["3"][0], 0)
    #                 self.assertEqual(recording.unit_firing_rate_array.shape, (46833, 2))
    def test_event_snippets(self):
        data_path = r"tests/test_data/test_recording_merged.rec/phy"
        recording = SpikeRecording(data_path)
        recording.event_dict = {"event": np.array([[0, 2000], [3000, 8000], [2338650, 2341650]])}
        recording.subject = 1
        recording.analyze(timebin=50, ignore_freq=0.5)
        whole_rec = recording.unit_firing_rate_array
        event_snippets = recording.__event_snippets__("event", whole_rec, 2)
        self.assertEqual(len(event_snippets), 3)
        self.assertEqual(len(event_snippets[0]), 40)

    def test_early_event_snippets(self):
        data_path = r"tests/test_data/test_recording_merged.rec/phy"
        recording = SpikeRecording(data_path)
        recording.event_dict = {"event": np.array([[0, 2000], [3000, 8000], [2338650, 2341650]])}
        recording.subject = 1
        recording.analyze(timebin=50, ignore_freq=0.5)
        whole_rec = recording.unit_firing_rate_array
        event_snippets = recording.__event_snippets__("event", whole_rec, 2, 1)
        self.assertEqual(len(event_snippets), 2)
        self.assertEqual(len(event_snippets[1]), 60)

    def test_late_event_snippets(self):
        data_path = r"tests/test_data/test_rec_fewgoodunits_merged.rec/phy"
        recording = SpikeRecording(data_path)
        recording.event_dict = {"event": np.array([[0, 2000], [3000, 8000], [2338650, 2341650]])}
        recording.subject = 1
        recording.analyze(timebin=50, ignore_freq=0.5)
        whole_rec = recording.unit_firing_rate_array
        event_snippets = recording.__event_snippets__("event", whole_rec, 4)
        self.assertEqual(len(event_snippets), 2)
        self.assertEqual(len(event_snippets[0]), 80)
