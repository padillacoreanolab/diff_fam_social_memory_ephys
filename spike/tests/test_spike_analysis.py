import unittest
import numpy as np
from spike_analysis.spike_analysis import SpikeAnalysis
from unittest.mock import patch
from spike_analysis.spike_collection import SpikeCollection
import io


class TestSpikeAnalyis(unittest.TestCase):
    def test_all_set_no_subjects_no_dicts(self):
        # to do create missing dictionaries, dictionaries, subjects, etc.
        with patch("sys.stdout", new=io.StringIO()) as fake_stdout:
            test_collection = SpikeCollection(r"tests/test_data")
            SpikeAnalysis(test_collection, timebin=50, ignore_freq=0.5)
            # Get what was printed
            printed_output = fake_stdout.getvalue()
            # Assert the expected message is in the output
            expected_message = "These recordings are missing subjects: ['test_rec2_merged.rec',"
            self.assertIn(expected_message, printed_output)
            self.assertIn("These recordings are missing event dictionaries:", printed_output)
            self.assertNotIn("All set to analyze", printed_output)

    def test_all_set_no_dicts(self):
        with patch("sys.stdout", new=io.StringIO()) as fake_stdout:
            test_collection = SpikeCollection(r"tests/test_data")
            i = 0
            for recording in test_collection.collection.values():
                recording.subject = i
                i += 1
            SpikeAnalysis(test_collection, timebin=50, ignore_freq=0.5)
            # Get what was printed
            printed_output = fake_stdout.getvalue()
            # Assert the expected message is in the output
            expected_message = "These recordings are missing subjects"
            self.assertNotIn(expected_message, printed_output)
            self.assertIn("These recordings are missing event dictionaries:", printed_output)
            self.assertNotIn("All set to analyze", printed_output)

    def test_all_set_no_subjects(self):
        with patch("sys.stdout", new=io.StringIO()) as fake_stdout:
            test_collection = SpikeCollection(r"tests/test_data")
            i = 0
            for recording in test_collection.collection.values():
                recording.event_dict = {"event": [i, i + 1]}
                i += 1
            SpikeAnalysis(test_collection, timebin=50, ignore_freq=0.5)
            # Get what was printed
            printed_output = fake_stdout.getvalue()
            # Assert the expected message is in the output
            expected_message = "These recordings are missing subjects"
            self.assertIn(expected_message, printed_output)
            self.assertNotIn("These recordings are missing event dictionaries:", printed_output)
            self.assertIn("Event arrays are not 2 dimensional numpy arrays of shape (n x 2).", printed_output)
            self.assertNotIn("All set to analyze", printed_output)

    def test_all_set_diff_events(self):
        with patch("sys.stdout", new=io.StringIO()) as fake_stdout:
            test_collection = SpikeCollection(r"tests/test_data")
            i = 0
            for recording in test_collection.collection.values():
                recording.event_dict = {f"event{i}": [i, i + 1]}
                recording.subject = i
                i += 1
            SpikeAnalysis(test_collection, timebin=50, ignore_freq=0.5)
            # Get what was printed
            printed_output = fake_stdout.getvalue()
            # Assert the expected message is in the output
            expected_message = "These recordings are missing subjects"
            self.assertNotIn(expected_message, printed_output)
            self.assertNotIn("These recordings are missing event dictionaries:", printed_output)
            self.assertIn("Your event dictionary keys are different across recordings.", printed_output)
            self.assertIn("Event arrays are not 2 dimensional numpy arrays of shape (n x 2).", printed_output)
            self.assertNotIn("All set to analyze", printed_output)

    def test_all_good(self):
        with patch("sys.stdout", new=io.StringIO()) as fake_stdout:
            test_collection = SpikeCollection(r"tests/test_data")
            i = 0
            for recording in test_collection.collection.values():
                recording.event_dict = {"event": np.array([[i, i + 1]])}
                recording.subject = i
                i += 1
            SpikeAnalysis(test_collection, timebin=50, ignore_freq=0.5)
            # Get what was printed
            printed_output = fake_stdout.getvalue()
            # Assert the expected message is in the output
            self.assertIn("All set to analyze", printed_output)

    def test_unit_keys(self):
        with patch("sys.stdout", new=io.StringIO()) as fake_stdout:
            test_collection = SpikeCollection(r"tests/test_data")
            i = 0
            for recording in test_collection.collection.values():
                recording.event_dict = {"event": [i, i + 1]}
                recording.subject = i
                i += 1
            SpikeAnalysis(test_collection, timebin=50, ignore_freq=0.5)
            # Get what was printed
            printed_output = fake_stdout.getvalue()
            # Assert the expected message is in the output
            self.assertIn("All set to analyze", printed_output)

    def test_freq_dict(self):
        with patch("sys.stdout", new=io.StringIO()):
            test_collection = SpikeCollection(r"tests/test_data")
            i = 0
            for name, recording in test_collection.collection.items():
                recording.event_dict = {"event": [i, i + 1]}
                recording.subject = i
                i += 1
            SpikeAnalysis(test_collection, timebin=50, ignore_freq=0.5)
            for key in recording.freq_dict.keys():
                self.assertIsInstance(key, str, f"Key {key} is not a string in freq_dict")
                if name == "test_recording_fewgoodunits_merged.rec":
                    self.assertEqual(len(recording.freq_dict.keys()), 2)
                    self.assertEqual(recording.freq_dict["3"], 2.74)

    def test_whole_spiketrain(self):
        with patch("sys.stdout", new=io.StringIO()):
            test_collection = SpikeCollection(r"tests/test_data")
            i = 0
            for name, recording in test_collection.collection.items():
                recording.event_dict = {"event": [i, i + 1]}
                recording.subject = i
                i += 1
            SpikeAnalysis(test_collection, timebin=50, ignore_freq=0.5)
            self.assertEqual(len(recording.spiketrain), 46833)

    def test_unit_spiketrain(self):
        with patch("sys.stdout", new=io.StringIO()):
            test_collection = SpikeCollection(r"tests/test_data")
            i = 0
            for name, recording in test_collection.collection.items():
                recording.event_dict = {"event": [i, i + 1]}
                recording.subject = i
                i += 1
            SpikeAnalysis(test_collection, timebin=50, ignore_freq=0.5)
            for key, value in recording.unit_spiketrains.items():
                self.assertIsInstance(key, str, f"Key {key} is not a string in unit_spiketrains")
                self.assertIsInstance(value, np.ndarray)
                if name == "test_recording_fewgoodunits_merged.rec":
                    self.assertEqual(len(recording.unit_spiketrains.keys()), 2)
                    self.assertEqual(len(recording.unit_spiketrains["3"], 46833))

    def test_unit_firingrates(self):
        with patch("sys.stdout", new=io.StringIO()):
            test_collection = SpikeCollection(r"tests/test_data")
            i = 0
            for name, recording in test_collection.collection.items():
                recording.event_dict = {"event": [i, i + 1]}
                recording.subject = i
                i += 1
            SpikeAnalysis(test_collection, timebin=50, ignore_freq=0.5)
            for key, value in recording.unit_firing_rates.items():
                self.assertIsInstance(key, str, f"Key {key} is not a string in unit_firing_rates")
                self.assertIsInstance(value, np.ndarray)
                if name == "test_recording_fewgoodunits_merged.rec":
                    self.assertEqual(len(recording.unit_firing_rates.keys()), 2)
                    self.assertEqual(len(recording.unit_firing_rates["3"]), 46833)
                    self.assertEqual(recording.unit_firing_rates["3"][0], 0)
                    self.assertEqual(recording.unit_firing_rate_array.shape, (46833, 2))

    def test_event_snippets(self):
        test_collection = SpikeCollection(r"tests/test_data")
        i = 0
        for recording in test_collection.collection.values():
            recording.event_dict = {"event": np.array([[0, 2000], [3000, 8000], [2338650, 2341650]])}
            recording.subject = i
            i += 1
        test_analysis = SpikeAnalysis(test_collection, timebin=50, ignore_freq=0.5)
        for recording in test_analysis.ephyscollection.collection.values():
            whole_rec = recording.unit_firing_rate_array
            event_snippets = test_analysis.__get_event_snippets__(recording, "event", whole_rec, 2)
            self.assertEqual(len(event_snippets), 3)
            self.assertEqual(len(event_snippets[0]), 40)

    def test_early_event_snippets(self):
        test_collection = SpikeCollection(r"tests/test_data")
        i = 0
        for recording in test_collection.collection.values():
            recording.event_dict = {"event": np.array([[0, 2000], [3000, 8000], [44800, 46820]])}
            recording.subject = i
            i += 1
        test_analysis = SpikeAnalysis(test_collection, timebin=50, ignore_freq=0.5)
        # test for events that start before the recording bc of prewindow get cut
        for recording in test_analysis.ephyscollection.collection.values():
            whole_rec = recording.unit_firing_rate_array
            event_snippets = test_analysis.__get_event_snippets__(recording, "event", whole_rec, 2, 1)
            self.assertEqual(len(event_snippets), 2)
            self.assertEqual(len(event_snippets[1]), 60)

    def test_late_event_snippets(self):
        test_collection = SpikeCollection(r"tests/test_data")
        i = 0
        for recording in test_collection.collection.values():
            recording.event_dict = {"event": np.array([[0, 2000], [3000, 8000], [44800, 46820]])}
            recording.subject = i
            i += 1
        test_analysis = SpikeAnalysis(test_collection, timebin=50, ignore_freq=0.5)
        # test that events that extend past the end of the recording get cut
        for name, recording in test_analysis.ephyscollection.collection.items():
            if name == "test_recording_fewgoodunits_merged.rec":
                whole_rec = recording.unit_firing_rate_array
                event_snippets = test_analysis.__get_event_snippets__(recording, "event", whole_rec, 4)
                self.assertEqual(len(event_snippets), 2)
                self.assertEqual(len(event_snippets[0]), 80)

    # def test_event_snippets_with_Nans(self):

    # def test_wilcox_baseline_v_event_stats(self):
