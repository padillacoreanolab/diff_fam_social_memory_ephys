import unittest
from spike_analysis.spike_collection import SpikeCollection
from unittest.mock import patch
import io
import numpy as np


class TestSpikeCollection(unittest.TestCase):
    def test_collection(self):
        with patch("sys.stdout", new=io.StringIO()) as fake_stdout:
            test_collection = SpikeCollection(r"tests/test_data")
            test_collection.make_collection()
            # Get what was printed
            printed_output = fake_stdout.getvalue()
            # Assert the expected message is in the output
            expected_message = "test_rec_nogoodunits_merged.rec has no good units"
            self.assertIn(expected_message, printed_output)
            self.assertIn("and will not be included in the collection", printed_output)
        self.assertEqual(len(test_collection.collection), 3)

    def test_all_set_no_subjects_no_dicts(self):
        # to do create missing dictionaries, dictionaries, subjects, etc.
        with patch("sys.stdout", new=io.StringIO()) as fake_stdout:
            test_collection = SpikeCollection(r"tests/test_data")
            test_collection.analyze(timebin=50, ignore_freq=0.5)
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
            for recording in test_collection.collection:
                recording.subject = i
                i += 1
            test_collection.analyze(timebin=50, ignore_freq=0.5)
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
            for recording in test_collection.collection:
                recording.event_dict = {"event": [i, i + 1]}
                i += 1
            test_collection.analyze(timebin=50, ignore_freq=0.5)
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
            for recording in test_collection.collection:
                recording.event_dict = {f"event{i}": [i, i + 1]}
                recording.subject = i
                i += 1
            test_collection.analyze(timebin=50, ignore_freq=0.5)
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
            for recording in test_collection.collection:
                recording.event_dict = {"event": np.array([[i, i + 1]])}
                recording.subject = i
                i += 1
            test_collection.analyze(timebin=50, ignore_freq=0.5)
            # Get what was printed
            printed_output = fake_stdout.getvalue()
            # Assert the expected message is in the output
            self.assertIn("All set to analyze", printed_output)
