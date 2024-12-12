import unittest
from spike_analysis.spike_collection import SpikeCollection
from unittest.mock import patch
import io


class TestSpikeCollection(unittest.TestCase):
    def test_collection(self):
        test_collection = SpikeCollection("./test_data")
        with patch("sys.stdout", new=io.StringIO()) as fake_stdout:
            test_collection.make_collection()
            # Get what was printed
            printed_output = fake_stdout.getvalue()
            # Assert the expected message is in the output
            expected_message = "test_rec_nogoodunits_merged.rec has no good units"
            self.assertIn(expected_message, printed_output)
            self.assertIn("and will not be included in the collection", printed_output)
