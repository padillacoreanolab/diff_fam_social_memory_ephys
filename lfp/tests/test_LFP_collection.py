import unittest
from lfp.lfp_analysis.LFP_collection import LFPCollection
from lfp.tests.utils import EXAMPLE_RECORDING_DIR


class TestLFPCollection(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.recording_to_behavior_dict = {"recording1": "behavior1", "recording2": "behavior2"}
        self.subject_to_channel_dict = {
            "subject1": {"channel1": 1, "channel2": 2},
            "subject2": {"channel1": 3, "channel2": 4},
        }
        self.data_path = EXAMPLE_RECORDING_DIR

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
            self.recording_to_behavior_dict,
            self.subject_to_channel_dict,
            self.data_path,
            {},
            4,
            "testing_trodes_directory/",
            **custom_params,
        )

        self.assertIsNotNone(lfp_collection.kwargs)

    def test_invalid_input(self):
        """Test LFPCollection initialization with invalid input"""
        with self.assertRaises(TypeError):
            LFPCollection(None, None, None)

    def test_make_recordings(self):
        """Test LFPCollection make_recordings method"""
        lfp_collection = LFPCollection(
            self.recording_to_behavior_dict,
            self.subject_to_channel_dict,
            self.data_path,
            {},
            0.75,
            "testing_trodes_directory/",
        )


if __name__ == "__main__":
    unittest.main()
