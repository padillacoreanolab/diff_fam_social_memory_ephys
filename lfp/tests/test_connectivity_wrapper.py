import numpy as np
import unittest
import os
import lfp_analysis.preprocessor as preprocessor
import lfp_analysis.connectivity_wrapper as connectivity_wrapper

SUBJECT_DICT = {"mPFC": 19, "vHPC": 31, "BLA": 30, "NAc": 28, "MD": 29}
SPIKE_GADGETS_MULTIPLIER = 0.6745


class test_lfp_recording_power(unittest.TestCase):
    def test_calculate_all_connectivity_small_file(self):
        traces_path = os.path.join("tests", "test_data", "test_traces.csv")
        all_traces_arr = np.loadtxt(traces_path, delimiter=",")
        rms_traces, brain_regions = preprocessor.preprocess(all_traces_arr, SUBJECT_DICT, 0.2, 0.195)

        connectivity, frequencies, power, coherence, granger = connectivity_wrapper.connectivity_wrapper(
            rms_traces, 200, 2, 1, 0.5
        )

        self.assertEqual(power.shape, (24, 100, 5))
        self.assertEqual(coherence.shape, (24, 100, 5, 5))
