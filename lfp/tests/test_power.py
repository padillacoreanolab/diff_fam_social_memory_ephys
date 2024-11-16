import numpy as np
import unittest
import os
import lfp_analysis.preprocessor as preprocessor

SUBJECT_DICT = {"mPFC": 19, "vHPC": 31, "BLA": 30, "NAc": 28, "MD": 29}
SPIKE_GADGETS_MULTIPLIER = 0.6745


class test_lfp_recording_power(unittest.TestCase):
    def test_calculate_power(self):
        traces_path = os.path.join("tests", "test_data", "test_traces.csv")
        all_traces_arr = np.loadtxt(traces_path, delimiter=",")
        power, frequencies = preprocessor.preprocess(all_traces_arr, SUBJECT_DICT, 0.2, 0.195)
        self.assertEqual(power.shape, (5, 2500))
